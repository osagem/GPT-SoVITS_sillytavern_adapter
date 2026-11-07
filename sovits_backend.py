import asyncio
import logging
import os
import threading
from typing import Any, AsyncIterator, Dict, NamedTuple, Optional, Tuple
from urllib.parse import urlparse

import httpx


logger = logging.getLogger("Backend")


class BackendError(Exception):
    def __init__(self, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class AudioResult(NamedTuple):
    content: bytes
    media_type: str


LANGUAGE_MAP = {
    "中文": "中文",
    "英文": "英文",
    "日文": "日文",
    "粤语": "粤语",
    "韩文": "韩文",
    "中英混合": "中英混合",
    "日英混合": "日英混合",
    "粤英混合": "粤英混合",
    "韩英混合": "韩英混合",
    "多语种混合": "多语种混合",
    "多语种混合(粤语)": "多语种混合(粤语)",
    "zh": "中英混合",
    "all_zh": "中文",
    "en": "英文",
    "ja": "日英混合",
    "all_ja": "日文",
    "yue": "粤英混合",
    "all_yue": "粤语",
    "ko": "韩英混合",
    "all_ko": "韩文",
    "auto": "多语种混合",
    "auto_yue": "多语种混合(粤语)",
}

CUT_METHOD_MAP = {
    "cut0": "不切",
    "cut1": "凑四句一切",
    "cut2": "凑50字一切",
    "cut3": "按中文句号。切",
    "cut4": "按英文句号.切",
    "cut5": "按标点符号切",
    "不切": "不切",
    "凑四句一切": "凑四句一切",
    "凑50字一切": "凑50字一切",
    "按中文句号。切": "按中文句号。切",
    "按英文句号.切": "按英文句号.切",
    "按标点符号切": "按标点符号切",
}


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def response_media_type(response: httpx.Response, fallback: str = "audio/wav") -> str:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    return content_type or fallback


def upstream_error(response: httpx.Response) -> BackendError:
    status_code = response.status_code if 400 <= response.status_code < 500 else 502
    detail = response.text[:2000]
    return BackendError(f"GPT-SoVITS 后端返回 {response.status_code}: {detail}", status_code)


def get_api_client_kwargs(base_url: str) -> Dict[str, Any]:
    """Disable proxies for localhost and the configured backend host."""
    mounts = {
        "http://127.0.0.1": httpx.AsyncHTTPTransport(proxy=None),
        "http://localhost": httpx.AsyncHTTPTransport(proxy=None),
        "https://127.0.0.1": httpx.AsyncHTTPTransport(proxy=None),
        "https://localhost": httpx.AsyncHTTPTransport(proxy=None),
    }
    try:
        parsed = urlparse(base_url)
        if parsed.hostname and parsed.hostname not in ["localhost", "127.0.0.1"]:
            scheme = parsed.scheme or "http"
            host = parsed.hostname
            host_prefix = f"{scheme}://{host}:{parsed.port}" if parsed.port else f"{scheme}://{host}"
            mounts[host_prefix] = httpx.AsyncHTTPTransport(proxy=None)
    except Exception as exc:
        logger.warning("解析后端地址失败，将仅使用默认本地代理绕过规则: %s", exc)
    return {
        "timeout": httpx.Timeout(120.0, connect=10.0),
        "mounts": mounts,
        "trust_env": True,
    }


class BaseBackend:
    serialize_requests = False

    def __init__(self, base_url: str):
        self.base_url = normalize_base_url(base_url)
        self.request_lock = asyncio.Lock()

    async def set_gpt_weights(self, weights_path: str) -> None:
        raise NotImplementedError

    async def set_sovits_weights(
        self, weights_path: str, prompt_lang: str = "zh", text_lang: str = "zh"
    ) -> None:
        raise NotImplementedError

    async def synthesize(self, request_data: Dict[str, Any]) -> AudioResult:
        raise NotImplementedError

    async def stream(self, request_data: Dict[str, Any]) -> Tuple[AsyncIterator[bytes], str]:
        raise NotImplementedError


class ApiV2Backend(BaseBackend):
    def __init__(self, base_url: str):
        super().__init__(base_url)
        self.client_kwargs = get_api_client_kwargs(self.base_url)

    async def _set_weights(self, endpoint: str, weights_path: str) -> None:
        try:
            async with httpx.AsyncClient(**self.client_kwargs) as client:
                response = await client.get(
                    f"{self.base_url}/{endpoint}", params={"weights_path": weights_path}
                )
        except (httpx.HTTPError, OSError) as exc:
            raise BackendError(f"连接 GPT-SoVITS 权重接口失败: {exc}", 502) from exc
        if response.status_code != 200:
            raise upstream_error(response)

    async def set_gpt_weights(self, weights_path: str) -> None:
        await self._set_weights("set_gpt_weights", weights_path)

    async def set_sovits_weights(
        self, weights_path: str, prompt_lang: str = "zh", text_lang: str = "zh"
    ) -> None:
        await self._set_weights("set_sovits_weights", weights_path)

    async def synthesize(self, request_data: Dict[str, Any]) -> AudioResult:
        try:
            async with httpx.AsyncClient(**self.client_kwargs) as client:
                response = await client.post(f"{self.base_url}/tts", json=request_data)
        except (httpx.HTTPError, OSError) as exc:
            raise BackendError(f"连接 GPT-SoVITS /tts 失败: {exc}", 502) from exc
        if response.status_code != 200:
            raise upstream_error(response)
        return AudioResult(response.content, response_media_type(response))

    async def stream(self, request_data: Dict[str, Any]) -> Tuple[AsyncIterator[bytes], str]:
        client = httpx.AsyncClient(**self.client_kwargs)
        stream_context = client.stream("POST", f"{self.base_url}/tts", json=request_data)
        try:
            response = await stream_context.__aenter__()
        except (httpx.HTTPError, OSError) as exc:
            await client.aclose()
            raise BackendError(f"连接 GPT-SoVITS /tts 流式接口失败: {exc}", 502) from exc
        except Exception:
            await client.aclose()
            raise
        if response.status_code != 200:
            try:
                await response.aread()
                error = upstream_error(response)
            finally:
                await stream_context.__aexit__(None, None, None)
                await client.aclose()
            raise error

        async def byte_stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await stream_context.__aexit__(None, None, None)
                await client.aclose()

        return byte_stream(), response_media_type(response)


class GradioBackend(BaseBackend):
    serialize_requests = True

    def __init__(
        self,
        base_url: str,
        output_dir: str,
        use_cuda_graph: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__(base_url)
        self.output_dir = output_dir
        self.use_cuda_graph = use_cuda_graph
        self.use_flash_attention = use_flash_attention
        self._client = None
        self._client_init_lock = threading.Lock()

    @staticmethod
    def map_language(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise BackendError("语言代码不能为空", 400)
        mapped = LANGUAGE_MAP.get(value) or LANGUAGE_MAP.get(value.lower())
        if mapped is None:
            raise BackendError(f"Gradio WebUI 不支持语言代码: {value}", 400)
        return mapped

    @staticmethod
    def map_cut_method(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise BackendError("文本切分方式不能为空", 400)
        mapped = CUT_METHOD_MAP.get(value) or CUT_METHOD_MAP.get(value.lower())
        if mapped is None:
            raise BackendError(f"Gradio WebUI 不支持文本切分方式: {value}", 400)
        return mapped

    def build_tts_args(self, request_data: Dict[str, Any]):
        from gradio_client import handle_file

        ref_audio_path = request_data.get("ref_audio_path", "")
        if not ref_audio_path or not os.path.isfile(ref_audio_path):
            raise BackendError(f"参考音频不存在: {ref_audio_path}", 400)
        prompt_text = (request_data.get("prompt_text") or "").strip()
        if not prompt_text:
            raise BackendError("Gradio/v3/v4/v5 模式需要有效的同名参考文本 (.txt)", 400)
        text = (request_data.get("text") or "").strip()
        if not text:
            raise BackendError("需要合成的文本不能为空", 400)

        sample_steps = request_data.get("sample_steps")
        cfg_rate = request_data.get("cfg_rate")
        return (
            handle_file(ref_audio_path),
            prompt_text,
            self.map_language(request_data.get("prompt_lang")),
            text,
            self.map_language(request_data.get("text_lang")),
            self.map_cut_method(request_data.get("text_split_method", "cut5")),
            int(request_data.get("top_k", 15)),
            float(request_data.get("top_p", 1)),
            float(request_data.get("temperature", 1)),
            False,
            float(request_data.get("speed_factor", 1.0)),
            False,
            None,
            None if sample_steps is None else int(sample_steps),
            bool(request_data.get("super_sampling", False)),
            float(request_data.get("fragment_interval", 0.3)),
            self.use_cuda_graph,
            self.use_flash_attention,
            None if cfg_rate is None else float(cfg_rate),
        )

    def _get_client(self):
        if self._client is None:
            with self._client_init_lock:
                if self._client is None:
                    from gradio_client import Client

                    os.makedirs(self.output_dir, exist_ok=True)
                    self._client = Client(
                        self.base_url,
                        verbose=False,
                        download_files=self.output_dir,
                        httpx_kwargs={
                            "timeout": httpx.Timeout(120.0, connect=10.0),
                            "trust_env": False,
                        },
                    )
        return self._client

    def _predict_sync(self, api_name: str, args):
        return self._get_client().predict(*args, api_name=api_name)

    async def _run_sync(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args))

    async def _predict(self, api_name: str, args):
        try:
            return await self._run_sync(self._predict_sync, api_name, args)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Gradio 调用 {api_name} 失败: {exc}", 502) from exc

    async def set_gpt_weights(self, weights_path: str) -> None:
        await self._predict("/change_gpt_weights", (weights_path,))

    async def set_sovits_weights(
        self, weights_path: str, prompt_lang: str = "zh", text_lang: str = "zh"
    ) -> None:
        await self._predict(
            "/change_sovits_weights",
            (weights_path, self.map_language(prompt_lang), self.map_language(text_lang)),
        )

    @staticmethod
    def _result_path(result: Any) -> str:
        if isinstance(result, (tuple, list)) and len(result) == 1:
            result = result[0]
        if isinstance(result, dict):
            result = result.get("path")
        elif not isinstance(result, (str, os.PathLike)) and hasattr(result, "path"):
            result = result.path
        if not isinstance(result, (str, os.PathLike)):
            raise BackendError("Gradio 未返回可识别的音频文件路径", 502)
        return os.fspath(result)

    def _read_audio_result(self, result: Any) -> bytes:
        path = self._result_path(result)
        if not os.path.isfile(path):
            raise BackendError(f"Gradio 输出音频不存在: {path}", 502)
        with open(path, "rb") as file:
            content = file.read()
        if len(content) < 12 or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
            raise BackendError("Gradio 输出不是有效的 WAV 音频", 502)
        return content

    async def synthesize(self, request_data: Dict[str, Any]) -> AudioResult:
        args = self.build_tts_args(request_data)
        result = await self._predict("/get_tts_wav", args)
        try:
            content = await self._run_sync(self._read_audio_result, result)
        except BackendError:
            raise
        except OSError as exc:
            raise BackendError(f"读取 Gradio 输出音频失败: {exc}", 502) from exc
        return AudioResult(content, "audio/wav")

    async def stream(self, request_data: Dict[str, Any]) -> Tuple[AsyncIterator[bytes], str]:
        result = await self.synthesize(request_data)

        async def byte_stream() -> AsyncIterator[bytes]:
            yield result.content

        return byte_stream(), result.media_type


def create_backend(
    mode: str,
    base_url: str,
    output_dir: str,
    use_cuda_graph: bool = True,
    use_flash_attention: bool = True,
) -> BaseBackend:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "api_v2":
        return ApiV2Backend(base_url)
    if normalized_mode == "gradio":
        return GradioBackend(base_url, output_dir, use_cuda_graph, use_flash_attention)
    raise ValueError(f"未知 BACKEND_MODE: {mode}，仅支持 gradio 或 api_v2")
