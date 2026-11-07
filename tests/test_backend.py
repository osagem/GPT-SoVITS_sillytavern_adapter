import asyncio
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from sovits_backend import (  # noqa: E402
    ApiV2Backend,
    AudioResult,
    BackendError,
    GradioBackend,
    create_backend,
)


TEST_ARTIFACT_ROOT = PROJECT_ROOT / "output"
TEST_ARTIFACT_ROOT.mkdir(exist_ok=True)


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(32000)
        wav_file.writeframes(b"\x00\x00" * 320)


class FakeGradioClient:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.calls = []

    def predict(self, *args, api_name):
        self.calls.append((api_name, args))
        if api_name == "/get_tts_wav":
            output_path = self.output_dir / "fake-gradio.wav"
            write_wav(output_path)
            return str(output_path)
        return None


class GradioMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=TEST_ARTIFACT_ROOT)
        self.root = Path(self.temp_dir.name)
        self.reference = self.root / "character.wav"
        write_wav(self.reference)
        self.backend = GradioBackend("http://127.0.0.1:9872/", str(self.root))

    def tearDown(self):
        self.temp_dir.cleanup()

    def request_data(self, **overrides):
        data = {
            "text": "测试文本",
            "text_lang": "zh",
            "ref_audio_path": str(self.reference),
            "prompt_lang": "all_ja",
            "prompt_text": "参考文本",
            "text_split_method": "cut5",
            "top_k": 15,
            "top_p": 1,
            "temperature": 1,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "sample_steps": None,
            "cfg_rate": None,
            "super_sampling": False,
        }
        data.update(overrides)
        return data

    def test_builds_19_arguments_in_webui_order(self):
        args = self.backend.build_tts_args(self.request_data())
        self.assertEqual(len(args), 19)
        self.assertEqual(args[1], "参考文本")
        self.assertEqual(args[2], "日文")
        self.assertEqual(args[3], "测试文本")
        self.assertEqual(args[4], "中英混合")
        self.assertEqual(args[5], "按标点符号切")
        self.assertEqual(args[9], False)
        self.assertIsNone(args[12])
        self.assertIsNone(args[13])
        self.assertIsNone(args[18])

    def test_language_and_cut_mappings(self):
        expected_languages = {
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
        for source, target in expected_languages.items():
            self.assertEqual(self.backend.map_language(source), target)
        for index, target in enumerate(["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"]):
            self.assertEqual(self.backend.map_cut_method(f"cut{index}"), target)

    def test_rejects_unknown_language_and_cut_method(self):
        with self.assertRaises(BackendError) as language_error:
            self.backend.build_tts_args(self.request_data(text_lang="unknown"))
        self.assertEqual(language_error.exception.status_code, 400)
        with self.assertRaises(BackendError) as cut_error:
            self.backend.build_tts_args(self.request_data(text_split_method="unknown"))
        self.assertEqual(cut_error.exception.status_code, 400)

    def test_requires_reference_file_text_and_target_text(self):
        with self.assertRaises(BackendError):
            self.backend.build_tts_args(self.request_data(ref_audio_path="missing.wav"))
        with self.assertRaises(BackendError):
            self.backend.build_tts_args(self.request_data(prompt_text=""))
        with self.assertRaises(BackendError):
            self.backend.build_tts_args(self.request_data(text=" "))

    def test_factory_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            create_backend("unknown", "http://127.0.0.1:9872", str(self.root))


class GradioAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=TEST_ARTIFACT_ROOT)
        self.root = Path(self.temp_dir.name)
        self.reference = self.root / "character.wav"
        write_wav(self.reference)
        self.backend = GradioBackend("http://127.0.0.1:9872", str(self.root))
        self.fake_client = FakeGradioClient(self.root)
        self.backend._client = self.fake_client
        self.request_data = {
            "text": "测试文本",
            "text_lang": "zh",
            "ref_audio_path": str(self.reference),
            "prompt_lang": "zh",
            "prompt_text": "参考文本",
            "text_split_method": "cut0",
        }

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def test_synthesize_downloads_and_returns_wav(self):
        result = await self.backend.synthesize(self.request_data)
        self.assertEqual(result.media_type, "audio/wav")
        self.assertTrue(result.content.startswith(b"RIFF"))
        self.assertEqual(self.fake_client.calls[0][0], "/get_tts_wav")
        self.assertEqual(len(self.fake_client.calls[0][1]), 19)

    async def test_stream_returns_one_complete_chunk(self):
        stream, media_type = await self.backend.stream(self.request_data)
        chunks = [chunk async for chunk in stream]
        self.assertEqual(media_type, "audio/wav")
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].startswith(b"RIFF"))

    async def test_weight_switch_uses_gradio_endpoints_and_mapped_languages(self):
        await self.backend.set_gpt_weights("gpt.ckpt")
        await self.backend.set_sovits_weights("sovits.pth", "all_zh", "ja")
        self.assertEqual(self.fake_client.calls[0], ("/change_gpt_weights", ("gpt.ckpt",)))
        self.assertEqual(
            self.fake_client.calls[1],
            ("/change_sovits_weights", ("sovits.pth", "中文", "日英混合")),
        )

    async def test_invalid_gradio_output_is_rejected(self):
        class InvalidClient(FakeGradioClient):
            def predict(self, *args, api_name):
                path = self.output_dir / "invalid.wav"
                path.write_bytes(b"not audio")
                return str(path)

        self.backend._client = InvalidClient(self.root)
        with self.assertRaises(BackendError) as error:
            await self.backend.synthesize(self.request_data)
        self.assertEqual(error.exception.status_code, 502)


class ApiV2Tests(unittest.IsolatedAsyncioTestCase):
    def backend_for(self, handler):
        backend = ApiV2Backend("http://127.0.0.1:9880/")
        backend.client_kwargs = {
            "transport": httpx.MockTransport(handler),
            "trust_env": False,
        }
        return backend

    async def test_non_streaming_preserves_content_type_and_payload(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, headers={"content-type": "audio/ogg"}, content=b"OggS-data")

        result = await self.backend_for(handler).synthesize({"text": "hello"})
        self.assertEqual(result, AudioResult(b"OggS-data", "audio/ogg"))
        self.assertEqual(requests[0].url.path, "/tts")
        self.assertEqual(requests[0].method, "POST")

    async def test_streaming_forwards_bytes(self):
        def handler(request):
            return httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"wav-bytes")

        stream, media_type = await self.backend_for(handler).stream({"text": "hello"})
        chunks = [chunk async for chunk in stream]
        self.assertEqual(b"".join(chunks), b"wav-bytes")
        self.assertEqual(media_type, "audio/wav")

    async def test_backend_error_is_not_returned_as_audio(self):
        def handler(request):
            return httpx.Response(400, json={"message": "bad request"})

        with self.assertRaises(BackendError) as error:
            await self.backend_for(handler).synthesize({"text": "hello"})
        self.assertEqual(error.exception.status_code, 400)
        self.assertIn("bad request", error.exception.detail)

    async def test_connection_error_becomes_backend_502(self):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        with self.assertRaises(BackendError) as error:
            await self.backend_for(handler).synthesize({"text": "hello"})
        self.assertEqual(error.exception.status_code, 502)

    async def test_weight_switch_uses_legacy_endpoints(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"message": "success"})

        backend = self.backend_for(handler)
        await backend.set_gpt_weights("gpt.ckpt")
        await backend.set_sovits_weights("sovits.pth")
        self.assertEqual([request.url.path for request in requests], ["/set_gpt_weights", "/set_sovits_weights"])
        self.assertEqual(requests[0].url.params["weights_path"], "gpt.ckpt")


if __name__ == "__main__":
    unittest.main()
