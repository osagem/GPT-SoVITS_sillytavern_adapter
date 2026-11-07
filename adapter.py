import argparse
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# GPT-SoVITS 的嵌入式 Python 可能不包含脚本目录，并会优先加载 v5 根目录的 config.py。
# 在导入项目模块前固定适配器路径和工作目录，确保从任意目录/启动器运行都一致。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from config import *
from pluginManager import plugin_manager
from sovits_backend import BackendError, create_backend


REF_AUDIO_DIR = os.path.join(BASE_DIR, REF_AUDIO_DIR_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
MODELS_CONFIG_PATH = os.path.join(BASE_DIR, MODELS_CONFIG_NAME)

# 初始化目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_AUDIO_DIR, exist_ok=True)

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Driver")
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)

# FastAPI 应用初始化
app = FastAPI(title="SillyTavern Adapter for GPT-SoVITS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/srt", StaticFiles(directory=OUTPUT_DIR), name="音频输出")

# 加载插件
plugin_manager.load_plugins_from_dir("plugins")

# 后端适配层：支持 inference_webui.py (Gradio) 和 api_v2.py
backend = create_backend(
    BACKEND_MODE,
    API_V2_URL,
    OUTPUT_DIR,
    GRADIO_USE_CUDA_GRAPH,
    GRADIO_USE_FLASH_ATTENTION,
)


@asynccontextmanager
async def backend_operation():
    """Gradio 中串行化切模与合成，避免共享模型在角色请求间竞争。"""
    if backend.serialize_requests:
        async with backend.request_lock:
            yield
    else:
        yield


CHARACTER_MODEL_MAP = {}
# 加载角色模型配置
MODELS_CONFIG_PATH_INFO = MODELS_CONFIG_PATH if not os.path.exists("models_local.json") else "models_local.json"
if os.path.exists(MODELS_CONFIG_PATH_INFO):
    try:
        with open(MODELS_CONFIG_PATH_INFO, "r", encoding="utf-8") as file:
            CHARACTER_MODEL_MAP = json.load(file)
        logger.info("已加载配置: 成功读取 %s 个角色配置", len(CHARACTER_MODEL_MAP))
    except Exception as exc:
        logger.error("加载 models.json 失败: %s", exc)

# 初始化当前加载的模型状态
CURRENT_LOADED_MODELS = {"gpt": None, "sovits": None}


class TTS_Request(BaseModel):
    text: str
    text_lang: str
    ref_audio_path: str = ""
    target_voice: str = ""
    aux_ref_audio_paths: list = []
    prompt_lang: str = ""
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: Optional[int] = None
    cfg_rate: Optional[float] = None
    super_sampling: bool = False

    @field_validator("streaming_mode", mode="before")
    def parse_streaming_mode(cls, value):
        if isinstance(value, str):
            return value.lower() == "true"
        return value


# 根据角色名称切换模型
async def switch_model(character_name: str, request_data: dict):
    if character_name not in CHARACTER_MODEL_MAP:
        return

    target_gpt = CHARACTER_MODEL_MAP[character_name].get("gpt")
    target_sovits = CHARACTER_MODEL_MAP[character_name].get("sovits")

    if target_gpt and CURRENT_LOADED_MODELS["gpt"] != target_gpt:
        logger.info("[%s] 切换 GPT: ...%s", character_name, os.path.basename(target_gpt)[-15:])
        await backend.set_gpt_weights(target_gpt)
        CURRENT_LOADED_MODELS["gpt"] = target_gpt
        logger.info("GPT 切换成功")

    if target_sovits and CURRENT_LOADED_MODELS["sovits"] != target_sovits:
        logger.info("[%s] 切换 SoVITS: ...%s", character_name, os.path.basename(target_sovits)[-15:])
        await backend.set_sovits_weights(
            target_sovits,
            request_data.get("prompt_lang", GLOBAL_DEFAULT_LANG),
            request_data.get("text_lang", GLOBAL_DEFAULT_LANG),
        )
        CURRENT_LOADED_MODELS["sovits"] = target_sovits
        logger.info("SoVITS 切换成功")


# 处理 SillyTavern 路径、角色 prompt 和文本清理
async def fix_request_path_and_load_prompt(request_data: dict):
    raw_path = request_data.get("ref_audio_path") or request_data.get("target_voice", "")
    filename = os.path.basename(raw_path)
    if request_data.get("media_type") == "auto":
        request_data["media_type"] = "wav"
    # 处理类似 filename.wav.mp3 的情况
    if len(split_filename := filename.split(".")) >= 3 and split_filename[-2] in ["wav", "mp3", "flac", "ogg"]:
        filename = ".".join(split_filename[:-2]) + "." + split_filename[-1]

    character_name = os.path.splitext(filename)[0]
    abs_file_path = os.path.join(REF_AUDIO_DIR, filename)
    if not os.path.isfile(abs_file_path):
        raise BackendError(f"参考音频不存在: {abs_file_path}", 400)
    request_data["ref_audio_path"] = abs_file_path

    # 加载角色信息
    target_lang = GLOBAL_DEFAULT_LANG
    if character_name in CHARACTER_MODEL_MAP:
        config_lang = CHARACTER_MODEL_MAP[character_name].get("prompt_lang")
        if config_lang:
            target_lang = config_lang
    request_data["prompt_lang"] = target_lang

    # 读取参考文本；不再把角色名当作参考文本
    txt_path = os.path.splitext(abs_file_path)[0] + ".txt"
    prompt_text = ""
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as file:
                prompt_text = file.read().strip()
            if prompt_text:
                logger.info(
                    "成功使用 [%s] 读取角色 %s 参考文本: \"%s...\"",
                    target_lang,
                    character_name,
                    prompt_text[:10],
                )
        except OSError as exc:
            logger.warning("读取角色 %s 参考文本失败: %s", character_name, exc)
    else:
        logger.warning("角色 %s 缺少同名参考文本: %s", character_name, txt_path)

    request_data["prompt_text"] = prompt_text
    if BACKEND_MODE.strip().lower() == "gradio" and not prompt_text:
        raise BackendError(f"角色 {character_name} 缺少有效的同名参考文本: {txt_path}", 400)

    # 清理文本中的垃圾内容，将其单独抽象出来形成插件
    request_data["text"] = await plugin_manager.run_hook(
        "on_clean_text",
        data=request_data.get("text", ""),
        character_name=character_name,
        target_lang=target_lang,
    )
    return request_data, character_name, target_lang


def raise_http_error(error: BackendError):
    logger.error("请求处理失败: %s", error.detail)
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


@app.post("/")
@app.post("/tts")
async def tts_stream_endpoint(request: TTS_Request):
    try:
        request_data = request.model_dump()
        request_data, character_name, target_lang = await fix_request_path_and_load_prompt(request_data)
        logger.debug("处理后的请求数据: %s", request_data)

        async with backend_operation():
            await switch_model(character_name, request_data)
            request_data = await plugin_manager.run_hook(
                "on_tts_request_streaming",
                data=request_data,
                target_lang=target_lang,
            )
            logger.debug("插件处理后的请求数据: %s", request_data)
            stream_data, media_type = await backend.stream(request_data)
    except BackendError as exc:
        raise_http_error(exc)

    stream_data = await plugin_manager.run_hook(
        "on_tts_response_streaming",
        data=stream_data,
        character_name=character_name,
        target_lang=target_lang,
    )
    return StreamingResponse(stream_data, media_type=media_type)


# 获取可用角色列表
@app.get("/speakers")
def speakers_endpoint():
    voices = []
    if os.path.exists(REF_AUDIO_DIR):
        for name in os.listdir(REF_AUDIO_DIR):
            if name.lower().endswith((".wav", ".mp3", ".ogg")):
                display_name = os.path.splitext(name)[0]
                voices.append({"name": display_name, "voice_id": name})
    return JSONResponse(voices)


@app.get("/speakers_list")
def speakers_list_endpoint():
    return JSONResponse(["female", "male"], 200)


@app.post("/srt")
async def tts_file_endpoint(request: TTS_Request, req_obj: Request):
    try:
        request_data = request.model_dump()
        request_data, character_name, target_lang = await fix_request_path_and_load_prompt(request_data)
        request_data["streaming_mode"] = False
        request_data["media_type"] = "wav"

        async with backend_operation():
            await switch_model(character_name, request_data)
            request_data = await plugin_manager.run_hook(
                "on_srt_request_streaming",
                data=request_data,
                target_lang=target_lang,
                character_name=character_name,
            )
            result = await backend.synthesize(request_data)
    except BackendError as exc:
        raise_http_error(exc)

    filename = "audio.wav"
    try:
        with open(os.path.join(OUTPUT_DIR, filename), "wb") as file:
            file.write(result.content)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"保存合成音频失败: {exc}") from exc

    base = f"http://{req_obj.url.hostname}:{req_obj.url.port}"
    return JSONResponse(
        {
            "code": "200",
            "srt": f"{base}/srt/tts-out.srt",
            "audio": f"{base}/srt/{filename}",
        }
    )


# 启动服务
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=9881)
    args = parser.parse_args()

    print(f"\n服务已启动, 监听端口: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
