# GPT-SoVITS SillyTavern Adapter

SillyTavern 与 GPT-SoVITS 之间的 TTS 中间服务。本次修改重点适配 GPT-SoVITS v5 项目的 Gradio 推理 WebUI，并兼容 SillyTavern 的 `GPT-SoVITS-Adapter` Provider。

## 本次修改内容

### GPT-SoVITS v5 / Gradio 兼容

- 支持 `GPT_SoVITS/inference_webui.py` 启动的 Gradio 服务，默认地址为 `http://127.0.0.1:9872`。
- 使用 `gradio_client 1.3.0` 调用 `/get_tts_wav`，按当前 WebUI 源码构造 19 个位置参数。
- 完成推理语言、切句方式、采样参数、参考音频和参考文本的映射。
- 下载并校验 Gradio 返回的 WAV 音频，不再把 Gradio 当作 FastAPI `/tts` 服务调用。
- 保留 `api_v2` 后端模式，可连接独立启动的 `api_v2.py` 服务。

### SillyTavern 兼容

- 支持 `GPT-SoVITS-Adapter` 使用 `target_voice` 指定角色音色。
- 保留旧请求的 `ref_audio_path` 处理方式。
- 将 `media_type=auto` 转换为 WAV；Gradio 模式统一返回完整 WAV。
- 保持 `POST /`、`POST /tts`、`POST /srt` 和 `GET /speakers` 接口。
- SillyTavern Provider Endpoint 仍指向中间服务，而不是直接指向 GPT-SoVITS WebUI。

### 稳定性与问题修复

- 支持 Gradio 模式下切换 GPT/SoVITS 权重，并串行化切换权重与合成过程，避免并发请求串用角色模型。
- 权重切换失败时中止合成，不再继续使用错误模型。
- api_v2 模式保留流式字节转发，并传递真实音频 MIME 和上游错误状态。
- 修复 `/srt` 预处理缺少 `await` 的问题。
- 修复参考音频扩展名错误覆盖输出格式的问题。
- 缺少同名参考文本时返回明确错误，不再使用角色名冒充参考文本。
- 修复 GPT-SoVITS 嵌入式 Python 启动时误加载 v5 项目 `config.py`、找不到适配器模块的问题。
- 新增 `run.bat`，可直接使用 GPT-SoVITS v5 自带 Python 启动中间服务。
- 新增协议、路由、错误处理和并发回归测试，目前共 23 项。

## 启动方法

1. 保持 GPT-SoVITS Gradio WebUI 运行在 `http://127.0.0.1:9872`。
2. 在 `voice` 目录放置角色参考音频和同名 UTF-8 文本，例如：

```text
voice/anno.wav
voice/anno.txt
```

参考音频应为 3–10 秒，文本内容应与参考音频实际语音一致。

3. 双击 `run.bat`，或在项目目录执行：

```bat
"H:\ai\GPT-Sovits\GPT-SoVITS-v5-20261006nvidia50x0\runtime\python.exe" -s adapter.py
```

中间服务默认监听 `http://127.0.0.1:9881`。如使用其他安装路径，请修改 `run.bat` 中的 Python 路径。

4. 在 SillyTavern 中选择 `GPT-SoVITS-Adapter`，Provider Endpoint 填写：

```text
http://127.0.0.1:9881
```

然后刷新并获取可用声音，为角色配置音色。

## 配置

`config.py` 默认使用：

```python
BACKEND_MODE = "gradio"
API_V2_URL = "http://127.0.0.1:9872"
```

如需连接独立 `api_v2.py` 服务，可将 `BACKEND_MODE` 改为 `api_v2`，并将地址改为对应服务地址，通常为 `http://127.0.0.1:9880`。

角色权重映射仍通过 `models.json` 或本地 `models_local.json` 配置；空配置表示沿用 WebUI 当前加载的模型。

## 测试

```bat
python -m unittest discover -s tests -v
```

测试使用模拟后端，不依赖 GPU。真实 Gradio 合成与 SillyTavern 播放仍需在运行环境中进行联调。
