# 更新日志

## [Unreleased]
### 新增
* 支持 GPT-SoVITS v5 项目的 `inference_webui.py` Gradio 后端，默认连接 9872。
* 保留 api_v2 双后端模式，并增加 Gradio 参数映射、权重切换和并发保护。
* 增加后端协议与 SillyTavern 路由单元测试。

### 修复
* 修复 `/srt` 预处理缺少 `await` 的问题。
* 解耦参考音频格式与输出 `media_type`，按实际后端响应设置 MIME。
* 缺失参考文本时返回明确错误，不再使用角色名作为参考文本。
* 后端错误不再作为 HTTP 200 音频正文返回。
* 修复 GPT-SoVITS 嵌入式 Python 未包含适配器脚本目录、误加载 v5 根目录 `config.py` 导致的启动失败。
* 兼容 SillyTavern `GPT-SoVITS-Adapter` 的 `target_voice` 请求字段，并将其 `media_type=auto` 映射为 WAV。

## [0.1.1] - 2025-12-15
### 修复
* 修复翻译插件无法使用代理连接互联网的问题。
* 修复因部分情况下产生的因音频格式错误而导致的语音合成失效问题。