import asyncio
import io
import os
import sys
import unittest
import uuid
import wave
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import adapter  # noqa: E402
from sovits_backend import AudioResult, BackendError  # noqa: E402


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(32000)
        wav_file.writeframes(b"\x00\x00" * 320)
    return buffer.getvalue()


class FakeAdapterBackend:
    serialize_requests = False

    def __init__(self):
        self.request_lock = asyncio.Lock()
        self.requests = []
        self.weight_calls = []
        self.audio = wav_bytes()

    async def set_gpt_weights(self, weights_path):
        self.weight_calls.append(("gpt", weights_path))

    async def set_sovits_weights(self, weights_path, prompt_lang="zh", text_lang="zh"):
        self.weight_calls.append(("sovits", weights_path, prompt_lang, text_lang))

    async def synthesize(self, request_data):
        self.requests.append(request_data)
        return AudioResult(self.audio, "audio/wav")

    async def stream(self, request_data):
        self.requests.append(request_data)

        async def generator():
            yield self.audio[:12]
            yield self.audio[12:]

        return generator(), "audio/wav"


class AdapterRouteTests(unittest.TestCase):
    def setUp(self):
        self.old_backend = adapter.backend
        self.old_model_map = adapter.CHARACTER_MODEL_MAP
        self.fake_backend = FakeAdapterBackend()
        adapter.backend = self.fake_backend
        adapter.CHARACTER_MODEL_MAP = {}
        self.client = TestClient(adapter.app)

        self.stem = f"test-{uuid.uuid4().hex}"
        self.reference = Path(adapter.REF_AUDIO_DIR) / f"{self.stem}.wav"
        self.prompt = self.reference.with_suffix(".txt")
        with wave.open(str(self.reference), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        self.prompt.write_text("这是参考文本。", encoding="utf-8")
        self.payload = {
            "text": "这是需要合成的文本。",
            "text_lang": "zh",
            "ref_audio_path": self.reference.name,
        }

    def tearDown(self):
        self.client.close()
        adapter.backend = self.old_backend
        adapter.CHARACTER_MODEL_MAP = self.old_model_map
        self.reference.unlink(missing_ok=True)
        self.prompt.unlink(missing_ok=True)

    def test_tts_keeps_streaming_contract_and_wav_response(self):
        response = self.client.post("/tts", json=self.payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, self.fake_backend.audio)
        self.assertEqual(self.fake_backend.requests[0]["prompt_text"], "这是参考文本。")

    def test_native_adapter_provider_uses_target_voice(self):
        payload = {
            "text": "这是需要合成的文本。",
            "card_name": "测试角色",
            "use_st_adapter": True,
            "target_voice": self.reference.name,
            "text_lang": "zh",
            "text_split_method": "cut5",
            "batch_size": 1,
            "media_type": "auto",
            "streaming_mode": "true",
        }
        response = self.client.post("/", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.fake_backend.requests[0]["ref_audio_path"], str(self.reference))
        self.assertEqual(self.fake_backend.requests[0]["prompt_text"], "这是参考文本。")
        self.assertEqual(self.fake_backend.requests[0]["media_type"], "wav")

    def test_srt_route_awaits_preprocessing_and_writes_wav(self):
        response = self.client.post("/srt", json=self.payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["code"], "200")
        output_path = Path(adapter.OUTPUT_DIR) / "audio.wav"
        self.assertEqual(output_path.read_bytes(), self.fake_backend.audio)
        self.assertEqual(self.fake_backend.requests[0]["streaming_mode"], False)
        self.assertEqual(self.fake_backend.requests[0]["media_type"], "wav")

    def test_missing_reference_text_returns_400(self):
        self.prompt.unlink()
        response = self.client.post("/tts", json=self.payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("参考文本", response.json()["detail"])

    def test_missing_reference_audio_returns_400(self):
        self.reference.unlink()
        response = self.client.post("/tts", json=self.payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("参考音频不存在", response.json()["detail"])

    def test_reference_suffix_does_not_change_output_media_type(self):
        mp3_reference = Path(adapter.REF_AUDIO_DIR) / f"{self.stem}.mp3"
        mp3_prompt = mp3_reference.with_suffix(".txt")
        mp3_reference.write_bytes(self.fake_backend.audio)
        mp3_prompt.write_text("这是参考文本。", encoding="utf-8")
        self.payload["ref_audio_path"] = mp3_reference.name
        try:
            response = self.client.post("/tts", json=self.payload)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.fake_backend.requests[0]["media_type"], "wav")
            self.assertEqual(response.headers["content-type"], "audio/wav")
        finally:
            mp3_reference.unlink(missing_ok=True)
            mp3_prompt.unlink(missing_ok=True)

    def test_weight_switch_failure_stops_synthesis(self):
        class FailingBackend(FakeAdapterBackend):
            async def set_gpt_weights(self, weights_path):
                raise BackendError("switch failed", 502)

        failing_backend = FailingBackend()
        adapter.backend = failing_backend
        adapter.CHARACTER_MODEL_MAP = {self.stem: {"gpt": "gpt.ckpt"}}
        response = self.client.post("/tts", json=self.payload)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(failing_backend.requests, [])

    def test_speakers_lists_reference_audio(self):
        response = self.client.get("/speakers")
        self.assertEqual(response.status_code, 200)
        self.assertIn({"name": self.stem, "voice_id": self.reference.name}, response.json())


class AdapterConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_gradio_operation_lock_serializes_model_and_synthesis_section(self):
        class LockedBackend(FakeAdapterBackend):
            serialize_requests = True

        old_backend = adapter.backend
        locked_backend = LockedBackend()
        adapter.backend = locked_backend
        active = 0
        maximum_active = 0

        async def worker():
            nonlocal active, maximum_active
            async with adapter.backend_operation():
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0.01)
                active -= 1

        try:
            await asyncio.gather(*(worker() for _ in range(3)))
        finally:
            adapter.backend = old_backend
        self.assertEqual(maximum_active, 1)


if __name__ == "__main__":
    unittest.main()
