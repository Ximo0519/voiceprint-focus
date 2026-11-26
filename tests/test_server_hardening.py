"""服务端硬化：上传解码不再「猜」、体积有上限、CORS 只放行本机来源。

修复前：
- soundfile 解码失败时把任意字节当裸 float32 PCM 读（mp3/m4a/aac 会变成刺耳噪声）；
- `await file.read()` 无上限；
- CORS allow_origins=["*"]，任意网页可跨域读本机注册列表。
"""
import io

import numpy as np
import pytest
import soundfile as sf

import server


def test_decode_rejects_compressed_bytes_as_pcm():
    with pytest.raises(ValueError):
        server._decode_audio(b"\x00\x01\x02\x03" * 40000, "voice.mp3")


def test_decode_accepts_explicit_raw_pcm():
    x = (0.01 * np.ones(16000, np.float32))
    out = server._decode_audio(x.tobytes(), "voice.f32")
    assert out.shape == (16000,)


def test_decode_still_handles_wav():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, np.float32), 16000, format="WAV")
    out = server._decode_audio(buf.getvalue(), "voice.wav")
    assert out.shape[0] == 16000


def test_upload_limit_is_configured():
    assert server.MAX_UPLOAD_BYTES >= 1024 * 1024


def test_cors_defaults_to_local_origins(monkeypatch):
    monkeypatch.delenv("VF_CORS_ORIGINS", raising=False)
    origins = server._cors_origins()
    assert "*" not in origins
    assert "http://127.0.0.1:8000" in origins
    assert "http://localhost:8000" in origins


def test_cors_env_extends_whitelist(monkeypatch):
    monkeypatch.setenv("VF_CORS_ORIGINS",
                       "http://192.168.1.5:5173, https://demo.example")
    origins = server._cors_origins()
    assert "http://192.168.1.5:5173" in origins
    assert "https://demo.example" in origins
