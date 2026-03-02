"""Tests for audio decoding utilities.

Requires torchaudio — skipped when not installed.
"""

import io
import struct

import numpy as np
import pytest

torchaudio = pytest.importorskip("torchaudio")

from granite_asr.audio import decode_audio_bytes  # noqa: E402


def _make_wav_bytes(
    samples: np.ndarray, sample_rate: int = 16000, num_channels: int = 1
) -> bytes:
    """Create a minimal WAV file in memory from float32 samples.

    Converts to 16-bit PCM for WAV encoding.
    """
    # Convert float32 [-1, 1] to int16
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    raw = pcm.tobytes()

    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(raw)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw)

    return buf.getvalue()


class TestDecodeAudioBytes:
    def test_decode_mono_16k_wav(self):
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        wav_bytes = _make_wav_bytes(samples, sample_rate=16000)
        result = decode_audio_bytes(wav_bytes)
        assert result.dtype == np.float32
        assert abs(len(result) - 16000) < 10  # ~1 second at 16kHz

    def test_resamples_from_48k(self):
        # 1 second at 48kHz
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 48000)).astype(np.float32)
        wav_bytes = _make_wav_bytes(samples, sample_rate=48000)
        result = decode_audio_bytes(wav_bytes)
        # Should be resampled to ~16000 samples
        assert abs(len(result) - 16000) < 100

    def test_stereo_downmixed_to_mono(self):
        left = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        right = np.sin(np.linspace(0, 2 * np.pi * 880, 16000)).astype(np.float32)
        stereo = np.stack([left, right])
        # Interleave for WAV
        interleaved = np.empty(32000, dtype=np.float32)
        interleaved[0::2] = left
        interleaved[1::2] = right
        wav_bytes = _make_wav_bytes(interleaved, sample_rate=16000, num_channels=2)
        result = decode_audio_bytes(wav_bytes)
        assert result.ndim == 1
        assert abs(len(result) - 16000) < 10

    def test_invalid_bytes_raises(self):
        with pytest.raises((ValueError, Exception)):
            decode_audio_bytes(b"not audio data")
