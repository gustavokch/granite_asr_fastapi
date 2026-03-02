"""Audio decoding utilities for Granite ASR."""

import io

import numpy as np
import torchaudio

SAMPLE_RATE = 16000


def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes (WAV, OGG, FLAC, etc.) to mono float32 numpy at 16kHz.

    Uses torchaudio which handles most container formats via its backends.
    """
    buf = io.BytesIO(audio_bytes)
    wav, sr = torchaudio.load(buf, normalize=True)

    # Downmix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        wav = resampler(wav)

    return wav.squeeze(0).numpy()


def decode_audio_file(file_path: str) -> np.ndarray:
    """Decode an audio file to mono float32 numpy at 16kHz."""
    wav, sr = torchaudio.load(file_path, normalize=True)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        wav = resampler(wav)

    return wav.squeeze(0).numpy()
