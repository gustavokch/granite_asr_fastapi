"""Granite ASR — IBM Granite Speech 3.3 2B transcription module.

Public API:
    load_model()              — Load the model into memory (call once).
    is_loaded()               — Check if the model is ready.
    transcribe(audio, ...)    — Batch transcription from bytes or file path.
    transcribe_stream(audio)  — Cumulative live transcription with silence-based
                                context reset.
"""

import logging
from pathlib import Path

from .config import GraniteSettings, get_settings
from .schemas import Segment, TranscriptionResult

logger = logging.getLogger("granite_asr")

__all__ = [
    "load_model",
    "is_loaded",
    "transcribe",
    "transcribe_stream",
    "GraniteSettings",
    "get_settings",
    "Segment",
    "TranscriptionResult",
]

SAMPLE_RATE = 16000


def load_model() -> None:
    """Load the Granite Speech model into memory. Call once at startup."""
    from .model import load_model as _load

    _load()


def is_loaded() -> bool:
    """Check if the model is loaded and ready for inference."""
    from .model import is_loaded as _is_loaded

    return _is_loaded()


def transcribe(
    audio: bytes | str | Path,
    *,
    language: str | None = None,
) -> TranscriptionResult:
    """Batch transcription — transcribe an entire audio input.

    Args:
        audio: Raw audio bytes (WAV/OGG/FLAC/etc.), or a file path (str or Path).
        language: BCP-47 tag. Defaults to settings.DEFAULT_LANGUAGE.

    Returns:
        TranscriptionResult with segments and metadata.
    """
    from .audio import decode_audio_bytes, decode_audio_file
    from .model import run_inference

    settings = get_settings()
    lang = language or settings.DEFAULT_LANGUAGE

    if isinstance(audio, (str, Path)):
        waveform = decode_audio_file(str(audio))
    else:
        waveform = decode_audio_bytes(audio)

    text, duration = run_inference(waveform, language=lang)

    segments = []
    if text:
        segments.append(Segment(start=0.0, end=duration, text=text))

    return TranscriptionResult(
        segments=segments,
        audio_duration_s=duration,
    )


def transcribe_stream(
    audio: bytes | str | Path,
    *,
    language: str | None = None,
) -> TranscriptionResult:
    """Cumulative live transcription with silence-based context reset.

    Designed for polling: the caller accumulates audio and passes the full
    buffer each time. This function finds the last silence boundary >= 1s
    and only transcribes audio after it, keeping the inference window bounded.

    Args:
        audio: Full accumulated audio buffer (bytes) or file path.
        language: BCP-47 tag. Defaults to settings.DEFAULT_LANGUAGE.

    Returns:
        TranscriptionResult with window_start_s indicating where the
        transcription window began within the full buffer.
    """
    from .audio import decode_audio_bytes, decode_audio_file
    from .model import run_inference
    from .silence import extract_transcription_window

    settings = get_settings()
    lang = language or settings.DEFAULT_LANGUAGE

    if isinstance(audio, (str, Path)):
        waveform = decode_audio_file(str(audio))
    else:
        waveform = decode_audio_bytes(audio)

    full_duration = len(waveform) / SAMPLE_RATE

    window_start_s, window_waveform = extract_transcription_window(
        waveform,
        threshold=settings.SILENCE_THRESHOLD_RMS,
        min_silence_s=settings.SILENCE_MIN_DURATION_S,
        frame_duration_s=settings.SILENCE_FRAME_DURATION_S,
    )

    text, window_duration = run_inference(window_waveform, language=lang)

    segments = []
    if text:
        segments.append(
            Segment(
                start=window_start_s,
                end=window_start_s + window_duration,
                text=text,
            )
        )

    return TranscriptionResult(
        segments=segments,
        window_start_s=window_start_s,
        audio_duration_s=full_duration,
    )
