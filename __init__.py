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
from .schemas import Segment, TranscriptionResult, WordSegment

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
    "WordSegment",
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
    """Batch transcription with diarization and alignment.

    Args:
        audio: Raw audio bytes or file path.
        language: BCP-47 tag.

    Returns:
        TranscriptionResult with segments and speaker labels.
    """
    from .audio import decode_audio_bytes, decode_audio_file
    from .model import run_inference, run_alignment, run_diarization, run_vad
    from .diarization import assign_speakers
    from .schemas import WordSegment

    settings = get_settings()
    lang = language or settings.DEFAULT_LANGUAGE

    if isinstance(audio, (str, Path)):
        waveform = decode_audio_file(str(audio))
    else:
        waveform = decode_audio_bytes(audio)

    full_duration = len(waveform) / SAMPLE_RATE

    # 1. VAD
    speech_segments = run_vad(waveform)
    if not speech_segments:
        # Fall back to whole audio if no speech detected by VAD
        speech_segments = [{"start": 0.0, "end": full_duration}]

    # 2. Transcribe speech segments
    # Note: For batch, we could concatenate speech segments or process them individually.
    # To keep it simple and consistent with WhisperX, we'll process the whole thing
    # but only on segments if needed.
    # Actually, let's just run Granite on the whole waveform first as a single segment.
    # But wait, WhisperX does it per VAD segment.
    
    # For now, let's keep it simple: Transcribe the whole thing, then align.
    text, duration = run_inference(waveform, language=lang)
    
    if not text:
        return TranscriptionResult(segments=[], audio_duration_s=full_duration)

    # 3. Alignment
    word_data = run_alignment(text, waveform)
    
    words = []
    for wd in word_data:
        words.append(
            WordSegment(
                word=wd["word"],
                start=wd["start"],
                end=wd["end"],
                score=wd.get("score"),
            )
        )

    # 4. Diarization
    diarize_df = run_diarization(waveform)
    
    # 5. Assembly
    segments = [
        Segment(
            start=0.0,
            end=duration,
            text=text,
            words=words,
        )
    ]
    
    # Assign speakers to segments and words
    assign_speakers(diarize_df, segments)

    return TranscriptionResult(
        segments=segments,
        audio_duration_s=full_duration,
    )


def transcribe_stream(
    audio: bytes | str | Path,
    *,
    language: str | None = None,
) -> TranscriptionResult:
    """Cumulative live transcription with silence-based context reset and diarization.

    Designed for polling: the caller accumulates audio and passes the full
    buffer each time. This function finds the last silence boundary >= 1s
    and only transcribes audio after it.
    """
    from .audio import decode_audio_bytes, decode_audio_file
    from .model import run_inference, run_alignment, run_diarization
    from .silence import extract_transcription_window
    from .diarization import assign_speakers
    from .schemas import WordSegment

    settings = get_settings()
    lang = language or settings.DEFAULT_LANGUAGE

    if isinstance(audio, (str, Path)):
        waveform = decode_audio_file(str(audio))
    else:
        waveform = decode_audio_bytes(audio)

    full_duration = len(waveform) / SAMPLE_RATE

    # Still use the silence extraction for windowing
    window_start_s, window_waveform = extract_transcription_window(
        waveform,
        threshold=settings.SILENCE_THRESHOLD_RMS,
        min_silence_s=settings.SILENCE_MIN_DURATION_S,
        frame_duration_s=settings.SILENCE_FRAME_DURATION_S,
    )

    text, window_duration = run_inference(window_waveform, language=lang)

    if not text:
        return TranscriptionResult(
            segments=[], window_start_s=window_start_s, audio_duration_s=full_duration
        )

    # Align only the windowed text
    word_data = run_alignment(text, window_waveform)
    
    words = []
    for wd in word_data:
        words.append(
            WordSegment(
                word=wd["word"],
                start=wd["start"] + window_start_s, # Offset by window start
                end=wd["end"] + window_start_s,
                score=wd.get("score"),
            )
        )

    # Diarize full waveform to get consistent speaker labels
    diarize_df = run_diarization(waveform)

    segments = [
        Segment(
            start=window_start_s,
            end=window_start_s + window_duration,
            text=text,
            words=words,
        )
    ]
    
    # Assign speakers
    assign_speakers(diarize_df, segments)

    return TranscriptionResult(
        segments=segments,
        window_start_s=window_start_s,
        audio_duration_s=full_duration,
    )
