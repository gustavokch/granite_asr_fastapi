"""Data classes for Granite ASR results."""

from dataclasses import dataclass


@dataclass
class WordSegment:
    """A single word with timestamps and speaker information."""

    word: str
    start: float
    end: float
    speaker: str = "Speaker 0"
    score: float | None = None


@dataclass
class Segment:
    """A single transcription segment with timestamps."""

    start: float
    end: float
    text: str
    speaker: str = "Speaker 0"
    confidence: float | None = None  # Granite does not produce confidence scores
    words: list[WordSegment] | None = None


@dataclass
class TranscriptionResult:
    """Result from a transcription call."""

    segments: list[Segment]
    window_start_s: float = 0.0
    audio_duration_s: float = 0.0
