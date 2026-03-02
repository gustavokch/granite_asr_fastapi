"""Tests for speaker diarization."""

import pandas as pd
import pytest
from granite_asr.diarization import assign_speakers
from granite_asr.schemas import Segment, WordSegment


def test_assign_speakers_simple():
    """Test speaker assignment with overlapping segments."""
    diarize_df = pd.DataFrame(
        [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_02"},
        ]
    )

    segments = [
        Segment(start=0.5, end=1.5, text="Hello"),
        Segment(start=2.5, end=3.5, text="World"),
    ]

    result = assign_speakers(diarize_df, segments)

    assert result[0].speaker == "SPEAKER_01"
    assert result[1].speaker == "SPEAKER_02"


def test_assign_speakers_with_words():
    """Test speaker assignment for individual words."""
    diarize_df = pd.DataFrame(
        [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_02"},
        ]
    )

    words = [
        WordSegment(word="Hello", start=0.5, end=1.0),
        WordSegment(word="there", start=2.5, end=3.0),
    ]
    
    segments = [
        Segment(start=0.5, end=3.0, text="Hello there", words=words)
    ]

    result = assign_speakers(diarize_df, segments)

    assert result[0].speaker == "SPEAKER_01" # Dominant speaker for whole segment
    assert result[0].words[0].speaker == "SPEAKER_01"
    assert result[0].words[1].speaker == "SPEAKER_02"


def test_assign_speakers_no_overlap():
    """Test speaker assignment when no diarization overlap exists."""
    diarize_df = pd.DataFrame(columns=["start", "end", "speaker"])

    segments = [
        Segment(start=0.5, end=1.5, text="Hello"),
    ]

    result = assign_speakers(diarize_df, segments)

    assert result[0].speaker == "Speaker 0"
