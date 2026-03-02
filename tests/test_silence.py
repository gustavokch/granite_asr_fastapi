"""Tests for silence detection and transcription window extraction."""

import numpy as np

from granite_asr.silence import (
    SAMPLE_RATE,
    compute_rms_frames,
    extract_transcription_window,
    find_silence_regions,
)


def _make_tone(duration_s: float, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine tone at 16kHz."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_silence(duration_s: float) -> np.ndarray:
    """Generate digital silence at 16kHz."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def _concat(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(arrays)


class TestComputeRmsFrames:
    def test_silence_has_zero_rms(self):
        silence = _make_silence(0.1)  # 100ms
        rms = compute_rms_frames(silence, frame_samples=320)
        assert len(rms) == 5  # 100ms / 20ms
        assert np.all(rms == 0.0)

    def test_tone_has_positive_rms(self):
        tone = _make_tone(0.1, amplitude=0.5)
        rms = compute_rms_frames(tone, frame_samples=320)
        assert len(rms) == 5
        assert np.all(rms > 0.1)

    def test_empty_waveform(self):
        rms = compute_rms_frames(np.array([], dtype=np.float32), frame_samples=320)
        assert len(rms) == 0

    def test_short_waveform_no_complete_frame(self):
        short = np.ones(100, dtype=np.float32)  # < 320 samples
        rms = compute_rms_frames(short, frame_samples=320)
        assert len(rms) == 0


class TestFindSilenceRegions:
    def test_no_silence_in_tone(self):
        tone = _make_tone(1.0)
        rms = compute_rms_frames(tone, frame_samples=320)
        regions = find_silence_regions(rms, 0.02, threshold=0.01, min_duration_s=1.0)
        assert len(regions) == 0

    def test_finds_long_silence(self):
        # 1s tone, 1.5s silence, 1s tone
        audio = _concat(_make_tone(1.0), _make_silence(1.5), _make_tone(1.0))
        rms = compute_rms_frames(audio, frame_samples=320)
        regions = find_silence_regions(rms, 0.02, threshold=0.01, min_duration_s=1.0)
        assert len(regions) == 1
        assert regions[0].duration_s >= 1.0
        assert abs(regions[0].start_s - 1.0) < 0.05

    def test_ignores_short_silence(self):
        # 1s tone, 0.5s silence, 1s tone
        audio = _concat(_make_tone(1.0), _make_silence(0.5), _make_tone(1.0))
        rms = compute_rms_frames(audio, frame_samples=320)
        regions = find_silence_regions(rms, 0.02, threshold=0.01, min_duration_s=1.0)
        assert len(regions) == 0

    def test_trailing_silence(self):
        audio = _concat(_make_tone(1.0), _make_silence(2.0))
        rms = compute_rms_frames(audio, frame_samples=320)
        regions = find_silence_regions(rms, 0.02, threshold=0.01, min_duration_s=1.0)
        assert len(regions) == 1
        assert abs(regions[0].start_s - 1.0) < 0.05

    def test_multiple_silence_regions(self):
        # tone(1s), silence(1.5s), tone(1s), silence(2s), tone(0.5s)
        audio = _concat(
            _make_tone(1.0),
            _make_silence(1.5),
            _make_tone(1.0),
            _make_silence(2.0),
            _make_tone(0.5),
        )
        rms = compute_rms_frames(audio, frame_samples=320)
        regions = find_silence_regions(rms, 0.02, threshold=0.01, min_duration_s=1.0)
        assert len(regions) == 2


class TestExtractTranscriptionWindow:
    def test_no_silence_returns_full_audio(self):
        tone = _make_tone(3.0)
        start, window = extract_transcription_window(tone, threshold=0.01, min_silence_s=1.0)
        assert start == 0.0
        assert len(window) == len(tone)

    def test_window_starts_after_last_silence(self):
        # 2s tone, 1.5s silence, 1s tone
        audio = _concat(_make_tone(2.0), _make_silence(1.5), _make_tone(1.0))
        start, window = extract_transcription_window(audio, threshold=0.01, min_silence_s=1.0)
        # Window should start around 3.5s (after the silence ends)
        assert start > 3.0
        assert start < 4.0
        # Window should be ~1s of audio
        window_duration = len(window) / SAMPLE_RATE
        assert 0.8 < window_duration < 1.2

    def test_trailing_silence_falls_back(self):
        # 1s tone, 1.5s silence, 1s tone, 2s silence (trailing)
        audio = _concat(
            _make_tone(1.0),
            _make_silence(1.5),
            _make_tone(1.0),
            _make_silence(2.0),
        )
        start, window = extract_transcription_window(audio, threshold=0.01, min_silence_s=1.0)
        # Should fall back to the first silence boundary (~2.5s)
        assert start > 2.0
        assert start < 3.0

    def test_empty_waveform(self):
        empty = np.array([], dtype=np.float32)
        start, window = extract_transcription_window(empty, threshold=0.01, min_silence_s=1.0)
        assert start == 0.0
        assert len(window) == 0

    def test_only_silence(self):
        silence = _make_silence(3.0)
        start, window = extract_transcription_window(silence, threshold=0.01, min_silence_s=1.0)
        # Full audio is silence — should return full waveform
        assert start == 0.0
        assert len(window) == len(silence)
