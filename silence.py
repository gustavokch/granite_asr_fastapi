"""Silence detection and transcription window extraction.

Analyzes audio waveforms to find silence boundaries >= 1s. Used by the
cumulative live transcription mode to cap the inference window — only audio
after the last significant silence is transcribed on each poll.
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class SilenceRegion:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def compute_rms_frames(waveform: np.ndarray, frame_samples: int) -> np.ndarray:
    """Per-frame RMS energy for a mono float32 waveform.

    Trailing samples that don't fill a complete frame are discarded.
    """
    num_frames = len(waveform) // frame_samples
    if num_frames == 0:
        return np.array([])
    frames = waveform[: num_frames * frame_samples].reshape(num_frames, frame_samples)
    return np.sqrt(np.mean(frames**2, axis=1))


def find_silence_regions(
    rms: np.ndarray,
    frame_duration_s: float,
    threshold: float,
    min_duration_s: float,
) -> list[SilenceRegion]:
    """Merge consecutive silent frames into regions.

    Returns only regions whose duration meets or exceeds min_duration_s.
    """
    regions: list[SilenceRegion] = []
    in_silence = False
    region_start = 0

    for i, energy in enumerate(rms):
        if energy < threshold:
            if not in_silence:
                in_silence = True
                region_start = i
        else:
            if in_silence:
                in_silence = False
                start_s = region_start * frame_duration_s
                end_s = i * frame_duration_s
                if (end_s - start_s) >= min_duration_s:
                    regions.append(SilenceRegion(start_s=start_s, end_s=end_s))

    # Trailing silence
    if in_silence:
        start_s = region_start * frame_duration_s
        end_s = len(rms) * frame_duration_s
        if (end_s - start_s) >= min_duration_s:
            regions.append(SilenceRegion(start_s=start_s, end_s=end_s))

    return regions


def extract_transcription_window(
    waveform: np.ndarray,
    threshold: float,
    min_silence_s: float,
    frame_duration_s: float = 0.02,
) -> tuple[float, np.ndarray]:
    """Find the last qualifying silence boundary and return the audio after it.

    Returns (window_start_s, window_waveform).
    If no qualifying silence exists, returns (0.0, full waveform).
    """
    frame_samples = int(frame_duration_s * SAMPLE_RATE)
    rms = compute_rms_frames(waveform, frame_samples)

    if len(rms) == 0:
        return 0.0, waveform

    regions = find_silence_regions(rms, frame_duration_s, threshold, min_silence_s)
    if not regions:
        return 0.0, waveform

    last_silence = regions[-1]
    window_start_s = last_silence.end_s
    window_start_sample = int(window_start_s * SAMPLE_RATE)

    remaining_samples = len(waveform) - window_start_sample
    if window_start_sample >= len(waveform) or remaining_samples < frame_samples:
        # Audio ends in (or near) silence — fall back to previous silence boundary
        if len(regions) >= 2:
            prev_silence = regions[-2]
            window_start_s = prev_silence.end_s
            window_start_sample = int(window_start_s * SAMPLE_RATE)
        else:
            return 0.0, waveform

    return window_start_s, waveform[window_start_sample:]
