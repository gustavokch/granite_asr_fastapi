"""Speaker Diarization using pyannote.audio.
Ported and simplified from WhisperX (C. Max Bain).
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline

SAMPLE_RATE = 16000


class DiarizationPipeline:
    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-3.1",
        token: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model = Pipeline.from_pretrained(model_id, token=token)
        if self.model is not None:
            self.model.to(torch.device(device))

    def __call__(
        self,
        waveform: np.ndarray,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run diarization on a waveform.

        Args:
            waveform: Mono float32 numpy array at 16kHz.
        """
        if self.model is None:
            return pd.DataFrame(columns=["start", "end", "speaker"])

        # Prepare input for pyannote
        audio_data = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        }

        # Run pipeline
        diarization = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # Convert to DataFrame
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                }
            )

        return pd.DataFrame(segments)


def assign_speakers(
    diarization_df: pd.DataFrame,
    segments: list,
) -> list:
    """Assign speakers to transcription segments based on overlap.

    Args:
        diarization_df: DataFrame with 'start', 'end', 'speaker'.
        segments: List of segment dicts/objects with 'start', 'end'.
    """
    if diarization_df.empty:
        for seg in segments:
            if isinstance(seg, dict):
                seg["speaker"] = "Speaker 0"
            else:
                seg.speaker = "Speaker 0"
        return segments

    for seg in segments:
        s_start = seg["start"] if isinstance(seg, dict) else seg.start
        s_end = seg["end"] if isinstance(seg, dict) else seg.end

        # Find overlapping diarization segments
        overlaps = diarization_df[
            (diarization_df["end"] > s_start) & (diarization_df["start"] < s_end)
        ].copy()

        if not overlaps.empty:
            # Calculate intersection duration
            overlaps["overlap"] = overlaps.apply(
                lambda row: min(row["end"], s_end) - max(row["start"], s_start), axis=1
            )
            # Pick speaker with maximum overlap
            best_speaker = overlaps.loc[overlaps["overlap"].idxmax(), "speaker"]
            if isinstance(seg, dict):
                seg["speaker"] = best_speaker
            else:
                seg.speaker = best_speaker
        else:
            if isinstance(seg, dict):
                seg["speaker"] = "Speaker 0"
            else:
                seg.speaker = "Speaker 0"

        # If segment has words, assign speakers to words too
        if isinstance(seg, dict) and "words" in seg and seg["words"]:
            for word in seg["words"]:
                w_start = word["start"]
                w_end = word["end"]
                w_overlaps = diarization_df[
                    (diarization_df["end"] > w_start) & (diarization_df["start"] < w_end)
                ].copy()
                if not w_overlaps.empty:
                    w_overlaps["overlap"] = w_overlaps.apply(
                        lambda row: min(row["end"], w_end) - max(row["start"], w_start),
                        axis=1,
                    )
                    word["speaker"] = w_overlaps.loc[
                        w_overlaps["overlap"].idxmax(), "speaker"
                    ]
                else:
                    word["speaker"] = "Speaker 0"
        elif not isinstance(seg, dict) and hasattr(seg, "words") and seg.words:
             for word in seg.words:
                w_start = word.start
                w_end = word.end
                w_overlaps = diarization_df[
                    (diarization_df["end"] > w_start) & (diarization_df["start"] < w_end)
                ].copy()
                if not w_overlaps.empty:
                    w_overlaps["overlap"] = w_overlaps.apply(
                        lambda row: min(row["end"], w_end) - max(row["start"], w_start),
                        axis=1,
                    )
                    word.speaker = w_overlaps.loc[
                        w_overlaps["overlap"].idxmax(), "speaker"
                    ]
                else:
                    word.speaker = "Speaker 0"

    return segments
