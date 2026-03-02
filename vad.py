"""Voice Activity Detection using Silero VAD.
"""

import numpy as np
import torch

SAMPLE_RATE = 16000


class VADPipeline:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.model.to(self.device)
        self.get_speech_timestamps = self.utils[0]

    def __call__(
        self,
        waveform: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ):
        """Find speech segments in a waveform.

        Args:
            waveform: Mono float32 numpy array at 16kHz.
        """
        waveform_tensor = torch.from_numpy(waveform).to(self.device)
        
        # silero-vad expects (T,) or (Batch, T)
        if len(waveform_tensor.shape) == 1:
            pass
        else:
            waveform_tensor = waveform_tensor.squeeze()

        speech_timestamps = self.get_speech_timestamps(
            waveform_tensor,
            self.model,
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        # Convert sample indices to seconds
        segments = []
        for ts in speech_timestamps:
            segments.append(
                {
                    "start": ts["start"] / SAMPLE_RATE,
                    "end": ts["end"] / SAMPLE_RATE,
                }
            )
        return segments
