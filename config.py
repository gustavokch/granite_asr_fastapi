"""Configuration for the Granite ASR module."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def resolve_device(device: str) -> str:
    """Map 'auto' to 'cuda' if available, else 'cpu'. Pass through other values."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


class GraniteSettings(BaseSettings):
    # Model
    MODEL_ID: str = "ibm-granite/granite-speech-3.3-2b"
    DEVICE: str = "auto"
    TORCH_DTYPE: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    MODEL_CACHE_DIR: str | None = None

    @property
    def resolved_device(self) -> str:
        """Return the effective device string ('cuda' or 'cpu')."""
        return resolve_device(self.DEVICE)

    @property
    def resolved_dtype(self) -> str:
        """Return the effective dtype string, defaulting to float16 on CUDA, float32 on CPU."""
        if self.TORCH_DTYPE != "auto":
            return self.TORCH_DTYPE
        return "float16" if self.resolved_device == "cuda" else "float32"

    # Alignment
    ALIGN_MODEL_ID: str = "alinerodrigues/wav2vec2-large-xlsr-grosman-53-words-phoneme-exp-1-v17"
    
    # Diarization
    DIARIZATION_MODEL_ID: str = "pyannote/speaker-diarization-3.1"
    HF_TOKEN: str | None = None

    # Inference
    MAX_NEW_TOKENS: int = 2048
    DEFAULT_LANGUAGE: str = "pt-BR"
    SYSTEM_PROMPT: str = (
        "Você é o Granite, desenvolvido pela IBM. "
        "Você é um assistente de transcrição de áudio médico preciso."
    )
    
    # Optimization
    USE_COMPILE: bool = False  # Enable torch.compile (requires PyTorch 2.0+)
    BATCH_SIZE: int = 1        # Max batch size for dynamic batching
    BATCH_TIMEOUT: float = 0.1 # Max wait time for batch accumulation (seconds)

    # Silence detection
    SILENCE_THRESHOLD_RMS: float = 0.01
    SILENCE_MIN_DURATION_S: float = 1.0
    SILENCE_FRAME_DURATION_S: float = 0.02  # 20ms frames (320 samples @ 16kHz)

    # Server (used by granite_asr.server)
    HOST: str = "0.0.0.0"
    PORT: int = 8010

    model_config = SettingsConfigDict(
        env_prefix="GRANITE_", env_file=".env", extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> GraniteSettings:
    return GraniteSettings()
