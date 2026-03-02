"""Configuration for the Granite ASR module."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class GraniteSettings(BaseSettings):
    # Model
    MODEL_ID: str = "ibm-granite/granite-speech-3.3-2b"
    DEVICE: str = "cpu"
    TORCH_DTYPE: str = "float16"  # "float16" | "bfloat16" | "float32"
    MODEL_CACHE_DIR: str | None = None

    # Inference
    MAX_NEW_TOKENS: int = 512
    DEFAULT_LANGUAGE: str = "pt-BR"
    SYSTEM_PROMPT: str = (
        "Você é o Granite, desenvolvido pela IBM. "
        "Você é um assistente de transcrição de áudio médico preciso."
    )

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
