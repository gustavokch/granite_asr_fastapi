"""Granite Speech 3.3 2B model loading and inference.

Loads the model once via load_model(), keeps it in memory, and provides
run_inference() for transcription. Thread-safe via a threading lock that
serializes all model and processor access.
"""

import logging
import threading

import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .config import get_settings

logger = logging.getLogger("granite_asr.model")

SAMPLE_RATE = 16000

_lock = threading.Lock()
_processor: AutoProcessor | None = None
_model: AutoModelForSpeechSeq2Seq | None = None

# New models
_align_model: torch.nn.Module | None = None
_align_dictionary: dict | None = None
_diarize_pipeline: object | None = None
_vad_pipeline: object | None = None


def load_model() -> None:
    """Load the Granite Speech model and other pipelines into memory.

    Blocks until fully loaded. Safe to call multiple times.
    """
    global _processor, _model, _align_model, _align_dictionary, _diarize_pipeline, _vad_pipeline

    with _lock:
        settings = get_settings()

        # Load Granite ASR
        if _processor is None or _model is None:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(settings.TORCH_DTYPE, torch.float16)

            logger.info(
                "Loading Granite Speech model: %s (device=%s, dtype=%s)",
                settings.MODEL_ID,
                settings.DEVICE,
                settings.TORCH_DTYPE,
            )

            kwargs = {}
            if settings.MODEL_CACHE_DIR:
                kwargs["cache_dir"] = settings.MODEL_CACHE_DIR

            _processor = AutoProcessor.from_pretrained(settings.MODEL_ID, **kwargs)
            _model = AutoModelForSpeechSeq2Seq.from_pretrained(
                settings.MODEL_ID,
                device_map=settings.DEVICE,
                dtype=dtype,
                **kwargs,
            )
            _model.train(False)
            logger.info("Granite Speech model loaded successfully")

        # Load Alignment model
        if _align_model is None:
            from .alignment import load_align_model as _load_align

            logger.info("Loading alignment model: %s", settings.ALIGN_MODEL_ID)
            _align_model, _align_dictionary = _load_align(
                settings.ALIGN_MODEL_ID, settings.DEVICE
            )
            logger.info("Alignment model loaded successfully")

        # Load Diarization pipeline
        if _diarize_pipeline is None:
            from .diarization import DiarizationPipeline

            logger.info("Loading diarization model: %s", settings.DIARIZATION_MODEL_ID)
            _diarize_pipeline = DiarizationPipeline(
                settings.DIARIZATION_MODEL_ID,
                token=settings.HF_TOKEN,
                device=settings.DEVICE,
            )
            logger.info("Diarization model loaded successfully")

        # Load VAD pipeline
        if _vad_pipeline is None:
            from .vad import VADPipeline

            logger.info("Loading VAD model (Silero)")
            _vad_pipeline = VADPipeline(device=settings.DEVICE)
            logger.info("VAD model loaded successfully")


def is_loaded() -> bool:
    return (
        _processor is not None
        and _model is not None
        and _align_model is not None
        and _diarize_pipeline is not None
        and _vad_pipeline is not None
    )


def _build_prompt(language: str) -> str:
    """Build the chat-template prompt with the <|audio|> placeholder."""
    settings = get_settings()
    tokenizer = _processor.tokenizer  # type: ignore[union-attr]

    lang_instructions = {
        "pt-BR": "Por favor, transcreva a fala a seguir no formato escrito em Português do Brasil.",
        "pt": "Por favor, transcreva a fala a seguir no formato escrito em Português.",
        "en": "Please transcribe the following speech into written format in English.",
        "es": "Por favor, transcribe el siguiente discurso en formato escrito en Español.",
        "fr": "Veuillez transcrire le discours suivant sous forme écrite en Français.",
        "de": "Bitte transkribieren Sie die folgende Rede in geschriebener Form auf Deutsch.",
    }
    instruction = lang_instructions.get(
        language, lang_instructions["pt-BR"]
    )
    user_prompt = f"<|audio|>{instruction}"

    chat = [
        {"role": "system", "content": settings.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def run_inference(
    waveform: np.ndarray,
    language: str = "pt-BR",
) -> tuple[str, float]:
    """Run Granite Speech inference on a waveform.

    Args:
        waveform: mono float32 numpy array at 16kHz.
        language: BCP-47 language tag for the transcription prompt.

    Returns:
        (transcribed_text, audio_duration_s) tuple.

    Raises:
        RuntimeError: if the model has not been loaded.
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)

    settings = get_settings()
    device = settings.DEVICE
    audio_duration_s = len(waveform) / SAMPLE_RATE

    with _lock:
        prompt_str = _build_prompt(language)
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)

        with torch.inference_mode():
            inputs = _processor(  # type: ignore[misc]
                prompt_str,
                wav_tensor,
                device=device,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            output_ids = _model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )

        num_input_tokens = inputs["input_ids"].shape[-1]
        new_token_ids = output_ids[0, num_input_tokens:]
        text = _processor.tokenizer.decode(  # type: ignore[union-attr]
            new_token_ids,
            skip_special_tokens=True,
            add_special_tokens=False,
        ).strip()

    return text, audio_duration_s


def run_alignment(text: str, waveform: np.ndarray) -> list:
    """Run forced alignment on a waveform.

    Args:
        text: Transcription text.
        waveform: Mono float32 numpy array at 16kHz.
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    from .alignment import align

    settings = get_settings()
    wav_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)

    with _lock:
        return align(
            text,
            wav_tensor,
            _align_model,  # type: ignore
            _align_dictionary,  # type: ignore
            settings.DEVICE,
        )


def run_diarization(waveform: np.ndarray) -> pd.DataFrame:
    """Run diarization on a waveform.

    Args:
        waveform: Mono float32 numpy array at 16kHz.
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    with _lock:
        return _diarize_pipeline(waveform)  # type: ignore


def run_vad(waveform: np.ndarray) -> list:
    """Run VAD on a waveform.

    Args:
        waveform: Mono float32 numpy array at 16kHz.
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    with _lock:
        return _vad_pipeline(waveform)  # type: ignore
