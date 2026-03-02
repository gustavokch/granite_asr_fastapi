"""Granite Speech 3.3 2B model loading and inference.

Loads the model once via load_model(), keeps it in memory, and provides
run_inference() for transcription. Thread-safe via a threading lock that
serializes all model and processor access.
"""

import logging
import threading

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .config import get_settings

logger = logging.getLogger("granite_asr.model")

SAMPLE_RATE = 16000

_lock = threading.Lock()
_processor: AutoProcessor | None = None
_model: AutoModelForSpeechSeq2Seq | None = None


def load_model() -> None:
    """Load the Granite Speech model and processor into memory.

    Blocks until fully loaded (~30-60s on CPU cold start).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _processor, _model

    with _lock:
        if _processor is not None and _model is not None:
            return

        settings = get_settings()

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(settings.TORCH_DTYPE, torch.float16)

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
            torch_dtype=torch_dtype,
            **kwargs,
        )
        # Switch to evaluation mode (disables dropout/batchnorm training behavior)
        _model.train(False)
        logger.info("Granite Speech model loaded successfully")


def is_loaded() -> bool:
    return _processor is not None and _model is not None


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
