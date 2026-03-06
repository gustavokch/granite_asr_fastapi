"""Granite Speech 3.3 2B model loading and inference.

Loads the model once via load_model(), keeps it in memory, and provides
run_inference() for transcription. Thread-safe via a threading lock that
serializes all model and processor access.
"""

import logging
import threading
import time

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

        # Optimize CPU threads if applicable
        if settings.DEVICE == "cpu":
            import os
            # Use around half of available cores for PyTorch to avoid contention
            cores = os.cpu_count() or 4
            threads = max(1, cores // 2)
            torch.set_num_threads(threads)
            logger.info("CPU optimization: set torch.set_num_threads(%d)", threads)

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
            
            # Optimization: torch.compile
            if settings.USE_COMPILE:
                if hasattr(torch, "compile"):
                    logger.info("Compiling model with torch.compile (mode='reduce-overhead')...")
                    # 'reduce-overhead' is good for small batches/CPU inference
                    _model = torch.compile(_model, mode="reduce-overhead")
                else:
                    logger.warning("torch.compile not available in this PyTorch version")

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

    logger.debug("Acquiring lock for inference...")
    with _lock:
        logger.debug("Lock acquired for inference. Building prompt...")
        prompt_str = _build_prompt(language)
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)

        logger.debug("Running model generate...")
        start_time = time.time()
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
        
        gen_time = time.time() - start_time
        logger.debug("Model generate complete in %.2fs", gen_time)

        num_input_tokens = inputs["input_ids"].shape[-1]
        new_token_ids = output_ids[0, num_input_tokens:]

        if len(new_token_ids) >= settings.MAX_NEW_TOKENS:
            logger.warning(
                "Generation reached MAX_NEW_TOKENS (%d). Transcription may be truncated.",
                settings.MAX_NEW_TOKENS,
            )

        text = _processor.tokenizer.decode(  # type: ignore[union-attr]
            new_token_ids,
            skip_special_tokens=True,
            add_special_tokens=False,
        ).strip()
    
    return text, audio_duration_s


def run_batch_inference(
    waveforms: list[np.ndarray],
    languages: list[str] | str = "pt-BR",
) -> list[tuple[str, float]]:
    """Run Granite Speech inference on a batch of waveforms.

    Args:
        waveforms: List of mono float32 numpy arrays at 16kHz.
        languages: List of BCP-47 language tags (one per waveform) or single tag.

    Returns:
        List of (transcribed_text, audio_duration_s) tuples.
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    if not waveforms:
        return []

    settings = get_settings()
    device = settings.DEVICE
    durations = [len(w) / SAMPLE_RATE for w in waveforms]
    
    # Handle languages
    if isinstance(languages, str):
        langs = [languages] * len(waveforms)
    else:
        if len(languages) != len(waveforms):
            raise ValueError("Length of languages must match waveforms")
        langs = languages

    # Ensure float32
    waveforms_f32 = [
        w.astype(np.float32) if w.dtype != np.float32 else w
        for w in waveforms
    ]

    logger.debug("Acquiring lock for batch inference...")
    with _lock:
        logger.debug("Lock acquired for batch inference. Building prompts...")
        
        # Build individual prompts
        prompts = [_build_prompt(l) for l in langs]
        
        start_time = time.time()
        logger.debug("Processing batch of %d waveforms...", len(waveforms))
        
        with torch.inference_mode():
            # Processor call for batch
            inputs = _processor(
                text=prompts,
                audio=waveforms_f32,
                device=device,
                return_tensors="pt",
                padding=True, # Pad audio to longest in batch
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logger.debug("Running model generate (batch)...")
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        
        gen_time = time.time() - start_time
        logger.debug("Batch generation complete in %.2fs", gen_time)

        # Decode batch
        num_input_tokens = inputs["input_ids"].shape[-1]
        new_token_ids = output_ids[:, num_input_tokens:]

        if new_token_ids.shape[-1] >= settings.MAX_NEW_TOKENS:
            logger.warning(
                "Batch generation reached MAX_NEW_TOKENS (%d). Transcription may be truncated.",
                settings.MAX_NEW_TOKENS,
            )
        
        texts = _processor.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # Strip texts
        texts = [t.strip() for t in texts]

    return list(zip(texts, durations))


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

    logger.debug("Acquiring lock for alignment...")
    with _lock:
        logger.debug("Lock acquired for alignment. Running align...")
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

    logger.debug("Acquiring lock for diarization...")
    with _lock:
        logger.debug("Lock acquired for diarization. Running pipeline...")
        return _diarize_pipeline(waveform)  # type: ignore


def run_vad(waveform: np.ndarray) -> list:
    """Run VAD on a waveform.

    Args:
        waveform: Mono float32 numpy array at 16kHz.
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    logger.debug("Acquiring lock for VAD...")
    with _lock:
        logger.debug("Lock acquired for VAD. Running pipeline...")
        return _vad_pipeline(waveform)  # type: ignore
