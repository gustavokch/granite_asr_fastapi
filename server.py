"""FastAPI microservice wrapping the granite_asr module.

Loads the Granite Speech 3.3 2B model once at startup and serves
transcription requests over HTTP.

Run with:
    python -m granite_asr.server
    uvicorn granite_asr.server:app --host 0.0.0.0 --port 8010
"""

import asyncio
import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .config import get_settings

logger = logging.getLogger("granite_asr.server")

# Serialise inference — PyTorch models are not re-entrant across threads
_inference_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class TranscribeRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded audio bytes")
    language: str = Field(default="pt-BR")


class WordResponse(BaseModel):
    word: str
    start: float
    end: float
    speaker: str = "Speaker 0"
    score: float | None = None


class SegmentResponse(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = "Speaker 0"
    confidence: float | None = None
    words: list[WordResponse] | None = None


class TranscribeResponse(BaseModel):
    segments: list[SegmentResponse]
    window_start_s: float = 0.0
    audio_duration_s: float = 0.0


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    device: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_response(result) -> TranscribeResponse:
    """Build a TranscribeResponse from a TranscriptionResult."""
    segments = []
    for s in result.segments:
        words = None
        if s.words:
            words = [
                WordResponse(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    speaker=w.speaker,
                    score=w.score,
                )
                for w in s.words
            ]
        segments.append(
            SegmentResponse(
                start=s.start,
                end=s.end,
                text=s.text,
                speaker=s.speaker,
                confidence=s.confidence,
                words=words,
            )
        )
    return TranscribeResponse(
        segments=segments,
        window_start_s=result.window_start_s,
        audio_duration_s=result.audio_duration_s,
    )


def _decode_b64_audio(audio_b64: str) -> bytes:
    try:
        return base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="audio_b64 is not valid base64")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Granite model on startup."""
    from . import load_model

    logger.info("Startup: loading Granite Speech model")
    await asyncio.to_thread(load_model)
    logger.info("Model ready")
    yield


app = FastAPI(
    title="Granite ASR Microservice",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from . import is_loaded

    settings = get_settings()
    loaded = is_loaded()
    return HealthResponse(
        status="ok" if loaded else "loading",
        model_loaded=loaded,
        model_id=settings.MODEL_ID,
        device=settings.DEVICE,
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """Batch transcription — transcribe entire audio."""
    from . import is_loaded, transcribe as _transcribe

    if not is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready",
        )

    audio_bytes = _decode_b64_audio(req.audio_b64)

    try:
        async with _inference_lock:
            result = await asyncio.to_thread(_transcribe, audio_bytes, language=req.language)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return _build_response(result)


@app.post("/transcribe/live", response_model=TranscribeResponse)
async def transcribe_live(req: TranscribeRequest) -> TranscribeResponse:
    """Cumulative live transcription with silence-based context reset."""
    from . import is_loaded, transcribe_stream

    if not is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready",
        )

    audio_bytes = _decode_b64_audio(req.audio_b64)

    try:
        async with _inference_lock:
            result = await asyncio.to_thread(
                transcribe_stream, audio_bytes, language=req.language
            )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return _build_response(result)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "granite_asr.server:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
        workers=1,
    )
