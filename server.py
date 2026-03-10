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
import time
from contextlib import asynccontextmanager
from typing import NamedTuple

from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from pydantic import BaseModel, Field

from .config import get_settings

logger = logging.getLogger("granite_asr.server")

# Global batch manager
_batch_manager = None

# ---------------------------------------------------------------------------
# Batching Logic
# ---------------------------------------------------------------------------

class BatchRequest(NamedTuple):
    waveform: object # numpy array
    language: str
    future: asyncio.Future


class BatchManager:
    def __init__(self, batch_size: int, timeout: float):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._process_batches())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, waveform, language: str):
        future = self.loop.create_future()
        await self.queue.put(BatchRequest(waveform, language, future))
        return await future

    async def _process_batches(self):
        from .model import run_batch_inference
        
        while True:
            batch: list[BatchRequest] = []
            try:
                # Wait for first item
                item = await self.queue.get()
                batch.append(item)
                
                # Collect more items with timeout
                deadline = time.time() + self.timeout
                while len(batch) < self.batch_size:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    continue

                # Prepare batch
                waveforms = [req.waveform for req in batch]
                languages = [req.language for req in batch]
                
                # Run inference
                try:
                    # Run in thread pool to avoid blocking event loop
                    results = await asyncio.to_thread(
                        run_batch_inference, waveforms, languages=languages
                    )
                    
                    # Resolve futures
                    for req, result in zip(batch, results):
                        if not req.future.done():
                            req.future.set_result(result)
                            
                except Exception as e:
                    logger.exception("Batch inference failed")
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(e)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in batch loop: %s", e)
                await asyncio.sleep(0.1) # Backoff


def run_inference_proxied(waveform, language="pt-BR"):
    """Thread-safe proxy for run_inference that submits to the batch manager."""
    if _batch_manager is None:
        raise RuntimeError("Batch manager not initialized")
        
    # Submit to batch manager running in main loop
    future = asyncio.run_coroutine_threadsafe(
        _batch_manager.submit(waveform, language), 
        _batch_manager.loop
    )
    return future.result()


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
    global _batch_manager
    from . import load_model
    import granite_asr.model

    logger.info("Startup: loading Granite Speech model")
    await asyncio.to_thread(load_model)
    
    # Initialize batch manager
    settings = get_settings()
    _batch_manager = BatchManager(
        batch_size=settings.BATCH_SIZE,
        timeout=settings.BATCH_TIMEOUT
    )
    _batch_manager.start()
    logger.info("Batch manager started (size=%d, timeout=%.2fs)", 
                settings.BATCH_SIZE, settings.BATCH_TIMEOUT)
    
    # Monkey-patch run_inference to use batch manager
    # This allows existing sync code (transcribe, transcribe_stream) to transparently batch
    original_run_inference = granite_asr.model.run_inference
    granite_asr.model.run_inference = run_inference_proxied
    
    logger.info("Model ready")
    yield
    
    # Cleanup
    if _batch_manager:
        await _batch_manager.stop()
    granite_asr.model.run_inference = original_run_inference


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
        device=settings.resolved_device,
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
        # Run in thread, run_inference inside will use batch manager
        result = await asyncio.to_thread(_transcribe, audio_bytes, language=req.language)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return _build_response(result)


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    audio: UploadFile = File(...), language: str = Form("pt-BR")
) -> TranscribeResponse:
    """Transcription via direct file upload (multipart/form-data)."""
    from . import is_loaded, transcribe as _transcribe

    if not is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready",
        )

    try:
        audio_bytes = await audio.read()
        # Run in thread, run_inference inside will use batch manager
        result = await asyncio.to_thread(_transcribe, audio_bytes, language=language)
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
        # Run in thread, run_inference inside will use batch manager
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
