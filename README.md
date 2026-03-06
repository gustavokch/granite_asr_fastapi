# granite_asr

Self-contained Python module for speech transcription using [IBM Granite Speech 3.3 2B](https://huggingface.co/ibm-granite/granite-speech-3.3-2b), a 3B-parameter speech-language model. Designed for CPU inference with optional GPU acceleration. Includes a FastAPI microservice wrapper for network access.

## Features

- **Batch transcription** — transcribe complete audio files in a single pass
- **Live (cumulative) transcription** — poll-based streaming with silence-based context reset to keep inference windows bounded on CPU
- **Dynamic Batching** — aggregates concurrent requests into efficient batches for higher throughput
- **Optimized Inference** — supports `torch.compile` for faster execution on modern PyTorch versions
- **Speaker Diarization** — separates speakers using `pyannote/speaker-diarization-3.1`
- **Forced Alignment** — generates word-level timestamps using `wav2vec2`
- **VAD (Voice Activity Detection)** — filters non-speech regions using `silero-vad`
- **RMS silence detection** — automatically finds speaker pauses (>=1s) and trims the transcription window, preventing unbounded inference growth during long recordings
- **Multi-format audio** — WAV, OGG/Opus, FLAC, MP3 via torchaudio backends; auto-resampling to 16kHz mono
- **Thread-safe inference** — all model access serialized under a single lock
- **Lazy imports** — `import granite_asr` does not load torch/torchaudio; heavy dependencies are deferred until first use
- **Configurable via environment** — all settings overridable with `GRANITE_` prefixed env vars

## Requirements

- Python 3.10+
- ~6 GB RAM for FP16 model on CPU
- Hugging Face Token (for diarization model access)

Install dependencies:

```bash
pip install -r requirements.txt
```

The default `requirements.txt` pulls CPU-only PyTorch wheels. For CUDA support, replace the `--extra-index-url` line:

```
--extra-index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### CLI Interface

```bash
export HF_TOKEN=your_token_here
python -m granite_asr.run audio.wav --diarize
```

### As a Python library

```python
import granite_asr

# Load model (blocking, ~30-60s on CPU cold start)
granite_asr.load_model()

# Batch transcription from file
result = granite_asr.transcribe("recording.wav", language="pt-BR")
for seg in result.segments:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}: {seg.text}")
    for word in seg.words:
        print(f"  {word.start:.2f}s: {word.word}")

# Batch transcription from bytes
with open("recording.ogg", "rb") as f:
    result = granite_asr.transcribe(f.read())

# Live (cumulative) transcription with silence-based context reset
result = granite_asr.transcribe_stream(audio_bytes, language="pt-BR")
print(f"Window starts at {result.window_start_s:.1f}s")
```

### As a microservice

```bash
# Start the server (loads model on startup)
python -m granite_asr.server

# Or with uvicorn directly
uvicorn granite_asr.server:app --host 0.0.0.0 --port 8010 --workers 1
```

> **Important:** Always use `workers=1`. PyTorch models are not re-entrant across processes, and the server already serializes concurrent async requests with an `asyncio.Lock`.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model status, device info |
| `POST` | `/transcribe` | Batch transcription (full audio) |
| `POST` | `/transcribe/live` | Cumulative transcription with silence-based window trimming |

#### Request format

```json
{
  "audio_b64": "<base64-encoded audio bytes>",
  "language": "pt-BR"
}
```

#### Response format

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Paciente apresenta dor torácica.",
      "speaker": "Speaker 0",
      "confidence": null
    }
  ],
  "window_start_s": 0.0,
  "audio_duration_s": 3.5
}
```

`window_start_s` is only meaningful for `/transcribe/live` — it indicates where the silence-trimmed transcription window begins within the cumulative buffer. For `/transcribe` it is always `0.0`.

## Public API

### Functions

| Function | Description |
|----------|-------------|
| `load_model()` | Load the Granite model into memory. Idempotent — safe to call multiple times. Blocks ~30-60s on CPU. |
| `is_loaded()` | Returns `True` if the model is ready for inference. |
| `transcribe(audio, *, language=None)` | Batch transcription. `audio` accepts `bytes`, `str` (file path), or `Path`. Returns `TranscriptionResult`. |
| `transcribe_stream(audio, *, language=None)` | Cumulative transcription with silence-based context reset. Same input types. Returns `TranscriptionResult` with `window_start_s` set. |
| `get_settings()` | Returns the cached `GraniteSettings` instance. |

### Data classes

```python
@dataclass
class Segment:
    start: float              # Start time in seconds
    end: float                # End time in seconds
    text: str                 # Transcribed text
    speaker: str = "Speaker 0"
    confidence: float | None = None  # Always None (Granite has no confidence scores)

@dataclass
class TranscriptionResult:
    segments: list[Segment]
    window_start_s: float = 0.0    # Silence-trimmed window offset (live mode)
    audio_duration_s: float = 0.0  # Total audio duration
```

## How Live Transcription Works

Live transcription is designed for polling-based streaming where a client accumulates audio and periodically sends the full buffer for transcription. Without silence detection, the inference window would grow unbounded as the recording progresses, making each poll slower.

The solution: **silence-based context reset**.

1. The cumulative audio buffer is analyzed for silence regions (RMS energy below threshold for >= 1 second)
2. The transcription window is set to start after the last qualifying silence boundary
3. Only the windowed audio is passed to the model for inference
4. Segment timestamps in the response are adjusted to be relative to the full buffer (not the window)

This keeps inference time roughly constant regardless of total recording duration.

```
Full buffer: [===SPEECH===][--silence--][===SPEECH===][--silence--][==SPEECH==]
                                                                    ↑
                                                            window_start_s
                                                       Only this portion is
                                                       sent to the model
```

### Silence detection parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SILENCE_THRESHOLD_RMS` | `0.01` | RMS energy below this is considered silence |
| `SILENCE_MIN_DURATION_S` | `1.0` | Minimum silence duration to qualify as a boundary |
| `SILENCE_FRAME_DURATION_S` | `0.02` | Analysis frame size (20ms = 320 samples at 16kHz) |

## Configuration

All settings use Pydantic `BaseSettings` with the `GRANITE_` environment variable prefix.

### Model settings

| Env var | Default | Description |
|---------|---------|-------------|
| `GRANITE_MODEL_ID` | `ibm-granite/granite-speech-3.3-2b` | HuggingFace model identifier |
| `GRANITE_DEVICE` | `cpu` | PyTorch device (`cpu`, `cuda`, `cuda:0`, `mps`) |
| `GRANITE_TORCH_DTYPE` | `float16` | Model precision: `float16`, `bfloat16`, `float32` |
| `GRANITE_MODEL_CACHE_DIR` | _(none)_ | Local directory for caching model weights |

### Inference settings

| Env var | Default | Description |
|---------|---------|-------------|
| `GRANITE_MAX_NEW_TOKENS` | `2048` | Maximum tokens generated per transcription |
| `GRANITE_DEFAULT_LANGUAGE` | `pt-BR` | Default language when not specified by caller |
| `GRANITE_SYSTEM_PROMPT` | _(Portuguese medical assistant)_ | System prompt for the chat template |

### Optimization settings

| Env var | Default | Description |
|---------|---------|-------------|
| `GRANITE_USE_COMPILE` | `false` | Enable `torch.compile` (requires PyTorch 2.0+) |
| `GRANITE_BATCH_SIZE` | `1` | Max requests to batch together |
| `GRANITE_BATCH_TIMEOUT` | `0.1` | Max wait time (seconds) to fill a batch |

### Silence detection settings

| Env var | Default | Description |
|---------|---------|-------------|
| `GRANITE_SILENCE_THRESHOLD_RMS` | `0.01` | RMS threshold for silence |
| `GRANITE_SILENCE_MIN_DURATION_S` | `1.0` | Minimum silence duration (seconds) |
| `GRANITE_SILENCE_FRAME_DURATION_S` | `0.02` | RMS frame size (seconds) |

### Diarization settings

| Env var | Default | Description |
|---------|---------|-------------|
| `GRANITE_DIARIZATION_MODEL_ID` | `pyannote/speaker-diarization-3.1` | Pyannote model identifier |
| `GRANITE_HF_TOKEN` | _(none)_ | HuggingFace access token (required for diarization) |
| `GRANITE_ALIGN_MODEL_ID` | `alinerodrigues/...` | Alignment model identifier |

### Server settings

| Env var | Default | Description |
|---------|---------|-------------|
| `GRANITE_HOST` | `0.0.0.0` | Server bind address |
| `GRANITE_PORT` | `8010` | Server port |

## Supported Languages

The model accepts a language parameter that is converted to a prompt instruction. Explicitly supported languages:

| Code | Language |
|------|----------|
| `pt-BR` | Brazilian Portuguese (default) |
| `pt` | Portuguese |
| `en` | English |
| `es` | Spanish |
| `fr` | French |
| `de` | German |

Other language codes fall back to the `pt-BR` instruction.

## Architecture

```
granite_asr/
├── __init__.py       # Public API with lazy imports
├── model.py          # Model loading + thread-safe inference
├── diarization.py    # Speaker diarization via pyannote
├── alignment.py      # Forced alignment via wav2vec2
├── vad.py            # Voice Activity Detection via Silero
├── silence.py        # RMS silence detection + window extraction
├── audio.py          # Audio decoding via torchaudio (format conversion, resampling)
├── config.py         # GraniteSettings (Pydantic BaseSettings, GRANITE_ prefix)
├── schemas.py        # Segment, TranscriptionResult, WordSegment dataclasses
├── server.py         # FastAPI microservice wrapper
├── run.py            # Command-line interface
├── requirements.txt  # Python dependencies (CPU PyTorch by default)
└── tests/
    ├── conftest.py
    ├── test_diarization.py
    ├── test_alignment.py
    ├── test_silence.py
    └── test_audio.py
```

### Thread safety

The module uses a tiered concurrency model:

1. **`threading.Lock`** in `model.py` — serializes all PyTorch access. All inference (including batch inference) runs inside this lock to ensure thread safety for the underlying model.
2. **`BatchManager`** in `server.py` — replaces the global request lock. Concurrent requests are collected into an `asyncio.Queue`, grouped into batches (up to `GRANITE_BATCH_SIZE`), and dispatched to the model as a single operation.

### Lazy imports

`import granite_asr` is lightweight — it does not import `torch`, `torchaudio`, or `transformers`. Heavy dependencies are loaded inside function bodies only when `load_model()`, `transcribe()`, or `transcribe_stream()` are called. This allows the module to be referenced without installing PyTorch (useful for the EEVA backend, which imports the module name for routing but communicates via HTTP).

## EEVA Backend Integration

When used within the EEVA application, the Granite microservice runs as a separate process. The backend communicates with it via HTTP through `graniteClient.py`:

```
┌─────────────┐     WebSocket      ┌─────────────────┐     HTTP POST      ┌─────────────────┐
│   Frontend   │ ──── audio ─────► │   EEVA Backend   │ ──── base64 ────► │  granite_asr     │
│  (browser)   │ ◄── events ────── │ (graniteClient)  │ ◄── segments ──── │  (server.py)     │
└─────────────┘                    └─────────────────┘                    └─────────────────┘
                                   Polls every 3s                         Runs inference on
                                   with cumulative                        silence-trimmed
                                   audio buffer                           audio window
```

### Backend settings

| Env var | Default | Description |
|---------|---------|-------------|
| `GRANITE_ASR_URL` | `http://localhost:8010` | Microservice base URL |
| `GRANITE_LANGUAGE` | `pt-BR` | Transcription language |
| `GRANITE_POLL_INTERVAL_S` | `3.0` | Live polling interval (seconds) |
| `GRANITE_TIMEOUT_S` | `120.0` | HTTP request timeout (seconds) |

### Enabling Granite as the ASR provider

In your `.env` file:

```bash
# Live transcription (streaming via polling)
ASR_LIVE_PROVIDER=granite

# Batch transcription (file uploads)
ASR_BATCH_PROVIDER=granite
```

## Testing

```bash
# Run all tests (silence detection tests run without PyTorch)
cd granite_asr
python -m pytest tests/ -v

# Audio decoding tests require torchaudio — they're skipped automatically if unavailable
```

Test coverage:
- **14 tests** for silence detection (RMS computation, silence region finding, window extraction)
- **4 tests** for audio decoding (mono/stereo, resampling, error handling) — skipped without torchaudio
- Edge cases: empty audio, sub-frame audio, trailing silence fallback, only-silence input

## Limitations

- **No confidence scores** — the `confidence` field is always `None`.
- **Greedy decoding** — uses `do_sample=False, num_beams=1` for deterministic, fast output.
- **Batched Inference** — requests are serialized and batched; concurrent throughput is high, but single-request latency includes batch accumulation time (`GRANITE_BATCH_TIMEOUT`).
- **Polling latency** — live transcription has a minimum latency of `GRANITE_POLL_INTERVAL_S` (default 3s), unlike WebSocket-based providers that stream results in real time.
