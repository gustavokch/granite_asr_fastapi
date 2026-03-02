"""Real-world tests for granite_asr against iPhone-recorded Portuguese medical encounters.

Requires:
    - Granite Speech 3.3 2B model weights (~6 GB RAM)
    - test-audios/iphone/*.opus files (9 encounters)

Run:
    cd granite_asr && python -m pytest tests/test_realworld.py -v -s
    cd granite_asr && python -m pytest tests/test_realworld.py -k ENCOUNTER_3 -v -s
    cd granite_asr && python -m pytest tests/ -m "not slow" -v   # skip these
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pytest

import granite_asr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "test-audios" / "iphone"

# Auto-discover .opus files, sorted by stem
_OPUS_FILES: list[Path] = sorted(_AUDIO_DIR.glob("*.opus")) if _AUDIO_DIR.is_dir() else []
_OPUS_IDS: list[str] = [p.stem for p in _OPUS_FILES]

# Common Portuguese words / accented characters used as content indicators.
_PT_INDICATORS = re.compile(
    r"(?:paciente|médic[oa]|hospital|dor|febre|pressão|coração|sangue"
    r"|exame|diagnóstico|tratamento|medicamento|sintoma|queixa"
    r"|não|está|também|é|já|até|você|como|para|mais"
    r"|ção|ções|ência|ância|ável|ível|mente"
    r"|[àáâãéêíóôõúç])",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

skip_no_audio = pytest.mark.skipif(
    not _OPUS_FILES,
    reason=f"No .opus files found in {_AUDIO_DIR}",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_portuguese_content(text: str, min_matches: int = 2) -> bool:
    """Return True if *text* contains at least *min_matches* Portuguese indicators."""
    return len(_PT_INDICATORS.findall(text)) >= min_matches


def _assert_valid_segments(
    result: granite_asr.TranscriptionResult,
    *,
    duration_tolerance: float = 1.0,
) -> None:
    """Common assertions on a successful TranscriptionResult."""
    assert result.segments, "Expected at least 1 segment"
    assert result.audio_duration_s > 0, "audio_duration_s must be positive"

    for seg in result.segments:
        assert seg.text.strip(), "Segment text must be non-empty"
        assert seg.start >= 0, f"start must be >= 0, got {seg.start}"
        assert seg.end > seg.start, f"end ({seg.end}) must be > start ({seg.start})"
        assert seg.end <= result.audio_duration_s + duration_tolerance, (
            f"end ({seg.end}) exceeds duration ({result.audio_duration_s}) + tolerance"
        )
        assert seg.speaker.startswith("SPEAKER_") or seg.speaker == "Speaker 0", f"Unexpected speaker: {seg.speaker}"
        assert seg.confidence is None, f"Granite should not produce confidence, got {seg.confidence}"
        
        # Check words if present
        if seg.words:
            for word in seg.words:
                assert word.word.strip(), "Word text must be non-empty"
                assert word.start >= 0
                assert word.end >= word.start
                assert word.speaker.startswith("SPEAKER_") or word.speaker == "Speaker 0"


ResultOrError = Union[granite_asr.TranscriptionResult, Exception]


def _unwrap(result: ResultOrError, file_id: str) -> granite_asr.TranscriptionResult:
    """Extract a TranscriptionResult or fail the test with the captured error."""
    if isinstance(result, Exception):
        pytest.fail(f"Inference failed for {file_id}: {result!r}")
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def granite_model():
    """Load the Granite Speech model once for the entire session."""
    try:
        granite_asr.load_model()
    except Exception as exc:
        pytest.skip(f"Cannot load Granite model: {exc}")

    assert granite_asr.is_loaded(), "Model should be loaded after load_model()"
    return True  # sentinel — tests just need the fixture to trigger loading


@pytest.fixture(scope="module")
def batch_results(granite_model) -> dict[str, ResultOrError]:
    """Run batch transcription on all opus files, caching results by stem."""
    results: dict[str, ResultOrError] = {}
    for path in _OPUS_FILES:
        try:
            results[path.stem] = granite_asr.transcribe(path, language="pt")
        except Exception as exc:
            results[path.stem] = exc
    return results


@pytest.fixture(scope="module")
def stream_results(granite_model) -> dict[str, ResultOrError]:
    """Run stream transcription on all opus files, caching results by stem."""
    results: dict[str, ResultOrError] = {}
    for path in _OPUS_FILES:
        try:
            results[path.stem] = granite_asr.transcribe_stream(path, language="pt")
        except Exception as exc:
            results[path.stem] = exc
    return results


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


@pytest.mark.slow
@skip_no_audio
class TestBatchTranscription:
    """Batch transcription (`transcribe`) on each encounter."""

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_produces_valid_segments(self, batch_results, file_id):
        result = _unwrap(batch_results[file_id], file_id)
        _assert_valid_segments(result)

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_positive_duration(self, batch_results, file_id):
        result = _unwrap(batch_results[file_id], file_id)
        assert result.audio_duration_s > 0

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_window_start_is_zero(self, batch_results, file_id):
        result = _unwrap(batch_results[file_id], file_id)
        assert result.window_start_s == 0.0, "Batch mode should always have window_start_s == 0"

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_portuguese_content(self, batch_results, file_id):
        result = _unwrap(batch_results[file_id], file_id)
        full_text = " ".join(seg.text for seg in result.segments)
        assert _has_portuguese_content(full_text), (
            f"Expected Portuguese content in batch output for {file_id}, got: {full_text[:200]}"
        )


@pytest.mark.slow
@skip_no_audio
class TestStreamTranscription:
    """Stream transcription (`transcribe_stream`) on each encounter."""

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_produces_valid_segments(self, stream_results, file_id):
        result = _unwrap(stream_results[file_id], file_id)
        _assert_valid_segments(result)

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_positive_duration(self, stream_results, file_id):
        result = _unwrap(stream_results[file_id], file_id)
        assert result.audio_duration_s > 0

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_window_start_non_negative(self, stream_results, file_id):
        result = _unwrap(stream_results[file_id], file_id)
        assert result.window_start_s >= 0, f"window_start_s must be >= 0, got {result.window_start_s}"

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_window_within_audio(self, stream_results, file_id):
        result = _unwrap(stream_results[file_id], file_id)
        assert result.window_start_s <= result.audio_duration_s, (
            f"window_start_s ({result.window_start_s}) exceeds duration ({result.audio_duration_s})"
        )

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_segments_after_window_start(self, stream_results, file_id):
        result = _unwrap(stream_results[file_id], file_id)
        for seg in result.segments:
            assert seg.start >= result.window_start_s, (
                f"Segment start ({seg.start}) < window_start ({result.window_start_s})"
            )

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_portuguese_content(self, stream_results, file_id):
        result = _unwrap(stream_results[file_id], file_id)
        full_text = " ".join(seg.text for seg in result.segments)
        assert _has_portuguese_content(full_text), (
            f"Expected Portuguese content in stream output for {file_id}, got: {full_text[:200]}"
        )


@pytest.mark.slow
@skip_no_audio
class TestBatchVsStream:
    """Cross-API consistency checks."""

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_same_audio_duration(self, batch_results, stream_results, file_id):
        batch = _unwrap(batch_results[file_id], file_id)
        stream = _unwrap(stream_results[file_id], file_id)
        assert batch.audio_duration_s == pytest.approx(stream.audio_duration_s, abs=0.1), (
            f"Duration mismatch: batch={batch.audio_duration_s}, stream={stream.audio_duration_s}"
        )

    @pytest.mark.parametrize("file_id", _OPUS_IDS)
    def test_both_produce_text(self, batch_results, stream_results, file_id):
        batch = _unwrap(batch_results[file_id], file_id)
        stream = _unwrap(stream_results[file_id], file_id)
        batch_text = " ".join(seg.text for seg in batch.segments)
        stream_text = " ".join(seg.text for seg in stream.segments)
        assert batch_text.strip(), f"Batch produced no text for {file_id}"
        assert stream_text.strip(), f"Stream produced no text for {file_id}"


@pytest.mark.slow
@skip_no_audio
class TestBytesInput:
    """Verify that both APIs accept raw bytes (not just file paths)."""

    def _smallest_file(self) -> Path:
        """Return the smallest .opus file to minimize inference time."""
        return min(_OPUS_FILES, key=lambda p: p.stat().st_size)

    def test_batch_from_bytes(self, granite_model):
        path = self._smallest_file()
        audio_bytes = path.read_bytes()
        result = granite_asr.transcribe(audio_bytes, language="pt")
        _assert_valid_segments(result)

    def test_stream_from_bytes(self, granite_model):
        path = self._smallest_file()
        audio_bytes = path.read_bytes()
        result = granite_asr.transcribe_stream(audio_bytes, language="pt")
        _assert_valid_segments(result)
