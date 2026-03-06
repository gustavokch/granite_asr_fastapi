"""Command-line interface for Granite ASR.

Usage:
    python run.py path/to/audio.wav --language en
    python run.py path/to/audio.wav --diarize
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Configure logging to show info from granite_asr
logging.basicConfig(level=logging.INFO)

try:
    import granite_asr
except ImportError:
    # Allow running from source without installation
    sys.path.insert(0, str(Path(__file__).parent))
    import granite_asr


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Granite Speech 3.3 2B")
    parser.add_argument("audio_path", type=Path, help="Path to input audio file")
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Language code (e.g. en, pt-BR). Defaults to environment setting.",
    )
    parser.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Enable speaker diarization (requires HF_TOKEN env var)",
    )
    parser.add_argument(
        "--compile", "-c",
        action="store_true",
        help="Enable torch.compile optimization (faster inference, slower startup)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Batch size for inference (default: env var or 1)",
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=None,
        help="Max new tokens to generate (default: env var or 2048)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output full JSON result instead of text",
    )
    args = parser.parse_args()

    # Apply settings via env vars before loading model
    # Note: These only work if get_settings() hasn't been called/cached yet.
    if args.compile:
        os.environ["GRANITE_USE_COMPILE"] = "true"
    if args.batch_size:
        os.environ["GRANITE_BATCH_SIZE"] = str(args.batch_size)
    if args.max_tokens:
        os.environ["GRANITE_MAX_NEW_TOKENS"] = str(args.max_tokens)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger("granite_asr").setLevel(log_level)
    
    if not args.audio_path.exists():
        print(f"Error: File not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model... (this may take a minute)")
    granite_asr.load_model()

    print(f"Transcribing {args.audio_path}...")
    
    # We will pass diarize=True if the user requested it, 
    # but first we need to implement it in the library.
    # For now, we'll pass it only if the library supports it.
    
    # Check if transcribe accepts 'diarize'
    import inspect
    sig = inspect.signature(granite_asr.transcribe)
    kwargs = {}
    if "diarize" in sig.parameters:
        kwargs["diarize"] = args.diarize
    elif args.diarize:
        print("Warning: Diarization requested but not supported by installed granite_asr version.", file=sys.stderr)

    result = granite_asr.transcribe(
        args.audio_path,
        language=args.language,
        **kwargs
    )

    if args.json:
        # Convert to dict for JSON serialization
        output = {
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker,
                }
                for s in result.segments
            ],
            "audio_duration_s": result.audio_duration_s,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        for seg in result.segments:
            time_str = f"[{seg.start:6.2f}s - {seg.end:6.2f}s]"
            print(f"{time_str} {seg.speaker}: {seg.text}")


if __name__ == "__main__":
    main()
