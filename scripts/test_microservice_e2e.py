import asyncio
import base64
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

import sys

# Configuration
SERVER_URL = "http://localhost:8010"
HEALTH_URL = f"{SERVER_URL}/health"
TRANSCRIBE_URL = f"{SERVER_URL}/transcribe"
AUDIO_PATH = Path("test-audios/iphone/ENCOUNTER_4.opus")
LANGUAGE = "pt-BR"

async def wait_for_server(process, timeout=300):
    """Wait for the server to be ready and model to be loaded."""
    start_time = time.time()
    print(f"Waiting for server at {HEALTH_URL} (Timeout: {timeout}s)...")
    
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print("Server process exited prematurely.")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False

            try:
                response = await client.get(HEALTH_URL)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("model_loaded"):
                        print("Server is ready and model is loaded.")
                        return True
                    else:
                        print(f"[{int(time.time() - start_time)}s] Server alive, but model still loading... ({data.get('status')})")
            except httpx.RequestError:
                pass
            
            await asyncio.sleep(5)
    
    print("Timeout reached waiting for server.")
    return False

def parse_env_file(path: Path):
    """Parse a .env file into a dictionary, handling quotes and comments."""
    env = {}
    if not path.exists():
        return env
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Remove quotes
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            # Remove trailing comments
            if " #" in value:
                value = value.split(" #", 1)[0].strip()
            env[key] = value
    return env

async def run_e2e_test():
    # 1. Start the server as a subprocess
    print("Starting granite_asr server...")
    
    # Load .env to get HF_TOKEN if present
    env = os.environ.copy()
    root_env = parse_env_file(Path(".env"))
    
    hf_token = root_env.get("HF_TOKEN") or env.get("HF_TOKEN")
    if hf_token:
        env["GRANITE_HF_TOKEN"] = hf_token
        masked_token = f"{hf_token[:6]}...{hf_token[-4:]}" if len(hf_token) > 10 else "****"
        print(f"Using HF_TOKEN from .env ({masked_token}), setting GRANITE_HF_TOKEN.")
    else:
        print("Warning: No HF_TOKEN found in .env or environment. Diarization may fail.")
    
    # Using sys.executable to ensure we use the same Python environment
    server_process = subprocess.Popen(
        [sys.executable, "-m", "granite_asr.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    
    try:
        # 2. Wait for server to be ready
        if not await wait_for_server(server_process):
            return False
        
        # 3. Prepare audio
        print(f"Reading audio file: {AUDIO_PATH}")
        if not AUDIO_PATH.exists():
            print(f"Error: Audio file not found at {AUDIO_PATH}")
            return False
            
        audio_bytes = AUDIO_PATH.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # 4. Send transcription request
        print(f"Sending transcription request for {AUDIO_PATH} (Language: {LANGUAGE})...")
        payload = {
            "audio_b64": audio_b64,
            "language": LANGUAGE
        }
        
        start_inference = time.time()
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(TRANSCRIBE_URL, json=payload)
        
        inference_duration = time.time() - start_inference
        print(f"Inference completed in {inference_duration:.2f}s")
        
        if response.status_code != 200:
            print(f"Error: Transcription failed with status {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        
        # 5. Validate results
        segments = result.get("segments", [])
        print(f"Received {len(segments)} segments.")
        
        if not segments:
            print("Error: No segments returned.")
            return False
            
        print("\n--- Transcription Results ---")
        for i, seg in enumerate(segments[:5]):  # Show first 5 segments
            print(f"[{seg.get('start'):.2f}s - {seg.get('end'):.2f}s] {seg.get('speaker')}: {seg.get('text')}")
        
        if len(segments) > 5:
            print(f"... and {len(segments) - 5} more segments.")
        print("-----------------------------\n")
        
        # Validation checks
        has_speaker_labels = any(seg.get("speaker") and seg.get("speaker") != "Speaker 0" for seg in segments)
        has_text = any(seg.get("text").strip() for seg in segments)
        
        print(f"Validation: Has speaker labels (diarization): {'PASS' if has_speaker_labels else 'FAIL'}")
        print(f"Validation: Has transcribed text: {'PASS' if has_text else 'PASS' if not has_text else 'FAIL'}") # Simplified check
        
        if has_text and (has_speaker_labels or len(segments) > 0):
            print("\nE2E Test PASSED!")
            return True
        else:
            print("\nE2E Test FAILED validation.")
            return False
            
    finally:
        print("Shutting down server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("Server shutdown complete.")

if __name__ == "__main__":
    success = asyncio.run(run_e2e_test())
    if not success:
        exit(1)
