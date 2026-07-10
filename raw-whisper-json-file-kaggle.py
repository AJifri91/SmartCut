# ============================================================
# CELL 1 — Install dependencies
# ============================================================
!pip install transformers accelerate librosa pydub soundfile torchaudio -q
!apt-get install -y ffmpeg -q

# ============================================================
# CELL 2 — Full pipeline (copy everything below into one cell)
# ============================================================

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from pydub import AudioSegment
from transformers import pipeline, Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — EDIT THIS
# ============================================================

# Set this to the path of your video in Kaggle.
# If you uploaded via "Add Data", the path will be:
#   /kaggle/input/your-dataset-name/your_video.mp4
VIDEO_PATH = "/kaggle/input/your-dataset-name/your_video.mp4"

# Merge segments to this duration (seconds). 20s gives good flexibility.
MIN_DURATION = 20.0

# ============================================================

# ── Constants ──────────────────────────────────────────────
MODEL_ID     = "openai/whisper-large-v3-turbo"
HOP_LENGTH   = 512

# Volume / intensity thresholds
_VOL_QUIET  = 0.1
_VOL_LOUD   = 0.3
_ZCR_HIGH   = 0.15
_CENT_HIGH  = 2_000.0


# ── Extract audio ──────────────────────────────────────────

def extract_audio(video_path: Path) -> Path:
    out = Path("/kaggle/working") / (video_path.stem + "_extracted.wav")
    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", str(out),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
    log.info("Audio extracted → %s", out)
    return out


# ── Load Whisper ───────────────────────────────────────────

def load_whisper(model_id: str = MODEL_ID) -> Pipeline:
    device    = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype     = torch.float16 if device == "cuda:0" else torch.float32
    log.info("Loading %s on %s", model_id, device)
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=dtype,
    )


# ── Feature matrices ───────────────────────────────────────

def compute_feature_matrices(audio_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    log.info("Computing audio feature matrices (single pass)...")
    y, sr = librosa.load(str(audio_path), sr=16_000, mono=True)
    rms  = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    zcr  = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    log.info("Feature matrices ready — %d frames", len(rms))
    return rms, zcr, cent, sr


def _classify_volume(v: float) -> str:
    if v < _VOL_QUIET: return "quiet"
    if v < _VOL_LOUD:  return "normal"
    return "loud"


def _classify_intensity(zcr: float, cent: float) -> str:
    return "high" if (zcr > _ZCR_HIGH and cent > _CENT_HIGH) else "normal"


def get_features(start: float, end: float, rms: np.ndarray, zcr: np.ndarray, cent: np.ndarray, sr: int) -> dict:
    s = max(0, int(start * sr / HOP_LENGTH))
    e = max(s + 1, int(end * sr / HOP_LENGTH))
    avg_rms  = float(np.mean(rms[s:e]))
    avg_zcr  = float(np.mean(zcr[s:e]))
    avg_cent = float(np.mean(cent[s:e]))
    return {
        "volume": {
            "level": _classify_volume(avg_rms),
            "value": avg_rms,
        },
        "characteristics": {
            "intensity":          _classify_intensity(avg_zcr, avg_cent),
            "zero_crossing_rate": avg_zcr,
            "spectral_centroid":  avg_cent,
        },
    }


# ── Transcribe ─────────────────────────────────────────────

def transcribe(pipe: Pipeline, audio_path: Path) -> list[dict]:
    log.info("Transcribing with %s ...", MODEL_ID)
    raw = pipe(
        str(audio_path),
        return_timestamps=True,
        generate_kwargs={"task": "transcribe"},
    )

    segments = []
    for chunk in raw["chunks"]:
        start, end = chunk["timestamp"]
        if start is None or end is None:
            log.warning("Skipping chunk with None timestamp: %r", chunk)
            continue
        if end <= start:
            end = start + 0.1
        segments.append({
            "start": start,
            "end":   end,
            "text":  chunk["text"],
        })

    log.info("Got %d raw segments from Whisper", len(segments))
    return segments


# ── Attach features + merge ────────────────────────────────

def _merge_features(bucket: list[dict]) -> dict:
    avg_rms  = float(np.mean([s["audio_features"]["volume"]["value"] for s in bucket]))
    avg_zcr  = float(np.mean([s["audio_features"]["characteristics"]["zero_crossing_rate"] for s in bucket]))
    avg_cent = float(np.mean([s["audio_features"]["characteristics"]["spectral_centroid"] for s in bucket]))
    return {
        "volume": {
            "level": _classify_volume(avg_rms),
            "value": avg_rms,
        },
        "characteristics": {
            "intensity":          _classify_intensity(avg_zcr, avg_cent),
            "zero_crossing_rate": avg_zcr,
            "spectral_centroid":  avg_cent,
        },
    }


def attach_and_merge(
    segments: list[dict],
    rms: np.ndarray, zcr: np.ndarray, cent: np.ndarray, sr: int,
    min_duration: float = MIN_DURATION,
    min_valid_duration: float = 0.05,
) -> list[dict]:
    # Attach features
    for seg in segments:
        seg["audio_features"] = get_features(
            seg["start"], seg["end"], rms, zcr, cent, sr
        )

    # Greedy merge
    merged: list[dict] = []
    bucket: list[dict] = []

    for seg in segments:
        bucket.append(seg)
        span = bucket[-1]["end"] - bucket[0]["start"]

        if span >= min_duration:
            merged.append(_flush_bucket(bucket, min_valid_duration))
            bucket = []

    if bucket:
        result = _flush_bucket(bucket, min_valid_duration)
        if result:
            merged.append(result)

    log.info("Merged into %d output segments (min %.1fs each)", len(merged), min_duration)
    return merged


def _flush_bucket(bucket: list[dict], min_valid_duration: float) -> Optional[dict]:
    duration = bucket[-1]["end"] - bucket[0]["start"]
    if duration < min_valid_duration:
        log.warning("Dropping ghost segment: %.4fs", duration)
        return None
    return {
        "start": bucket[0]["start"],
        "end":   bucket[-1]["end"],
        "text":  " ".join(s["text"].strip() for s in bucket),
        "audio_features": _merge_features(bucket),
    }


# ── Save JSON ──────────────────────────────────────────────

def save_json(segments: list[dict], out_path: Path) -> None:
    out_path.write_text(
        json.dumps(segments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved %d segments → %s", len(segments), out_path)


# ── Orchestrator ───────────────────────────────────────────

def run():
    video_path = Path(VIDEO_PATH)

    if not video_path.exists():
        log.error("Video file not found: %s", video_path)
        log.error("Check VIDEO_PATH at the top of the notebook.")
        return

    audio_path = extract_audio(video_path)
    pipe = load_whisper()
    segments = transcribe(pipe, audio_path)
    rms, zcr, cent, sr = compute_feature_matrices(audio_path)
    final_segments = attach_and_merge(segments, rms, zcr, cent, sr, min_duration=MIN_DURATION)

    out_path = Path("/kaggle/working") / f"{video_path.stem}.min{int(MIN_DURATION)}.enhanced_transcription.json"
    save_json(final_segments, out_path)
    log.info("Done. Saved to %s", out_path)


# ── Run ────────────────────────────────────────────────────
run()