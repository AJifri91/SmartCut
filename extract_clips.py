#!/usr/bin/env python3
"""
Extract video clips using boundaries from a JSON file.

Default mode uses ffmpeg with -c copy (no re-encode) — fast and lossless,
but cuts snap to the nearest keyframe (typically ±1-2s drift).

Pass --accurate to re-encode with libx264 for frame-perfect cuts
(slower, minor quality loss).

Usage:
  python extract_clips.py --video video.mp4 --boundaries boundaries.json
  python extract_clips.py --video video.mp4 --boundaries boundaries.json --accurate
  python extract_clips.py --video video.mp4 --boundaries boundaries.json --outdir clips --min-duration 24 --max-duration 60
"""

import json
import os
import sys
import argparse
import shutil
import subprocess


def _check_ffmpeg():
    """Fail early with a clear message if ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ERROR: ffmpeg not found on PATH.\n"
            "Install it:\n"
            "  - Windows: choco install ffmpeg  OR  winget install ffmpeg\n"
            "  - macOS:   brew install ffmpeg\n"
            "  - Linux:   sudo apt install ffmpeg"
        )


def _safe_reason(reason: str, score: int) -> str:
    """Sanitize the reason string for use in a filename."""
    safe = "".join(c for c in reason if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
    return safe if safe else f"score{score}"


def _build_output_name(index: int, reason: str, score: int, ext: str = "mp4") -> str:
    return f"clip_{index + 1:02d}_{_safe_reason(reason, score)}.{ext}"


def _extract_copy(video_path: str, start: float, end: float, out_path: str) -> None:
    """Fast stream-copy cut. Snaps to keyframes. No re-encode."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", video_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg (copy) failed: {proc.stderr.strip()}")


def _extract_accurate(video_path: str, start: float, end: float, out_path: str) -> None:
    """Frame-accurate cut via re-encode with libx264. Slower, lossy."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "18",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg (accurate) failed: {proc.stderr.strip()}")


def extract_clips(video_path, boundaries_json, output_dir, min_duration, max_duration, accurate=False):
    """Cut video at start/end times from JSON, saving clips as MP4."""
    # ── Validate inputs ──
    if not os.path.exists(video_path):
        sys.exit(f"Video file not found: {video_path}")
    if not os.path.exists(boundaries_json):
        sys.exit(f"Boundaries JSON not found: {boundaries_json}")
    _check_ffmpeg()

    # ── Load boundaries ──
    try:
        with open(boundaries_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.exit(f"Error reading {boundaries_json}: {e}")

    clips = data.get("clips", [])
    if not clips:
        print("Warning: No clips found in JSON. Nothing to extract.")
        return

    os.makedirs(output_dir, exist_ok=True)

    mode = "ACCURATE (re-encode, slow, frame-perfect)" if accurate else "FAST (stream copy, no re-encode, ±keyframe drift)"
    print(f"Mode: {mode}\n")

    saved = 0
    skipped = 0
    failed = 0

    for i, clip_info in enumerate(clips):
        start = clip_info.get("start")
        end = clip_info.get("end")
        if start is None or end is None:
            print(f"Skipping clip {i+1}: missing start or end time.")
            skipped += 1
            continue

        duration = end - start
        if duration < min_duration or duration > max_duration:
            print(f"Skipping clip {i+1}: duration {duration:.1f}s (not between {min_duration}-{max_duration}s)")
            skipped += 1
            continue

        score = clip_info.get("score", 0)
        reason = clip_info.get("reason", "funny")
        out_name = _build_output_name(i, reason, score)
        out_path = os.path.join(output_dir, out_name)

        extract_fn = _extract_accurate if accurate else _extract_copy
        try:
            extract_fn(video_path, start, end, out_path)
            print(f"Saved: {out_path} ({duration:.1f}s, score {score})")
            saved += 1
        except Exception as e:
            print(f"Failed to extract clip {i+1}: {e}")
            failed += 1

    print(f"\nDone. Saved {saved} clip(s), skipped {skipped}, failed {failed} → '{output_dir}/'.")


def main():
    parser = argparse.ArgumentParser(description="Extract video clips using a boundaries JSON file")
    parser.add_argument("--video",        required=True,                help="Original video file (MP4)")
    parser.add_argument("--boundaries",   required=True,                help="JSON file with clip boundaries")
    parser.add_argument("--outdir",       default="funny_clips",        help="Output directory for clips (default: funny_clips)")
    parser.add_argument("--min-duration", type=float, default=24.0,      help="Minimum clip duration in seconds (default: 24)")
    parser.add_argument("--max-duration", type=float, default=80.0,      help="Maximum clip duration in seconds (default: 80)")
    parser.add_argument("--accurate",     action="store_true",          help="Re-encode with libx264 for frame-perfect cuts (slower)")
    args = parser.parse_args()

    extract_clips(
        video_path=args.video,
        boundaries_json=args.boundaries,
        output_dir=args.outdir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        accurate=args.accurate,
    )


if __name__ == "__main__":
    main()
