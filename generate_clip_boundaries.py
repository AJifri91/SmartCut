#!/usr/bin/env python3
"""
Generate clip boundaries using DeepSeek (OpenRouter) from raw Whisper JSON.
Splits transcript into overlapping windows, processes each, merges and re-ranks.

Usage:
  python generate_clip_boundaries.py --input raw_segments.json --output boundaries.json
  python generate_clip_boundaries.py --input raw_segments.json --window_minutes 10
"""

import json
import os
import re
import sys
import argparse
import time
from openai import OpenAI
from dotenv import load_dotenv

# ─── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# ─── Constants ─────────────────────────────────────────────────────────────────
TOL = 0.05  # Timestamp tolerance for floating-point drift (50ms)

# ─── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
ROLE
You are an expert comedy video editor. Your job is to scan a transcript
and find the moments that would make the best viral short-form clips.

INPUT FORMAT
You will receive a plain-text transcript where each line represents
one audio segment in this exact format:

[start-end] vol=<level>, int=<intensity> | <transcript text>

Example:
[120.5-128.2] vol=normal, int=high | So I told him, "That's not a cat, that's my uncle!"

Fields:
- start / end : timestamps in seconds (one decimal place)
- vol         : quiet | normal | loud
- int         : normal | high
- text        : the spoken words (may be any language)

TASK
Find the strongest viral clip candidates from this transcript window.
Each clip must:
- Represent a complete comedic unit: a full joke, a funny story,
  a clever observation, a sarcastic comeback, or a moment of physical
  comedy that is well described in the text.
- Be formed by merging contiguous segments
- Have end - start between 24 and 80 seconds (hard constraint)
- Not overlap another clip in your output by more than 5 seconds

UNIVERSAL COMEDY SIGNALS — in order of weight
1. Explicit laughter or audience reaction indicated in the text
   (e.g. "(laughter)", "[applause]", "hahaha", "laughs", "crowd erupts")
2. A clear setup → punchline structure, even if subtle
3. Unexpected twists, ironic reversals, or absurd conclusions
4. Wordplay, puns, double meanings, or phonetic jokes
5. Sarcasm, deadpan delivery, or mock seriousness
6. Exaggeration, understatement, or comedic repetition
7. Observational humour about everyday life (relationships,
   work, technology, social norms)
8. Physical comedy or character voices described in the text
9. High energy shift (vol=loud or int=high) that coincides with
   a comedic peak — use only as confirmation, never as primary signal
10. Callback to an earlier joke — use only if the callback itself
    is self-contained enough to be understood without context

SCORING RUBRIC (0‑10)
1–3 : Mildly amusing, no clear punchline or complete arc
4–5 : A decent joke or funny moment, but payoff is weak or too slow
6–7 : Good joke with identifiable setup, build-up, and clear payoff
8–9 : Excellent joke, strong comedic arc, likely audience reaction,
      highly shareable
10   : Perfect viral moment — massive audience reaction, flawless
       structure, universally funny, could stand alone without context

TIMESTAMP RULES
- Use start and end times exactly as they appear in the transcript
- Do not interpolate, round, or estimate
- Every clip's start must equal a segment's start value
- Every clip's end must equal a segment's end value

OUTPUT RULES
- Return JSON only — no markdown, no explanation, no text outside
  the JSON object
- Never return a clip where end - start < 24 or end - start > 80
- Never cut a joke in half — the punchline must be inside the clip
- Return between 3 and 10 clips per window — fewer only if the
  window genuinely contains fewer than 3 funny moments
- Clips must be ordered by score descending

OUTPUT STRUCTURE
{
  "clips": [
    {
      "start": 120.5,
      "end": 155.0,
      "score": 9,
      "reason": "audience erupts at punchline — sarcastic workplace observation with perfect deadpan delivery"
    }
  ]
}
"""

# ─── Segment loader ───────────────────────────────────────────────────────────

def load_segments(json_path):
    """Load and validate raw Whisper JSON. Returns list of segments."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except FileNotFoundError:
        sys.exit(f"ERROR: File not found: {json_path}")
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: Invalid JSON in {json_path}: {e}")
    except Exception as e:
        sys.exit(f"ERROR reading {json_path}: {e}")

    if not isinstance(segments, list):
        sys.exit(f"ERROR: Expected a JSON list of segments, got {type(segments)}")
    if len(segments) == 0:
        sys.exit("ERROR: Segment list is empty.")

    return segments


# ─── Transcript builder ───────────────────────────────────────────────────────

def build_window_transcript(segments, win_start, win_end):
    """
    Build flat transcript string and valid timestamp sets
    for segments overlapping [win_start, win_end].
    """
    lines = []
    valid_starts = set()
    valid_ends = set()

    for seg in segments:
        seg_start = seg.get("start")
        seg_end   = seg.get("end")
        text      = seg.get("text", "").strip()

        if seg_start is None or seg_end is None or not text:
            continue
        if seg_end < win_start or seg_start > win_end:
            continue

        features  = seg.get("audio_features", {})
        vol       = features.get("volume", {}).get("level", "unknown")
        intensity = features.get("characteristics", {}).get("intensity", "unknown")

        lines.append(f"[{seg_start:.1f}-{seg_end:.1f}] vol={vol}, int={intensity} | {text}")
        valid_starts.add(round(seg_start, 1))
        valid_ends.add(round(seg_end, 1))

    return "\n".join(lines), valid_starts, valid_ends


# ─── Helper: timestamp tolerance ─────────────────────────────────────────────

def _near(value: float, valid_set: set) -> bool:
    """Check if value is within TOL of any timestamp in valid_set."""
    return any(abs(value - v) < TOL for v in valid_set)


# ─── DeepSeek call ────────────────────────────────────────────────────────────

def call_deepseek(transcript_text, api_key, window_label, max_retries=3):
    """Call DeepSeek for one window. Returns list of clip dicts."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    user_message = f"TRANSCRIPT:\n{transcript_text}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,
                max_tokens=6000,
            )
            content = response.choices[0].message.content

            # Strip markdown fences robustly
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            content = content.strip()

            data = json.loads(content)

            if not isinstance(data, dict) or "clips" not in data:
                raise ValueError("Response missing 'clips' key")
            if not isinstance(data["clips"], list):
                raise ValueError("'clips' is not a list")

            return data["clips"]

        except Exception as e:
            print(f"    Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                print(f"    WARNING: window {window_label} returned no clips after "
                      f"{max_retries} attempts — this time range will not be represented.")
                return []
            time.sleep(2 * (attempt + 1))

    return []


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_clips(clips, valid_starts, valid_ends, window_label):
    """
    Drop clips with invalid duration or timestamps.
    Uses TOL tolerance for floating-point drift.
    Returns (validated_clips, dropped_count).
    """
    validated = []
    dropped = 0

    for clip in clips:
        start = round(clip.get("start", 0), 1)
        end = round(clip.get("end", 0), 1)
        duration = end - start

        # Duration check
        if not (24 <= duration <= 80):
            print(f"    Dropping [{window_label}] {start}–{end}: duration {duration:.1f}s outside 24–60s")
            dropped += 1
            continue

        # Timestamp check with tolerance
        if not _near(start, valid_starts):
            print(f"    Dropping [{window_label}] {start}–{end}: start {start} not found in transcript (tolerance {TOL}s)")
            dropped += 1
            continue

        if not _near(end, valid_ends):
            print(f"    Dropping [{window_label}] {start}–{end}: end {end} not found in transcript (tolerance {TOL}s)")
            dropped += 1
            continue

        validated.append(clip)

    if dropped > 0:
        print(f"    [{window_label}] Dropped {dropped} invalid clips, kept {len(validated)}")

    return validated, dropped


# ─── Deduplication ────────────────────────────────────────────────────────────

def deduplicate_clips(all_clips, overlap_threshold=10):
    """
    Remove overlapping and near-duplicate clips, keeping higher-scored one.

    Two clips are considered duplicates if:
    - They truly overlap (clip starts before previous clip ends), OR
    - Both start and end timestamps are within overlap_threshold seconds
      of each other (same joke, slightly different cut from window overlap)
    """
    if not all_clips:
        return []

    all_clips.sort(key=lambda x: x["start"])

    merged = []
    for clip in all_clips:
        if not merged:
            merged.append(clip)
            continue

        last = merged[-1]

        truly_overlaps = clip["start"] < last["end"] - 2

        near_duplicate = (
            abs(clip["start"] - last["start"]) < overlap_threshold and
            abs(clip["end"]   - last["end"])   < overlap_threshold
        )

        if truly_overlaps or near_duplicate:
            if clip.get("score", 0) > last.get("score", 0):
                merged[-1] = clip
        else:
            merged.append(clip)

    return merged


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract funny clip boundaries from a Whisper transcript using DeepSeek."
    )
    parser.add_argument("--input",          required=True,
                        help="Raw Whisper JSON file (list of segments)")
    parser.add_argument("--output",         default="clip_boundaries.json",
                        help="Output JSON file (default: clip_boundaries.json)")
    parser.add_argument("--window_minutes", type=float, default=15.0,
                        help="Duration of each processing window in minutes (default: 15)")
    parser.add_argument("--api-key",        default=None,
                        help="OpenRouter API key (overrides .env and environment variable)")
    args = parser.parse_args()

    # ── Get API key from: CLI arg > .env > environment variable ──
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv("OPEN_ROUTER_KEY")
    if not api_key:
        sys.exit("ERROR: No API key. Set OPEN_ROUTER_KEY in .env, as an environment variable, or pass --api-key.")

    # ── Load ──
    print(f"Loading transcript: {args.input}")
    segments = load_segments(args.input)
    total_duration = segments[-1].get("end", 0)
    if total_duration == 0:
        sys.exit("ERROR: Could not determine total duration from last segment.")
    print(f"Total duration: {total_duration:.0f}s ({total_duration/60:.1f} min), "
          f"{len(segments)} segments.\n")

    # ── Build windows ──
    window_sec  = args.window_minutes * 60
    overlap_sec = 15   # overlap between windows to avoid cutting jokes at boundaries
    min_advance = 30   # minimum forward progress per window

    if window_sec <= overlap_sec + min_advance:
        sys.exit(f"ERROR: --window_minutes too small. Must be > "
                 f"{(overlap_sec + min_advance) / 60:.1f} min.")

    windows = []
    win_start = 0.0
    while win_start < total_duration:
        win_end = min(win_start + window_sec, total_duration)
        windows.append((win_start, win_end))
        next_start = win_end - overlap_sec
        if next_start <= win_start:
            break
        win_start = next_start

    print(f"Split into {len(windows)} window(s) of {args.window_minutes:.0f} min "
          f"with {overlap_sec}s overlap.\n")

    # ── Process windows ──
    all_clips = []
    failed_windows = []
    total_dropped = 0

    for i, (win_start, win_end) in enumerate(windows):
        label = f"{win_start/60:.1f}–{win_end/60:.1f} min"
        print(f"Window {i+1}/{len(windows)}: {label}")

        transcript_text, valid_starts, valid_ends = build_window_transcript(
            segments, win_start, win_end
        )

        if not transcript_text.strip():
            print("  No segments in this window — skipping.")
            continue

        clips = call_deepseek(transcript_text, api_key, label)

        if not clips:
            failed_windows.append(label)
            continue

        print(f"  Received {len(clips)} clip(s) — validating...")
        clips, dropped = validate_clips(clips, valid_starts, valid_ends, label)
        total_dropped += dropped

        if clips:
            print(f"  {len(clips)} clip(s) passed validation.")
            all_clips.extend(clips)
        else:
            print(f"  No clips passed validation (dropped {dropped}).")

    # ── Merge and rank ──
    print(f"\nTotal raw clips collected : {len(all_clips)}")
    print(f"Total invalid clips dropped : {total_dropped}")
    deduped = deduplicate_clips(all_clips)
    print(f"After deduplication       : {len(deduped)}")
    deduped.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_clips = deduped[:20]
    print(f"Top clips selected        : {len(top_clips)}")

    # ── Report failed windows ──
    if failed_windows:
        print(f"\nWARNING: These windows produced no clips and are unrepresented in output:")
        for w in failed_windows:
            print(f"  - {w}")
        print("Re-run with smaller --window_minutes or inspect those segments manually.")

    # ── Save ──
    output_data = {"clips": top_clips}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(top_clips)} clips to: {args.output}")


if __name__ == "__main__":
    main()
