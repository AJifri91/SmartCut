# ✂️ SmartCut — AI‑Powered Funny Clip Extractor

**SmartCut** extracts the funniest moments from any video using Whisper transcription and an AI comedy curator.  
The pipeline:

- 🎤 Transcribes audio with **Whisper Large‑v3 Turbo** (on Kaggle, free GPU) – merges short Whisper segments into **20–30 second chunks** for coherent LLM analysis
- 🤖 Uses **DeepSeek V3.2** via OpenRouter to detect complete jokes and define clip boundaries (configurable – you can also use V4 Flash)
- ✂️ Cuts the top 20 clips (24–80 seconds, configurable) directly from the video

> Inspired by the original [Clipception](https://github.com/msylvester/Clipception) – SmartCut improves transcription quality by using a large cloud‑based Whisper model and gives the LLM control over segment merging.

---

## 📋 Requirements

- **Python 3.8+**
- **ffmpeg** (for video cutting)
- **OpenRouter API key** with a small credit balance ([get one here](https://openrouter.ai/keys))
- **Kaggle account** (for the transcription notebook – provides a free T4 GPU)

---

## 🛠️ Installation

```bash
git clone https://github.com/AJifri91/SmartCut.git
cd SmartCut
python -m venv venv
source venv/bin/activate      # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OpenRouter key:
#   OPEN_ROUTER_KEY=sk-or-v1-...
```

`requirements.txt` contains:

```
openai>=1.0.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

`ffmpeg` must be installed and on your PATH.
MoviePy is optional – only needed if you want frame‑perfect cuts (see --accurate below).

## 🚀 Workflow

### Step 1: Generate the Whisper segment JSON (on Kaggle)

The transcription step runs on Kaggle because it requires a GPU (free T4) to run Whisper Large‑v3 Turbo.

**1.1 Go to Kaggle**

Go to kaggle.com and sign in.
Click "Create" → "New Notebook".

**1.2 Enable GPU**

In the right sidebar, click "Settings" (gear icon).
Under "Accelerator", select "GPU T4 x2".
Turn on "Internet" toggle (required to download models).

**1.3 Upload your video**

Click the "Add Data" button in the right sidebar.
Select "Upload" and choose your video file (e.g., my_video.mp4).
Kaggle will create a folder like `/kaggle/input/your-dataset-name/`.

**1.4 Copy the notebook code**

Open the provided `raw-whisper-json-file-kaggle.py` file in this repository and paste its contents into a code cell in your Kaggle notebook.

🔴 **CRITICAL – EDIT THE VIDEO PATH:**

At the top of the notebook, update:

```python
VIDEO_PATH = "/kaggle/input/your-dataset-name/your_video.mp4"
```

To find the correct path, run this in a separate Kaggle cell:

```python
!ls /kaggle/input/
```

**1.5 Run the notebook**

Run all cells (Runtime → "Run All").

The notebook will:

- Extract audio from your video
- Run Whisper Large‑v3 Turbo with segment‑level timestamps
- Merge segments into 20–30 second chunks (configurable via `MIN_DURATION`)
- Attach audio features (volume, intensity) to each merged segment
- Output `{video_stem}.min{MIN_DURATION}.enhanced_transcription.json`

**1.6 Download the JSON**

In the "Output" tab (right sidebar), navigate to `/kaggle/working/`.
Download the JSON file and place it in your local project folder.

**Example:**
If your video is `live_asa_hanem_06-05-2026.mp4` and `MIN_DURATION = 20`, the file will be:
`live_asa_hanem_06-05-2026.min20.enhanced_transcription.json`

### Step 2: Generate clip boundaries (locally)

```bash
python generate_clip_boundaries.py \
    --input live_asa_hanem_06-05-2026.min20.enhanced_transcription.json \
    --output boundaries.json \
    --window_minutes 15
```

This script:

- Splits the transcript into 15‑minute windows (15s overlap)
- Sends each window to DeepSeek V3.2 via OpenRouter
- Returns a JSON list of the top 20 clip boundaries (start, end, score, reason)

### Step 3: Extract video clips (locally)

```bash
python extract_clips.py \
    --video live_asa_hanem_06-05-2026.mp4 \
    --boundaries boundaries.json \
    --outdir funny_clips \
    --min-duration 24 \
    --max-duration 80
```

Two cutting modes:

| Mode | Command | Speed | Quality | Accuracy |
|------|---------|-------|---------|----------|
| Fast (default) | No flag | ~1‑2s/clip | Lossless | ±1‑2s (keyframe snap) |
| Accurate | `--accurate` | ~10‑30s/clip | Re‑encodes (slight loss) | Frame‑perfect |

- **Fast mode** uses ffmpeg `-c copy` – no re‑encoding, preserves original quality.
- **Accurate mode** uses ffmpeg re‑encoding with libx264 – slower but frame‑exact.
- (If you prefer MoviePy for accurate cuts, install `moviepy>=1.0.3` and modify the script.)

The clips are saved in `funny_clips/`.

---

## ⚙️ Configuration

| What to change | Where | How |
|----------------|-------|-----|
| Clip duration range | `extract_clips.py` | `--min-duration` / `--max-duration` |
| Cut accuracy | `extract_clips.py` | `--accurate` flag |
| Window size | `generate_clip_boundaries.py` | `--window_minutes` (default 15) |
| AI model | `generate_clip_boundaries.py` | Edit `model=` in `call_deepseek()` – default `deepseek/deepseek-v3.2` |
| Comedy prompt | `generate_clip_boundaries.py` | Edit `SYSTEM_PROMPT` variable |
| Number of clips | `generate_clip_boundaries.py` | Change `top_clips = deduped[:20]` to any number |
| Kaggle merge duration | `raw-whisper-json-file-kaggle.py` | Change `MIN_DURATION` (default 20.0) |

Current AI model: `deepseek/deepseek-v3.2` – excellent performance for Arabic content.

---

## 📁 Files

| File | Description |
|------|-------------|
| `generate_clip_boundaries.py` | AI boundary detection (uses DeepSeek via OpenRouter) |
| `extract_clips.py` | Video cutting via ffmpeg (fast by default; `--accurate` for re‑encode) |
| `raw-whisper-json-file-kaggle.py` | Kaggle script for Whisper transcription + merging + feature extraction |
| `.env.example` | Template for API key |
| `requirements.txt` | Python dependencies (MoviePy optional) |

---

## 🙏 Credits

This project builds upon the original [Clipception](https://github.com/msylvester/Clipception) by:

- Replacing the local tiny Whisper model with Large‑v3 Turbo running on Kaggle – significantly improving transcription accuracy, especially for Arabic and other non‑English languages
- Using DeepSeek V3.2 for cost‑effective, high‑quality AI comedy analysis (also tested with V4 Flash)
- Merging Whisper segments into coherent chunks before sending to the LLM – giving it the context needed for accurate comedy detection
- Using a universal comedy prompt that works for any language
- Streamlining the workflow into two clear steps

Many thanks to the original authors for their brilliant concept!

---

## 📄 License

MIT – see the `LICENSE` file.
