# Whisper Confidence Dispatch

This project runs the real local `openai/whisper` model first, extracts per-token confidence and timing, visualizes the result, and dispatches low-confidence spans to an online OpenAI transcription model.

The goal is quality improvement, not speed optimization.

## What It Does

1. Run Whisper Tiny locally on CPU from a vendored `openai/whisper` source tree.
2. Recover per-token timestamps and confidence scores.
3. Render an interactive HTML visualization with:
   - waveform
   - audio player
   - moving playhead
   - active word/token highlighting during playback
4. Identify low-confidence spans.
5. Optionally send only those spans to OpenAI as fallback transcription requests.

## Project Layout

```text
.
├── confidence_dispatch/
│   └── dispatch.py              # span grouping, clip export, transcript patch helpers
├── kernels/
│   ├── openai_whisper.py        # local Whisper wrapper
│   ├── vendor_whisper.py        # vendored Whisper import/cache helpers
│   └── base.py                  # optional backend interface
├── scripts/
│   ├── bootstrap_local.sh
│   ├── doctor.py
│   ├── visualize_token_confidence.py
│   ├── dispatch_low_confidence.py
│   └── prefetch_whisper_tiny.sh
├── tests/
├── vendor/whisper/              # vendored upstream Whisper source
└── README.md
```

## Setup

Requirements:

- `python3`
- `ffmpeg`

Bootstrap locally:

```bash
./scripts/bootstrap_local.sh
source .venv/bin/activate
python scripts/doctor.py
```

That installs:

- local Python dependencies from `requirements.txt`
- the vendored Whisper package via `pip install -e vendor/whisper`

## Visualize Token Confidence

Run the visualizer on the bundled JFK sample:

```bash
source .venv/bin/activate
python scripts/visualize_token_confidence.py
```

Outputs:

- `results/token-confidence/jfk-token-confidence.json`
- `results/token-confidence/jfk-token-confidence.html`
- `results/token-confidence/jfk.flac`

Open the HTML file in a browser. It includes playback, a vertical playhead, and active highlighting of the current token and word.

Run on your own audio:

```bash
python scripts/visualize_token_confidence.py --audio /path/to/audio.wav
```

## Dispatch Low-Confidence Spans

Create a dry-run dispatch plan from an analysis JSON:

```bash
source .venv/bin/activate
python scripts/dispatch_low_confidence.py \
  --analysis-json results/token-confidence/jfk-token-confidence.json \
  --threshold 0.6 \
  --output results/dispatch-plan.json
```

This will:

- find tokens below the threshold
- merge nearby low-confidence tokens into spans
- add a little audio context around each span
- write a dispatch plan JSON

To actually send those spans to OpenAI:

```bash
export OPENAI_API_KEY=...
python scripts/dispatch_low_confidence.py \
  --analysis-json results/token-confidence/jfk-token-confidence.json \
  --threshold 0.6 \
  --run-openai \
  --output results/dispatch-openai.json
```

Current default fallback model:

- `whisper-1`

## Confidence Definition

The token confidence shown in the visualization is:

- the softmax probability assigned to each emitted text token during Whisper's alignment pass over the final transcript for that segment

This means the confidence values are tied to Whisper's own final token sequence, and the token timestamps come from the same alignment machinery Whisper uses for word timestamps.

## Tests

Run:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Current tests cover:

- vendored Whisper source presence
- low-confidence span grouping
- transcript patching logic for dispatched spans

## Notes

- The local-first model is real Whisper source code from `vendor/whisper`.
- The dispatch path is selective: it targets only low-confidence spans, not the whole file.
- If you want higher quality later, the next logical step is to add a transcript reconciliation policy for replacing local spans with the online model output more intelligently than simple span substitution.
