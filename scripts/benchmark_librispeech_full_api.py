#!/usr/bin/env python3
"""Benchmark full-audio OpenAI transcription on LibriSpeech."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from confidence_dispatch.dispatch import dispatch_with_openai
from scripts.benchmark_librispeech_dispatch import _iter_librispeech_items, _resolve_subset_root, _wer


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark full-audio OpenAI transcription on LibriSpeech")
    parser.add_argument("--input", type=Path, required=True, help="Path to LibriSpeech root or subset directory")
    parser.add_argument("--subset", default="dev-other", help="LibriSpeech subset to benchmark")
    parser.add_argument("--max-files", type=int, default=25, help="Maximum number of files to benchmark")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed; must match comparison runs")
    parser.add_argument("--language", default="en", help="Language hint passed to OpenAI")
    parser.add_argument("--openai-model", default="whisper-1", help="OpenAI transcription model")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "benchmark-dev-other-full-api.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    subset_root = _resolve_subset_root(args.input, args.subset)
    items = list(_iter_librispeech_items(subset_root))
    if not items:
        print(f"Error: no LibriSpeech utterances found under {subset_root}", file=sys.stderr)
        return 1

    if args.max_files and len(items) > args.max_files:
        import random

        rng = random.Random(args.seed)
        items = rng.sample(items, args.max_files)
        items.sort(key=lambda item: item["utterance_id"])

    results = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['utterance_id']}")
        start = time.perf_counter()
        transcript = dispatch_with_openai(
            item["audio_path"],
            args.language,
            args.openai_model,
        )
        wall_sec = time.perf_counter() - start
        results.append(
            {
                "utterance_id": item["utterance_id"],
                "audio_path": str(item["audio_path"]),
                "reference": item["reference"],
                "full_api_transcript": transcript,
                "full_api_wer": round(_wer(item["reference"], transcript), 6),
                "full_api_wall_sec": round(wall_sec, 3),
            }
        )

    avg_full_api_wer = round(sum(item["full_api_wer"] for item in results) / len(results), 6)
    payload = {
        "input": str(args.input.resolve()),
        "subset_root": str(subset_root),
        "subset": args.subset,
        "max_files": len(results),
        "seed": args.seed,
        "language": args.language,
        "openai_model": args.openai_model,
        "summary": {
            "files": len(results),
            "avg_full_api_wer": avg_full_api_wer,
            "full_api_accuracy": round(1.0 - avg_full_api_wer, 6),
            "total_full_api_wall_sec": round(sum(item["full_api_wall_sec"] for item in results), 3),
        },
        "files": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output.resolve(), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output.resolve()}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
