#!/usr/bin/env python3
"""Benchmark local Whisper vs confidence-based OpenAI dispatch on LibriSpeech."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from confidence_dispatch.analyze import ConfidenceAnalyzer
from confidence_dispatch.dispatch import dispatch_analysis, normalize_text


def _normalize_for_wer(text: str) -> List[str]:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def _wer(reference: str, hypothesis: str) -> float:
    ref_words = _normalize_for_wer(reference)
    hyp_words = _normalize_for_wer(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1] / len(ref_words)


def _resolve_subset_root(input_path: Path, subset: str) -> Path:
    input_path = input_path.resolve()
    if input_path.name == subset:
        return input_path
    candidate = input_path / subset
    if candidate.is_dir():
        return candidate
    candidate = input_path / "LibriSpeech" / subset
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Could not find LibriSpeech subset '{subset}' under {input_path}")


def _iter_librispeech_items(subset_root: Path) -> Iterable[dict]:
    for transcript_path in sorted(subset_root.glob("*/*/*.trans.txt")):
        utterances = {}
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(" ", 1)
            utterances[utt_id] = text.strip()

        for utt_id, text in utterances.items():
            audio_path = transcript_path.parent / f"{utt_id}.flac"
            if audio_path.is_file():
                yield {
                    "utterance_id": utt_id,
                    "audio_path": audio_path,
                    "reference": text,
                }


def _summarize(items: List[dict]) -> dict:
    if not items:
        return {
            "files": 0,
            "avg_local_wer": None,
            "avg_dispatched_wer": None,
            "avg_wer_improvement": None,
            "total_dispatch_spans": 0,
            "total_dispatch_audio_sec": 0.0,
        }

    local = sum(item["local_wer"] for item in items) / len(items)
    dispatched = sum(item["dispatched_wer"] for item in items) / len(items)
    return {
        "files": len(items),
        "avg_local_wer": round(local, 6),
        "avg_dispatched_wer": round(dispatched, 6),
        "avg_wer_improvement": round(local - dispatched, 6),
        "total_dispatch_spans": sum(item["dispatch_span_count"] for item in items),
        "total_dispatch_audio_sec": round(sum(item["dispatch_audio_sec"] for item in items), 3),
        "total_local_wall_sec": round(sum(item["local_wall_sec"] for item in items), 3),
        "total_dispatch_wall_sec": round(sum(item["dispatch_wall_sec"] for item in items), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark confidence dispatch on LibriSpeech")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to LibriSpeech root or directly to the test-other directory",
    )
    parser.add_argument(
        "--subset",
        default="test-other",
        help="LibriSpeech subset to benchmark",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="Local Whisper model name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Dispatch tokens below this confidence threshold",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=0.15,
        help="Merge nearby low-confidence tokens if the gap is below this value",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=0.2,
        help="Add this much audio context around dispatched spans",
    )
    parser.add_argument(
        "--left-context-sec",
        type=float,
        default=None,
        help="Left-side audio context in seconds; defaults to --context-sec",
    )
    parser.add_argument(
        "--right-context-sec",
        type=float,
        default=None,
        help="Right-side audio context in seconds; defaults to --context-sec",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=1,
        help="Minimum low-confidence token count for a dispatched span",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum low-confidence word count for a dispatched span",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.0,
        help="Minimum audio duration for a dispatched span",
    )
    parser.add_argument(
        "--prompt-prefix-words",
        type=int,
        default=0,
        help="Number of preceding words to pass as prompt context to OpenAI",
    )
    parser.add_argument(
        "--following-context-words",
        type=int,
        default=3,
        help="Number of following local words used to trim remote overlap",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of files to benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed when max-files is smaller than the dataset",
    )
    parser.add_argument(
        "--run-openai",
        action="store_true",
        help="Actually dispatch low-confidence spans to OpenAI",
    )
    parser.add_argument(
        "--openai-model",
        default="whisper-1",
        help="OpenAI transcription model for fallback spans",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "benchmark-test-other.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    subset_root = _resolve_subset_root(args.input, args.subset)
    items = list(_iter_librispeech_items(subset_root))
    if not items:
        print(f"Error: no LibriSpeech utterances found under {subset_root}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    if args.max_files and len(items) > args.max_files:
        items = rng.sample(items, args.max_files)
        items.sort(key=lambda item: item["utterance_id"])

    analyzer = ConfidenceAnalyzer(model_name=args.model, device="cpu")
    results = []

    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['utterance_id']}")

        local_start = time.perf_counter()
        analysis = analyzer.analyze(item["audio_path"])
        local_wall_sec = time.perf_counter() - local_start

        dispatch_start = time.perf_counter()
        try:
            dispatch_report = dispatch_analysis(
                analysis,
                threshold=args.threshold,
                merge_gap_sec=args.merge_gap_sec,
                context_sec=args.context_sec,
                left_context_sec=args.left_context_sec,
                right_context_sec=args.right_context_sec,
                min_tokens=args.min_tokens,
                min_words=args.min_words,
                min_duration_sec=args.min_duration_sec,
                prompt_prefix_words=args.prompt_prefix_words,
                following_context_words=args.following_context_words,
                run_openai=args.run_openai,
                openai_model=args.openai_model,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        dispatch_wall_sec = time.perf_counter() - dispatch_start

        dispatch_spans = dispatch_report["dispatch_spans"]
        dispatch_audio_sec = sum(
            span["dispatch_end"] - span["dispatch_start"] for span in dispatch_spans
        )
        local_text = dispatch_report["local_transcript"]
        dispatched_text = dispatch_report["dispatched_transcript"]
        reference = item["reference"]

        row = {
            "utterance_id": item["utterance_id"],
            "audio_path": str(item["audio_path"]),
            "reference": reference,
            "local_transcript": local_text,
            "dispatched_transcript": dispatched_text,
            "local_wer": round(_wer(reference, local_text), 6),
            "dispatched_wer": round(_wer(reference, dispatched_text), 6),
            "dispatch_span_count": len(dispatch_spans),
            "dispatch_audio_sec": round(dispatch_audio_sec, 3),
            "local_wall_sec": round(local_wall_sec, 3),
            "dispatch_wall_sec": round(dispatch_wall_sec, 3),
        }
        if dispatch_spans:
            row["dispatch_spans"] = dispatch_spans
        results.append(row)

    payload = {
        "input": str(args.input.resolve()),
        "subset_root": str(subset_root),
        "subset": args.subset,
        "model": args.model,
        "threshold": args.threshold,
        "merge_gap_sec": args.merge_gap_sec,
        "context_sec": args.context_sec,
        "left_context_sec": args.left_context_sec if args.left_context_sec is not None else args.context_sec,
        "right_context_sec": args.right_context_sec if args.right_context_sec is not None else args.context_sec,
        "min_tokens": args.min_tokens,
        "min_words": args.min_words,
        "min_duration_sec": args.min_duration_sec,
        "prompt_prefix_words": args.prompt_prefix_words,
        "following_context_words": args.following_context_words,
        "run_openai": args.run_openai,
        "openai_model": args.openai_model,
        "max_files": args.max_files,
        "seed": args.seed,
        "summary": _summarize(results),
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
