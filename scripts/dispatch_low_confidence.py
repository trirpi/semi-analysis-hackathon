#!/usr/bin/env python3
"""Dispatch low-confidence Whisper spans to OpenAI transcription."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from confidence_dispatch.dispatch import (
    dispatch_analysis,
    load_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch low-confidence spans to OpenAI")
    parser.add_argument(
        "--analysis-json",
        type=Path,
        default=REPO_ROOT / "results" / "token-confidence" / "jfk-token-confidence.json",
        help="Confidence analysis JSON produced by visualize_token_confidence.py",
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
        help="Merge nearby low-confidence tokens into one span if the gap is below this value",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=0.2,
        help="Pad each dispatched span on both sides by this many seconds",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=1,
        help="Minimum number of low-confidence tokens required for a dispatched span",
    )
    parser.add_argument(
        "--openai-model",
        default="whisper-1",
        help="OpenAI transcription model for fallback dispatch",
    )
    parser.add_argument(
        "--run-openai",
        action="store_true",
        help="Actually send spans to OpenAI; otherwise only produce a dry-run dispatch plan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "dispatch-plan.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    analysis_path = args.analysis_json.resolve()
    if not analysis_path.is_file():
        print(f"Error: analysis JSON not found: {analysis_path}", file=sys.stderr)
        return 1

    analysis = load_analysis(analysis_path)
    audio_path = Path(analysis["audio"]).resolve()
    if not audio_path.is_file():
        print(f"Error: source audio not found: {audio_path}", file=sys.stderr)
        return 1

    try:
        report = dispatch_analysis(
            analysis,
            threshold=args.threshold,
            merge_gap_sec=args.merge_gap_sec,
            context_sec=args.context_sec,
            min_tokens=args.min_tokens,
            run_openai=args.run_openai,
            openai_model=args.openai_model,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    report.update(
        {
            "analysis_json": str(analysis_path),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output.resolve(), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output.resolve()}")
    print(f"Dispatch spans: {len(report['dispatch_spans'])}")
    if report["dispatch_spans"]:
        for span in report["dispatch_spans"][:10]:
            print(
                f"  {span['dispatch_start']:.2f}-{span['dispatch_end']:.2f}s "
                f"local={span['local_text']!r} min_p={span['min_probability']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
