#!/usr/bin/env python3
"""Dispatch low-confidence Whisper spans to OpenAI transcription."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from confidence_dispatch.dispatch import (
    build_dispatched_transcript,
    collect_low_confidence_spans,
    extract_audio_clip,
    load_analysis,
    normalize_text,
    save_wav_clip,
)


def _dispatch_with_openai(clip_path: Path, language: str | None, model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    with open(clip_path, "rb") as handle:
        response = client.audio.transcriptions.create(
            model=model_name,
            file=handle,
            language=language,
            temperature=0,
        )
    text = getattr(response, "text", None)
    if text is None and isinstance(response, dict):
        text = response.get("text")
    return normalize_text(text or "")


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

    spans = collect_low_confidence_spans(
        analysis,
        threshold=args.threshold,
        merge_gap_sec=args.merge_gap_sec,
        context_sec=args.context_sec,
        min_tokens=args.min_tokens,
    )

    report = {
        "analysis_json": str(analysis_path),
        "audio": str(audio_path),
        "threshold": args.threshold,
        "merge_gap_sec": args.merge_gap_sec,
        "context_sec": args.context_sec,
        "min_tokens": args.min_tokens,
        "run_openai": args.run_openai,
        "openai_model": args.openai_model,
        "local_transcript": normalize_text(analysis.get("text", "")),
        "dispatch_spans": spans,
    }

    if args.run_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY is required with --run-openai", file=sys.stderr)
            return 1

        dispatched: List[dict] = []
        with TemporaryDirectory(prefix="dispatch-clips-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, span in enumerate(spans):
                clip_audio = extract_audio_clip(
                    audio_path,
                    span["dispatch_start"],
                    span["dispatch_end"],
                )
                clip_path = temp_root / f"span-{index:03d}.wav"
                save_wav_clip(clip_audio, clip_path)
                remote_text = _dispatch_with_openai(
                    clip_path,
                    analysis.get("language"),
                    args.openai_model,
                )
                dispatched.append({**span, "remote_text": remote_text})
        report["dispatch_spans"] = dispatched
        report["dispatched_transcript"] = build_dispatched_transcript(
            analysis,
            dispatched,
        )
    else:
        report["dispatched_transcript"] = report["local_transcript"]

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
