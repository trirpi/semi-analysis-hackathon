#!/usr/bin/env python3
"""Run conservative dispatch sweeps and save summary plots."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_one(
    *,
    input_path: Path,
    subset: str,
    max_files: int,
    threshold: float,
    prefix_words: int,
    left_context_sec: float,
    right_context_sec: float,
    min_duration_sec: float,
    output_path: Path,
) -> dict:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_librispeech_dispatch.py"),
        "--input",
        str(input_path),
        "--subset",
        subset,
        "--max-files",
        str(max_files),
        "--threshold",
        str(threshold),
        "--prompt-prefix-words",
        str(prefix_words),
        "--left-context-sec",
        str(left_context_sec),
        "--right-context-sec",
        str(right_context_sec),
        "--min-words",
        "1",
        "--min-duration-sec",
        str(min_duration_sec),
        "--run-openai",
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def render_svg_plot(
    *,
    title: str,
    subtitle: str,
    thresholds: list[float],
    series: dict[str, list[float]],
    baseline: float,
    y_label: str,
    output_path: Path,
) -> None:
    width = 1100
    height = 640
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [baseline]
    for values in series.values():
        all_values.extend(values)
    y_min = min(all_values)
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_min -= 0.01
        y_max += 0.01
    padding = max(0.005, (y_max - y_min) * 0.12)
    y_min -= padding
    y_max += padding

    def x_pos(index: int) -> float:
        if len(thresholds) == 1:
            return margin_left + plot_width / 2.0
        return margin_left + plot_width * index / (len(thresholds) - 1)

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1.0 - (value - y_min) / (y_max - y_min))

    colors = {
        "1": "#2563eb",
        "3": "#16a34a",
        "5": "#dc2626",
    }

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #111; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis { stroke: #111827; stroke-width: 1.5; }",
        ".tick { font-size: 12px; fill: #374151; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #4b5563; }",
        ".legend { font-size: 13px; }",
        "</style>",
    ]

    svg_parts.append(f"<text class='title' x='{margin_left}' y='34'>{title}</text>")
    svg_parts.append(f"<text class='subtitle' x='{margin_left}' y='56'>{subtitle}</text>")

    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5.0
        y = y_pos(value)
        svg_parts.append(f"<line class='grid' x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' />")
        svg_parts.append(f"<text class='tick' x='{margin_left - 10}' y='{y + 4:.2f}' text-anchor='end'>{value:.3f}</text>")

    for index, threshold in enumerate(thresholds):
        x = x_pos(index)
        svg_parts.append(f"<line class='grid' x1='{x:.2f}' y1='{margin_top}' x2='{x:.2f}' y2='{height - margin_bottom}' />")
        svg_parts.append(f"<text class='tick' x='{x:.2f}' y='{height - margin_bottom + 24}' text-anchor='middle'>{threshold:.2f}</text>")

    svg_parts.append(f"<line class='axis' x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' />")
    svg_parts.append(f"<line class='axis' x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' />")

    baseline_y = y_pos(baseline)
    svg_parts.append(
        f"<line x1='{margin_left}' y1='{baseline_y:.2f}' x2='{width - margin_right}' y2='{baseline_y:.2f}' "
        "stroke='#6b7280' stroke-width='2' stroke-dasharray='8 6' />"
    )
    svg_parts.append(
        f"<text class='legend' x='{width - margin_right - 8}' y='{baseline_y - 8:.2f}' text-anchor='end'>local baseline</text>"
    )

    legend_y = margin_top + 12
    legend_x = width - margin_right - 210
    for label, values in sorted(series.items(), key=lambda item: int(item[0])):
        points = " ".join(
            f"{x_pos(index):.2f},{y_pos(value):.2f}"
            for index, value in enumerate(values)
        )
        color = colors.get(label, "#111827")
        svg_parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{points}' />")
        for index, value in enumerate(values):
            x = x_pos(index)
            y = y_pos(value)
            svg_parts.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.5' fill='{color}' />")
        svg_parts.append(f"<line x1='{legend_x}' y1='{legend_y}' x2='{legend_x + 22}' y2='{legend_y}' stroke='{color}' stroke-width='3' />")
        svg_parts.append(f"<text class='legend' x='{legend_x + 30}' y='{legend_y + 4}'>prefix words = {label}</text>")
        legend_y += 22

    svg_parts.append(f"<text class='legend' x='{margin_left + plot_width / 2:.2f}' y='{height - 18}' text-anchor='middle'>confidence threshold</text>")
    svg_parts.append(
        f"<text class='legend' transform='translate(24 {margin_top + plot_height / 2:.2f}) rotate(-90)' text-anchor='middle'>{y_label}</text>"
    )
    svg_parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(svg_parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep dispatch thresholds and prefix lengths")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--subset", default="dev-other")
    parser.add_argument("--max-files", type=int, default=25)
    parser.add_argument("--thresholds", default="0.2,0.25,0.3,0.35")
    parser.add_argument("--prefix-words", default="1,3,5")
    parser.add_argument("--left-context-sec", type=float, default=0.5)
    parser.add_argument("--right-context-sec", type=float, default=0.25)
    parser.add_argument("--min-duration-sec", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dispatch-sweep",
    )
    args = parser.parse_args()

    thresholds = parse_csv_floats(args.thresholds)
    prefix_words_list = parse_csv_ints(args.prefix_words)
    output_dir = args.output_dir.resolve()
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    local_baseline_accuracy = None

    for prefix_words in prefix_words_list:
        for threshold in thresholds:
            output_path = runs_dir / f"{args.subset}-p{prefix_words}-th{threshold:.2f}.json"
            result = run_one(
                input_path=args.input.resolve(),
                subset=args.subset,
                max_files=args.max_files,
                threshold=threshold,
                prefix_words=prefix_words,
                left_context_sec=args.left_context_sec,
                right_context_sec=args.right_context_sec,
                min_duration_sec=args.min_duration_sec,
                output_path=output_path,
            )
            summary = result["summary"]
            local_accuracy = 1.0 - float(summary["avg_local_wer"])
            dispatched_accuracy = 1.0 - float(summary["avg_dispatched_wer"])
            local_baseline_accuracy = local_accuracy
            summaries.append(
                {
                    "prefix_words": prefix_words,
                    "threshold": threshold,
                    "avg_local_wer": summary["avg_local_wer"],
                    "avg_dispatched_wer": summary["avg_dispatched_wer"],
                    "avg_wer_improvement": summary["avg_wer_improvement"],
                    "local_accuracy": round(local_accuracy, 6),
                    "dispatched_accuracy": round(dispatched_accuracy, 6),
                    "total_dispatch_spans": summary["total_dispatch_spans"],
                    "total_dispatch_audio_sec": summary["total_dispatch_audio_sec"],
                    "result_json": str(output_path),
                }
            )

    grouped_accuracy = {
        str(prefix_words): [
            next(
                item["dispatched_accuracy"]
                for item in summaries
                if item["prefix_words"] == prefix_words and math.isclose(item["threshold"], threshold)
            )
            for threshold in thresholds
        ]
        for prefix_words in prefix_words_list
    }
    grouped_improvement = {
        str(prefix_words): [
            next(
                item["avg_wer_improvement"]
                for item in summaries
                if item["prefix_words"] == prefix_words and math.isclose(item["threshold"], threshold)
            )
            for threshold in thresholds
        ]
        for prefix_words in prefix_words_list
    }

    payload = {
        "input": str(args.input.resolve()),
        "subset": args.subset,
        "max_files": args.max_files,
        "thresholds": thresholds,
        "prefix_words": prefix_words_list,
        "left_context_sec": args.left_context_sec,
        "right_context_sec": args.right_context_sec,
        "min_duration_sec": args.min_duration_sec,
        "summaries": summaries,
    }
    save_json(output_dir / "summary.json", payload)

    render_svg_plot(
        title=f"{args.subset} dispatch accuracy sweep",
        subtitle=f"{args.max_files} sampled files, conservative dispatch policy",
        thresholds=thresholds,
        series=grouped_accuracy,
        baseline=local_baseline_accuracy or 0.0,
        y_label="accuracy (1 - WER)",
        output_path=output_dir / "accuracy-vs-threshold.svg",
    )
    render_svg_plot(
        title=f"{args.subset} dispatch improvement sweep",
        subtitle=f"{args.max_files} sampled files, conservative dispatch policy",
        thresholds=thresholds,
        series=grouped_improvement,
        baseline=0.0,
        y_label="WER improvement over local baseline",
        output_path=output_dir / "improvement-vs-threshold.svg",
    )

    print(f"Wrote sweep summary to {output_dir / 'summary.json'}")
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
