#!/usr/bin/env python3
"""Run conservative dispatch sweeps and save summary plots."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def build_plot_spec(
    *,
    thresholds: list[float],
    series: dict[str, list[float]],
    baseline: float,
    extra_values: list[float] | None = None,
) -> dict:
    width = 1100
    height = 640
    margin_left = 90
    margin_right = 50
    margin_top = 80
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [baseline]
    if extra_values:
        all_values.extend(extra_values)
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

    return {
        "width": width,
        "height": height,
        "margin_left": margin_left,
        "margin_right": margin_right,
        "margin_top": margin_top,
        "margin_bottom": margin_bottom,
        "plot_width": plot_width,
        "plot_height": plot_height,
        "y_min": y_min,
        "y_max": y_max,
        "x_pos": x_pos,
        "y_pos": y_pos,
    }


def render_svg_plot(
    *,
    title: str,
    subtitle: str,
    thresholds: list[float],
    series: dict[str, list[float]],
    references: list[dict],
    y_label: str,
    output_path: Path,
) -> None:
    reference_values = [float(item["value"]) for item in references]
    spec = build_plot_spec(
        thresholds=thresholds,
        series=series,
        baseline=min(reference_values) if reference_values else 0.0,
        extra_values=reference_values,
    )
    width = spec["width"]
    height = spec["height"]
    margin_left = spec["margin_left"]
    margin_right = spec["margin_right"]
    margin_top = spec["margin_top"]
    margin_bottom = spec["margin_bottom"]
    plot_width = spec["plot_width"]
    plot_height = spec["plot_height"]
    y_min = spec["y_min"]
    y_max = spec["y_max"]
    x_pos = spec["x_pos"]
    y_pos = spec["y_pos"]

    colors = {
        "1": "#2563eb",
        "3": "#16a34a",
        "5": "#dc2626",
        "7": "#f59e0b",
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

    for reference in references:
        reference_y = y_pos(float(reference["value"]))
        reference_color = reference.get("color", "#6b7280")
        reference_dash = reference.get("dasharray", "8 6")
        svg_parts.append(
            f"<line x1='{margin_left}' y1='{reference_y:.2f}' x2='{width - margin_right}' y2='{reference_y:.2f}' "
            f"stroke='{reference_color}' stroke-width='2' stroke-dasharray='{reference_dash}' />"
        )
        svg_parts.append(
            f"<text class='legend' x='{width - margin_right - 8}' y='{reference_y - 8:.2f}' text-anchor='end'>{reference['label']}</text>"
        )

    legend_y = margin_top + 12
    legend_x = width - margin_right - 290
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


def render_png_plot(
    *,
    title: str,
    subtitle: str,
    thresholds: list[float],
    series: dict[str, list[float]],
    references: list[dict],
    y_label: str,
    output_path: Path,
) -> None:
    reference_values = [float(item["value"]) for item in references]
    spec = build_plot_spec(
        thresholds=thresholds,
        series=series,
        baseline=min(reference_values) if reference_values else 0.0,
        extra_values=reference_values,
    )
    width = spec["width"]
    height = spec["height"]
    margin_left = spec["margin_left"]
    margin_right = spec["margin_right"]
    margin_top = spec["margin_top"]
    margin_bottom = spec["margin_bottom"]
    plot_width = spec["plot_width"]
    plot_height = spec["plot_height"]
    y_min = spec["y_min"]
    y_max = spec["y_max"]
    x_pos = spec["x_pos"]
    y_pos = spec["y_pos"]

    colors = {
        "1": "#2563eb",
        "3": "#16a34a",
        "5": "#dc2626",
        "7": "#f59e0b",
    }

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    def text_size(text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top

    def draw_centered_text(x: float, y: float, text: str, font: ImageFont.ImageFont, fill: str) -> None:
        w, h = text_size(text, font)
        draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)

    def draw_right_aligned_text(x: float, y: float, text: str, font: ImageFont.ImageFont, fill: str) -> None:
        w, h = text_size(text, font)
        draw.text((x - w, y - h / 2), text, font=font, fill=fill)

    def draw_dashed_line(x1: float, y1: float, x2: float, y2: float, *, fill: str, dash: int = 8, gap: int = 6, width_px: int = 2) -> None:
        total = x2 - x1
        current = x1
        while current < x2:
            end = min(current + dash, x2)
            draw.line((current, y1, end, y2), fill=fill, width=width_px)
            current += dash + gap

    draw.text((margin_left, 24), title, font=title_font, fill="#111111")
    draw.text((margin_left, 46), subtitle, font=body_font, fill="#4b5563")

    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5.0
        y = y_pos(value)
        draw.line((margin_left, y, width - margin_right, y), fill="#e5e7eb", width=1)
        draw_right_aligned_text(margin_left - 12, y, f"{value:.3f}", body_font, "#374151")

    for index, threshold in enumerate(thresholds):
        x = x_pos(index)
        draw.line((x, margin_top, x, height - margin_bottom), fill="#e5e7eb", width=1)
        draw_centered_text(x, height - margin_bottom + 22, f"{threshold:.2f}", body_font, "#374151")

    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="#111827", width=2)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="#111827", width=2)

    for reference in references:
        reference_y = y_pos(float(reference["value"]))
        reference_color = reference.get("color", "#6b7280")
        dash_values = [int(part) for part in str(reference.get("dasharray", "8 6")).split()]
        dash = dash_values[0] if dash_values else 8
        gap = dash_values[1] if len(dash_values) > 1 else 6
        draw_dashed_line(
            margin_left,
            reference_y,
            width - margin_right,
            reference_y,
            fill=reference_color,
            dash=dash,
            gap=gap,
        )
        draw_right_aligned_text(
            width - margin_right - 8,
            reference_y - 8,
            reference["label"],
            body_font,
            "#374151",
        )

    legend_y = margin_top + 12
    legend_x = width - margin_right - 290
    for label, values in sorted(series.items(), key=lambda item: int(item[0])):
        color = colors.get(label, "#111827")
        points = [(x_pos(index), y_pos(value)) for index, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.line((legend_x, legend_y, legend_x + 22, legend_y), fill=color, width=3)
        draw.text((legend_x + 30, legend_y - 8), f"prefix words = {label}", font=body_font, fill="#111111")
        legend_y += 22

    draw_centered_text(margin_left + plot_width / 2, height - 18, "confidence threshold", body_font, "#111111")

    y_label_image = Image.new("RGBA", (40, 320), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label_image)
    y_w, y_h = y_draw.textbbox((0, 0), y_label, font=body_font)[2:]
    y_draw.text(((40 - y_w) / 2, (320 - y_h) / 2), y_label, font=body_font, fill="#111111")
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (10, int(margin_top + plot_height / 2 - rotated.size[1] / 2)), rotated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_grouped_series(payload: dict) -> tuple[list[float], list[int], list[dict], dict[str, list[float]], dict[str, list[float]], float]:
    thresholds = [float(value) for value in payload["thresholds"]]
    prefix_words_list = [int(value) for value in payload["prefix_words"]]
    summaries = payload["summaries"]
    local_baseline_accuracy = 1.0 - float(summaries[0]["avg_local_wer"]) if summaries else 0.0

    grouped_accuracy = {
        str(prefix_words): [
            next(
                item["dispatched_accuracy"]
                for item in summaries
                if int(item["prefix_words"]) == prefix_words and math.isclose(float(item["threshold"]), threshold)
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
                if int(item["prefix_words"]) == prefix_words and math.isclose(float(item["threshold"]), threshold)
            )
            for threshold in thresholds
        ]
        for prefix_words in prefix_words_list
    }
    return thresholds, prefix_words_list, summaries, grouped_accuracy, grouped_improvement, local_baseline_accuracy


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep dispatch thresholds and prefix lengths")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--subset", default="dev-other")
    parser.add_argument("--max-files", type=int, default=25)
    parser.add_argument("--thresholds", default="0.2,0.25,0.3,0.35")
    parser.add_argument("--prefix-words", default="1,3,5")
    parser.add_argument("--left-context-sec", type=float, default=0.5)
    parser.add_argument("--right-context-sec", type=float, default=0.25)
    parser.add_argument("--min-duration-sec", type=float, default=0.25)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Use an existing sweep summary to render plots without rerunning benchmarks",
    )
    parser.add_argument(
        "--full-api-json",
        type=Path,
        help="Optional JSON benchmark for full-audio OpenAI transcription reference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dispatch-sweep",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.summary_json is not None:
        with open(args.summary_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        if args.input is None:
            raise SystemExit("--input is required unless --summary-json is provided")
        thresholds = parse_csv_floats(args.thresholds)
        prefix_words_list = parse_csv_ints(args.prefix_words)
        runs_dir = output_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        summaries: list[dict] = []
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

    if args.full_api_json is not None:
        with open(args.full_api_json, "r", encoding="utf-8") as handle:
            full_api_payload = json.load(handle)
        avg_full_api_wer = float(full_api_payload["summary"]["avg_full_api_wer"])
        payload["full_api_reference"] = {
            "avg_full_api_wer": round(avg_full_api_wer, 6),
            "full_api_accuracy": round(1.0 - avg_full_api_wer, 6),
            "local_minus_full_api_wer": 0.0,
            "source_json": str(args.full_api_json.resolve()),
        }

    thresholds, prefix_words_list, summaries, grouped_accuracy, grouped_improvement, local_baseline_accuracy = build_grouped_series(payload)
    if payload.get("full_api_reference"):
        payload["full_api_reference"]["local_minus_full_api_wer"] = round(
            local_baseline_accuracy - (1.0 - float(payload["full_api_reference"]["avg_full_api_wer"])),
            6,
        )
    subset = payload["subset"]
    max_files = int(payload["max_files"])
    accuracy_references = [
        {
            "label": f"local-only Whisper Tiny baseline ({local_baseline_accuracy:.3f} accuracy)",
            "value": local_baseline_accuracy,
            "color": "#6b7280",
            "dasharray": "8 6",
        }
    ]
    improvement_references = [
        {
            "label": "local-only baseline reference (0.000 improvement)",
            "value": 0.0,
            "color": "#6b7280",
            "dasharray": "8 6",
        }
    ]
    if payload.get("full_api_reference"):
        full_api_wer = float(payload["full_api_reference"]["avg_full_api_wer"])
        full_api_accuracy = 1.0 - full_api_wer
        full_api_improvement = float(payload["full_api_reference"]["local_minus_full_api_wer"])
        accuracy_references.append(
            {
                "label": f"full-audio OpenAI transcription ({full_api_accuracy:.3f} accuracy)",
                "value": full_api_accuracy,
                "color": "#7c3aed",
                "dasharray": "12 6",
            }
        )
        improvement_references.append(
            {
                "label": f"full-audio OpenAI transcription ({full_api_improvement:.3f} improvement)",
                "value": full_api_improvement,
                "color": "#7c3aed",
                "dasharray": "12 6",
            }
        )

    render_svg_plot(
        title=f"{subset} dispatch accuracy sweep",
        subtitle=f"{max_files} sampled files, conservative dispatch policy",
        thresholds=thresholds,
        series=grouped_accuracy,
        references=accuracy_references,
        y_label="accuracy (1 - WER)",
        output_path=output_dir / "dispatch-accuracy-sweep.svg",
    )
    render_png_plot(
        title=f"{subset} dispatch accuracy sweep",
        subtitle=f"{max_files} sampled files, conservative dispatch policy",
        thresholds=thresholds,
        series=grouped_accuracy,
        references=accuracy_references,
        y_label="accuracy (1 - WER)",
        output_path=output_dir / "dispatch-accuracy-sweep.png",
    )
    render_svg_plot(
        title=f"{subset} dispatch improvement sweep",
        subtitle=f"{max_files} sampled files, conservative dispatch policy",
        thresholds=thresholds,
        series=grouped_improvement,
        references=improvement_references,
        y_label="WER improvement over local baseline",
        output_path=output_dir / "dispatch-improvement-sweep.svg",
    )
    render_png_plot(
        title=f"{subset} dispatch improvement sweep",
        subtitle=f"{max_files} sampled files, conservative dispatch policy",
        thresholds=thresholds,
        series=grouped_improvement,
        references=improvement_references,
        y_label="WER improvement over local baseline",
        output_path=output_dir / "dispatch-improvement-sweep.png",
    )

    if args.summary_json is None:
        print(f"Wrote sweep summary to {output_dir / 'summary.json'}")
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
