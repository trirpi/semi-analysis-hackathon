#!/usr/bin/env python3
"""Visualize Whisper token confidence, timing, and words on CPU."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernels.vendor_whisper import import_vendor_whisper, whisper_cache_dir


def _decode_token_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.encoding.decode([token_id])
    except Exception:
        raw = tokenizer.encoding.decode_single_token_bytes(token_id)
        return raw.decode("utf-8", errors="replace")


def _find_token_alignment(whisper, model, tokenizer, text_tokens, mel, num_frames: int):
    timing = importlib.import_module("whisper.timing")
    model_module = importlib.import_module("whisper.model")

    if len(text_tokens) == 0:
        return []

    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ],
        device=model.device,
    )

    qks = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, __, outs, index=i: qks.__setitem__(index, outs[-1][0])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    try:
        with torch.no_grad(), model_module.disable_sdpa():
            logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]
            sampled_logits = logits[len(tokenizer.sot_sequence) :, : tokenizer.eot]
            token_probs = sampled_logits.softmax(dim=-1)
            text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens]
            text_token_probs = text_token_probs.tolist()

        weights = torch.stack(
            [qks[layer][head] for layer, head in model.alignment_heads.indices().T]
        )
        weights = weights[:, :, : num_frames // 2]
        weights = weights.softmax(dim=-1)
        std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
        weights = (weights - mean) / std
        weights = timing.median_filter(weights, 7)

        matrix = weights.mean(axis=0)
        matrix = matrix[len(tokenizer.sot_sequence) : -1]
        text_indices, time_indices = timing.dtw(-matrix)

        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] / timing.TOKENS_PER_SECOND
        expected_boundaries = len(text_tokens) + 1
        max_time = num_frames * timing.HOP_LENGTH / timing.SAMPLE_RATE
        if len(jump_times) < expected_boundaries:
            jump_times = np.pad(
                jump_times,
                (0, expected_boundaries - len(jump_times)),
                constant_values=max_time,
            )
        elif len(jump_times) > expected_boundaries:
            jump_times = jump_times[:expected_boundaries]

        details = []
        for idx, token_id in enumerate(text_tokens):
            start = float(np.clip(jump_times[idx], 0.0, max_time))
            end = float(np.clip(jump_times[idx + 1], start, max_time))
            details.append(
                {
                    "token_id": int(token_id),
                    "token_text": _decode_token_text(tokenizer, int(token_id)),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "probability": round(float(text_token_probs[idx]), 6),
                }
            )
        return details
    finally:
        for hook in hooks:
            hook.remove()


def _run_transcription(whisper, audio_path: Path, model_name: str, device: str) -> dict:
    model = whisper.load_model(
        model_name,
        device=device,
        download_root=str(whisper_cache_dir()),
    )

    transcribe_module = importlib.import_module("whisper.transcribe")
    original_add_word_timestamps = transcribe_module.add_word_timestamps

    def patched_add_word_timestamps(**kwargs):
        return _patched_add_word_timestamps_impl(
            whisper=whisper,
            original_add_word_timestamps=original_add_word_timestamps,
            **kwargs,
        )

    transcribe_module.add_word_timestamps = patched_add_word_timestamps
    try:
        result = model.transcribe(
            str(audio_path),
            fp16=False,
            verbose=None,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        return result
    finally:
        transcribe_module.add_word_timestamps = original_add_word_timestamps


def _patched_add_word_timestamps_impl(
    whisper,
    original_add_word_timestamps,
    *,
    segments,
    model,
    tokenizer,
    mel,
    num_frames,
    prepend_punctuations="\"'“¿([{-",
    append_punctuations="\"'.。,，!！?？:：”)]}、",
    last_speech_timestamp,
    **kwargs,
):
    original_add_word_timestamps(
        segments=segments,
        model=model,
        tokenizer=tokenizer,
        mel=mel,
        num_frames=num_frames,
        prepend_punctuations=prepend_punctuations,
        append_punctuations=append_punctuations,
        last_speech_timestamp=last_speech_timestamp,
        **kwargs,
    )

    time_offset = segments[0]["seek"] * 0.01 if segments else 0.0
    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot]
        for segment in segments
    ]
    flat_text_tokens = [
        token for segment_tokens in text_tokens_per_segment for token in segment_tokens
    ]
    token_details = _find_token_alignment(
        whisper=whisper,
        model=model,
        tokenizer=tokenizer,
        text_tokens=flat_text_tokens,
        mel=mel,
        num_frames=num_frames,
    )

    cursor = 0
    for segment, segment_tokens in zip(segments, text_tokens_per_segment):
        segment_token_details = []
        for token_detail in token_details[cursor : cursor + len(segment_tokens)]:
            segment_token_details.append(
                {
                    **token_detail,
                    "start": round(time_offset + token_detail["start"], 3),
                    "end": round(time_offset + token_detail["end"], 3),
                }
            )
        segment["token_details"] = segment_token_details
        cursor += len(segment_tokens)


def _build_validation(result: dict) -> dict:
    total_tokens = 0
    monotonic = True
    probabilities_ok = True
    text_matches = True
    lowest_tokens: List[dict] = []

    for segment in result.get("segments", []):
        token_details = segment.get("token_details", [])
        total_tokens += len(token_details)
        previous_end = None
        reconstructed = "".join(token["token_text"] for token in token_details).strip()
        if reconstructed != segment.get("text", "").strip():
            text_matches = False
        for token in token_details:
            prob = float(token["probability"])
            probabilities_ok = probabilities_ok and 0.0 <= prob <= 1.0
            if previous_end is not None and token["start"] + 1e-6 < previous_end:
                monotonic = False
            previous_end = token["end"]
            lowest_tokens.append(
                {
                    "segment_id": segment.get("id"),
                    "token_text": token["token_text"],
                    "token_id": token["token_id"],
                    "start": token["start"],
                    "end": token["end"],
                    "probability": token["probability"],
                }
            )

    lowest_tokens.sort(key=lambda item: item["probability"])
    return {
        "total_segments": len(result.get("segments", [])),
        "total_tokens": total_tokens,
        "monotonic_token_timestamps": monotonic,
        "token_probabilities_in_range": probabilities_ok,
        "segment_text_matches_token_concat": text_matches,
        "lowest_confidence_tokens": lowest_tokens[:15],
    }


def _waveform_svg(audio: np.ndarray, duration_sec: float, width: int = 1200, height: int = 180) -> str:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return ""

    bins = np.array_split(audio, min(width, max(1, audio.size)))
    peaks = np.array([float(np.max(np.abs(chunk))) if chunk.size else 0.0 for chunk in bins])
    if peaks.max() > 0:
        peaks = peaks / peaks.max()

    mid = height / 2.0
    xs = np.linspace(0, width, num=len(peaks), endpoint=False)
    bars = []
    for x, peak in zip(xs, peaks):
        bar_height = max(1.0, peak * (height * 0.46))
        y = mid - bar_height
        bars.append(
            f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{y:.2f}' y2='{(mid + bar_height):.2f}' />"
        )

    tick_count = 10
    ticks = []
    for idx in range(tick_count + 1):
        x = width * idx / tick_count
        t = duration_sec * idx / tick_count
        ticks.append(
            f"<text x='{x:.2f}' y='{height - 6}' class='tick'>{t:.1f}s</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' class='waveform'>"
        f"<g class='bars'>{''.join(bars)}</g>"
        f"<g class='ticks'>{''.join(ticks)}</g>"
        "</svg>"
    )


def _color_for_probability(probability: float) -> str:
    prob = max(0.0, min(1.0, probability))
    hue = 120.0 * prob
    return f"hsl({hue:.1f}, 75%, 45%)"


def _token_boxes(result: dict, duration_sec: float) -> str:
    boxes = []
    for segment in result.get("segments", []):
        for token in segment.get("token_details", []):
            left = 100.0 * token["start"] / duration_sec if duration_sec else 0.0
            width = max(0.3, 100.0 * (token["end"] - token["start"]) / duration_sec) if duration_sec else 0.3
            label = html.escape(token["token_text"] or "<empty>")
            title = html.escape(
                f"token={token['token_text']!r} id={token['token_id']} "
                f"start={token['start']:.3f}s end={token['end']:.3f}s "
                f"p={token['probability']:.4f}"
            )
            boxes.append(
                "<div class='token-box' "
                f"data-start='{token['start']:.3f}' data-end='{token['end']:.3f}' "
                f"style='left:{left:.4f}%;width:{width:.4f}%;background:{_color_for_probability(token['probability'])}' "
                f"title=\"{title}\">"
                f"<span>{label}</span>"
                f"<em>{token['probability']:.2f}</em>"
                "</div>"
            )
    return "".join(boxes)


def _word_boxes(result: dict, duration_sec: float) -> str:
    boxes = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            left = 100.0 * word["start"] / duration_sec if duration_sec else 0.0
            width = max(0.5, 100.0 * (word["end"] - word["start"]) / duration_sec) if duration_sec else 0.5
            label = html.escape(word["word"])
            title = html.escape(
                f"word={word['word']!r} start={word['start']:.3f}s "
                f"end={word['end']:.3f}s p={word['probability']:.4f}"
            )
            boxes.append(
                "<div class='word-box' "
                f"data-start='{word['start']:.3f}' data-end='{word['end']:.3f}' "
                f"style='left:{left:.4f}%;width:{width:.4f}%;border-color:{_color_for_probability(word['probability'])}' "
                f"title=\"{title}\">{label}</div>"
            )
    return "".join(boxes)


def _build_html(result: dict, audio_path: Path, audio_src: str, duration_sec: float, waveform_svg: str) -> str:
    lowest = result["validation"]["lowest_confidence_tokens"]
    lowest_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['token_text'])}</td>"
        f"<td>{item['token_id']}</td>"
        f"<td>{item['start']:.3f}</td>"
        f"<td>{item['end']:.3f}</td>"
        f"<td>{item['probability']:.4f}</td>"
        "</tr>"
        for item in lowest
    )
    transcript = html.escape(result.get("text", "").strip())
    escaped_audio_src = html.escape(audio_src)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Whisper token confidence</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ color: #444; margin-bottom: 16px; }}
    .waveform-track, .lane {{ position: relative; }}
    .waveform {{ display: block; width: 100%; height: auto; background: #f7f7f7; border: 1px solid #ddd; }}
    .waveform .bars line {{ stroke: #5b6ee1; stroke-width: 1; opacity: 0.85; }}
    .waveform .ticks text {{ fill: #666; font-size: 10px; text-anchor: middle; }}
    .lane {{ height: 82px; border: 1px solid #ddd; margin-top: 12px; overflow: hidden; background: #fcfcfc; }}
    .lane-label {{ margin-top: 16px; font-weight: 600; }}
    .token-box, .word-box {{ position: absolute; top: 8px; min-width: 18px; box-sizing: border-box; padding: 4px 6px; border-radius: 4px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
    .token-box {{ color: white; height: 64px; transition: transform 0.05s linear, box-shadow 0.05s linear, outline-color 0.05s linear; }}
    .token-box span {{ display: block; font-size: 12px; font-weight: 600; }}
    .token-box em {{ display: block; font-size: 11px; margin-top: 4px; font-style: normal; }}
    .word-box {{ top: 20px; background: rgba(255,255,255,0.92); border: 2px solid #999; font-size: 12px; transition: transform 0.05s linear, box-shadow 0.05s linear; }}
    .token-box.active, .word-box.active {{ transform: translateY(-2px); box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.15); z-index: 3; }}
    .token-box.active {{ outline: 3px solid rgba(255, 255, 255, 0.95); }}
    .word-box.active {{ background: #fff7cc; }}
    .playhead {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #d40000; transform: translateX(-1px); pointer-events: none; z-index: 4; display: none; }}
    table {{ border-collapse: collapse; margin-top: 12px; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }}
    .transcript {{ white-space: pre-wrap; background: #fafafa; border: 1px solid #ddd; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Whisper token confidence visualization</h1>
  <div class="meta">
    <div><strong>Audio:</strong> <code>{html.escape(str(audio_path))}</code></div>
    <div><strong>Duration:</strong> {duration_sec:.2f}s</div>
    <div><strong>Language:</strong> {html.escape(result.get("language", ""))}</div>
    <div><strong>Device:</strong> {html.escape(result.get("device", ""))}</div>
    <div><strong>Confidence definition:</strong> {html.escape(result.get("confidence_definition", ""))}</div>
  </div>
  <audio id="audio-player" controls preload="metadata" src="{escaped_audio_src}"></audio>
  <h2>Waveform</h2>
  <div class="waveform-track">
    {waveform_svg}
    <div class="playhead"></div>
  </div>
  <div class="lane-label">Word confidence</div>
  <div class="lane" id="word-lane">{_word_boxes(result, duration_sec)}<div class="playhead"></div></div>
  <div class="lane-label">Token confidence</div>
  <div class="lane" id="token-lane">{_token_boxes(result, duration_sec)}<div class="playhead"></div></div>
  <h2>Transcript</h2>
  <div class="transcript">{transcript}</div>
  <h2>Lowest-confidence tokens</h2>
  <table>
    <thead>
      <tr><th>Token</th><th>ID</th><th>Start</th><th>End</th><th>Probability</th></tr>
    </thead>
    <tbody>{lowest_rows}</tbody>
  </table>
  <script>
    (() => {{
      const audio = document.getElementById("audio-player");
      const duration = {duration_sec:.6f};
      const playheads = Array.from(document.querySelectorAll(".playhead"));
      const tokenBoxes = Array.from(document.querySelectorAll(".token-box"));
      const wordBoxes = Array.from(document.querySelectorAll(".word-box"));

      function setPlayhead(currentTime) {{
        const pct = duration > 0 ? Math.max(0, Math.min(100, (currentTime / duration) * 100)) : 0;
        for (const playhead of playheads) {{
          playhead.style.display = "block";
          playhead.style.left = `${{pct}}%`;
        }}
      }}

      function setActive(items, currentTime) {{
        for (const item of items) {{
          const start = Number(item.dataset.start || "0");
          const end = Number(item.dataset.end || "0");
          const active = currentTime >= start && currentTime <= end;
          item.classList.toggle("active", active);
        }}
      }}

      function updateVisualization() {{
        const currentTime = audio.currentTime || 0;
        setPlayhead(currentTime);
        setActive(tokenBoxes, currentTime);
        setActive(wordBoxes, currentTime);
      }}

      audio.addEventListener("loadedmetadata", updateVisualization);
      audio.addEventListener("timeupdate", updateVisualization);
      audio.addEventListener("play", updateVisualization);
      audio.addEventListener("seeked", updateVisualization);
      audio.addEventListener("pause", updateVisualization);
      audio.addEventListener("ended", () => {{
        updateVisualization();
        for (const item of tokenBoxes.concat(wordBoxes)) {{
          item.classList.remove("active");
        }}
      }});
      updateVisualization();
    }})();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize Whisper token confidence on CPU")
    parser.add_argument(
        "--audio",
        type=Path,
        default=REPO_ROOT / "vendor/whisper/tests/jfk.flac",
        help="Audio file to transcribe",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="Whisper model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "token-confidence",
        help="Directory for output JSON and HTML",
    )
    args = parser.parse_args()

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    whisper = import_vendor_whisper()
    audio = whisper.load_audio(str(audio_path))
    duration_sec = len(audio) / 16000.0

    result = _run_transcription(
        whisper=whisper,
        audio_path=audio_path,
        model_name=args.model,
        device="cpu",
    )
    result["device"] = "cpu"
    result["audio"] = str(audio_path)
    result["duration_sec"] = round(duration_sec, 3)
    result["confidence_definition"] = (
        "Per-token confidence is the softmax probability assigned to each emitted text token "
        "during Whisper's alignment pass over the final transcript for that segment."
    )
    result["validation"] = _build_validation(result)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{audio_path.stem}-token-confidence.json"
    html_path = output_dir / f"{audio_path.stem}-token-confidence.html"
    copied_audio_path = output_dir / audio_path.name

    if audio_path != copied_audio_path:
        shutil.copy2(audio_path, copied_audio_path)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    waveform_svg = _waveform_svg(audio=audio, duration_sec=duration_sec)
    html_text = _build_html(
        result=result,
        audio_path=audio_path,
        audio_src=copied_audio_path.name,
        duration_sec=duration_sec,
        waveform_svg=waveform_svg,
    )
    html_path.write_text(html_text, encoding="utf-8")

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote HTML: {html_path}")
    print(f"Segments: {result['validation']['total_segments']}")
    print(f"Tokens: {result['validation']['total_tokens']}")
    print(
        "Validation:"
        f" monotonic={result['validation']['monotonic_token_timestamps']}"
        f" probs_ok={result['validation']['token_probabilities_in_range']}"
        f" text_match={result['validation']['segment_text_matches_token_concat']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
