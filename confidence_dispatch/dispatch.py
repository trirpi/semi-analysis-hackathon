from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from kernels.vendor_whisper import import_vendor_whisper


def load_analysis(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def collect_low_confidence_spans(
    analysis: dict,
    *,
    threshold: float,
    merge_gap_sec: float = 0.15,
    context_sec: float = 0.2,
    min_tokens: int = 1,
) -> List[dict]:
    duration_sec = float(analysis.get("duration_sec") or 0.0)
    spans: List[dict] = []

    for segment in analysis.get("segments", []):
        tokens = segment.get("token_details", [])
        current = None

        for token_idx, token in enumerate(tokens):
            prob = float(token["probability"])
            if prob >= threshold:
                if current is not None:
                    spans.append(_finalize_span(current, duration_sec, context_sec))
                    current = None
                continue

            token_payload = {
                "segment_id": int(segment.get("id", 0)),
                "token_index": token_idx,
                "token_id": int(token["token_id"]),
                "token_text": token["token_text"],
                "start": float(token["start"]),
                "end": float(token["end"]),
                "probability": prob,
            }

            if current is None:
                current = _start_span(token_payload)
                continue

            gap = token_payload["start"] - current["raw_end"]
            if gap <= merge_gap_sec:
                _extend_span(current, token_payload)
            else:
                spans.append(_finalize_span(current, duration_sec, context_sec))
                current = _start_span(token_payload)

        if current is not None:
            spans.append(_finalize_span(current, duration_sec, context_sec))

    return [span for span in spans if span["token_count"] >= min_tokens]


def _start_span(token: dict) -> dict:
    return {
        "segment_id": token["segment_id"],
        "start_token_index": token["token_index"],
        "end_token_index": token["token_index"] + 1,
        "raw_start": token["start"],
        "raw_end": token["end"],
        "local_text": token["token_text"],
        "token_count": 1,
        "min_probability": token["probability"],
        "avg_probability": token["probability"],
        "tokens": [token],
    }


def _extend_span(span: dict, token: dict) -> None:
    total_prob = span["avg_probability"] * span["token_count"] + token["probability"]
    span["end_token_index"] = token["token_index"] + 1
    span["raw_end"] = token["end"]
    span["local_text"] += token["token_text"]
    span["token_count"] += 1
    span["min_probability"] = min(span["min_probability"], token["probability"])
    span["avg_probability"] = total_prob / span["token_count"]
    span["tokens"].append(token)


def _finalize_span(span: dict, duration_sec: float, context_sec: float) -> dict:
    start = max(0.0, span["raw_start"] - context_sec)
    end = min(duration_sec, span["raw_end"] + context_sec) if duration_sec else span["raw_end"] + context_sec
    return {
        "segment_id": span["segment_id"],
        "start_token_index": span["start_token_index"],
        "end_token_index": span["end_token_index"],
        "raw_start": round(span["raw_start"], 3),
        "raw_end": round(span["raw_end"], 3),
        "dispatch_start": round(start, 3),
        "dispatch_end": round(end, 3),
        "local_text": span["local_text"],
        "token_count": span["token_count"],
        "min_probability": round(span["min_probability"], 6),
        "avg_probability": round(span["avg_probability"], 6),
        "tokens": span["tokens"],
    }


def extract_audio_clip(audio_path: Path, start_sec: float, end_sec: float) -> np.ndarray:
    whisper = import_vendor_whisper()
    audio = whisper.load_audio(str(audio_path))
    start_idx = max(0, int(round(start_sec * 16000)))
    end_idx = min(len(audio), int(round(end_sec * 16000)))
    return np.asarray(audio[start_idx:end_idx], dtype=np.float32)


def save_wav_clip(audio: np.ndarray, output_path: Path, sample_rate: int = 16000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def build_dispatched_transcript(analysis: dict, dispatched_spans: Sequence[dict]) -> str:
    by_segment: Dict[int, List[dict]] = {}
    for span in dispatched_spans:
        by_segment.setdefault(int(span["segment_id"]), []).append(span)

    parts: List[str] = []
    for segment in analysis.get("segments", []):
        token_details = segment.get("token_details", [])
        spans = sorted(by_segment.get(int(segment.get("id", 0)), []), key=lambda item: item["start_token_index"])
        cursor = 0
        segment_parts: List[str] = []

        for span in spans:
            for token in token_details[cursor : span["start_token_index"]]:
                segment_parts.append(token["token_text"])
            replacement = span.get("remote_text") or span["local_text"]
            segment_parts.append(replacement)
            cursor = span["end_token_index"]

        for token in token_details[cursor:]:
            segment_parts.append(token["token_text"])

        parts.append("".join(segment_parts))

    return normalize_text(" ".join(parts))

