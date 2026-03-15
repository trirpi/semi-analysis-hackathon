from __future__ import annotations

import json
import os
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


def trailing_words(text: str, count: int) -> str:
    if count <= 0:
        return ""
    matches = list(re.finditer(r"[A-Za-z0-9']+", text))
    if not matches:
        return ""
    start = matches[max(0, len(matches) - count)].start()
    return normalize_text(text[start:])


def leading_words(text: str, count: int) -> str:
    if count <= 0:
        return ""
    matches = list(re.finditer(r"[A-Za-z0-9']+", text))
    if not matches:
        return ""
    end = matches[min(len(matches), count) - 1].end()
    return normalize_text(text[:end])


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def has_alnum(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", text))


def _normalize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", normalize_text(text).lower())


def _drop_prefix_words(text: str, count: int) -> str:
    if count <= 0:
        return normalize_text(text)
    words = normalize_text(text).split()
    return " ".join(words[count:]).strip()


def _drop_suffix_words(text: str, count: int) -> str:
    if count <= 0:
        return normalize_text(text)
    words = normalize_text(text).split()
    if count >= len(words):
        return ""
    return " ".join(words[:-count]).strip()


def _trim_context_overlap(remote_text: str, prefix_text: str, suffix_text: str) -> str:
    remote = normalize_text(remote_text)
    if not remote:
        return ""

    remote_words = _normalize_words(remote)
    prefix_words = _normalize_words(prefix_text)
    suffix_words = _normalize_words(suffix_text)

    max_prefix_overlap = min(len(prefix_words), len(remote_words), 6)
    for overlap in range(max_prefix_overlap, 0, -1):
        if prefix_words[-overlap:] == remote_words[:overlap]:
            remote = _drop_prefix_words(remote, overlap)
            remote_words = _normalize_words(remote)
            break

    max_suffix_overlap = min(len(suffix_words), len(remote_words), 6)
    for overlap in range(max_suffix_overlap, 0, -1):
        if suffix_words[:overlap] == remote_words[-overlap:]:
            remote = _drop_suffix_words(remote, overlap)
            break

    return normalize_text(remote)


def collect_low_confidence_spans(
    analysis: dict,
    *,
    threshold: float,
    merge_gap_sec: float = 0.15,
    context_sec: float = 0.2,
    left_context_sec: float | None = None,
    right_context_sec: float | None = None,
    min_tokens: int = 1,
    min_words: int = 1,
    min_duration_sec: float = 0.0,
    skip_punctuation_only: bool = True,
) -> List[dict]:
    duration_sec = float(analysis.get("duration_sec") or 0.0)
    spans: List[dict] = []
    left_context_sec = context_sec if left_context_sec is None else left_context_sec
    right_context_sec = context_sec if right_context_sec is None else right_context_sec

    for segment_position, segment in enumerate(analysis.get("segments", [])):
        words = _segment_words_with_tokens(segment, segment_position)
        current = None

        for word in words:
            if float(word["probability"]) >= threshold:
                if current is not None:
                    spans.append(
                        _finalize_span(
                            current,
                            duration_sec,
                            left_context_sec,
                            right_context_sec,
                        )
                    )
                    current = None
                continue

            if current is None:
                current = _start_span(word)
                continue

            gap = word["start"] - current["raw_end"]
            if gap <= merge_gap_sec:
                _extend_span(current, word)
            else:
                spans.append(
                    _finalize_span(
                        current,
                        duration_sec,
                        left_context_sec,
                        right_context_sec,
                    )
                )
                current = _start_span(word)

        if current is not None:
            spans.append(
                _finalize_span(
                    current,
                    duration_sec,
                    left_context_sec,
                    right_context_sec,
                )
            )

    filtered = []
    for span in spans:
        duration = span["raw_end"] - span["raw_start"]
        if span["token_count"] < min_tokens:
            continue
        if span["word_count"] < min_words:
            continue
        if duration < min_duration_sec:
            continue
        if skip_punctuation_only and not has_alnum(span["local_text"]):
            continue
        filtered.append(span)
    return filtered


def _segment_words_with_tokens(segment: dict, segment_position: int) -> List[dict]:
    token_details = segment.get("token_details", [])
    words = segment.get("words", [])
    if not words:
        return [
            {
                "segment_id": int(segment.get("id", 0)),
                "segment_position": segment_position,
                "start_token_index": idx,
                "end_token_index": idx + 1,
                "raw_start": float(token["start"]),
                "raw_end": float(token["end"]),
                "start": float(token["start"]),
                "end": float(token["end"]),
                "local_text": token["token_text"],
                "probability": float(token["probability"]),
                "word_count": 1 if has_alnum(token["token_text"]) else 0,
                "tokens": [{**token, "segment_position": segment_position, "token_index": idx}],
            }
            for idx, token in enumerate(token_details)
        ]

    mapped_words: List[dict] = []
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        overlapping = [
            idx
            for idx, token in enumerate(token_details)
            if float(token["end"]) > start + 1e-6 and float(token["start"]) < end - 1e-6
        ]
        if not overlapping:
            nearest = min(
                range(len(token_details)),
                key=lambda idx: abs(float(token_details[idx]["start"]) - start),
            )
            overlapping = [nearest]

        start_idx = overlapping[0]
        end_idx = overlapping[-1] + 1
        tokens = [
            {
                **token_details[idx],
                "segment_position": segment_position,
                "token_index": idx,
            }
            for idx in range(start_idx, end_idx)
        ]
        mapped_words.append(
            {
                "segment_id": int(segment.get("id", 0)),
                "segment_position": segment_position,
                "start_token_index": start_idx,
                "end_token_index": end_idx,
                "raw_start": start,
                "raw_end": end,
                "start": start,
                "end": end,
                "local_text": "".join(token["token_text"] for token in tokens),
                "probability": float(word["probability"]),
                "word_count": 1 if has_alnum(word["word"]) else 0,
                "tokens": tokens,
            }
        )
    return mapped_words


def _start_span(token: dict) -> dict:
    return {
        "segment_id": token["segment_id"],
        "segment_position": token["segment_position"],
        "start_token_index": token["start_token_index"],
        "end_token_index": token["end_token_index"],
        "raw_start": token["raw_start"],
        "raw_end": token["raw_end"],
        "local_text": token["local_text"],
        "token_count": len(token["tokens"]),
        "word_count": token["word_count"],
        "min_probability": token["probability"],
        "avg_probability": token["probability"],
        "tokens": list(token["tokens"]),
    }


def _extend_span(span: dict, token: dict) -> None:
    total_prob = span["avg_probability"] * span["word_count"] + token["probability"]
    span["end_token_index"] = token["end_token_index"]
    span["raw_end"] = token["raw_end"]
    span["local_text"] += token["local_text"]
    span["token_count"] += len(token["tokens"])
    span["word_count"] += token["word_count"]
    span["min_probability"] = min(span["min_probability"], token["probability"])
    span["avg_probability"] = total_prob / max(1, span["word_count"])
    span["tokens"].extend(token["tokens"])


def _finalize_span(
    span: dict,
    duration_sec: float,
    left_context_sec: float,
    right_context_sec: float,
) -> dict:
    start = max(0.0, span["raw_start"] - left_context_sec)
    end = (
        min(duration_sec, span["raw_end"] + right_context_sec)
        if duration_sec
        else span["raw_end"] + right_context_sec
    )
    return {
        "segment_id": span["segment_id"],
        "segment_position": span["segment_position"],
        "start_token_index": span["start_token_index"],
        "end_token_index": span["end_token_index"],
        "raw_start": round(span["raw_start"], 3),
        "raw_end": round(span["raw_end"], 3),
        "dispatch_start": round(start, 3),
        "dispatch_end": round(end, 3),
        "local_text": span["local_text"],
        "token_count": span["token_count"],
        "word_count": span["word_count"],
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


def build_prompt_prefix(analysis: dict, span: dict, prompt_prefix_words: int) -> str:
    if prompt_prefix_words <= 0:
        return ""

    pieces: List[str] = []
    for index, segment in enumerate(analysis.get("segments", [])):
        token_details = segment.get("token_details", [])
        if index < span["segment_position"]:
            pieces.extend(token["token_text"] for token in token_details)
            continue
        if index == span["segment_position"]:
            pieces.extend(
                token["token_text"]
                for token in token_details[: span["start_token_index"]]
            )
            break

    return trailing_words("".join(pieces), prompt_prefix_words)


def build_following_context(analysis: dict, span: dict, following_words: int) -> str:
    if following_words <= 0:
        return ""

    pieces: List[str] = []
    started = False
    for index, segment in enumerate(analysis.get("segments", [])):
        token_details = segment.get("token_details", [])
        if index < span["segment_position"]:
            continue
        if index == span["segment_position"]:
            pieces.extend(
                token["token_text"]
                for token in token_details[span["end_token_index"] :]
            )
            started = True
            continue
        if started:
            pieces.extend(token["token_text"] for token in token_details)

    return leading_words("".join(pieces), following_words)


def dispatch_with_openai(
    clip_path: Path,
    language: str | None,
    model_name: str,
    prompt: str = "",
) -> str:
    from openai import OpenAI

    client = OpenAI()
    with open(clip_path, "rb") as handle:
        response = client.audio.transcriptions.create(
            model=model_name,
            file=handle,
            language=language,
            prompt=prompt or None,
            temperature=0,
        )
    text = getattr(response, "text", None)
    if text is None and isinstance(response, dict):
        text = response.get("text")
    return normalize_text(text or "")


def _accept_remote_text(
    remote_text: str,
    local_text: str,
    local_word_count: int,
    prompt_prefix_text: str,
    following_context_text: str,
    max_extra_words: int = 2,
) -> str:
    cleaned = _trim_context_overlap(remote_text, prompt_prefix_text, following_context_text)
    if not cleaned:
        return ""
    if not has_alnum(cleaned) and has_alnum(local_text):
        return ""

    local_words = max(1, word_count(local_text))
    remote_words = word_count(cleaned)
    if remote_words == 0 and local_words > 0:
        return ""
    if local_word_count <= 1 and remote_words > 1:
        return ""
    if remote_words > max(local_words + max_extra_words, local_words * 2):
        return ""
    if len(cleaned) > max(len(normalize_text(local_text)) * 3, 40):
        return ""
    return cleaned


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


def dispatch_analysis(
    analysis: dict,
    *,
    threshold: float,
    merge_gap_sec: float = 0.15,
    context_sec: float = 0.2,
    left_context_sec: float | None = None,
    right_context_sec: float | None = None,
    min_tokens: int = 1,
    min_words: int = 1,
    min_duration_sec: float = 0.0,
    skip_punctuation_only: bool = True,
    prompt_prefix_words: int = 0,
    following_context_words: int = 3,
    run_openai: bool = False,
    openai_model: str = "whisper-1",
) -> dict:
    audio_path = Path(analysis["audio"]).resolve()
    spans = collect_low_confidence_spans(
        analysis,
        threshold=threshold,
        merge_gap_sec=merge_gap_sec,
        context_sec=context_sec,
        left_context_sec=left_context_sec,
        right_context_sec=right_context_sec,
        min_tokens=min_tokens,
        min_words=min_words,
        min_duration_sec=min_duration_sec,
        skip_punctuation_only=skip_punctuation_only,
    )

    report = {
        "audio": str(audio_path),
        "threshold": threshold,
        "merge_gap_sec": merge_gap_sec,
        "context_sec": context_sec,
        "left_context_sec": context_sec if left_context_sec is None else left_context_sec,
        "right_context_sec": context_sec if right_context_sec is None else right_context_sec,
        "min_tokens": min_tokens,
        "min_words": min_words,
        "min_duration_sec": min_duration_sec,
        "skip_punctuation_only": skip_punctuation_only,
        "prompt_prefix_words": prompt_prefix_words,
        "following_context_words": following_context_words,
        "run_openai": run_openai,
        "openai_model": openai_model,
        "local_transcript": normalize_text(analysis.get("text", "")),
        "dispatch_spans": [
            {
                **span,
                "prompt_prefix_text": build_prompt_prefix(
                    analysis,
                    span,
                    prompt_prefix_words,
                ),
                "following_context_text": build_following_context(
                    analysis,
                    span,
                    following_context_words,
                ),
            }
            for span in spans
        ],
    }

    if not run_openai:
        report["dispatched_transcript"] = report["local_transcript"]
        return report

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required when run_openai=True")

    from tempfile import TemporaryDirectory

    dispatched: List[dict] = []
    with TemporaryDirectory(prefix="dispatch-clips-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, span in enumerate(report["dispatch_spans"]):
            clip_audio = extract_audio_clip(
                audio_path,
                span["dispatch_start"],
                span["dispatch_end"],
            )
            clip_path = temp_root / f"span-{index:03d}.wav"
            save_wav_clip(clip_audio, clip_path)
            remote_text = dispatch_with_openai(
                clip_path,
                analysis.get("language"),
                openai_model,
                prompt=span.get("prompt_prefix_text", ""),
            )
            accepted_text = _accept_remote_text(
                remote_text,
                span["local_text"],
                span["word_count"],
                span.get("prompt_prefix_text", ""),
                span.get("following_context_text", ""),
            )
            dispatched.append(
                {
                    **span,
                    "remote_text": accepted_text or span["local_text"],
                    "remote_text_raw": remote_text,
                    "accepted_remote_text": bool(accepted_text),
                }
            )

    report["dispatch_spans"] = dispatched
    report["dispatched_transcript"] = build_dispatched_transcript(analysis, dispatched)
    return report

