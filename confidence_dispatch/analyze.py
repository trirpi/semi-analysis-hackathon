from __future__ import annotations

import importlib
from pathlib import Path
from typing import List

import numpy as np
import torch

from kernels.vendor_whisper import import_vendor_whisper, whisper_cache_dir


def decode_token_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.encoding.decode([token_id])
    except Exception:
        raw = tokenizer.encoding.decode_single_token_bytes(token_id)
        return raw.decode("utf-8", errors="replace")


def find_token_alignment(whisper, model, tokenizer, text_tokens, mel, num_frames: int):
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
                    "token_text": decode_token_text(tokenizer, int(token_id)),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "probability": round(float(text_token_probs[idx]), 6),
                }
            )
        return details
    finally:
        for hook in hooks:
            hook.remove()


def patched_add_word_timestamps_impl(
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
    flat_text_tokens = [token for segment_tokens in text_tokens_per_segment for token in segment_tokens]
    token_details = find_token_alignment(
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


def build_validation(result: dict) -> dict:
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


class ConfidenceAnalyzer:
    def __init__(self, model_name: str = "tiny", device: str = "cpu") -> None:
        self.whisper = import_vendor_whisper()
        self.model_name = model_name
        self.device = device
        self.model = self.whisper.load_model(
            model_name,
            device=device,
            download_root=str(whisper_cache_dir()),
        )

    def analyze(self, audio_path: Path) -> dict:
        audio = self.whisper.load_audio(str(audio_path))
        duration_sec = len(audio) / 16000.0

        transcribe_module = importlib.import_module("whisper.transcribe")
        original_add_word_timestamps = transcribe_module.add_word_timestamps

        def patched_add_word_timestamps(**kwargs):
            return patched_add_word_timestamps_impl(
                whisper=self.whisper,
                original_add_word_timestamps=original_add_word_timestamps,
                **kwargs,
            )

        transcribe_module.add_word_timestamps = patched_add_word_timestamps
        try:
            result = self.model.transcribe(
                str(audio_path),
                fp16=False,
                verbose=None,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
            )
        finally:
            transcribe_module.add_word_timestamps = original_add_word_timestamps

        result["device"] = self.device
        result["audio"] = str(audio_path)
        result["duration_sec"] = round(duration_sec, 3)
        result["confidence_definition"] = (
            "Per-token confidence is the softmax probability assigned to each emitted text token "
            "during Whisper's alignment pass over the final transcript for that segment."
        )
        result["validation"] = build_validation(result)
        return result


def analyze_audio(audio_path: Path, model_name: str = "tiny", device: str = "cpu") -> dict:
    return ConfidenceAnalyzer(model_name=model_name, device=device).analyze(audio_path)
