"""Recover a single JSON object; reject ambiguity and invalid score structures."""
import json
import math
from typing import Any

from .config import WEIGHTS


def extract_json(text: str) -> dict[str, Any]:
    """Accept surrounding prose/fences, but never guess between multiple objects."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty JSON response")
    decoder = json.JSONDecoder()
    candidates = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        candidates.append(value)
        index = start + length
    if len(candidates) != 1:
        raise ValueError("Expected exactly one valid JSON object")
    return candidates[0]


def required_text(data: dict, field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing text: {field}")
    return value.strip()


def weighted_score(scores: dict[str, float]) -> float:
    """Compute a 0–5 score locally using the published rubric."""
    return round(sum(scores[key] * weight for key, weight in WEIGHTS.items()), 4)


def parse_judgment(text: str) -> dict:
    """Coerce finite numeric strings, clamp to 0–5 and record every correction."""
    data = extract_json(text)
    scores, reasons, warnings = {}, {}, []
    for key in WEIGHTS:
        item = data.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"Missing metric object: {key}")
        raw = item.get("score")
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValueError(f"Invalid score: {key}")
        try:
            number = float(raw)
        except ValueError as exc:
            raise ValueError(f"Nonnumeric score: {key}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite score: {key}")
        scores[key] = min(5.0, max(0.0, number))
        if isinstance(raw, str):
            warnings.append(f"{key}: numeric string converted")
        if number != scores[key]:
            warnings.append(f"{key}: {number} clamped to {scores[key]}")
        reasons[key] = required_text(item, "reason")
    overall = weighted_score(scores)
    return {"scores": scores, "reasons": reasons, "overall_score": overall,
            "score_100": round(overall * 20, 2),
            "overall_reason": required_text(data, "overall_reason"),
            "improvement": required_text(data, "improvement"),
            "warnings": warnings}
