"""Extract explicit size options from command text without interpreting prose."""

from __future__ import annotations

import re
from typing import NamedTuple

from ..shared.constants import SUPPORTED_ASPECT_RATIOS, SUPPORTED_RESOLUTIONS


class GenerationOptions(NamedTuple):
    prompt: str
    aspect_ratio: str | None = None
    resolution: str | None = None
    error: str | None = None


_KEY = r"宽高比|比例|画幅|分辨率"
# Prefer complete numeric expressions (including internal whitespace), but also
# capture invalid explicit values so they cannot silently fall back to defaults.
_VALUE = (
    r"(?:\d+\s*[:：]\s*\d+|\d+(?:\.\d+)?\s*[kK])"
    r"(?![A-Za-z0-9_.:：])|[^\s，,。；;！？!?、=：:]+"
)
_REVERSE = re.compile(
    rf"(?:使用\s*)?(?P<value>\d+\s*[:：]\s*\d+|"
    rf"\d+(?:\.\d+)?\s*[kK]|[A-Za-z0-9_.:+-]+|不指定)\s*(?P<key>{_KEY})"
)
_FORWARD = re.compile(
    # Embedded prose such as "高分辨率 猫" is not an option. Embedded keys
    # still work when followed by a number or an explicit assignment marker.
    rf"(?:(?<!\w)|(?=(?:使用\s*)?(?:{_KEY})\s*(?:为|是|[:：=]|[0-9]|不指定)))"
    rf"(?:使用\s*)?(?P<key>{_KEY})"
    rf"(?:\s*(?:为|是|[:：=])\s*|\s+|(?=[0-9]|不指定))"
    rf"(?P<value>{_VALUE})?"
)
_SUFFIX = re.compile(
    r"(?<!\S)(?P<value>\d+\s*[:：]\s*\d+|\d+(?:\.\d+)?\s*[kK])\s*$"
)
_SEPARATORS = " \t\r\n，,；;、。"


def _clean_spans(prompt: str, spans: list[tuple[int, int]]) -> str:
    """Remove matched clauses, touching punctuation only at their boundaries."""
    for start, end in sorted(spans, reverse=True):
        left, right = prompt[:start].rstrip(), prompt[end:].lstrip()
        if not right.strip(_SEPARATORS):
            prompt = left.rstrip(_SEPARATORS)
        elif not left.strip(_SEPARATORS):
            prompt = right.lstrip(_SEPARATORS)
        elif left[-1] in _SEPARATORS:
            prompt = left + right.lstrip(_SEPARATORS)
        elif right[0] in _SEPARATORS:
            prompt = left + right
        else:
            prompt = left + " " + right
    return prompt


def parse_generation_options(prompt: str) -> GenerationOptions:
    """Return cleaned text and validated overrides; leave unmarked prose alone.

    The command's trailing image count must be removed by the caller first.
    Conflicting or invalid explicit values reject the whole request.
    """
    matches: list[tuple[int, int, str, str]] = []
    candidates: list[tuple[int, int, str, str]] = []
    for pattern in (_REVERSE, _FORWARD):
        for match in pattern.finditer(prompt):
            start, end = match.span()
            key = "resolution" if match["key"] == "分辨率" else "aspect_ratio"
            candidates.append((start, end, key, match["value"] or ""))
    # Resolve in text order so adjacent forward clauses do not become a reverse
    # phrase: "比例 16:9 分辨率 2K" must not consume "16:9 分辨率".
    for candidate in sorted(candidates):
        if not matches or candidate[0] >= matches[-1][1]:
            matches.append(candidate)

    # Match only a contiguous suffix of independent numeric size tokens. Do not
    # scan isolated values in the body, or across an explicit parameter clause.
    suffix_end = len(prompt)
    while match := _SUFFIX.search(prompt[:suffix_end]):
        start, end = match.span()
        if any(
            start < used_end and end > used_start
            for used_start, used_end, _, _ in matches
        ):
            break
        value = match["value"]
        key = "aspect_ratio" if ":" in value or "：" in value else "resolution"
        matches.append((start, end, key, value))
        suffix_end = start

    values: dict[str, str] = {}
    for _start, _end, key, raw in sorted(matches):
        value = re.sub(r"\s+", "", raw).replace("：", ":").upper()
        supported = (
            SUPPORTED_RESOLUTIONS if key == "resolution" else SUPPORTED_ASPECT_RATIOS
        )
        label = "分辨率" if key == "resolution" else "宽高比"
        if value not in supported:
            return GenerationOptions(
                prompt, error=f"{label}参数无效：{raw or '未填写'}。支持的值：{'、'.join(supported)}"
            )
        if key in values and values[key] != value:
            return GenerationOptions(
                prompt, error=f"{label}参数冲突：{values[key]} 与 {value}，请只指定一个值。"
            )
        values[key] = value

    return GenerationOptions(
        _clean_spans(prompt, [(start, end) for start, end, _, _ in matches]),
        values.get("aspect_ratio"),
        values.get("resolution"),
    )
