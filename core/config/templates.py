"""Prompt template helpers for presets and personas."""

from __future__ import annotations

import json
from typing import Any

from astrbot.api import logger

from ..generation.reference_collector import normalize_string_items
from ..shared.logging import log_prefix
from .models import PersonaTemplate

LOG = log_prefix("Config")


def build_generation_prompt(
    *,
    preset_prompts: list[str] | None = None,
    persona_prompts: list[str] | None = None,
    extra_prompt: str = "",
) -> str:
    """Build a structured generation prompt from multiple sources.

    Args:
        preset_prompts: Prompt fragments from matched presets.
        persona_prompts: Prompt fragments from matched personas.
        extra_prompt: User-provided prompt for the current request.

    Returns:
        Original text for single-source prompts, otherwise a lightweight
        sectioned prompt that keeps source intent clear for the model.
    """
    preset_parts = [part.strip() for part in preset_prompts or [] if part.strip()]
    persona_parts = [part.strip() for part in persona_prompts or [] if part.strip()]
    extra_text = str(extra_prompt or "").strip()

    sections: list[tuple[str, list[str]]] = []
    if persona_parts:
        sections.append(("人物设定", persona_parts))
    if preset_parts:
        sections.append(("预设提示词", preset_parts))
    if extra_text:
        sections.append(("附加提示词", [extra_text]))

    if not sections:
        return ""
    if len(sections) == 1 and len(sections[0][1]) == 1:
        return sections[0][1][0]

    blocks: list[str] = []
    for title, parts in sections:
        blocks.append(f"[{title}]\n" + "\n".join(parts))
    return "\n\n".join(blocks).strip()


def normalize_name_items(raw: Any) -> list[str]:
    """Normalize one or many preset/persona names from tool arguments."""
    names: list[str] = []
    seen: set[str] = set()
    for item in normalize_string_items(raw):
        for name in item.split():
            normalized = name.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            names.append(normalized)
    return names


def find_named_entry(entries: dict[str, Any], token: str) -> str | None:
    """Find an entry by exact or case-insensitive name."""
    if token in entries:
        return token
    lowered_token = token.lower()
    for name in entries:
        if name.lower() == lowered_token:
            return name
    return None


def _ascii_token_boundary_ok(text: str, start: int, end: int) -> bool:
    """Return whether an ASCII template name sits on a token boundary.

    CJK names are matched as substrings. ASCII names must not match inside a
    longer ASCII word, so ``cat`` does not match ``category``.
    """
    name = text[start:end]
    if not name.isascii():
        return True
    if (
        start > 0
        and text[start - 1].isascii()
        and text[start - 1].isalnum()
        and name[0].isalnum()
    ):
        return False
    if (
        end < len(text)
        and text[end].isascii()
        and text[end].isalnum()
        and name[-1].isalnum()
    ):
        return False
    return True


def extract_templates_from_prompt(
    prompt: str,
    presets: dict[str, Any],
    personas: dict[str, PersonaTemplate],
    *,
    aspect_ratio: str,
    resolution: str,
    skip_names: list[str] | None = None,
) -> tuple[
    str,
    str,
    str,
    list[str],
    list[str],
    list[str],
    list[str],
    list[tuple[str, str]],
]:
    """Match configured preset/persona names anywhere in prompt text.

    Longer names win at the same position. Each template is applied at most
    once. Scanning is linear in prompt length times template count and stays
    cheap for typical plugin-scale lists.

    Args:
        prompt: User-provided prompt that may contain template names.
        presets: Configured preset name-to-content mapping.
        personas: Configured persona templates.
        aspect_ratio: Current aspect ratio, updated by JSON presets.
        resolution: Current resolution, updated by JSON presets.
        skip_names: Template names already applied by explicit parameters.

    Returns:
        Remaining extra prompt, aspect ratio, resolution, preset prompt
        fragments, persona prompt fragments, matched preset names, matched
        persona names, and persona reference images.
    """
    extra_source = str(prompt or "").strip()
    empty: tuple[
        str,
        str,
        str,
        list[str],
        list[str],
        list[str],
        list[str],
        list[tuple[str, str]],
    ] = ("", aspect_ratio, resolution, [], [], [], [], [])
    if not extra_source:
        return empty

    needles: dict[str, tuple[str, str]] = {}
    for name in personas:
        key = name.strip().lower()
        if key:
            needles[key] = ("persona", name)
    for name in presets:
        key = name.strip().lower()
        if key:
            needles[key] = ("preset", name)
    ordered_needles = sorted(
        ((key, kind, original) for key, (kind, original) in needles.items()),
        key=lambda item: (-len(item[0]), item[0]),
    )
    if not ordered_needles:
        return extra_source, aspect_ratio, resolution, [], [], [], [], []

    skip = {name.strip().lower() for name in skip_names or [] if str(name).strip()}
    lowered = extra_source.lower()
    preset_prompts: list[str] = []
    persona_prompts: list[str] = []
    matched_presets: list[str] = []
    matched_personas: list[str] = []
    persona_images: list[tuple[str, str]] = []
    leftover_chars: list[str] = []
    index = 0
    length = len(extra_source)
    while index < length:
        found: tuple[str, str, int] | None = None
        for key, kind, original in ordered_needles:
            end = index + len(key)
            if end > length or lowered[index:end] != key:
                continue
            if not _ascii_token_boundary_ok(extra_source, index, end):
                continue
            found = (kind, original, end)
            break
        if found is None:
            leftover_chars.append(extra_source[index])
            index += 1
            continue

        kind, original, end = found
        if original.lower() in skip:
            index = end
            continue
        skip.add(original.lower())
        if kind == "preset":
            preset_prompt, aspect_ratio, resolution = parse_preset_prompt(
                presets[original],
                aspect_ratio,
                resolution,
            )
            if preset_prompt:
                preset_prompts.append(preset_prompt)
            matched_presets.append(original)
        else:
            persona = personas[original]
            persona_prompt = persona.prompt.strip()
            if persona_prompt:
                persona_prompts.append(persona_prompt)
            if persona.image:
                persona_images.append((original, persona.image))
            matched_personas.append(original)
        index = end

    extra = extra_source
    if matched_presets or matched_personas:
        extra = " ".join("".join(leftover_chars).split())
    return (
        extra,
        aspect_ratio,
        resolution,
        preset_prompts,
        persona_prompts,
        matched_presets,
        matched_personas,
        persona_images,
    )


def parse_preset_prompt(
    preset_content: Any,
    aspect_ratio: str,
    resolution: str,
) -> tuple[str, str, str]:
    """Parse a preset prompt and optional generation overrides."""
    preset_prompt = str(preset_content or "").strip()
    if not preset_prompt.startswith("{"):
        return preset_prompt, aspect_ratio, resolution

    try:
        preset_data = json.loads(preset_prompt)
    except json.JSONDecodeError:
        return preset_prompt, aspect_ratio, resolution

    if not isinstance(preset_data, dict):
        return preset_prompt, aspect_ratio, resolution

    preset_prompt = str(preset_data.get("prompt", "") or "").strip()
    aspect_ratio = str(preset_data.get("aspect_ratio") or aspect_ratio)
    resolution = str(preset_data.get("resolution") or resolution)
    return preset_prompt, aspect_ratio, resolution


def format_template_summary(
    matched_presets: list[str],
    matched_personas: list[str],
) -> tuple[str | None, str]:
    """Format matched preset/persona names for task metadata."""
    if matched_presets and matched_personas:
        return (
            "；".join(
                (
                    f"预设: {'、'.join(matched_presets)}",
                    f"人设: {'、'.join(matched_personas)}",
                )
            ),
            "预设/人设",
        )
    if matched_presets:
        return "、".join(matched_presets), "预设"
    if matched_personas:
        return "、".join(matched_personas), "人设"
    return None, "预设"


class ConfigTemplateStoreMixin:
    """Mixin for reading and writing preset/persona template config."""

    def _load_presets(self, presets_config: list[Any]) -> dict[str, Any]:
        """Load configured prompt presets."""
        presets: dict[str, Any] = {}
        if not isinstance(presets_config, list):
            return presets

        for preset_str in presets_config:
            if isinstance(preset_str, str) and ":" in preset_str:
                name, prompt = preset_str.split(":", 1)
                if name.strip() and prompt.strip():
                    presets[name.strip()] = prompt.strip()
        return presets

    def _get_writable_prompt_templates_config(self) -> dict[str, Any]:
        """Return the grouped prompt-template config for command-side updates."""
        value = self._config.setdefault("prompt_templates", {})
        if isinstance(value, dict):
            return value
        logger.warning(f"{LOG} prompt_templates 配置格式错误，已重置为空对象")
        value = {}
        self._config["prompt_templates"] = value
        return value

    def _save_presets_config(self) -> None:
        prompt_templates_cfg = self._get_writable_prompt_templates_config()
        prompt_templates_cfg["presets"] = [
            f"{k}:{v}" for k, v in self._plugin_config.presets.items()
        ]
        self._config.save_config()

    def _load_personas(self, personas_config: Any) -> dict[str, PersonaTemplate]:
        """Load configured persona templates."""
        personas: dict[str, PersonaTemplate] = {}
        if not isinstance(personas_config, list):
            return personas

        for item in personas_config:
            if not isinstance(item, dict):
                continue

            name = str(item.get("persona_name") or item.get("name") or "").strip()
            prompt = str(item.get("persona_prompt") or item.get("prompt") or "").strip()
            image = self._parse_file_value(
                item.get("persona_image")
                or item.get("image")
                or item.get("reference_image")
            )
            if name and (prompt or image):
                personas[name] = PersonaTemplate(name=name, prompt=prompt, image=image)
        return personas

    def _parse_file_value(self, raw: Any) -> str:
        """Extract the first usable file path or URL from a file config value."""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, list):
            for item in raw:
                if parsed := self._parse_file_value(item):
                    return parsed
            return ""
        if isinstance(raw, dict):
            for key in ("path", "file", "url", "name"):
                if parsed := self._parse_file_value(raw.get(key)):
                    return parsed
        return ""

    def save_preset(self, name: str, content: str) -> None:
        """Save a prompt preset."""
        self._plugin_config.presets[name] = content
        self._save_presets_config()

    def delete_preset(self, name: str) -> bool:
        """Delete a prompt preset and return whether it existed."""
        if name in self._plugin_config.presets:
            del self._plugin_config.presets[name]
            self._save_presets_config()
            return True
        return False


__all__ = (
    "build_generation_prompt",
    "ConfigTemplateStoreMixin",
    "extract_templates_from_prompt",
    "find_named_entry",
    "format_template_summary",
    "normalize_name_items",
    "parse_preset_prompt",
)
