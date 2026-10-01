"""Preset and persona text helpers for LLM tools.

These functions only normalize aliases, format query results, and validate
preset JSON. Tool classes and their schemas stay in ``tools.py``.
"""

from __future__ import annotations

import json
from typing import Any

from ..shared.constants import SUPPORTED_ASPECT_RATIOS, SUPPORTED_RESOLUTIONS

ASPECT_RATIO_OPTIONS = list(SUPPORTED_ASPECT_RATIOS)
RESOLUTION_OPTIONS = list(SUPPORTED_RESOLUTIONS)


def normalize_preset_query_category(category: str) -> str:
    """Normalize preset query category aliases."""
    normalized = category.strip().lower()
    if normalized in {
        "人设",
        "persona",
        "personas",
        "list_persona",
        "list_personas",
        "get_persona",
        "persona_list",
    }:
        return "persona"
    return "preset"


def normalize_preset_edit_action(action: str) -> str:
    """Normalize preset edit action aliases."""
    normalized = action.strip().lower()
    if normalized in {"添加", "新增", "保存", "save", "create", "add", "add_preset"}:
        return "create_preset"
    if normalized in {
        "删除",
        "移除",
        "remove",
        "del",
        "delete",
        "delete_preset",
    }:
        return "delete_preset"
    return normalized


def normalize_task_action(action: str) -> str:
    """Normalize image task management action aliases."""
    normalized = action.strip().lower()
    if normalized in {"", "列表", "查看列表", "list", "list_tasks", "tasks"}:
        return "list"
    if normalized in {"详情", "查看", "查询", "detail", "get", "show"}:
        return "detail"
    if normalized in {"取消", "cancel", "cancel_task"}:
        return "cancel"
    return normalized


def _format_preset_detail(name: str, content: Any) -> str:
    """Format one preset's full content for query results."""
    content_text = str(content or "").strip()
    lines = [f"📋 预设详情: {name}"]
    if content_text.startswith("{"):
        try:
            preset_data = json.loads(content_text)
        except json.JSONDecodeError:
            lines.append("格式: 高级 JSON（解析失败，将按原文展示）")
            lines.append(f"内容: {content_text}")
            return "\n".join(lines)

        if isinstance(preset_data, dict):
            lines.append("格式: 高级 JSON")
            if prompt := str(preset_data.get("prompt", "") or "").strip():
                lines.append(f"提示词: {prompt}")
            if aspect_ratio := str(preset_data.get("aspect_ratio", "") or "").strip():
                lines.append(f"宽高比: {aspect_ratio}")
            if resolution := str(preset_data.get("resolution", "") or "").strip():
                lines.append(f"分辨率: {resolution}")
            if description := str(preset_data.get("description", "") or "").strip():
                lines.append(f"描述: {description}")
            lines.append(f"原始内容: {content_text}")
            return "\n".join(lines)

    lines.append("格式: 简单提示词")
    lines.append(f"内容: {content_text}")
    return "\n".join(lines)


def _format_persona_detail(name: str, persona: Any) -> str:
    """Format one persona's full content for query results."""
    lines = [f"👤 人设详情: {name}"]
    lines.append(f"提示词: {persona.prompt}")
    lines.append(f"参考图: {persona.image or '无'}")
    return "\n".join(lines)


def _validate_preset_content(content: str) -> str | None:
    """Validate preset content when it is written by an LLM tool."""
    if not content.startswith("{"):
        return None

    try:
        preset_data = json.loads(content)
    except json.JSONDecodeError as exc:
        return f"高级 JSON 预设格式错误: {exc}"

    if not isinstance(preset_data, dict):
        return "高级 JSON 预设必须是对象"
    if not str(preset_data.get("prompt", "") or "").strip():
        return "高级 JSON 预设必须包含非空 prompt 字段"

    aspect_ratio = str(preset_data.get("aspect_ratio", "") or "").strip()
    if aspect_ratio and aspect_ratio not in ASPECT_RATIO_OPTIONS:
        return f"高级 JSON 预设的 aspect_ratio 不支持: {aspect_ratio}"

    resolution = str(preset_data.get("resolution", "") or "").strip()
    if resolution and resolution not in RESOLUTION_OPTIONS:
        return f"高级 JSON 预设的 resolution 不支持: {resolution}"

    return None
