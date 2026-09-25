"""Logging helpers for the image generation plugin."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import parse_qsl, urlparse

from .constants import (
    LOG_PREFIX,
    MASK_MIN_LENGTH,
    MASK_PLACEHOLDER,
    MASK_VISIBLE_CHARS,
)


def log_prefix(component: str | None = None, task_id: str | None = None) -> str:
    """Build a consistent log prefix with optional component and task id."""
    parts = [LOG_PREFIX]
    if component:
        parts.append(f"[{component}]")
    if task_id:
        parts.append(f"[{task_id}]")
    return " ".join(parts)


def mask_sensitive(
    value: object,
    visible_chars: int = MASK_VISIBLE_CHARS,
    min_length: int = MASK_MIN_LENGTH,
    placeholder: str = MASK_PLACEHOLDER,
) -> str:
    """Mask sensitive identifiers before logging."""
    text = str(value or "")
    if len(text) <= min_length:
        return placeholder
    return f"{text[:visible_chars]}{placeholder}{text[-visible_chars:]}"


def safe_log_text(value: object, limit: int = 120) -> str:
    """Return a single-line, length-limited text summary for logs."""
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...({len(text)} chars)"


_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)"
    r"([\"']?\s*[:=]\s*[\"']?)"
    r"([^\"'\s,}]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._~+\-/=]{8,})")
_DATA_URL_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")
_LONG_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b")
SENSITIVE_LOG_FIELD_NAMES = {
    "authorization",
    "api-key",
    "apikey",
    "api_key",
    "access-token",
    "accesstoken",
    "access_token",
    "refresh-token",
    "refreshtoken",
    "refresh_token",
    "token",
    "secret",
    "password",
    "x-goog-api-key",
}
SUMMARY_PRIORITY_KEYS = (
    "message",
    "msg",
    "error",
    "detail",
    "details",
    "code",
    "type",
    "param",
    "status",
)
DETAIL_FIELD_NAMES = {
    "message",
    "msg",
    "error",
    "detail",
    "details",
    "reason",
    "code",
    "type",
    "param",
    "status",
}
LARGE_PAYLOAD_FIELD_HINTS = (
    "base64",
    "b64",
    "bytes",
    "image",
    "file",
    "content",
    "payload",
    "data",
    "document",
    "图片",
    "文件",
    "内容",
)


def is_sensitive_log_field(name: object) -> bool:
    """Return whether a field name should always be masked in debug logs."""
    normalized = str(name or "").strip().lower()
    if normalized in SENSITIVE_LOG_FIELD_NAMES:
        return True
    normalized = normalized.replace("_", "-")
    return normalized in SENSITIVE_LOG_FIELD_NAMES


def _normalize_field_name(name: object) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _is_large_payload_field(name: object) -> bool:
    normalized = _normalize_field_name(name)
    if not normalized or normalized in DETAIL_FIELD_NAMES:
        return False
    return any(hint in normalized for hint in LARGE_PAYLOAD_FIELD_HINTS)


def _sanitize_error_text(value: object) -> str:
    text = str(value or "")
    text = _DATA_URL_RE.sub(lambda m: f"data-url({len(m.group(0))} chars)", text)
    text = _LONG_BASE64_RE.sub(lambda m: f"base64({len(m.group(0))} chars)", text)
    text = _BEARER_RE.sub(lambda m: f"Bearer {mask_sensitive(m.group(1))}", text)

    def _mask_assignment(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{mask_sensitive(match.group(3))}"

    return _SENSITIVE_ASSIGNMENT_RE.sub(_mask_assignment, text)


def _try_parse_json_text(value: object) -> object | None:
    text = str(value or "").strip()
    if not text or text[0] not in '[{"':
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def _summarize_safe_value(
    value: object,
    *,
    limit: int,
    field_limit: int = 160,
    max_fields: int = 8,
    max_list_items: int = 3,
    max_depth: int = 3,
    depth: int = 0,
    field_name: object = "",
    parse_json: bool = True,
) -> str:
    if isinstance(value, dict):
        if set(value.keys()) == {"error"}:
            return _summarize_safe_value(
                value.get("error"),
                limit=limit,
                field_limit=field_limit,
                max_fields=max_fields,
                max_list_items=max_list_items,
                max_depth=max_depth,
                depth=depth + 1,
                field_name="error",
                parse_json=parse_json,
            )

        ordered_keys = [key for key in SUMMARY_PRIORITY_KEYS if key in value]
        ordered_keys.extend(key for key in value if key not in ordered_keys)

        parts: list[str] = []
        for key in ordered_keys[:max_fields]:
            raw_item = value.get(key)
            if is_sensitive_log_field(key):
                item = mask_sensitive(raw_item)
            else:
                item = _summarize_safe_value(
                    raw_item,
                    limit=field_limit,
                    field_limit=field_limit,
                    max_fields=max_fields,
                    max_list_items=max_list_items,
                    max_depth=max_depth,
                    depth=depth + 1,
                    field_name=key,
                    parse_json=parse_json,
                )
            if item:
                parts.append(f"{key}={item}")
        if len(ordered_keys) > max_fields:
            parts.append(f"+{len(ordered_keys) - max_fields} fields")
        return safe_log_text("，".join(parts) or safe_log_mapping(value), limit)

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, dict | list) for item in value[:max_list_items]):
            items = [
                _summarize_safe_value(
                    item,
                    limit=field_limit,
                    field_limit=field_limit,
                    max_fields=max_fields,
                    max_list_items=max_list_items,
                    max_depth=max_depth,
                    depth=depth + 1,
                    field_name=field_name,
                    parse_json=parse_json,
                )
                for item in value[:max_list_items]
            ]
            suffix = (
                f", +{len(value) - max_list_items} items"
                if len(value) > max_list_items
                else ""
            )
            return "[" + ", ".join(items) + suffix + "]"
        return f"list(len={len(value)})"

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)

    text = _sanitize_error_text(value).replace("\r", " ").replace("\n", " ").strip()
    parsed = _try_parse_json_text(text) if parse_json and depth < max_depth else None
    if parsed is not None:
        return _summarize_safe_value(
            parsed,
            limit=limit,
            field_limit=field_limit,
            max_fields=max_fields,
            max_list_items=max_list_items,
            max_depth=max_depth,
            depth=depth + 1,
            field_name=field_name,
            parse_json=parse_json,
        )
    if _is_large_payload_field(field_name) and len(text) > field_limit:
        field_label = _normalize_field_name(field_name) or "payload"
        if text.startswith(("base64(", "data-url(")):
            return text
        return f"{field_label}(len={len(text)})"
    return safe_log_text(text, field_limit)


def safe_log_summary(
    value: object,
    limit: int = 200,
    *,
    field_limit: int = 120,
    max_fields: int = 8,
    max_list_items: int = 3,
    parse_json: bool = True,
) -> str:
    """Return a sanitized summary for logs and user-facing diagnostics.

    Args:
        value: Raw value, provider body, exception, mapping, list, or text.
        limit: Maximum length of the final summary.
        field_limit: Maximum length for one scalar field.
        max_fields: Maximum number of mapping fields to include.
        max_list_items: Maximum number of simple list items to include.
        parse_json: Whether to parse JSON strings and nested JSON strings.

    Returns:
        A one-line summary with secrets and large payload fields compacted.
    """
    if isinstance(value, dict | list):
        return _summarize_safe_value(
            value,
            limit=limit,
            field_limit=field_limit,
            max_fields=max_fields,
            max_list_items=max_list_items,
            parse_json=parse_json,
        )

    text = _sanitize_error_text(value)
    parsed = _try_parse_json_text(text) if parse_json else None
    if parsed is not None:
        return _summarize_safe_value(
            parsed,
            limit=limit,
            field_limit=field_limit,
            max_fields=max_fields,
            max_list_items=max_list_items,
            parse_json=parse_json,
        )
    return safe_log_text(text, limit=limit)


def safe_log_error_body(value: object, limit: int = 200) -> str:
    """Return a compact provider error body summary for logs.

    The plugin keeps provider error bodies readable for administrators, but still
    removes values that are almost always secrets or binary payloads.
    """
    return safe_log_summary(value, limit=limit, field_limit=min(120, limit))


def safe_user_error_detail(value: object, limit: int = 600) -> str:
    """Return a structured, sanitized provider error summary for users.

    Args:
        value: Raw provider error body or exception text.
        limit: Maximum length of the final detail string.

    Returns:
        A sanitized summary that preserves important JSON fields when possible.
    """
    return safe_log_summary(
        value,
        limit=limit,
        field_limit=180,
        max_fields=10,
        max_list_items=4,
    )


def _query_summary(query: str, max_keys: int = 6) -> str:
    """Return query parameter names without values for URL logs."""
    if not query:
        return ""
    keys: list[str] = []
    for key, _ in parse_qsl(query, keep_blank_values=True):
        if key and key not in keys:
            keys.append(key)
    if not keys:
        return f"query_len={len(query)}"
    visible = keys[:max_keys]
    suffix = f",+{len(keys) - max_keys}" if len(keys) > max_keys else ""
    return f"query_keys={','.join(visible)}{suffix};query_len={len(query)}"


def safe_log_url(value: object, limit: int = 80) -> str:
    """Mask URLs, data URLs, and local paths before writing logs."""
    text = str(value or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered.startswith("data:"):
        return f"data-url({len(text)} chars)"
    if lowered.startswith("base64://"):
        return f"base64-url({len(text)} chars)"

    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc or "<unknown>"
        path = parsed.path or "/"
        if len(path) > limit:
            path = f"{path[:limit]}..."
        query = f"?<{_query_summary(parsed.query)}>" if parsed.query else ""
        return f"{parsed.scheme}://{host}{path}{query}"

    if parsed.scheme == "file" or os.path.isabs(text) or ":\\" in text:
        normalized_path = text.replace("\\", "/")
        return safe_log_text(normalized_path, limit=max(limit, 160))

    return safe_log_text(text, limit=limit)


def safe_log_mapping(value: object, limit: int = 120) -> str:
    """Return a compact mapping/list summary without dumping provider payloads."""
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return safe_log_text(value, limit=limit)


def format_seconds(value: float | None) -> str:
    """Format seconds using Chinese units for human-facing logs."""
    if value is None:
        return "未知"
    return f"{max(0.0, value):.2f}秒"


def format_sub_request(index: int | None, count: int | None) -> str:
    """Format a generation sub-request index for task logs."""
    if not index or not count or count <= 1:
        return ""
    return f"子请求={max(1, index)}/{max(1, count)}"


def format_optional(value: object, empty: str = "无", limit: int = 120) -> str:
    """Format optional values for readable Chinese logs."""
    text = safe_log_text(value, limit=limit)
    return text if text else empty


def format_log_fields(**fields: object) -> str:
    """Format optional key-value fields for debug logs."""
    parts: list[str] = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={safe_log_text(value, 160)}")
    return ", ".join(parts)


def format_cn_log_fields(**fields: object) -> str:
    """Format optional fields with Chinese labels for readable logs."""
    parts: list[str] = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={safe_log_text(value, 160)}")
    return "，".join(parts)


def format_log_event(event: str, **fields: object) -> str:
    """Format a stable event name followed by optional Chinese fields."""
    details = format_cn_log_fields(**fields)
    if not details:
        return f"事件={safe_log_text(event, 80)}"
    return f"事件={safe_log_text(event, 80)}，{details}"
