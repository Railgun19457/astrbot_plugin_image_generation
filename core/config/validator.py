"""Configuration content validation helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from astrbot.api import logger

from ..shared.logging import log_prefix

LOG = log_prefix("Config")
# Key written by the dashboard template_list editor for every template entry.
TEMPLATE_KEY_FIELD = "__template_key"

SCHEMA_DEFAULT_FACTORIES: dict[str, Any] = {
    "int": int,
    "float": float,
    "bool": bool,
    "string": str,
    "text": str,
    "list": list,
    "file": list,
    "template_list": list,
}

MIN_NUMBER_VALUES: dict[str, int | float] = {
    "api_providers.*.timeout": 0,
    "api_providers.*.max_retry_attempts": 0,
    "api_providers.*.steps": 0,
    "api_providers.*.async_poll_interval": 1,
    "api_providers.*.async_poll_timeout": 1,
    "generation_runtime.timeout": 1,
    "generation_runtime.max_retry_attempts": 0,
    "generation_runtime.non_retryable_status_codes": 100,
    "generation_runtime.non_retryable_status_codes.*": 100,
    "generation_runtime.max_concurrent_tasks": 1,
    "generation_runtime.max_running_generation_tasks": 1,
    "generation_runtime.max_queued_generation_tasks": 1,
    "generation_task_history.generation_task_history_limit": 1,
    "generation_task_history.generation_task_history_retention_days": 0,
    "generation.default_image_count": 1,
    "generation.max_image_count": 1,
    "generation.max_images_per_message": 1,
    "user_limits.rate_limit_seconds": 0,
    "user_limits.max_image_size_mb": 1,
    "user_limits.daily_limit_count": 1,
    "user_limits.group_daily_limit_count": 1,
    "user_limits.user_daily_limit_count": 1,
    "safety_audit.prompt_audit.max_retry_attempts": 1,
    "safety_audit.image_audit.max_retry_attempts": 1,
}


class ConfigValidator:
    """Validate supported config values with schema metadata."""

    def __init__(self, schema: Mapping[str, Any] | None):
        self._schema = schema if isinstance(schema, Mapping) else {}

    def validate(self, config: dict[str, Any]) -> bool:
        """Validate config values in place and return whether anything changed."""
        if not self._schema:
            return False

        changed = self._validate_object(config, self._schema, path="")
        if changed:
            logger.info(f"{LOG} 已校验并修正不合理配置值")
        return changed

    def _validate_object(
        self,
        raw: Any,
        schema: Mapping[str, Any],
        *,
        path: str,
    ) -> bool:
        if not isinstance(raw, dict):
            return False

        changed = False
        for key, meta in schema.items():
            if key not in raw:
                continue
            value_changed, value = self._validate_value(
                raw[key],
                meta,
                path=self._join_path(path, key),
            )
            if value_changed:
                raw[key] = value
                changed = True
        return changed

    def _validate_value(
        self,
        raw: Any,
        meta: Any,
        *,
        path: str,
    ) -> tuple[bool, Any]:
        if not isinstance(meta, Mapping):
            return False, raw

        meta_type = str(meta.get("type") or "")
        if meta_type == "object":
            items = meta.get("items")
            if isinstance(raw, dict) and isinstance(items, Mapping):
                return self._validate_object(raw, items, path=path), raw
            return True, self._schema_default(meta)

        if meta_type == "template_list":
            return self._validate_template_list(raw, meta, path=path)

        return self._validate_leaf_value(raw, meta, path=path)

    def _validate_template_list(
        self,
        raw: Any,
        meta: Mapping[str, Any],
        *,
        path: str,
    ) -> tuple[bool, Any]:
        if not isinstance(raw, list):
            return True, self._schema_default(meta)

        templates = meta.get("templates")
        if not isinstance(templates, Mapping):
            return False, raw

        changed = False
        validated_items: list[Any] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                changed = True
                continue

            template_key = str(item.get(TEMPLATE_KEY_FIELD) or "").strip()
            template_meta = templates.get(template_key)
            if not isinstance(template_meta, Mapping):
                changed = True
                continue

            item_schema = template_meta.get("items")
            if not isinstance(item_schema, Mapping):
                validated_items.append(item)
                continue

            item_changed = self._validate_object(
                item,
                item_schema,
                path=f"{path}[{index}]",
            )
            changed |= item_changed
            validated_items.append(item)

        if len(validated_items) != len(raw):
            changed = True
        return changed, validated_items

    def _validate_leaf_value(
        self,
        raw: Any,
        meta: Mapping[str, Any],
        *,
        path: str,
    ) -> tuple[bool, Any]:
        default = self._schema_default(meta)
        meta_type = str(meta.get("type") or "")
        value = self._coerce_schema_value(raw, meta_type, default)
        changed = value != raw

        if meta_type == "list" and meta.get("items_type") == "int":
            value, list_changed = self._validate_int_list(value, default, path)
            changed |= list_changed
        else:
            value, options_changed = self._validate_options(value, meta, default)
            changed |= options_changed

        value, range_changed = self._validate_number_minimum(value, path, default)
        changed |= range_changed
        return changed, value

    def _coerce_schema_value(self, raw: Any, meta_type: str, default: Any) -> Any:
        if meta_type == "int":
            return self._coerce_number(raw, default, int)
        if meta_type == "float":
            return self._coerce_number(raw, default, float)
        if meta_type == "bool":
            return self._coerce_bool(raw, default)
        if meta_type in {"string", "text"}:
            return raw if isinstance(raw, str) else default if raw is None else str(raw)
        if meta_type == "file":
            return self._coerce_file_value(raw, default)
        if meta_type == "list":
            return copy.deepcopy(raw) if isinstance(raw, list) else default
        return copy.deepcopy(raw)

    def _coerce_number(self, raw: Any, default: Any, parser: Any) -> Any:
        if isinstance(raw, bool):
            return default
        try:
            return parser(raw)
        except (TypeError, ValueError):
            return default

    def _coerce_bool(self, raw: Any, default: bool) -> bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        return default

    def _coerce_file_value(self, raw: Any, default: Any) -> list[Any]:
        if isinstance(raw, list):
            return copy.deepcopy(raw)
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        return default

    def _validate_options(
        self,
        value: Any,
        meta: Mapping[str, Any],
        default: Any,
    ) -> tuple[Any, bool]:
        options = meta.get("options")
        if not isinstance(options, list):
            return value, False

        option_set = set(options)
        if isinstance(value, list):
            filtered = [item for item in value if item in option_set]
            if filtered:
                return filtered, filtered != value
            return copy.deepcopy(default), value != default

        if value in option_set:
            return value, False
        return copy.deepcopy(default), value != default

    def _validate_int_list(
        self,
        values: list[Any],
        default: Any,
        path: str,
    ) -> tuple[list[int], bool]:
        """Coerce and validate list items as integers."""
        result: list[int] = []
        changed = False
        for index, item in enumerate(values):
            if isinstance(item, bool):
                changed = True
                continue
            try:
                parsed = int(item)
            except (TypeError, ValueError):
                changed = True
                continue
            parsed, min_changed = self._validate_number_minimum(
                parsed,
                f"{path}[{index}]",
                None,
            )
            if min_changed or parsed is None:
                changed = True
                continue
            if parsed in result:
                changed = True
                continue
            result.append(parsed)
            changed |= parsed != item
        if result:
            return result, changed or result != values
        fallback = copy.deepcopy(default) if isinstance(default, list) else []
        return fallback, values != fallback

    def _validate_number_minimum(
        self,
        value: Any,
        path: str,
        default: Any,
    ) -> tuple[Any, bool]:
        if not isinstance(value, int | float) or isinstance(value, bool):
            return value, False

        min_value = self._number_minimum_for_path(path)
        if min_value is None:
            return value, False

        if value < min_value:
            return copy.deepcopy(default), value != default
        return value, False

    def _number_minimum_for_path(self, path: str) -> int | float | None:
        if path in MIN_NUMBER_VALUES:
            return MIN_NUMBER_VALUES[path]
        normalized_path = self._normalize_template_list_path(path)
        if normalized_path in MIN_NUMBER_VALUES:
            return MIN_NUMBER_VALUES[normalized_path]
        return None

    def _normalize_template_list_path(self, path: str) -> str:
        parts = path.split(".")
        normalized_parts = []
        for part in parts:
            bracket_index = part.find("[")
            if bracket_index >= 0:
                if part[:bracket_index]:
                    normalized_parts.append(part[:bracket_index])
                normalized_parts.append("*")
            else:
                normalized_parts.append(part)
        return ".".join(part for part in normalized_parts if part)

    def _schema_default(self, meta: Any) -> Any:
        if not isinstance(meta, Mapping):
            return None

        meta_type = str(meta.get("type") or "")
        if meta_type == "object":
            items = meta.get("items")
            if not isinstance(items, Mapping):
                return {}
            return {key: self._schema_default(value) for key, value in items.items()}

        if "default" in meta:
            return copy.deepcopy(meta["default"])

        default_factory = SCHEMA_DEFAULT_FACTORIES.get(meta_type)
        return default_factory() if default_factory else None

    def _join_path(self, base: str, key: str) -> str:
        return f"{base}.{key}" if base else key
