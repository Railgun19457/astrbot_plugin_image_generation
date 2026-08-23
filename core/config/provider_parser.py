"""Provider configuration parsing helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from astrbot.api import logger

from ..shared.constants import (
    DEFAULT_MAX_RETRY_ATTEMPTS,
    DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS,
    DEFAULT_NON_RETRYABLE_STATUS_CODES,
    DEFAULT_TIMEOUT,
)
from ..shared.logging import log_prefix, safe_log_text
from ..shared.types import AdapterConfig, AdapterType

PROVIDER_COMMON_FIELDS = frozenset(
    {
        "__template_key",
        "name",
        "base_url",
        "proxy",
        "api_keys",
        "available_models",
        "capability_options",
        "timeout",
        "max_retry_attempts",
    }
)

ADAPTER_EXTRA_DEFAULTS: dict[AdapterType, dict[str, Any]] = {
    AdapterType.OPENAI_CHAT: {
        "prompt_prefix": "Generate an image: ",
        "modalities": ["image", "text"],
    },
    AdapterType.OPENAI: {
        "model_family": "auto",
        "enable_streaming": True,
        "output_format": "png",
    },
    AdapterType.AGNES_AI: {"response_format": "base64"},
}
LOG = log_prefix("Config")


class ConfigProviderParserMixin:
    """Mixin for parsing image provider configuration."""

    def _load_provider_configs(
        self, raw_providers: Any, gen_cfg: dict[str, Any]
    ) -> list[AdapterConfig]:
        """Parse all provider templates into normalized adapter configs."""
        if not isinstance(raw_providers, list):
            logger.warning(f"{LOG} api_providers 配置格式错误，已按空列表处理")
            return []

        provider_configs: list[AdapterConfig] = []
        for provider_item in raw_providers:
            if not isinstance(provider_item, dict):
                continue
            if parsed := self._parse_provider_config(provider_item, gen_cfg):
                provider_configs.append(parsed)
        return provider_configs

    def _parse_provider_config(
        self,
        provider_item: dict[str, Any],
        gen_cfg: dict[str, Any],
    ) -> AdapterConfig | None:
        """Parse one provider item with global fallback and provider overrides."""
        adapter_type = self._parse_adapter_type(provider_item)
        if not adapter_type:
            return None

        base_url = str(provider_item.get("base_url") or "").strip()
        proxy = str(provider_item.get("proxy") or "").strip() or None

        return AdapterConfig(
            type=adapter_type,
            name=str(provider_item.get("name", "")).strip(),
            base_url=self._clean_base_url(
                base_url,
                preserve_version_path=adapter_type == AdapterType.CUSTOM_HTTP,
            ),
            api_keys=self._parse_string_list(provider_item.get("api_keys", [])),
            available_models=self._parse_string_list(
                provider_item.get("available_models", [])
            ),
            proxy=proxy,
            timeout=self._get_provider_int_override(
                provider_item,
                gen_cfg,
                "timeout",
                DEFAULT_TIMEOUT,
                min_value=1,
            ),
            max_retry_attempts=self._get_provider_int_override(
                provider_item,
                gen_cfg,
                "max_retry_attempts",
                DEFAULT_MAX_RETRY_ATTEMPTS,
                min_value=0,
            ),
            debug_request_logging=self._get_bool(
                gen_cfg, "debug_request_logging", False
            ),
            show_user_error_details=self._get_bool(
                gen_cfg, "show_user_error_details", False
            ),
            non_retryable_status_codes=self._parse_int_list(
                gen_cfg.get(
                    "non_retryable_status_codes",
                    list(DEFAULT_NON_RETRYABLE_STATUS_CODES),
                ),
                list(DEFAULT_NON_RETRYABLE_STATUS_CODES),
            ),
            non_retryable_error_keywords=self._parse_string_list_config(
                gen_cfg.get(
                    "non_retryable_error_keywords",
                    list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS),
                ),
                list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS),
            ),
            capability_options=self._parse_capability_options(provider_item),
            extra=self._parse_provider_extra(adapter_type, provider_item),
        )

    def _parse_adapter_type(self, provider_item: dict[str, Any]) -> AdapterType | None:
        """Parse and validate the provider template key."""
        adapter_type_str = str(provider_item.get("__template_key") or "").strip()
        if not adapter_type_str:
            return None

        try:
            return AdapterType(adapter_type_str)
        except ValueError:
            logger.warning(
                f"{LOG} 忽略未知适配器类型: {safe_log_text(adapter_type_str)}"
            )
            return None

    def _get_provider_int_override(
        self,
        provider_item: dict[str, Any],
        gen_cfg: dict[str, Any],
        key: str,
        default: int,
        *,
        min_value: int,
    ) -> int:
        """Resolve an integer provider override, using global config by default."""
        global_value = self._coerce_int(
            gen_cfg.get(key, default), default, min_value=min_value
        )
        if key not in provider_item:
            return global_value

        raw_value = provider_item.get(key)
        if raw_value in (None, ""):
            return global_value

        provider_value = self._coerce_int(raw_value, global_value, min_value=0)
        if provider_value <= 0:
            return global_value
        return max(min_value, provider_value)

    def _parse_provider_extra(
        self,
        adapter_type: AdapterType,
        provider_item: dict[str, Any],
    ) -> dict[str, Any]:
        """Collect adapter-specific settings without changing parser code later."""
        extra = dict(ADAPTER_EXTRA_DEFAULTS.get(adapter_type, {}))

        for key in extra:
            if key in provider_item:
                extra[key] = self._normalize_extra_value(provider_item[key])

        for key, value in provider_item.items():
            if key in PROVIDER_COMMON_FIELDS or key.startswith("__"):
                continue
            extra.setdefault(key, self._normalize_extra_value(value))

        return extra

    def _normalize_extra_value(self, value: Any) -> Any:
        """Normalize adapter-specific values before storing them in extra."""
        if isinstance(value, str):
            return value.strip()
        return value

    def _select_adapter_config(
        self, provider_configs: list[AdapterConfig], model_setting: str
    ) -> AdapterConfig | None:
        """Select active provider config and attach full model choices."""
        matched_config: AdapterConfig | None = None
        current_model = ""

        if "/" in model_setting:
            target_provider_name, target_model = model_setting.split("/", 1)
            for cfg in provider_configs:
                if cfg.name == target_provider_name:
                    matched_config = cfg
                    current_model = target_model
                    break

        if not matched_config and provider_configs:
            matched_config = provider_configs[0]
            current_model = (
                matched_config.available_models[0]
                if matched_config.available_models
                else ""
            )
            logger.info(
                f"{LOG} 未匹配到当前模型配置，默认使用: {safe_log_text(matched_config.name)}/{safe_log_text(current_model)}"
            )

        if not matched_config:
            logger.error(f"{LOG} 未找到任何有效的生图模型配置")
            return None

        return replace(
            matched_config,
            model=current_model,
            available_models=self._build_model_choices(provider_configs),
        )

    def _build_model_choices(self, provider_configs: list[AdapterConfig]) -> list[str]:
        """Build display model choices in provider/model format."""
        choices: list[str] = []
        for cfg in provider_configs:
            choices.extend(f"{cfg.name}/{model}" for model in cfg.available_models)
        return choices

    def _parse_capability_options(
        self, provider_item: dict[str, Any]
    ) -> dict[str, bool]:
        """Parse provider capability options from configuration."""
        raw = provider_item.get("capability_options", [])

        supported_keys = (
            "text_to_image",
            "image_to_image",
            "aspect_ratio",
            "resolution",
        )

        if not isinstance(raw, list):
            logger.warning(f"{LOG} capability_options 配置格式错误，已按空列表处理")
            raw = []

        capability_alias_map = {
            "文生图": "text_to_image",
            "图生图": "image_to_image",
            "宽高比": "aspect_ratio",
            "分辨率": "resolution",
            "text_to_image": "text_to_image",
            "image_to_image": "image_to_image",
            "aspect_ratio": "aspect_ratio",
            "resolution": "resolution",
        }

        selected: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            key = capability_alias_map.get(item.strip())
            if key:
                selected.add(key)

        return {key: key in selected for key in supported_keys}

    def _clean_base_url(self, url: str, *, preserve_version_path: bool = False) -> str:
        """Clean a base URL and optionally remove trailing /v1 paths."""
        if not url:
            return ""
        url = url.rstrip("/")
        if preserve_version_path:
            return url
        if "/v1" in url:
            url = url.split("/v1", 1)[0]
        return url.rstrip("/")


__all__ = (
    "ADAPTER_EXTRA_DEFAULTS",
    "PROVIDER_COMMON_FIELDS",
    "ConfigProviderParserMixin",
)
