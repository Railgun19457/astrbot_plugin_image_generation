"""Plugin configuration management module."""

from __future__ import annotations

from typing import Any

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig

from .validator import ConfigValidator
from .models import (
    GenerationSettings,
    ImageAuditSettings,
    PersonaTemplate,
    PluginConfig,
    PromptAuditSettings,
    SafetyAuditSettings,
    UsageSettings,
)
from .provider_parser import ConfigProviderParserMixin
from .templates import ConfigTemplateStoreMixin
from ..shared.constants import (
    ALL_LLM_HANDLED_COMMANDS,
    ALL_LLM_TOOLS,
    ALL_RESULT_INFO_ITEMS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS,
    DEFAULT_DAILY_LIMIT_COUNT,
    DEFAULT_ENABLE_GENERATION_TASK_HISTORY,
    DEFAULT_GENERATION_IMAGE_COUNT,
    DEFAULT_GENERATION_TASK_HISTORY_LIMIT,
    DEFAULT_GENERATION_TASK_HISTORY_RETENTION_DAYS,
    DEFAULT_MAX_CONCURRENT_TASKS,
    DEFAULT_MAX_GENERATION_IMAGE_COUNT,
    DEFAULT_MAX_IMAGE_SIZE_MB,
    DEFAULT_MAX_IMAGES_PER_MESSAGE,
    DEFAULT_MAX_QUEUED_GENERATION_TASKS,
    DEFAULT_MAX_RUNNING_GENERATION_TASKS,
    DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS,
    DEFAULT_NON_RETRYABLE_STATUS_CODES,
    DEFAULT_RATE_LIMIT_SECONDS,
    DEFAULT_RESOLUTION,
    DEFAULT_RESULT_INFO_ITEMS,
    LLM_TOOL_IMAGE_GENERATION,
    LLM_TOOL_PRESET_EDIT,
    LLM_TOOL_PRESET_QUERY,
    LLM_TOOL_TASK_MANAGEMENT,
    RESULT_INFO_COUNT,
    RESULT_INFO_DURATION,
    RESULT_INFO_MODEL,
    RESULT_INFO_TASK_ID,
    RESULT_INFO_USAGE,
)
from ..shared.logging import log_prefix, safe_log_text
from ..shared.types import AdapterConfig, AdapterType

__all__ = (
    "ConfigManager",
    "GenerationSettings",
    "ImageAuditSettings",
    "LLM_TOOL_IMAGE_GENERATION",
    "LLM_TOOL_PRESET_EDIT",
    "LLM_TOOL_PRESET_QUERY",
    "LLM_TOOL_TASK_MANAGEMENT",
    "PersonaTemplate",
    "PluginConfig",
    "PromptAuditSettings",
    "RESULT_INFO_COUNT",
    "RESULT_INFO_DURATION",
    "RESULT_INFO_MODEL",
    "RESULT_INFO_TASK_ID",
    "RESULT_INFO_USAGE",
    "SafetyAuditSettings",
    "UsageSettings",
)


LOG = log_prefix("Config")


class ConfigManager(ConfigProviderParserMixin, ConfigTemplateStoreMixin):
    """Plugin configuration manager."""

    def __init__(self, config: AstrBotConfig):
        self._config = config
        self._config_validator = ConfigValidator(getattr(config, "schema", None))
        self._plugin_config: PluginConfig = PluginConfig()
        self._all_provider_configs: list[
            AdapterConfig
        ] = []  # Store all provider configs.
        self.load()

    def load(self) -> PluginConfig:
        """Load and parse plugin configuration."""
        self._validate_config_values()

        gen_cfg = self._get_config_section("generation")
        generation_runtime_cfg = self._get_config_section("generation_runtime")
        user_limits_cfg = self._get_config_section("user_limits")
        safety_cfg = self._get_config_section("safety_audit")
        prompt_templates_cfg = self._get_config_section("prompt_templates")
        generation_task_history_cfg = self._get_config_section(
            "generation_task_history"
        )
        api_providers_raw = self._config.get("api_providers", [])

        fallback_runtime_cfg = (
            gen_cfg if not generation_runtime_cfg else generation_runtime_cfg
        )
        all_provider_configs = self._load_provider_configs(
            api_providers_raw,
            fallback_runtime_cfg,
        )
        self._all_provider_configs = all_provider_configs

        self._plugin_config = PluginConfig(
            adapter_config=self._select_adapter_config(
                all_provider_configs,
                self._get_str(gen_cfg, "model", ""),
            ),
            usage_settings=self._parse_usage_settings(user_limits_cfg),
            generation_settings=self._parse_generation_settings(
                gen_cfg,
                generation_runtime_cfg,
                generation_task_history_cfg,
            ),
            safety_audit_settings=self._parse_safety_audit_settings(safety_cfg),
            presets=self._load_presets(prompt_templates_cfg.get("presets", [])),
            personas=self._load_personas(prompt_templates_cfg.get("personas", [])),
            enabled_llm_tools=set(
                self._parse_enabled_llm_tools(
                    self._config.get("enable_llm_tool", list(ALL_LLM_TOOLS))
                )
            ),
            llm_handles_commands=self._get_bool(
                self._config, "llm_handles_commands", False
            ),
            llm_handled_commands=set(
                self._parse_llm_handled_commands(
                    self._config.get(
                        "llm_handled_commands", list(ALL_LLM_HANDLED_COMMANDS)
                    )
                )
            ),
        )

        return self._plugin_config

    def _parse_usage_settings(self, cfg: dict[str, Any]) -> UsageSettings:
        """Parse user limit settings from normalized config."""
        return UsageSettings(
            rate_limit_seconds=self._get_int(
                cfg,
                "rate_limit_seconds",
                DEFAULT_RATE_LIMIT_SECONDS,
                min_value=0,
            ),
            enable_daily_limit=self._get_bool(cfg, "enable_daily_limit", False),
            daily_limit_count=self._get_int(
                cfg,
                "daily_limit_count",
                DEFAULT_DAILY_LIMIT_COUNT,
                min_value=1,
            ),
            max_image_size_mb=self._get_int(
                cfg,
                "max_image_size_mb",
                DEFAULT_MAX_IMAGE_SIZE_MB,
                min_value=1,
            ),
            umo_blacklist=self._parse_string_list(cfg.get("umo_blacklist", [])),
            admin_bypass_limits=self._get_bool(cfg, "admin_bypass_limits", True),
            umo_whitelist=self._parse_string_list(cfg.get("umo_whitelist", [])),
            blacklist_block_message=self._get_str(
                cfg,
                "blacklist_block_message",
                UsageSettings.blacklist_block_message,
            ),
        )

    def _parse_generation_settings(
        self,
        cfg: dict[str, Any],
        runtime_cfg: dict[str, Any],
        history_cfg: dict[str, Any],
    ) -> GenerationSettings:
        """Parse image generation behavior settings."""
        # Prefer the standalone runtime and task-history sections, but keep
        # reading old generation.* values when users upgrade from older versions.
        fallback_runtime_cfg = cfg if not runtime_cfg else runtime_cfg
        fallback_history_cfg = cfg if not history_cfg else history_cfg
        return GenerationSettings(
            default_aspect_ratio=self._get_str(
                cfg,
                "default_aspect_ratio",
                DEFAULT_ASPECT_RATIO,
            ),
            default_resolution=self._get_str(
                cfg,
                "default_resolution",
                DEFAULT_RESOLUTION,
            ),
            default_image_count=self._get_int(
                cfg,
                "default_image_count",
                DEFAULT_GENERATION_IMAGE_COUNT,
                min_value=1,
            ),
            max_image_count=self._get_int(
                cfg,
                "max_image_count",
                DEFAULT_MAX_GENERATION_IMAGE_COUNT,
                min_value=1,
            ),
            max_images_per_message=self._get_int(
                cfg,
                "max_images_per_message",
                DEFAULT_MAX_IMAGES_PER_MESSAGE,
                min_value=1,
            ),
            max_concurrent_tasks=self._get_int(
                fallback_runtime_cfg,
                "max_concurrent_tasks",
                DEFAULT_MAX_CONCURRENT_TASKS,
                min_value=1,
            ),
            max_running_generation_tasks=self._get_int(
                fallback_runtime_cfg,
                "max_running_generation_tasks",
                DEFAULT_MAX_RUNNING_GENERATION_TASKS,
                min_value=1,
            ),
            max_queued_generation_tasks=self._get_int(
                fallback_runtime_cfg,
                "max_queued_generation_tasks",
                DEFAULT_MAX_QUEUED_GENERATION_TASKS,
                min_value=1,
            ),
            enable_generation_task_history=self._get_bool(
                fallback_history_cfg,
                "enable_generation_task_history",
                DEFAULT_ENABLE_GENERATION_TASK_HISTORY,
            ),
            generation_task_history_limit=self._get_int(
                fallback_history_cfg,
                "generation_task_history_limit",
                DEFAULT_GENERATION_TASK_HISTORY_LIMIT,
                min_value=1,
            ),
            generation_task_history_retention_days=self._get_int(
                fallback_history_cfg,
                "generation_task_history_retention_days",
                DEFAULT_GENERATION_TASK_HISTORY_RETENTION_DAYS,
                min_value=0,
            ),
            debug_request_logging=self._get_bool(
                fallback_runtime_cfg,
                "debug_request_logging",
                False,
            ),
            show_user_error_details=self._get_bool(
                fallback_runtime_cfg,
                "show_user_error_details",
                False,
            ),
            non_retryable_status_codes=self._parse_int_list(
                fallback_runtime_cfg.get(
                    "non_retryable_status_codes",
                    list(DEFAULT_NON_RETRYABLE_STATUS_CODES),
                ),
                list(DEFAULT_NON_RETRYABLE_STATUS_CODES),
            ),
            non_retryable_error_keywords=self._parse_string_list_config(
                fallback_runtime_cfg.get(
                    "non_retryable_error_keywords",
                    list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS),
                ),
                list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS),
            ),
            result_info_items=self._parse_result_info_items(cfg),
            start_task_message_template=self._get_str(
                cfg,
                "start_task_message_template",
                GenerationSettings.start_task_message_template,
            ),
        )

    def _parse_result_info_items(self, cfg: dict[str, Any]) -> set[str]:
        """Parse selected result information items."""
        selected = self._parse_string_list(
            cfg.get("result_info_items", list(DEFAULT_RESULT_INFO_ITEMS))
        )
        valid_items = set(ALL_RESULT_INFO_ITEMS)
        return {item for item in selected if item in valid_items}

    def _parse_safety_audit_settings(self, cfg: dict[str, Any]) -> SafetyAuditSettings:
        """Parse prompt and image audit settings."""
        prompt_audit_cfg = self._get_nested_section(cfg, "prompt_audit")
        image_audit_cfg = self._get_nested_section(cfg, "image_audit")

        return SafetyAuditSettings(
            umo_whitelist=self._parse_string_list(cfg.get("umo_whitelist", [])),
            prompt_audit=self._parse_prompt_audit_settings(prompt_audit_cfg),
            image_audit=self._parse_image_audit_settings(image_audit_cfg),
        )

    def _parse_prompt_audit_settings(self, cfg: dict[str, Any]) -> PromptAuditSettings:
        """Parse prompt audit settings."""
        return PromptAuditSettings(
            blocked_words=self._parse_string_list(cfg.get("blocked_words", [])),
            enable_ai_audit=self._get_bool(cfg, "enable_ai_audit", False),
            ai_provider_id=self._get_str(cfg, "ai_provider_id", ""),
            max_retry_attempts=self._get_int(
                cfg,
                "max_retry_attempts",
                DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS,
                min_value=1,
            ),
            ai_prompt=self._get_str(cfg, "ai_prompt", PromptAuditSettings.ai_prompt),
        )

    def _parse_image_audit_settings(self, cfg: dict[str, Any]) -> ImageAuditSettings:
        """Parse image audit settings."""
        return ImageAuditSettings(
            enable_ai_audit=self._get_bool(cfg, "enable_ai_audit", False),
            ai_provider_id=self._get_str(cfg, "ai_provider_id", ""),
            max_retry_attempts=self._get_int(
                cfg,
                "max_retry_attempts",
                DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS,
                min_value=1,
            ),
            ai_prompt=self._get_str(cfg, "ai_prompt", ImageAuditSettings.ai_prompt),
        )

    def reload(self) -> PluginConfig:
        """Reload plugin configuration."""
        return self.load()

    def _validate_config_values(self) -> None:
        """Validate config values and persist corrected values."""
        changed = self._config_validator.validate(self._config)
        if not changed:
            return
        self._config.save_config()

    def _get_config_section(self, name: str) -> dict[str, Any]:
        """Return a dictionary config section, falling back to an empty dict."""
        value = self._config.get(name, {})
        if isinstance(value, dict):
            return value
        logger.warning(f"{LOG} 配置项 {safe_log_text(name)} 格式错误，已按空对象处理")
        return {}

    def _get_nested_section(self, cfg: dict[str, Any], key: str) -> dict[str, Any]:
        """Return a nested dictionary section, falling back to an empty dict."""
        value = cfg.get(key, {})
        if isinstance(value, dict):
            return value
        logger.warning(f"{LOG} 配置项 {key} 格式错误，已按空对象处理")
        return {}

    def _get_str(
        self,
        cfg: dict[str, Any],
        key: str,
        default: str,
        *,
        strip: bool = True,
    ) -> str:
        """Read a config value as string."""
        value = cfg.get(key, default)
        if value is None:
            value = default
        parsed = str(value)
        return parsed.strip() if strip else parsed

    def _get_bool(self, cfg: dict[str, Any], key: str, default: bool) -> bool:
        """Read a config value as bool without treating arbitrary strings as true."""
        value = cfg.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        return default

    def _get_int(
        self,
        cfg: dict[str, Any],
        key: str,
        default: int,
        *,
        min_value: int,
    ) -> int:
        """Read a config value as int and clamp it."""
        return self._coerce_int(cfg.get(key, default), default, min_value=min_value)

    def _parse_enabled_llm_tools(self, raw: Any) -> list[str]:
        """Parse enabled LLM tool names from list config."""
        if isinstance(raw, bool):
            return list(ALL_LLM_TOOLS) if raw else []

        if not isinstance(raw, list):
            logger.warning(f"{LOG} enable_llm_tool 配置格式错误，已按空列表处理")
            return []

        selected: list[str] = []
        for item in raw:
            tool_name = str(item).strip()
            if tool_name in ALL_LLM_TOOLS and tool_name not in selected:
                selected.append(tool_name)
        return selected

    def _parse_llm_handled_commands(self, raw: Any) -> list[str]:
        """Parse command labels that may be handed off to the session LLM."""
        if isinstance(raw, bool):
            return list(ALL_LLM_HANDLED_COMMANDS) if raw else []

        if not isinstance(raw, list):
            logger.warning(
                f"{LOG} llm_handled_commands 配置格式错误，已按默认列表处理"
            )
            return list(ALL_LLM_HANDLED_COMMANDS)

        selected: list[str] = []
        for item in raw:
            command_name = str(item).strip()
            if (
                command_name in ALL_LLM_HANDLED_COMMANDS
                and command_name not in selected
            ):
                selected.append(command_name)
        return selected

    def _coerce_int(self, value: Any, default: int, *, min_value: int) -> int:
        """Safely coerce a value to int and clamp it."""
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, parsed)

    def _parse_int_list(self, raw: Any, default: list[int]) -> list[int]:
        """Parse a list-like config value into unique integers."""
        if not isinstance(raw, list):
            return list(default)

        result: list[int] = []
        for item in raw:
            if isinstance(item, bool):
                continue
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value not in result:
                result.append(value)
        return result

    def _parse_string_list(self, raw: Any) -> list[str]:
        """Parse a list-like config value into non-empty strings."""
        if not isinstance(raw, list):
            return []
        return [item for item in (str(v).strip() for v in raw) if item]

    def _parse_string_list_config(self, raw: Any, default: list[str]) -> list[str]:
        """Parse a string list config value while preserving explicit empty lists."""
        if not isinstance(raw, list):
            return list(default)
        return self._parse_string_list(raw)

    def save_model_setting(self, model: str) -> None:
        """Save the active model setting."""
        self._config.setdefault("generation", {})["model"] = model
        self._config.save_config()

    # Convenience property accessors.
    @property
    def adapter_config(self) -> AdapterConfig | None:
        """Return the active adapter configuration."""
        return self._plugin_config.adapter_config

    @property
    def presets(self) -> dict[str, Any]:
        """Return configured prompt presets."""
        return self._plugin_config.presets

    @property
    def personas(self) -> dict[str, PersonaTemplate]:
        """Return configured persona templates."""
        return self._plugin_config.personas

    def is_llm_tool_enabled(self, tool_name: str) -> bool:
        """Return whether an LLM tool is enabled."""
        return tool_name in self._plugin_config.enabled_llm_tools

    def should_llm_handle_command(self, command_name: str) -> bool:
        """Return whether a slash command should be handed off to the session LLM."""
        if not self._plugin_config.llm_handles_commands:
            return False
        if not self.is_llm_tool_enabled(LLM_TOOL_IMAGE_GENERATION):
            return False
        return command_name in self._plugin_config.llm_handled_commands

    @property
    def default_aspect_ratio(self) -> str:
        """Return the default aspect ratio."""
        return self._plugin_config.generation_settings.default_aspect_ratio

    @property
    def default_resolution(self) -> str:
        """Return the default resolution."""
        return self._plugin_config.generation_settings.default_resolution

    @property
    def default_image_count(self) -> int:
        """Return the default image count per request."""
        return min(
            self._plugin_config.generation_settings.default_image_count,
            self.max_image_count,
        )

    @property
    def max_image_count(self) -> int:
        """Return the maximum image count per request."""
        return self._plugin_config.generation_settings.max_image_count

    @property
    def max_images_per_message(self) -> int:
        """Return the maximum number of images sent per message."""
        return self._plugin_config.generation_settings.max_images_per_message

    @property
    def max_concurrent_tasks(self) -> int:
        """Return the maximum concurrent adapter requests."""
        return self._plugin_config.generation_settings.max_concurrent_tasks

    @property
    def max_running_generation_tasks(self) -> int:
        """Return the maximum number of concurrently running generation tasks."""
        return self._plugin_config.generation_settings.max_running_generation_tasks

    @property
    def max_queued_generation_tasks(self) -> int:
        """Return the maximum number of queued generation tasks."""
        return self._plugin_config.generation_settings.max_queued_generation_tasks

    @property
    def enable_generation_task_history(self) -> bool:
        """Return whether generation task history persistence is enabled."""
        return self._plugin_config.generation_settings.enable_generation_task_history

    @property
    def generation_task_history_limit(self) -> int:
        """Return the generation task history item limit."""
        return self._plugin_config.generation_settings.generation_task_history_limit

    @property
    def generation_task_history_retention_days(self) -> int:
        """Return generation task history retention days."""
        return self._plugin_config.generation_settings.generation_task_history_retention_days

    @property
    def result_info_items(self) -> set[str]:
        """Return result metadata items shown after successful generation."""
        return self._plugin_config.generation_settings.result_info_items

    def should_show_result_info(self, item: str) -> bool:
        """Return whether one result metadata item is enabled."""
        return item in self.result_info_items

    @property
    def start_task_message_template(self) -> str:
        """Return the start-task message template."""
        return self._plugin_config.generation_settings.start_task_message_template

    @property
    def show_user_error_details(self) -> bool:
        """Return whether detailed user-facing errors are enabled."""
        return self._plugin_config.generation_settings.show_user_error_details

    @property
    def usage_settings(self) -> UsageSettings:
        """Return usage limit settings."""
        return self._plugin_config.usage_settings

    @property
    def safety_audit_settings(self) -> SafetyAuditSettings:
        """Return safety audit settings."""
        return self._plugin_config.safety_audit_settings

    # Provider lookup helpers.
    def get_provider_config(self, adapter_type: AdapterType) -> AdapterConfig | None:
        """Return provider configuration for one adapter type.

        Args:
            adapter_type: Adapter type to look up.

        Returns:
            Matching provider config, or None when unavailable.
        """
        for cfg in self._all_provider_configs:
            if cfg.type == adapter_type:
                return cfg
        return None
