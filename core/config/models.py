"""Configuration data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..shared.constants import (
    ALL_LLM_HANDLED_COMMANDS,
    ALL_LLM_TOOLS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS,
    DEFAULT_DAILY_LIMIT_COUNT,
    DEFAULT_ENABLE_GENERATION_TASK_HISTORY,
    DEFAULT_GENERATION_IMAGE_COUNT,
    DEFAULT_GENERATION_TASK_HISTORY_LIMIT,
    DEFAULT_GENERATION_TASK_HISTORY_RETENTION_DAYS,
    DEFAULT_IMAGE_AUDIT_PROMPT,
    DEFAULT_MAX_CONCURRENT_TASKS,
    DEFAULT_MAX_GENERATION_IMAGE_COUNT,
    DEFAULT_MAX_IMAGE_SIZE_MB,
    DEFAULT_MAX_IMAGES_PER_MESSAGE,
    DEFAULT_MAX_QUEUED_GENERATION_TASKS,
    DEFAULT_MAX_RUNNING_GENERATION_TASKS,
    DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS,
    DEFAULT_NON_RETRYABLE_STATUS_CODES,
    DEFAULT_PROMPT_AUDIT_PROMPT,
    DEFAULT_RATE_LIMIT_SECONDS,
    DEFAULT_RESOLUTION,
    DEFAULT_RESULT_INFO_ITEMS,
)
from ..shared.types import AdapterConfig


@dataclass
class UsageSettings:
    """User usage limit settings."""

    rate_limit_seconds: int = DEFAULT_RATE_LIMIT_SECONDS
    enable_daily_limit: bool = False
    daily_limit_count: int = DEFAULT_DAILY_LIMIT_COUNT
    max_image_size_mb: int = DEFAULT_MAX_IMAGE_SIZE_MB
    umo_blacklist: list[str] = field(default_factory=list)
    admin_bypass_limits: bool = True
    umo_whitelist: list[str] = field(default_factory=list)
    blacklist_block_message: str = "❌ 当前会话已被加入黑名单，无法使用生图功能"


@dataclass
class GenerationSettings:
    """Image generation behavior settings."""

    default_aspect_ratio: str = DEFAULT_ASPECT_RATIO
    default_resolution: str = DEFAULT_RESOLUTION
    default_image_count: int = DEFAULT_GENERATION_IMAGE_COUNT
    max_image_count: int = DEFAULT_MAX_GENERATION_IMAGE_COUNT
    max_images_per_message: int = DEFAULT_MAX_IMAGES_PER_MESSAGE
    max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS
    max_running_generation_tasks: int = DEFAULT_MAX_RUNNING_GENERATION_TASKS
    max_queued_generation_tasks: int = DEFAULT_MAX_QUEUED_GENERATION_TASKS
    enable_generation_task_history: bool = DEFAULT_ENABLE_GENERATION_TASK_HISTORY
    generation_task_history_limit: int = DEFAULT_GENERATION_TASK_HISTORY_LIMIT
    generation_task_history_retention_days: int = (
        DEFAULT_GENERATION_TASK_HISTORY_RETENTION_DAYS
    )
    debug_request_logging: bool = False
    show_user_error_details: bool = False
    non_retryable_status_codes: list[int] = field(
        default_factory=lambda: list(DEFAULT_NON_RETRYABLE_STATUS_CODES)
    )
    non_retryable_error_keywords: list[str] = field(
        default_factory=lambda: list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS)
    )
    result_info_items: set[str] = field(
        default_factory=lambda: set(DEFAULT_RESULT_INFO_ITEMS)
    )
    start_task_message_template: str = "已开始生图任务{reference_images_block}{preset_block}{persona_block}{image_count_block} [任务ID: {task_id}]"


@dataclass
class PersonaTemplate:
    """Image generation persona template."""

    name: str
    prompt: str
    image: str = ""


@dataclass
class PromptAuditSettings:
    """Pre-generation prompt audit settings."""

    blocked_words: list[str] = field(default_factory=list)
    enable_ai_audit: bool = False
    ai_provider_id: str = ""
    max_retry_attempts: int = DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS
    ai_prompt: str = DEFAULT_PROMPT_AUDIT_PROMPT


@dataclass
class ImageAuditSettings:
    """Post-generation image audit settings."""

    enable_ai_audit: bool = False
    ai_provider_id: str = ""
    max_retry_attempts: int = DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS
    ai_prompt: str = DEFAULT_IMAGE_AUDIT_PROMPT


@dataclass
class SafetyAuditSettings:
    """Safety audit settings."""

    umo_whitelist: list[str] = field(default_factory=list)
    prompt_audit: PromptAuditSettings = field(default_factory=PromptAuditSettings)
    image_audit: ImageAuditSettings = field(default_factory=ImageAuditSettings)


@dataclass
class PluginConfig:
    """Complete plugin configuration."""

    adapter_config: AdapterConfig | None = None
    usage_settings: UsageSettings = field(default_factory=UsageSettings)
    generation_settings: GenerationSettings = field(default_factory=GenerationSettings)
    safety_audit_settings: SafetyAuditSettings = field(
        default_factory=SafetyAuditSettings
    )
    presets: dict[str, Any] = field(default_factory=dict)
    personas: dict[str, PersonaTemplate] = field(default_factory=dict)
    enabled_llm_tools: set[str] = field(default_factory=lambda: set(ALL_LLM_TOOLS))
    llm_handles_commands: bool = False
    llm_handled_commands: set[str] = field(
        default_factory=lambda: set(ALL_LLM_HANDLED_COMMANDS)
    )


__all__ = (
    "GenerationSettings",
    "ImageAuditSettings",
    "PersonaTemplate",
    "PluginConfig",
    "PromptAuditSettings",
    "SafetyAuditSettings",
    "UsageSettings",
)
