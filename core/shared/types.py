from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class AdapterType(str, enum.Enum):
    """Supported image generation adapter types."""

    GEMINI = "gemini"
    OPENAI_CHAT = "openai_chat"
    OPENAI = "openai"
    SILICONFLOW = "siliconflow_adapter"
    VOLCENGINE_ARK = "volcengine_ark"
    GITEE_AI = "gitee_ai"
    AGNES_AI = "agnes_ai"
    GROK = "grok"
    CODEX_RESPONSES = "codex_responses"
    MODELSCOPE = "modelscope"
    CUSTOM_HTTP = "custom_http"
    COMFYUI = "comfyui"


class ImageCapability(enum.Flag):
    """Capabilities supported by image generation adapters."""

    NONE = 0
    TEXT_TO_IMAGE = enum.auto()  # Text-to-image generation.
    IMAGE_TO_IMAGE = enum.auto()  # Image-to-image generation.
    RESOLUTION = enum.auto()  # Explicit resolution support.
    ASPECT_RATIO = enum.auto()  # Explicit aspect-ratio support.


@dataclass
class AdapterConfig:
    """Configuration required to construct an adapter."""

    type: AdapterType = AdapterType.GEMINI
    name: str = ""  # Provider display name.
    base_url: str | None = None
    api_keys: list[str] = field(default_factory=list)
    model: str = ""
    available_models: list[str] = field(default_factory=list)
    proxy: str | None = None
    timeout: int = 180
    max_retry_attempts: int = 3
    debug_request_logging: bool = False
    show_user_error_details: bool = False
    non_retryable_status_codes: list[int] = field(default_factory=list)
    non_retryable_error_keywords: list[str] = field(default_factory=list)
    safety_settings: str | None = None
    capability_options: dict[str, bool] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # Adapter-specific config.


@dataclass
class ImageData:
    """Image bytes with MIME type and optional source URL."""

    data: bytes
    mime_type: str
    source_url: str | None = None


@dataclass
class GenerationRequest:
    """User image generation request."""

    prompt: str
    images: list[ImageData] = field(default_factory=list)
    aspect_ratio: str | None = None
    resolution: str | None = None
    task_id: str | None = None
    batch_index: int = 1
    batch_count: int = 1
    retry_status_callback: Callable[[int, int], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    gif_requested: bool = False


@dataclass
class GenerationResult:
    """Result of one image generation attempt."""

    images: list[bytes] | None = None
    error: str | None = None
    auxiliary_images: list[bytes] = field(default_factory=list)
