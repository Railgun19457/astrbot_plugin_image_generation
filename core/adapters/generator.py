from __future__ import annotations

from astrbot.api import logger

from ..generation.image_utils import convert_images_batch
from ..shared.logging import format_cn_log_fields, log_prefix, safe_log_text
from ..shared.types import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
    GenerationResult,
    ImageData,
)


def create_adapter(config: AdapterConfig):
    """Create one adapter instance from configuration.

    Args:
        config: Adapter configuration to instantiate.

    Returns:
        The adapter instance matching `config.type`.

    Raises:
        ValueError: If the configured adapter type is not supported.
    """
    from ...adapter import (
        AgnesAIAdapter,
        CodexResponsesAdapter,
        ComfyUIAdapter,
        CustomHTTPAdapter,
        GeminiAdapter,
        GiteeAIAdapter,
        GrokAdapter,
        ModelScopeAdapter,
        OpenAIAdapter,
        OpenAIChatAdapter,
        SiliconFlowAdapter,
        VolcengineArkAdapter,
    )

    adapter_map: dict[AdapterType, type] = {
        AdapterType.GEMINI: GeminiAdapter,
        AdapterType.OPENAI_CHAT: OpenAIChatAdapter,
        AdapterType.OPENAI: OpenAIAdapter,
        AdapterType.SILICONFLOW: SiliconFlowAdapter,
        AdapterType.VOLCENGINE_ARK: VolcengineArkAdapter,
        AdapterType.GITEE_AI: GiteeAIAdapter,
        AdapterType.AGNES_AI: AgnesAIAdapter,
        AdapterType.GROK: GrokAdapter,
        AdapterType.CODEX_RESPONSES: CodexResponsesAdapter,
        AdapterType.MODELSCOPE: ModelScopeAdapter,
        AdapterType.CUSTOM_HTTP: CustomHTTPAdapter,
        AdapterType.COMFYUI: ComfyUIAdapter,
    }

    adapter_cls = adapter_map.get(config.type)
    if not adapter_cls:
        raise ValueError(f"不支持的适配器类型: {config.type}")
    return adapter_cls(config)


class ImageGenerator:
    """Adapter orchestrator for dispatching image generation requests."""

    def __init__(self, adapter_config: AdapterConfig):
        self.adapter_config = adapter_config
        self.adapter = create_adapter(adapter_config)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images for one request."""
        if not self.adapter:
            return GenerationResult(images=None, error="适配器未初始化")

        converted_images = await self.convert_reference_images(request.images)
        return await self.generate_preconverted(request, images=converted_images)

    async def convert_reference_images(
        self, images: list[ImageData]
    ) -> list[ImageData]:
        """Convert reference images into adapter-compatible formats once."""
        if not images:
            return []
        return await convert_images_batch(images)

    def build_generation_request(
        self,
        request: GenerationRequest,
        *,
        images: list[ImageData],
    ) -> GenerationRequest:
        """Build a request with already converted reference images."""
        return GenerationRequest(
            prompt=request.prompt,
            images=images,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            task_id=request.task_id,
            batch_index=request.batch_index,
            batch_count=request.batch_count,
            retry_status_callback=request.retry_status_callback,
            gif_requested=request.gif_requested,
        )

    async def generate_preconverted(
        self,
        request: GenerationRequest,
        *,
        images: list[ImageData],
    ) -> GenerationResult:
        """Generate an image request with preconverted reference images."""
        if not self.adapter:
            return GenerationResult(images=None, error="适配器未初始化")

        patched_request = self.build_generation_request(request, images=images)
        logger.debug(
            f"{log_prefix('Generator', request.task_id)} 分发生图请求: "
            + format_cn_log_fields(
                适配器=self.adapter.__class__.__name__,
                模型=self.adapter.model,
                进度=f"{request.batch_index}/{request.batch_count}",
                参考图=f"{len(images)}张",
                宽高比=request.aspect_ratio,
                分辨率=request.resolution,
            )
        )

        try:
            return await self.adapter.generate(patched_request)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"{log_prefix('Generator', request.task_id)} 生成失败: {safe_log_text(exc, 200)}",
                exc_info=True,
            )
            return GenerationResult(images=None, error=str(exc))

    async def update_adapter(self, adapter_config: AdapterConfig) -> None:
        """Update adapter configuration and recreate the adapter.

        The previous adapter is closed to release resources.
        """
        if self.adapter:
            await self.adapter.close()
        self.adapter_config = adapter_config
        self.adapter = create_adapter(adapter_config)

    async def close(self) -> None:
        """Close the active adapter."""
        if self.adapter:
            await self.adapter.close()
