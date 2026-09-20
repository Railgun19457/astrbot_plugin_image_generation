from __future__ import annotations

import base64
import time
from typing import Any

import aiohttp

from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import (
    GITEE_AI_DEFAULT_BASE_URL,
    RESOLUTION_1K_MAP,
    RESOLUTION_2K_MAP,
    UNSPECIFIED_OPTION,
)
from ..core.shared.logging import (
    safe_log_error_body,
    safe_log_mapping,
    safe_log_text,
    safe_log_url,
)
from ..core.shared.types import (
    GenerationRequest,
    GenerationResult,
    ImageCapability,
    ImageData,
)


class GiteeAIAdapter(BaseImageAdapter):
    """General Gitee AI image generation and editing adapter."""

    DEFAULT_BASE_URL = GITEE_AI_DEFAULT_BASE_URL
    DEFAULT_TEXT_MODEL = "z-image-turbo"
    DEFAULT_EDIT_MODEL = "LongCat-Image-Edit"
    EDIT_MODELS = {
        "animesharp",
        "dreamo",
        "flux.1-dev",
        "flux.1-kontext-dev",
        "flux.2-dev",
        "flux.2-klein-4b",
        "flux.2-klein-9b",
        "kolors",
        "longcat-image-edit",
        "qwen-image-edit",
    }
    IMAGE_EXTENSIONS = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/heic": "heic",
        "image/heif": "heif",
    }

    EXTRA_1K_SIZE_MAP = {
        "4:5": "832x1024",
        "5:4": "1024x832",
        "21:9": "1024x448",
    }
    EXTRA_2K_SIZE_MAP = {
        "4:5": "1632x2048",
        "5:4": "2048x1632",
        "21:9": "2048x864",
    }

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""
        return self._get_configured_capabilities()

    # generate() is provided by the base class via the template method pattern.

    def _pre_generate(self, request: GenerationRequest) -> GenerationResult | None:
        """Log a Gitee AI request overview."""
        prefix = self._get_log_prefix(request.task_id)
        mode = "图片编辑" if self._should_use_edits(request) else "文本生成图片"
        logger.debug(
            f"{prefix} 准备 Gitee AI 请求: 模式={mode}，模型={safe_log_text(self._model_name(request))}"
        )
        return None

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one image generation request."""
        start_time = time.time()
        session = self._get_session()

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "X-Failover-Enabled": "true",
        }

        if self._should_use_edits(request):
            form, fields = self._build_edit_form(request)
            url = self._endpoint_url("images/edits")
            kwargs: dict[str, Any] = {"data": form}
            self._log_request_overview(request, url, form_fields=fields)
        else:
            payload = self._build_payload(request)
            url = self._endpoint_url("images/generations")
            headers["Content-Type"] = "application/json"
            kwargs = {"json": payload}
            self._log_request_overview(request, url, payload=payload)
            self._log_debug_json("请求", payload, request.task_id)

        try:
            async with session.post(
                url,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
                **kwargs,
            ) as resp:
                duration = time.time() - start_time
                self._log_response_status(request, resp.status, duration)
                if resp.status != 200:
                    error_text = await resp.text()
                    self._log_debug_json_text("响应", error_text, request.task_id)
                    self._log_api_error(request, resp.status, duration, error_text)
                    return None, self._format_api_error_message(
                        resp.status,
                        error_text,
                    )

                data = await self._read_response_json(resp, request.task_id)
                return await self._extract_images(data, request.task_id)
        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            self._log_request_exception(request, duration, e)
            return None, safe_log_error_body(e)

    def _endpoint_url(self, path: str) -> str:
        """Build a Gitee AI v1 image endpoint URL."""
        base = (self.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        for suffix in ("/v1/images/generations", "/v1/images/edits"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        if base.endswith("/v1"):
            return f"{base}/{path}"
        return f"{base}/v1/{path}"

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the image generation request payload."""
        payload: dict[str, Any] = {
            "model": self._model_name(request),
            "prompt": request.prompt,
            "n": 1,
        }
        if size := self._resolve_size(request):
            payload["size"] = size

        if request.images:
            self._add_generation_image(payload, request.images, request.task_id)

        return payload

    def _build_edit_form(
        self, request: GenerationRequest
    ) -> tuple[aiohttp.FormData, list[str]]:
        """Build the image editing multipart/form-data request."""
        form = aiohttp.FormData()
        fields: list[str] = []

        def add_field(name: str, value: str) -> None:
            form.add_field(name, value)
            fields.append(name)

        add_field("model", self._model_name(request))
        add_field("prompt", request.prompt)
        if size := self._resolve_size(request):
            add_field("size", size)
        self._log_debug_json("请求表单尺寸", {"size": size}, request.task_id)
        add_field("n", "1")

        for index, image in enumerate(request.images[:1], start=1):
            form.add_field(
                "image",
                image.data,
                filename=self._image_filename(index, image.mime_type),
                content_type=image.mime_type or "image/png",
            )
            fields.append("image")

        return form, fields

    def _resolve_size(self, request: GenerationRequest) -> str | None:
        """Resolve the Gitee AI size parameter from aspect ratio and resolution."""
        if (
            not request.aspect_ratio
            or request.aspect_ratio == UNSPECIFIED_OPTION
            or not request.resolution
            or request.resolution == UNSPECIFIED_OPTION
        ):
            logger.debug(
                f"{self._get_log_prefix(request.task_id)} 参数: size=未指定, "
                f"宽高比={request.aspect_ratio or UNSPECIFIED_OPTION}, "
                f"分辨率={request.resolution or UNSPECIFIED_OPTION}"
            )
            return None
        aspect_ratio = request.aspect_ratio

        size = "1024x1024"
        if request.resolution in ("2K", "4K"):
            size_map = {**RESOLUTION_2K_MAP, **self.EXTRA_2K_SIZE_MAP}
            size = size_map.get(aspect_ratio, "2048x2048")
        else:
            size_map = {**RESOLUTION_1K_MAP, **self.EXTRA_1K_SIZE_MAP}
            size = size_map.get(aspect_ratio, "1024x1024")

        logger.debug(
            f"{self._get_log_prefix(request.task_id)} 参数: size={size}, "
            f"宽高比={aspect_ratio}, 分辨率={request.resolution}"
        )
        return size

    def _add_generation_image(
        self, payload: dict[str, Any], images: list[ImageData], task_id: str | None
    ) -> None:
        """Add a reference image to /images/generations payloads."""
        if not images:
            return
        if len(images) > 1:
            logger.debug(
                f"{self._get_log_prefix(task_id)} /images/generations 仅发送第一张参考图"
            )
        payload["image"] = base64.b64encode(images[0].data).decode("ascii")

    def _model_name(self, request: GenerationRequest | None = None) -> str:
        """Return the active model name."""
        if request and self._should_use_edits(request):
            edit_model = str(self.config.extra.get("edit_model") or "").strip()
            if edit_model:
                return edit_model
            if self._looks_like_edit_model(self.model):
                return self.model
            return self.DEFAULT_EDIT_MODEL
        if self.model:
            return self.model
        return self.DEFAULT_TEXT_MODEL

    def _should_use_edits(self, request: GenerationRequest) -> bool:
        """Return whether the request should use the image editing endpoint."""
        if not request.images:
            return False
        endpoint_mode = self._image_endpoint_mode()
        if endpoint_mode == "generations":
            return False
        if endpoint_mode == "edits":
            return True
        if str(self.config.extra.get("edit_model") or "").strip():
            return True
        return self._looks_like_edit_model(self.model)

    def _image_endpoint_mode(self) -> str:
        """Return the configured image-to-image endpoint mode."""
        endpoint_mode = (
            str(self.config.extra.get("image_endpoint") or "auto").strip().lower()
        )
        if endpoint_mode in {"generations", "edits"}:
            return endpoint_mode
        return "auto"

    def _looks_like_edit_model(self, model: str | None) -> bool:
        """Return whether a model name looks like a Gitee AI edit model."""
        model_name = (model or "").lower()
        if model_name in self.EDIT_MODELS:
            return True
        return any(
            marker in model_name
            for marker in ("edit", "kontext", "dreamo", "animesharp")
        )

    def _image_filename(self, index: int, mime_type: str) -> str:
        """Build an upload filename from the MIME type."""
        extension = self.IMAGE_EXTENSIONS.get((mime_type or "").lower(), "png")
        return f"image_{index}.{extension}"

    async def _extract_images(
        self, data: dict, task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """Extract image bytes from an API response."""
        prefix = self._get_log_prefix(task_id)

        if response_error := data.get("error"):
            if isinstance(response_error, dict):
                message = response_error.get("message") or response_error.get("code")
                return None, f"API 错误: {message}"
            return None, f"API 错误: {response_error}"

        if "data" not in data:
            return None, f"响应格式错误: {safe_log_mapping(data)}"

        images: list[bytes] = []
        for item in data["data"]:
            if not isinstance(item, dict):
                logger.warning(
                    f"{prefix} 跳过无法识别的响应项: {safe_log_mapping(item)}"
                )
                continue

            if "b64_json" in item:
                if img_bytes := self._decode_base64_image(item["b64_json"], task_id):
                    images.append(img_bytes)
            elif "url" in item:
                # Download URL results when the provider returns URLs.
                url = str(item["url"])
                if self.debug_request_logging:
                    logger.debug(f"{prefix} 正在下载图像: {safe_log_url(url)}")
                if url.startswith("data:image/"):
                    if img_bytes := self._decode_base64_image(url, task_id):
                        images.append(img_bytes)
                elif img_bytes := await self._download_image(url, task_id):
                    images.append(img_bytes)
            else:
                logger.warning(
                    f"{prefix} 无法从响应项中提取图像: {safe_log_mapping(item)}"
                )

        if not images:
            return None, "未生成任何图像"

        if self.debug_request_logging:
            logger.debug(f"{prefix} 成功提取 {len(images)} 张图像")
        return images, None

    def _decode_base64_image(
        self, value: Any, task_id: str | None = None
    ) -> bytes | None:
        """Decode a b64_json or data URL image value."""
        data = str(value or "")
        if ";base64," in data:
            _, _, data = data.partition(";base64,")
        try:
            return base64.b64decode(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"{self._get_log_prefix(task_id)} Base64 解码失败: {safe_log_error_body(exc)}"
            )
            return None

    async def _download_image(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        """Download an image URL."""
        session = self._get_session()
        prefix = self._get_log_prefix(task_id)
        try:
            async with session.get(
                url, proxy=self.proxy, timeout=self._get_download_timeout()
            ) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    if self.debug_request_logging:
                        logger.debug(f"{prefix} 图像下载成功: {len(data)} bytes")
                    return data
                logger.error(
                    f"{prefix} 下载图像失败 ({resp.status}): {safe_log_url(url)}"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"{prefix} 下载图像异常: {safe_log_error_body(e)}")
        return None
