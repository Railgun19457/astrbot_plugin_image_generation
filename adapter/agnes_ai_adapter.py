from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import (
    RESOLUTION_1K_MAP,
    RESOLUTION_2K_MAP,
    UNSPECIFIED_OPTION,
)
from ..core.shared.logging import (
    safe_log_error_body,
    safe_log_mapping,
)
from ..core.shared.types import GenerationRequest, ImageCapability, ImageData


class AgnesAIAdapter(BaseImageAdapter):
    """Agnes AI image generation adapter."""

    DEFAULT_BASE_URL = "https://apihub.agnes-ai.com"
    DEFAULT_MODEL = "agnes-image-2.1-flash"

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
        configured = self._get_configured_capabilities()
        defaults = (
            ImageCapability.TEXT_TO_IMAGE
            | ImageCapability.IMAGE_TO_IMAGE
            | ImageCapability.ASPECT_RATIO
            | ImageCapability.RESOLUTION
        )
        if configured is ImageCapability.NONE:
            return defaults
        return configured | ImageCapability.ASPECT_RATIO

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one image generation request."""
        start_time = time.time()
        payload = self._build_payload(request)
        url = self._endpoint_url()
        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
        }

        self._log_request_overview(request, url, payload=payload)
        self._log_debug_json("请求", payload, request.task_id)

        try:
            async with self._get_session().post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
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

    def _endpoint_url(self) -> str:
        """Build the Agnes AI image generation endpoint URL."""
        base = (self.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        suffix = "/v1/images/generations"
        if base.endswith(suffix):
            return base
        if base.endswith("/v1"):
            return f"{base}/images/generations"
        return f"{base}{suffix}"

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the request payload."""
        payload: dict[str, Any] = {
            "model": self._model_name(),
            "prompt": request.prompt,
        }
        if size := self._resolve_size(request):
            payload["size"] = size

        response_format = self._response_format()
        extra_body: dict[str, Any] = {}
        if response_format == "url":
            extra_body["response_format"] = "url"
        elif request.images:
            extra_body["response_format"] = "b64_json"
        else:
            payload["return_base64"] = True

        if request.images:
            extra_body["image"] = self._build_image_refs(request.images)
        if extra_body:
            payload["extra_body"] = extra_body

        return payload

    def _model_name(self) -> str:
        """Return the active model name."""
        return self.model or self.DEFAULT_MODEL

    def _response_format(self) -> str:
        """Resolve the Agnes response format setting."""
        value = str(self.config.extra.get("response_format") or "base64")
        normalized = value.strip().lower()
        if normalized == "url":
            return "url"
        return "base64"

    def _resolve_size(self, request: GenerationRequest) -> str | None:
        """Resolve the Agnes AI size parameter from aspect ratio and resolution."""
        if not request.aspect_ratio or request.aspect_ratio == UNSPECIFIED_OPTION:
            return None

        aspect_ratio = request.aspect_ratio
        resolution = (request.resolution or "").upper()
        if resolution in ("2K", "4K"):
            size_map = {**RESOLUTION_2K_MAP, **self.EXTRA_2K_SIZE_MAP}
            return size_map.get(aspect_ratio, "2048x2048")

        size_map = {**RESOLUTION_1K_MAP, **self.EXTRA_1K_SIZE_MAP}
        return size_map.get(aspect_ratio, "1024x1024")

    def _build_image_refs(self, images: list[ImageData]) -> list[str]:
        """Build Agnes AI image-to-image reference data URLs.

        Platform image URLs are often temporary links that require platform-side
        authentication, so the adapter sends data URLs with the request instead.
        """
        refs: list[str] = []
        for image in images:
            if not image.data:
                continue
            mime_type = image.mime_type or "image/png"
            b64_data = base64.b64encode(image.data).decode("ascii")
            refs.append(f"data:{mime_type};base64,{b64_data}")
        return refs

    async def _extract_images(
        self, response: dict[str, Any], task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """Extract image bytes from the response payload."""
        if response_error := response.get("error"):
            if isinstance(response_error, dict):
                message = response_error.get("message") or response_error.get("code")
                return None, f"API 错误: {message}"
            return None, f"API 错误: {response_error}"

        data_items = response.get("data")
        if not isinstance(data_items, list):
            return None, f"响应中未找到 data 字段: {safe_log_mapping(response)}"

        images: list[bytes] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue

            b64_json = item.get("b64_json")
            if b64_json:
                try:
                    images.append(base64.b64decode(b64_json))
                    continue
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"{self._get_log_prefix(task_id)} 跳过无效 b64_json 图片数据: {safe_log_error_body(e)}"
                    )

            image_url = item.get("url") or item.get("image_url")
            if isinstance(image_url, str) and image_url:
                if image_url.startswith("data:image/"):
                    if data := self._decode_base64_image(image_url, task_id):
                        images.append(data)
                elif data := await self._download_image(image_url, task_id):
                    images.append(data)

        if not images:
            return None, "未找到有效的图片数据"

        return images, None

    async def _download_image(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        """Download a temporary image URL returned by Agnes AI."""
        prefix = self._get_log_prefix(task_id)
        try:
            async with self._get_session().get(
                url, proxy=self.proxy, timeout=self._get_download_timeout()
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                logger.warning(f"{prefix} 下载 Agnes AI 图片失败: HTTP {resp.status}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"{prefix} 下载 Agnes AI 图片异常: {safe_log_error_body(exc)}"
            )
        return None
