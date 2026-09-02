from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

import aiohttp

from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import UNSPECIFIED_OPTION
from ..core.shared.logging import safe_log_error_body, safe_log_url
from ..core.shared.types import GenerationRequest, ImageCapability


class OpenAIAdapter(BaseImageAdapter):
    """OpenAI image generation adapter for DALL-E and GPT Image models."""

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""
        return self._get_configured_capabilities()

    def _is_gpt_image_model(self) -> bool:
        """Return whether the active model is a GPT Image model."""
        model_family = self.config.extra.get("model_family", "auto")
        if model_family == "gpt-image":
            return True
        if model_family == "dall-e":
            return False
        # auto: infer the family from the model name.
        return self.model is not None and "gpt-image" in self.model

    def _image_filename(self, mime_type: str, index: int) -> str:
        """Return a filename whose extension matches the uploaded image bytes."""
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/heic": ".heic",
            "image/heif": ".heif",
        }.get((mime_type or "").lower(), ".png")
        return f"reference_{index}{extension}"

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one image generation request."""
        start_time = time.time()
        prefix = self._get_log_prefix(request.task_id)

        is_gpt = self._is_gpt_image_model()
        use_edit = bool(request.images) and is_gpt
        if request.images and not is_gpt:
            logger.warning(
                f"{prefix} 提供了参考图但当前模型不支持图生图，仅 GPT Image 系列支持图生图，参考图将被忽略"
            )
        session = self._get_session()
        base = self.base_url.rstrip("/") if self.base_url else "https://api.openai.com"
        headers = {"Authorization": f"Bearer {self._get_current_api_key()}"}

        if use_edit:
            url = f"{base}/v1/images/edits"
            form = aiohttp.FormData()
            form.add_field("model", self.model or "gpt-image-1")
            form.add_field("prompt", request.prompt)
            form.add_field("n", "1")
            if size := self._map_aspect_ratio_to_size(
                request.aspect_ratio, gpt_model=True
            ):
                form.add_field("size", size)
            for index, img in enumerate(request.images, start=1):
                form.add_field(
                    "image[]",
                    img.data,
                    content_type=img.mime_type,
                    filename=self._image_filename(img.mime_type, index),
                )
            kwargs: dict = {"data": form}
        else:
            url = f"{base}/v1/images/generations"
            headers["Content-Type"] = "application/json"
            payload = self._build_payload(request)
            enable_streaming = self.config.extra.get("enable_streaming", True)
            if isinstance(enable_streaming, str):
                enable_streaming = enable_streaming.strip().lower() not in {
                    "false",
                    "0",
                    "no",
                    "off",
                    "",
                }
            if enable_streaming:
                payload["stream"] = True
            kwargs = {"json": payload}
            self._log_request_overview(request, url, payload=payload)
            self._log_debug_json("请求", payload, request.task_id)
        if use_edit:
            self._log_request_overview(
                request,
                url,
                form_fields=["model", "prompt", "n", "size", "image[]"],
            )

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
                if "text/event-stream" in resp.headers.get("Content-Type", "").lower():
                    data = await self._read_stream_response(resp, request.task_id)
                else:
                    data = await self._read_response_json(resp, request.task_id)
                return await self._extract_images(data, request.task_id)
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_exception(request, duration, e)
            error_detail = safe_log_error_body(e) or type(e).__name__
            return None, error_detail

    async def _read_stream_response(
        self, response: aiohttp.ClientResponse, task_id: str | None
    ) -> dict[str, Any]:
        """Collect completed image events from an OpenAI-compatible SSE response."""
        images: list[dict[str, Any]] = []
        buffer = ""
        data_lines: list[str] = []
        stream_error: str | None = None
        completed = False

        def collect_event(payload_text: str) -> bool:
            nonlocal stream_error
            if not payload_text or payload_text == "[DONE]":
                return False
            try:
                event = json.loads(payload_text)
            except json.JSONDecodeError:
                return False
            if not isinstance(event, dict):
                return False

            event_type = str(event.get("type", "")).strip().lower()
            if event_type in {"error", "upstream_error"} or event.get("error"):
                error = event.get("error")
                if isinstance(error, dict):
                    stream_error = str(error.get("message") or error)
                else:
                    stream_error = str(event.get("message") or error)
            elif event_type in {
                "image_generation.completed",
                "image_edit.completed",
            }:
                if isinstance(event.get("data"), list):
                    images.extend(
                        item for item in event["data"] if isinstance(item, dict)
                    )
                else:
                    images.append(event)
                return True
            elif isinstance(event.get("data"), list):
                images.extend(item for item in event["data"] if isinstance(item, dict))
            return False

        async for chunk in response.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                if line or not data_lines:
                    continue

                completed = collect_event("\n".join(data_lines).strip())
                data_lines.clear()
                if completed:
                    break
            if completed:
                break

        if not completed:
            if buffer.rstrip("\r").startswith("data:"):
                data_lines.append(buffer.rstrip("\r")[5:].lstrip())
            if data_lines:
                collect_event("\n".join(data_lines).strip())

        if stream_error:
            raise RuntimeError(f"Image stream error: {stream_error}")

        self._log_debug_json(
            "SSE response",
            {
                "event_count": len(images),
                "image_fields": [list(item) for item in images],
            },
            task_id,
        )
        return {"data": images}

    def _build_payload(self, request: GenerationRequest) -> dict:
        """Build the request payload."""
        gpt = self._is_gpt_image_model()
        payload: dict[str, Any] = {
            "model": self.model or "dall-e-3",
            "prompt": request.prompt,
            "n": 1,
        }

        if size := self._map_aspect_ratio_to_size(request.aspect_ratio, gpt_model=gpt):
            payload["size"] = size
        # OpenAI models do not support the plugin resolution setting; quality is separate.
        if gpt:
            output_format = str(self.config.extra.get("output_format") or "png").lower()
            payload["output_format"] = (
                output_format if output_format in {"png", "jpeg", "webp"} else "png"
            )
        else:
            # Keep DALL-E results self-contained for local persistence.
            payload["response_format"] = "b64_json"

        return payload

    def _map_aspect_ratio_to_size(
        self, aspect_ratio: str | None, gpt_model: bool
    ) -> str | None:
        """Map an aspect ratio to an OpenAI-supported size parameter."""
        if not aspect_ratio or aspect_ratio == UNSPECIFIED_OPTION:
            return None

        if gpt_model:
            # GPT Image models support only square, landscape, and portrait sizes.
            # Map unsupported ratios to the closest supported size.
            mapping = {
                "1:1": "1024x1024",
                "3:2": "1536x1024",
                "16:9": "1536x1024",
                "4:3": "1536x1024",
                "5:4": "1536x1024",
                "21:9": "1536x1024",
                "2:3": "1024x1536",
                "3:4": "1024x1536",
                "9:16": "1024x1536",
                "4:5": "1024x1536",
            }
        else:
            # DALL-E 3 supports only square, landscape, and portrait sizes.
            # Map unsupported ratios to the closest supported size.
            mapping = {
                "1:1": "1024x1024",
                "3:2": "1792x1024",
                "16:9": "1792x1024",
                "4:3": "1792x1024",
                "5:4": "1792x1024",
                "21:9": "1792x1024",
                "2:3": "1024x1792",
                "3:4": "1024x1792",
                "9:16": "1024x1792",
                "4:5": "1024x1792",
            }
        return mapping.get(aspect_ratio)

    async def _extract_images(
        self, response: dict, task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """Extract image bytes from the response payload."""
        if "data" not in response:
            return None, "响应中未找到 data 字段"

        images = []
        download_error: str | None = None
        for item in response["data"]:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                # Download URL results even though b64_json is requested.
                image_url = str(item["url"])
                prefix = self._get_log_prefix(task_id)
                for attempt in range(1, 4):
                    download_start = time.time()
                    logger.debug(
                        f"{prefix} 图片下载开始: 尝试={attempt}/3, "
                        f"地址={safe_log_url(image_url)}"
                    )
                    try:
                        async with self._get_session().get(
                            image_url,
                            proxy=self.proxy,
                            timeout=self._get_timeout(),
                        ) as resp:
                            duration = time.time() - download_start
                            logger.debug(
                                f"{prefix} 图片下载响应: 状态码={resp.status}, "
                                f"尝试={attempt}/3, 耗时={duration:.2f}秒, "
                                f"地址={safe_log_url(image_url)}"
                            )
                            if resp.status == 200:
                                images.append(await resp.read())
                                download_error = None
                                break
                            download_error = f"HTTP {resp.status}"
                    except Exception as exc:
                        duration = time.time() - download_start
                        detail = safe_log_error_body(exc) or type(exc).__name__
                        download_error = detail
                        logger.warning(
                            f"{prefix} 图片下载失败: 尝试={attempt}/3, "
                            f"耗时={duration:.2f}秒, 错误={detail}, "
                            f"地址={safe_log_url(image_url)}"
                        )
                    if attempt < 3:
                        await asyncio.sleep(2 ** (attempt - 1))

        if not images:
            if download_error:
                return (
                    None,
                    f"图片下载失败（生成请求已完成，请勿自动重试）: {download_error}",
                )
            return None, "未找到有效的图片数据"

        return images, None

    def _is_retryable_error(self, error: str) -> bool:
        """Avoid resubmitting a generation after only its result download failed."""
        if error.startswith("图片下载失败（生成请求已完成"):
            return False
        return super()._is_retryable_error(error)
