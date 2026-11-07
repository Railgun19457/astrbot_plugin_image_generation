from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import Any, ClassVar

import aiohttp
from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import UNSPECIFIED_OPTION
from ..core.shared.logging import safe_log_error_body, safe_log_url
from ..core.shared.types import GenerationRequest, GenerationResult, ImageCapability


class OpenAIAdapter(BaseImageAdapter):
    """OpenAI image generation adapter for GPT Image 2-series models."""

    # GPT Image 2 / 2.5 size presets. Every edge is a multiple of 16, the
    # longest edge is at most 3840, and total pixels stay within the
    # documented 655,360-8,294,400 range.
    RESOLUTION_SIZE_MAPS: ClassVar[dict[str, dict[str, str]]] = {
        "1K": {
            "1:1": "1024x1024",
            "4:3": "1024x768",
            "3:4": "768x1024",
            "3:2": "1056x704",
            "2:3": "704x1056",
            "5:4": "1040x832",
            "4:5": "832x1040",
            "16:9": "1152x640",
            "9:16": "640x1152",
            "21:9": "1344x576",
        },
        "1.5K": {
            "1:1": "1536x1536",
            "4:3": "1536x1152",
            "3:4": "1152x1536",
            "3:2": "1536x1024",
            "2:3": "1024x1536",
            "5:4": "1520x1216",
            "4:5": "1216x1520",
            "16:9": "1536x864",
            "9:16": "864x1536",
            "21:9": "1680x720",
        },
        "2K": {
            "1:1": "2048x2048",
            "4:3": "2048x1536",
            "3:4": "1536x2048",
            "3:2": "2064x1376",
            "2:3": "1376x2064",
            "5:4": "2080x1664",
            "4:5": "1664x2080",
            "16:9": "2048x1152",
            "9:16": "1152x2048",
            "21:9": "2352x1008",
        },
        "4K": {
            "1:1": "2880x2880",
            "4:3": "3264x2448",
            "3:4": "2448x3264",
            "3:2": "3504x2336",
            "2:3": "2336x3504",
            "5:4": "3200x2560",
            "4:5": "2560x3200",
            "16:9": "3840x2160",
            "9:16": "2160x3840",
            "21:9": "3696x1584",
        },
    }

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""
        return self._get_configured_capabilities()

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

    def _pre_generate(self, request: GenerationRequest) -> GenerationResult | None:
        """Reject invalid GIF settings before sending or retrying a request."""
        try:
            self._build_payload(request)
        except ValueError as exc:
            return GenerationResult(error=str(exc))
        return None

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one image generation request."""
        start_time = time.time()
        payload = self._build_payload(request)

        use_edit = bool(request.images)
        session = self._get_session()
        base = self.base_url.rstrip("/") if self.base_url else "https://api.openai.com"
        headers = {"Authorization": f"Bearer {self._get_current_api_key()}"}

        if use_edit:
            url = f"{base}/v1/images/edits"
            form = aiohttp.FormData()
            for key, value in payload.items():
                form.add_field(key, str(value))
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
            enable_streaming = self.config.extra.get("enable_streaming", True)
            if isinstance(enable_streaming, str):
                enable_streaming = enable_streaming.strip().lower() not in {
                    "false",
                    "0",
                    "no",
                    "off",
                    "",
                }
            if enable_streaming and payload["output_format"] != "gif":
                payload["stream"] = True
            kwargs = {"json": payload}
            self._log_request_overview(request, url, payload=payload)
            self._log_debug_json("请求", payload, request.task_id)
        if use_edit:
            self._log_request_overview(
                request,
                url,
                form_fields=[*payload, "image[]"],
            )
            self._log_debug_json("请求表单参数", payload, request.task_id)

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
        payload: dict[str, Any] = {
            "model": self.model or "gpt-image-2",
            "prompt": request.prompt,
            "n": 1,
        }

        output_format = str(self.config.extra.get("output_format") or "png").strip().lower()
        payload["output_format"] = (
            output_format if output_format in {"png", "jpeg", "webp", "gif"} else "png"
        )
        if output_format == "gif":
            for key, default, minimum, maximum in (
                ("frame_count", 9, 2, 16),
                ("frame_duration_ms", 200, 20, 2000),
            ):
                try:
                    value = int(str(self.config.extra.get(key, default)))
                    if not minimum <= value <= maximum:
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        f"GIF {key} 必须是 {minimum}～{maximum} 的整数"
                    ) from None
                payload[key] = value

            size = str(self.config.extra.get("gif_size") or "auto").strip().lower()
            if size != "auto":
                dimensions = re.fullmatch(r"([0-9]+)x([0-9]+)", size)
                if not dimensions or any(
                    int(edge) < 16 or int(edge) % 16 for edge in dimensions.groups()
                ):
                    raise ValueError(
                        "GIF gif_size 必须为 auto 或宽x高，宽高均须为正整数且为 16 的倍数"
                    )
            payload["size"] = size
        elif size := self._map_size(request.aspect_ratio, request.resolution):
            payload["size"] = size

        return payload

    def _map_size(self, aspect_ratio: str | None, resolution: str | None) -> str | None:
        """Map the requested aspect ratio and resolution to an Images API size."""
        if resolution and resolution != UNSPECIFIED_OPTION:
            ratio = (
                aspect_ratio
                if aspect_ratio and aspect_ratio != UNSPECIFIED_OPTION
                else "1:1"
            )
            size_map = self.RESOLUTION_SIZE_MAPS.get(resolution)
            if size_map:
                return size_map.get(ratio, size_map["1:1"])

        return self._map_aspect_ratio_to_size(aspect_ratio)

    def _map_aspect_ratio_to_size(self, aspect_ratio: str | None) -> str | None:
        """Map an aspect ratio to an OpenAI-supported size parameter."""
        if not aspect_ratio or aspect_ratio == UNSPECIFIED_OPTION:
            return None

        # Map unsupported ratios to the documented preset sizes. Custom
        # dimensions are not exposed by the plugin for this adapter.
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
