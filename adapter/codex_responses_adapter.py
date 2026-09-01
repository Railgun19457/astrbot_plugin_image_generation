from __future__ import annotations

import base64
import binascii
import json
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.logging import safe_log_error_body, safe_log_url, safe_log_text
from ..core.shared.types import GenerationRequest, GenerationResult, ImageCapability


class CodexResponsesAdapter(BaseImageAdapter):
    """Adapter for synchronous Codex Responses image generation endpoints."""

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities from provider configuration options."""
        return self._get_configured_capabilities()

    def _pre_generate(self, _request: GenerationRequest) -> GenerationResult | None:
        """Validate the fixed endpoint configuration before making a request."""
        if not (self.model or "").strip():
            return GenerationResult(images=None, error="未配置 Codex Responses 模型")
        _, error = self._build_responses_url()
        if error:
            return GenerationResult(images=None, error=error)
        return None

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Send one synchronous image-generation request to Codex Responses."""
        url, url_error = self._build_responses_url()
        if url_error:
            return None, url_error

        payload = self._build_payload(request)
        api_key = self._get_current_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, image/*",
        }
        start_time = time.time()
        self._log_request_overview(
            request,
            url,
            payload=payload,
            extra={
                "API Key": "已配置" if api_key else "未配置",
                "超时配置": f"{self.timeout}秒",
            },
        )
        self._log_debug_json("请求", payload, request.task_id)

        try:
            async with self._get_session().post(
                url,
                json=payload,
                headers=headers,
                timeout=self._get_timeout(),
                proxy=self.proxy,
            ) as response:
                duration = time.time() - start_time
                body = await response.read()
                self._log_response_status(request, response.status, duration)
                if not 200 <= response.status < 300:
                    error_text = body.decode("utf-8", errors="replace")
                    self._log_debug_json_text("响应", error_text, request.task_id)
                    self._log_api_error(
                        request,
                        response.status,
                        duration,
                        error_text,
                        label="Codex Responses 错误",
                    )
                    return None, self._format_http_error(response.status, error_text)

                content_type = response.headers.get("Content-Type", "").lower()
                if content_type.startswith("image/"):
                    return [body], None

                response_data, parse_error = self._parse_response_json(
                    body, request.task_id
                )
                if parse_error:
                    return None, parse_error
                if not isinstance(response_data, dict):
                    return None, "Codex Responses 响应必须是 JSON 对象"

                if error := self._response_error_message(response_data):
                    return None, error

                images = await self._extract_images(response_data, request.task_id)
                if images:
                    return images, None

                self._log_missing_image_response(response_data, request.task_id)
                return None, self._no_image_error(response_data)
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            self._log_request_exception(
                request,
                duration,
                exc,
                label=f"Codex Responses 请求异常 (配置超时 {self.timeout}s)",
            )
            return None, safe_log_error_body(exc)

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Build a text-only or multimodal Responses request payload."""
        payload: dict[str, Any] = {
            "model": self.model,
            "tools": [self._build_image_generation_tool()],
        }
        if not request.images:
            payload["input"] = request.prompt
            return payload

        content: list[dict[str, str]] = [{"type": "input_text", "text": request.prompt}]
        for image in request.images:
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image.mime_type};base64,{encoded}",
                }
            )
        payload["input"] = [{"role": "user", "content": content}]
        return payload

    def _build_image_generation_tool(self) -> dict[str, str]:
        """Build the verified image-generation tool declaration."""
        return {
            "type": "image_generation",
            "output_format": "png",
        }

    def _build_responses_url(self) -> tuple[str, str | None]:
        """Join the configured service root with the Responses endpoint path."""
        base = (self.base_url or "").strip()
        if not base:
            return "", "未配置 Codex Responses 接口地址"

        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "", "Codex Responses 接口地址必须是有效的 http(s) URL"
        if parsed.query or parsed.fragment:
            return "", "Codex Responses 接口地址不能包含查询参数或片段"

        endpoint = str(self.config.extra.get("endpoint") or "/codex/responses").strip()
        if not endpoint:
            endpoint = "/codex/responses"
        if endpoint.startswith(("http://", "https://")):
            return endpoint, None

        endpoint_path = f"{parsed.path.rstrip('/')}/" + endpoint.lstrip("/")
        return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", "")), None

    def _parse_response_json(
        self, body: bytes, task_id: str | None
    ) -> tuple[Any, str | None]:
        """Decode a JSON response while keeping debug logs safely summarized."""
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return None, "Codex Responses 响应不是 JSON 且不是图片数据"

        try:
            response_data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"{self._get_log_prefix(task_id)} Codex Responses JSON 解析失败: "
                f"{safe_log_error_body(exc)}"
            )
            return None, "Codex Responses 响应 JSON 解析失败"

        self._log_debug_json("响应", response_data, task_id)
        return response_data, None

    def _format_http_error(self, status: int, error_text: str) -> str:
        """Preserve retryable upstream failures wrapped in a 4xx response."""
        normalized = error_text.lower()
        upstream_statuses = re.findall(r"http\s+(502|503|504)\b", normalized)
        if (
            "upstream_error" in normalized
            or "upstream request failed" in normalized
            or upstream_statuses
        ):
            upstream_status = upstream_statuses[-1] if upstream_statuses else "502"
            return f"API 错误 ({upstream_status}): 上游服务暂时不可用"
        return self._format_api_error_message(status, error_text)

    def _response_error_message(self, response_data: dict[str, Any]) -> str | None:
        """Return a concise error for completed error responses before image parsing."""
        error = response_data.get("error")
        if error not in (None, "", {}, []):
            return self._format_response_error("Codex Responses 返回错误", error)

        status = str(response_data.get("status") or "").strip().lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            detail = (
                response_data.get("message")
                or response_data.get("reason")
                or response_data.get("code")
                or status
            )
            return self._format_response_error("Codex Responses 请求失败", detail)
        return None

    def _format_response_error(self, prefix: str, detail: Any) -> str:
        """Add an optional sanitized remote detail to a response error."""
        if not self.show_user_error_details:
            return prefix
        safe_detail = safe_log_text(detail, 300)
        return f"{prefix}: {safe_detail}" if safe_detail else prefix

    async def _extract_images(
        self, response_data: dict[str, Any], task_id: str | None
    ) -> list[bytes]:
        """Extract images from standard Responses output and proxy-compatible fallbacks."""
        images: list[bytes] = []
        output = response_data.get("output")
        if isinstance(output, list):
            for item in output:
                if (
                    not isinstance(item, dict)
                    or item.get("type") != "image_generation_call"
                ):
                    continue
                images.extend(await self._decode_call_images(item, task_id))

        if images:
            return images

        for field in ("result", "result_b64", "b64_json", "base64", "url"):
            if field in response_data:
                decoded = await self._decode_image_value(response_data[field], task_id)
                if decoded:
                    images.extend(decoded)

        data = response_data.get("data")
        if isinstance(data, list):
            for item in data:
                decoded = await self._decode_image_value(item, task_id)
                if decoded:
                    images.extend(decoded)

        return images

    async def _decode_call_images(
        self, call: dict[str, Any], task_id: str | None
    ) -> list[bytes]:
        """Decode one image_generation_call, preferring its documented result field."""
        status = str(call.get("status") or "").strip().lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            logger.warning(
                f"{self._get_log_prefix(task_id)} image_generation_call 失败: "
                f"{safe_log_text(call.get('error') or call.get('message') or status, 200)}"
            )
            return []

        images: list[bytes] = []
        for field in ("result", "result_b64", "b64_json", "base64", "url"):
            if field not in call:
                continue
            decoded = await self._decode_image_value(call[field], task_id)
            if decoded:
                images.extend(decoded)
        return images

    async def _decode_image_value(self, value: Any, task_id: str | None) -> list[bytes]:
        """Decode image candidates represented as bytes, arrays, objects, URLs, or Base64."""
        if value is None:
            return []
        if isinstance(value, bytes):
            return [value]
        if isinstance(value, list):
            images: list[bytes] = []
            for item in value:
                images.extend(await self._decode_image_value(item, task_id))
            return images
        if isinstance(value, dict):
            images: list[bytes] = []
            for field in (
                "result",
                "result_b64",
                "b64_json",
                "base64",
                "image",
                "data",
                "url",
            ):
                if field in value:
                    images.extend(await self._decode_image_value(value[field], task_id))
            image_url = value.get("image_url")
            if isinstance(image_url, dict):
                images.extend(
                    await self._decode_image_value(image_url.get("url"), task_id)
                )
            elif image_url:
                images.extend(await self._decode_image_value(image_url, task_id))
            return images
        if not isinstance(value, str):
            return []

        candidate = value.strip()
        if not candidate:
            return []
        if candidate.startswith(("http://", "https://")):
            downloaded = await self._download_image_from_url(candidate, task_id)
            return [downloaded] if downloaded else []
        decoded = self._decode_base64_image(candidate, task_id)
        return [decoded] if decoded else []

    def _decode_base64_image(self, value: str, task_id: str | None) -> bytes | None:
        """Decode a raw Base64 value or image data URI without logging its contents."""
        candidate = value.strip()
        if candidate.lower().startswith("data:"):
            header, separator, candidate = candidate.partition(",")
            if not separator or not header.lower().startswith("data:image/"):
                return None
            if ";base64" not in header.lower():
                return None

        candidate = re.sub(r"\s+", "", candidate)
        # Skip short or non-base64-looking strings to avoid decoding error text.
        if len(candidate) < 32 or not re.fullmatch(r"[A-Za-z0-9+/=]+", candidate):
            return None
        try:
            return base64.b64decode(
                candidate + "=" * (-len(candidate) % 4), validate=True
            )
        except (ValueError, binascii.Error) as exc:
            logger.warning(
                f"{self._get_log_prefix(task_id)} 图片 Base64 解码失败: "
                f"{safe_log_error_body(exc)}"
            )
            return None

    async def _download_image_from_url(
        self, url: str, task_id: str | None
    ) -> bytes | None:
        """Download an absolute image URL without forwarding the provider API key."""
        prefix = self._get_log_prefix(task_id)
        try:
            async with self._get_session().get(
                url,
                proxy=self.proxy,
                timeout=self._get_download_timeout(),
            ) as response:
                if response.status == 200:
                    return await response.read()
                logger.warning(
                    f"{prefix} Codex Responses 图片下载失败: {response.status} - "
                    f"{safe_log_url(url)}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"{prefix} Codex Responses 图片下载异常: {safe_log_error_body(exc)}"
            )
        return None

    def _log_missing_image_response(
        self, response_data: dict[str, Any], task_id: str | None
    ) -> None:
        """Log a safe response-shape summary when no image can be extracted."""
        output_summary: list[dict[str, Any]] = []
        output = response_data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    output_summary.append({"value_type": type(item).__name__})
                    continue
                result = item.get("result")
                output_summary.append(
                    {
                        "type": item.get("type"),
                        "status": item.get("status"),
                        "keys": sorted(item.keys()),
                        "result_type": type(result).__name__,
                        "result_length": len(result)
                        if isinstance(result, str)
                        else None,
                    }
                )
        logger.warning(
            f"{self._get_log_prefix(task_id)} Codex Responses 未返回可用图片: "
            f"{json.dumps({'status': response_data.get('status'), 'error': response_data.get('error'), 'output': output_summary}, ensure_ascii=False, separators=(',', ':'))}"
        )

    def _no_image_error(self, response_data: dict[str, Any]) -> str:
        """Describe an otherwise successful response that contains no usable image."""
        output = response_data.get("output")
        if isinstance(output, list) and any(
            isinstance(item, dict) and item.get("type") == "image_generation_call"
            for item in output
        ):
            return "image_generation_call 中未找到可解码的图片结果"
        if response_data.get("status") in {"in_progress", "queued"}:
            return "Codex Responses 返回未完成任务，当前适配器仅支持同步图片响应"
        return "Codex Responses 响应中未找到图片数据"
