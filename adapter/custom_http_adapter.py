from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import UNSPECIFIED_OPTION
from ..core.shared.logging import safe_log_error_body, safe_log_url
from ..core.shared.types import GenerationRequest, ImageCapability, ImageData

PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z0-9_]+)\}")
EXACT_PLACEHOLDER_PATTERN = re.compile(r"^\{([a-zA-Z0-9_]+)\}$")


class CustomHTTPAdapter(BaseImageAdapter):
    """User-defined HTTP JSON image generation adapter."""

    requires_api_key = False

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""
        return self._get_configured_capabilities()

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one custom HTTP image generation request."""
        start_time = time.time()
        context = self._build_placeholder_context(request)

        url = self._build_url(context)
        if not url:
            return None, "未配置请求 URL"

        headers, headers_error = self._load_json_object_template(
            "headers_json",
            default={
                "Authorization": "Bearer {api_key}",
                "Content-Type": "application/json",
            },
            context=context,
        )
        if headers_error:
            return None, headers_error

        query_params, query_error = self._load_json_object_template(
            "query_params_json",
            default={},
            context=context,
        )
        if query_error:
            return None, query_error

        payload, payload_error = self._load_json_object_template(
            "payload_json",
            default={"model": "{model}", "prompt": "{prompt}"},
            context=context,
        )
        if payload_error:
            return None, payload_error

        method = self._request_method()
        kwargs: dict[str, Any] = {
            "headers": self._normalize_headers(headers),
            "params": query_params or None,
            "proxy": self.proxy,
            "timeout": self._get_timeout(),
        }
        if method not in {"GET", "DELETE"}:
            kwargs["json"] = payload

        self._log_request_overview(
            request,
            url,
            method=method,
            payload=payload,
        )
        if method not in {"GET", "DELETE"}:
            self._log_debug_json("请求", payload, request.task_id)

        try:
            async with self._get_session().request(method, url, **kwargs) as response:
                duration = time.time() - start_time
                body = await response.read()
                self._log_response_status(request, response.status, duration)
                if response.status not in self._success_status_codes():
                    error_text = body.decode("utf-8", errors="replace")
                    self._log_api_error(
                        request,
                        response.status,
                        duration,
                        error_text,
                        label="自定义 HTTP 错误",
                    )
                    return None, self._format_api_error_message(
                        response.status,
                        error_text,
                    )

                content_type = response.headers.get("Content-Type", "").lower()
                if content_type.startswith("image/"):
                    return [body], None

                response_data, parse_error = self._parse_json_response(body)
                if parse_error:
                    return None, parse_error

                if error_message := self._extract_error_message(response_data):
                    return None, error_message

                images = await self._extract_images(response_data, request.task_id)
                if images:
                    return images, None
                return None, "响应中未找到图片数据"
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            self._log_request_exception(
                request, duration, exc, label="自定义 HTTP 请求异常"
            )
            return None, safe_log_error_body(exc)

    def _request_method(self) -> str:
        method = str(self.config.extra.get("request_method") or "POST").strip().upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            return "POST"
        return method

    def _success_status_codes(self) -> set[int]:
        raw = self.config.extra.get("success_status_codes", [200])
        if isinstance(raw, int):
            return {raw}
        if isinstance(raw, str):
            raw = re.split(r"[,，\s]+", raw.strip())
        if not isinstance(raw, list):
            return {200}

        result: set[int] = set()
        for item in raw:
            try:
                result.add(int(item))
            except (TypeError, ValueError):
                continue
        return result or {200}

    def _build_url(self, context: dict[str, Any]) -> str:
        endpoint = str(self.config.extra.get("endpoint") or "").strip()
        if endpoint.startswith(("http://", "https://")):
            return self._render_template_string(endpoint, context)

        base = (self.base_url or "").rstrip("/")
        if not base:
            return ""
        if not endpoint:
            return self._render_template_string(base, context)
        return self._render_template_string(f"{base}/{endpoint.lstrip('/')}", context)

    def _build_placeholder_context(self, request: GenerationRequest) -> dict[str, Any]:
        image_base64 = [self._image_to_base64(image) for image in request.images]
        image_data_urls = [
            f"data:{image.mime_type};base64,{encoded}"
            for image, encoded in zip(request.images, image_base64, strict=False)
        ]
        image_mime_types = [image.mime_type for image in request.images]

        aspect_ratio = self._placeholder_optional(request.aspect_ratio)
        resolution = self._placeholder_optional(request.resolution)
        first_base64 = image_base64[0] if image_base64 else ""
        first_data_url = image_data_urls[0] if image_data_urls else ""
        first_mime_type = image_mime_types[0] if image_mime_types else ""

        return {
            "prompt": request.prompt,
            "model": self.model or "",
            "api_key": self._get_current_api_key(),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "task_id": request.task_id or "",
            "batch_index": request.batch_index,
            "batch_count": request.batch_count,
            "requested_count": request.batch_count,
            "image_count": 1,
            "count": 1,
            "reference_image_count": len(request.images),
            "reference_images_count": len(request.images),
            "reference_images": image_data_urls,
            "reference_images_base64": image_base64,
            "reference_images_base64_nullable": image_base64 if len(image_base64) > 0 else None,
            "reference_images_data_url": image_data_urls,
            "reference_images_data_url_nullable": image_data_urls if len(image_data_urls) > 0 else None,
            "reference_images_mime_types": image_mime_types,
            "reference_image_0": first_data_url,
            "reference_image_0_base64": first_base64,
            "reference_image_0_data_url": first_data_url,
            "reference_image_0_mime_type": first_mime_type,
        }

    def _placeholder_optional(self, value: str | None) -> str:
        if not value or value == UNSPECIFIED_OPTION:
            return ""
        return value

    def _image_to_base64(self, image: ImageData) -> str:
        return base64.b64encode(image.data).decode("ascii")

    def _load_json_object_template(
        self,
        key: str,
        *,
        default: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        raw = self.config.extra.get(key)
        if raw in (None, ""):
            parsed: Any = default
        elif isinstance(raw, dict):
            parsed = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(f"{self._get_log_prefix()} {key} JSON 解析失败: {exc}")
                return {}, f"{key} JSON 解析失败: {exc}"
        else:
            return {}, f"{key} 必须是 JSON 对象"

        rendered = self._render_template_value(parsed, context)
        if not isinstance(rendered, dict):
            return {}, f"{key} 必须是 JSON 对象"
        return rendered, None

    def _render_template_value(self, value: Any, context: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            rendered: dict[str, Any] = {}
            for key, item in value.items():
                if (
                    isinstance(item, str)
                    and (match := EXACT_PLACEHOLDER_PATTERN.match(item))
                    and match.group(1) in context
                    and context[match.group(1)] is None
                ):
                    continue
                rendered[self._render_template_string(str(key), context)] = (
                    self._render_template_value(item, context)
                )
            return rendered
        if isinstance(value, list):
            return [self._render_template_value(item, context) for item in value]
        if isinstance(value, str):
            if match := EXACT_PLACEHOLDER_PATTERN.match(value):
                return context.get(match.group(1), value)
            return self._render_template_string(value, context)
        return value

    def _render_template_string(self, value: str, context: dict[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            placeholder_value = context.get(match.group(1), match.group(0))
            if isinstance(placeholder_value, (dict, list)):
                return json.dumps(placeholder_value, ensure_ascii=False)
            if placeholder_value is None:
                return ""
            return str(placeholder_value)

        return PLACEHOLDER_PATTERN.sub(replace, value)

    def _normalize_headers(self, headers: dict[str, Any]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in headers.items():
            header_name = str(key).strip()
            header_value = "" if value is None else str(value)
            if not header_name:
                continue
            if (
                header_name.lower() == "authorization"
                and header_value.strip().lower() == "bearer"
            ):
                continue
            normalized[header_name] = header_value
        return normalized

    def _parse_json_response(self, body: bytes) -> tuple[Any, str | None]:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return None, "响应不是 JSON 且不是图片数据"

        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            logger.warning(f"{self._get_log_prefix()} 响应 JSON 解析失败: {exc}")
            return None, f"响应 JSON 解析失败: {exc}"

    def _extract_error_message(self, response_data: Any) -> str | None:
        configured_path = str(self.config.extra.get("error_result_path") or "").strip()
        paths = [configured_path] if configured_path else ["error.message", "error"]
        for path in paths:
            for value in self._extract_path_values(response_data, path):
                if value in (None, "", [], {}):
                    continue
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)
                return str(value)
        return None

    async def _extract_images(
        self, response_data: Any, task_id: str | None
    ) -> list[bytes] | None:
        images: list[bytes] = []
        result_type = str(self.config.extra.get("image_result_type") or "auto").strip()
        for path in self._image_result_paths():
            for value in self._extract_path_values(response_data, path):
                images.extend(
                    await self._decode_image_value(value, result_type, task_id)
                )
        return images or None

    def _image_result_paths(self) -> list[str]:
        raw = self.config.extra.get("image_result_path") or self.config.extra.get(
            "image_result_paths"
        )
        if isinstance(raw, list):
            paths = [str(item).strip() for item in raw]
        else:
            raw_text = str(raw or "data.*.b64_json")
            paths = [item.strip() for item in re.split(r"[\n,，]+", raw_text)]
        return [path for path in paths if path]

    def _extract_path_values(self, data: Any, path: str) -> list[Any]:
        normalized_path = path.strip()
        if not normalized_path or normalized_path == "$":
            return [data]
        if normalized_path.startswith("$."):
            normalized_path = normalized_path[2:]

        values = [data]
        for part in normalized_path.split("."):
            if not part:
                continue
            next_values: list[Any] = []
            for value in values:
                if part == "*":
                    if isinstance(value, dict):
                        next_values.extend(value.values())
                    elif isinstance(value, list):
                        next_values.extend(value)
                    continue
                if isinstance(value, dict) and part in value:
                    next_values.append(value[part])
                elif isinstance(value, list) and part.isdigit():
                    index = int(part)
                    if 0 <= index < len(value):
                        next_values.append(value[index])
            values = next_values
            if not values:
                break
        return values

    async def _decode_image_value(
        self, value: Any, result_type: str, task_id: str | None
    ) -> list[bytes]:
        if value is None:
            return []
        if isinstance(value, bytes):
            return [value]
        if isinstance(value, list):
            images: list[bytes] = []
            for item in value:
                images.extend(
                    await self._decode_image_value(item, result_type, task_id)
                )
            return images
        if isinstance(value, dict):
            return await self._decode_image_dict(value, result_type, task_id)
        if not isinstance(value, str):
            return []

        raw = value.strip()
        if not raw:
            return []
        if raw.startswith(("http://", "https://")):
            downloaded = await self._download_image_from_url(raw, task_id)
            return [downloaded] if downloaded else []
        if raw.startswith("data:image/"):
            decoded = self._decode_base64(raw)
            return [decoded] if decoded else []
        if result_type in {"auto", "base64", "b64_json"}:
            decoded = self._decode_base64(raw)
            return [decoded] if decoded else []
        return []

    async def _decode_image_dict(
        self, value: dict[str, Any], result_type: str, task_id: str | None
    ) -> list[bytes]:
        candidates: list[Any] = []
        for key in ("b64_json", "base64", "image", "data", "url"):
            if key in value:
                candidates.append(value[key])
        image_url = value.get("image_url")
        if isinstance(image_url, dict):
            candidates.append(image_url.get("url"))
        elif image_url:
            candidates.append(image_url)

        images: list[bytes] = []
        for candidate in candidates:
            images.extend(
                await self._decode_image_value(candidate, result_type, task_id)
            )
        return images

    def _decode_base64(self, value: str) -> bytes | None:
        raw = value.strip()
        if ";base64," in raw:
            raw = raw.partition(";base64,")[2]
        raw = re.sub(r"\s+", "", raw)
        if not raw:
            return None
        padding = "=" * (-len(raw) % 4)
        try:
            return base64.b64decode(raw + padding)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"{self._get_log_prefix()} Base64 解码失败: {safe_log_error_body(exc)}"
            )
            return None

    async def _download_image_from_url(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        prefix = self._get_log_prefix(task_id)
        try:
            async with self._get_session().get(
                url,
                proxy=self.proxy,
                timeout=self._get_download_timeout(),
            ) as response:
                if response.status == 200:
                    return await response.read()
                logger.error(
                    f"{prefix} 下载图像失败: {response.status} - {safe_log_url(url)}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 下载图像出错: {safe_log_error_body(exc)}")
        return None
