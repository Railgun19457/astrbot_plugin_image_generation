from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import re
from typing import Any

import aiohttp

from astrbot.api import logger

from ..shared.constants import DEFAULT_DOWNLOAD_TIMEOUT
from ..shared.logging import (
    format_cn_log_fields,
    format_log_event,
    format_seconds,
    format_sub_request,
    is_sensitive_log_field,
    log_prefix,
    mask_sensitive,
    safe_log_error_body,
    safe_log_text,
    safe_user_error_detail,
    safe_log_url,
)
from ..shared.types import (
    AdapterConfig,
    GenerationRequest,
    GenerationResult,
    ImageCapability,
)

API_STATUS_ERROR_PATTERN = re.compile(r"API 错误\s*\((\d{3})\)")
DEBUG_JSON_STRING_LIMIT = 1000
DEBUG_JSON_EDGE_CHARS = 120
DEBUG_JSON_LIST_LIMIT = 50
BASE64_VALUE_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


class BaseImageAdapter(abc.ABC):
    """Base class for image generation adapters."""

    requires_api_key = True

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.api_keys = config.api_keys or []
        self.current_key_index = 0
        self.base_url = (config.base_url or "").rstrip("/")
        self.model = config.model
        self.proxy = config.proxy
        self.timeout = config.timeout
        self.download_timeout = DEFAULT_DOWNLOAD_TIMEOUT
        self.max_retry_attempts = max(1, config.max_retry_attempts)
        self.debug_request_logging = config.debug_request_logging
        self.show_user_error_details = config.show_user_error_details
        self.non_retryable_status_codes = set(config.non_retryable_status_codes)
        self.non_retryable_error_keywords = [
            keyword.lower()
            for keyword in config.non_retryable_error_keywords
            if keyword
        ]
        self.safety_settings = config.safety_settings
        self._session: aiohttp.ClientSession | None = None

    @abc.abstractmethod
    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""

    def _get_configured_capabilities(self) -> ImageCapability:
        """Build adapter capabilities from configuration options."""
        capability_map: dict[str, ImageCapability] = {
            "text_to_image": ImageCapability.TEXT_TO_IMAGE,
            "image_to_image": ImageCapability.IMAGE_TO_IMAGE,
            "aspect_ratio": ImageCapability.ASPECT_RATIO,
            "resolution": ImageCapability.RESOLUTION,
        }

        result = ImageCapability.NONE
        for key, capability_flag in capability_map.items():
            if self.config.capability_options.get(key, False):
                result |= capability_flag
        return result

    async def close(self) -> None:
        """Close the underlying HTTP session."""

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Return or create the HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_current_api_key(self) -> str:
        """Return the current API key."""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _get_log_prefix(self, task_id: str | None = None) -> str:
        """Return the adapter log prefix."""
        adapter_name = self.__class__.__name__.replace("Adapter", "")
        return log_prefix(adapter_name, task_id)

    def _format_request_context(self, request: GenerationRequest) -> dict[str, str]:
        """Build reusable context fields for adapter request logs.

        Args:
            request: Generation request currently sent to the provider.

        Returns:
            A mapping of optional Chinese log fields.
        """
        return {
            "子请求": format_sub_request(request.batch_index, request.batch_count),
            "模型": safe_log_text(self.model),
            "参考图": f"{len(request.images)}张",
        }

    def _log_request_overview(
        self,
        request: GenerationRequest,
        url: str,
        *,
        method: str = "POST",
        payload: Any = None,
        form_fields: list[str] | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Log safe adapter request metadata without dumping payload values.

        Args:
            request: Generation request currently sent to the provider.
            url: Provider request URL.
            method: HTTP method.
            payload: Optional JSON payload used only for field names.
            form_fields: Optional form field names.
            extra: Additional safe metadata fields.
        """
        fields: dict[str, object] = {
            **self._format_request_context(request),
            "方法": method,
            "地址": safe_log_url(url),
        }
        if not self.debug_request_logging:
            if isinstance(payload, dict):
                fields["请求字段"] = list(payload.keys())
            elif form_fields:
                fields["表单字段"] = form_fields
        if extra:
            fields.update(extra)
        logger.debug(
            f"{self._get_log_prefix(request.task_id)} "
            + format_log_event("适配器请求", **fields)
        )

    def _log_response_status(
        self,
        request: GenerationRequest,
        status: int,
        duration: float,
    ) -> None:
        """Log safe adapter response status metadata.

        Args:
            request: Generation request currently sent to the provider.
            status: HTTP response status code.
            duration: Request duration in seconds.
        """
        logger.debug(
            f"{self._get_log_prefix(request.task_id)} "
            + format_log_event(
                "适配器请求状态",
                **self._format_request_context(request),
                状态码=status,
                耗时=format_seconds(duration),
            )
        )

    def _log_api_error(
        self,
        request: GenerationRequest,
        status: int,
        duration: float,
        error_body: object,
        *,
        label: str = "API 错误",
    ) -> None:
        """Log provider API failure with sanitized error body.

        Args:
            request: Generation request currently sent to the provider.
            status: HTTP response status code.
            duration: Request duration in seconds.
            error_body: Provider error response body or exception.
            label: Human-readable error label.
        """
        logger.error(
            f"{self._get_log_prefix(request.task_id)} "
            + format_log_event(
                "适配器请求失败",
                类型=label,
                **self._format_request_context(request),
                状态码=status,
                耗时=format_seconds(duration),
                错误=safe_log_error_body(error_body),
            )
        )

    def _log_request_exception(
        self,
        request: GenerationRequest,
        duration: float,
        exc: object,
        *,
        label: str = "请求异常",
    ) -> None:
        """Log provider request exception with sanitized error text.

        Args:
            request: Generation request currently sent to the provider.
            duration: Request duration in seconds.
            exc: Exception or error object.
            label: Human-readable error label.
        """
        error_text = safe_log_error_body(exc)
        if not error_text:
            error_text = type(exc).__name__
        logger.error(
            f"{self._get_log_prefix(request.task_id)} "
            + format_log_event(
                "适配器请求异常",
                类型=label,
                **self._format_request_context(request),
                耗时=format_seconds(duration),
                错误=error_text,
            )
        )

    def _format_api_error_message(
        self,
        status: int,
        error_body: object | None = None,
    ) -> str:
        """Format user-facing API error with optional sanitized details.

        Args:
            status: HTTP response status code.
            error_body: Raw provider error body to summarize when enabled.

        Returns:
            A concise user-facing API error message.
        """
        message = f"API 错误 ({status})"
        if self.show_user_error_details and error_body:
            detail = safe_user_error_detail(error_body, 600)
            if detail:
                return f"{message}: {detail}"
        return message

    def _get_timeout(self) -> aiohttp.ClientTimeout:
        """Return the request timeout configuration."""
        return aiohttp.ClientTimeout(total=self.timeout)

    def _get_download_timeout(self) -> aiohttp.ClientTimeout:
        """Return the download timeout configuration."""
        return aiohttp.ClientTimeout(total=self.download_timeout)

    def _log_debug_json(
        self, label: str, value: Any, task_id: str | None = None
    ) -> None:
        """Log JSON debug data when enabled, compacting long strings."""
        if not self.debug_request_logging:
            return

        prefix = self._get_log_prefix(task_id)
        safe_value = self._sanitize_debug_json(value)
        json_text = json.dumps(
            safe_value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        logger.debug(f"{prefix} {label} JSON: {json_text}")

    def _sanitize_debug_json(self, value: Any, field_name: str | None = None) -> Any:
        """Preserve JSON structure while truncating image Base64 and long strings."""
        if field_name and is_sensitive_log_field(field_name):
            return mask_sensitive(value)
        if isinstance(value, dict):
            return {
                key: self._sanitize_debug_json(item, field_name=str(key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            items = [
                self._sanitize_debug_json(item, field_name=field_name)
                for item in value[:DEBUG_JSON_LIST_LIMIT]
            ]
            if len(value) > DEBUG_JSON_LIST_LIMIT:
                omitted_count = len(value) - DEBUG_JSON_LIST_LIMIT
                items.append(f"<list truncated: {omitted_count} items omitted>")
            return items
        if isinstance(value, str):
            return self._sanitize_debug_string(value)
        return value

    def _sanitize_debug_string(self, value: str) -> str:
        """Truncate strings that could flood debug JSON logs."""
        if len(value) <= DEBUG_JSON_STRING_LIMIT:
            return value

        digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]
        if value.startswith("data:"):
            media_type = value.split(",", 1)[0]
            return (
                f"<data-url omitted: {media_type}, len={len(value)}, sha256={digest}>"
            )
        if BASE64_VALUE_RE.fullmatch(value):
            return f"<base64 omitted: len={len(value)}, sha256={digest}>"

        head = value[:DEBUG_JSON_EDGE_CHARS]
        tail = value[-DEBUG_JSON_EDGE_CHARS:]
        return f"{head}<string truncated: len={len(value)}, sha256={digest}>{tail}"

    def _log_debug_json_text(
        self, label: str, value: str, task_id: str | None = None
    ) -> None:
        """Log text responses as JSON debug data when they can be parsed."""
        if not self.debug_request_logging:
            return

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return
        self._log_debug_json(label, parsed, task_id)

    async def _read_response_json(
        self, response: aiohttp.ClientResponse, task_id: str | None = None
    ) -> Any:
        """Read response JSON and log a safe debug summary when enabled."""
        data = await response.json()
        self._log_debug_json("响应", data, task_id)
        return data

    def _rotate_api_key(self) -> None:
        """Rotate to the next API key."""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.debug(
                f"{self._get_log_prefix()} 轮换 API Key -> 索引 {self.current_key_index}"
            )

    def update_model(self, model: str) -> None:
        """Update the active model."""
        self.model = model

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run the image generation template method with retry logic.

        Subclasses implement provider-specific logic in `_generate_once()`.
        Override `_pre_generate()` when provider-specific prechecks are needed.
        """
        if self.requires_api_key and not self.api_keys:
            return GenerationResult(images=None, error="未配置 API Key")

        prefix = self._get_log_prefix(request.task_id)
        logger.debug(
            f"{prefix} 准备生图请求: "
            + format_cn_log_fields(
                模型=self.model,
                子请求=format_sub_request(request.batch_index, request.batch_count),
                参考图=f"{len(request.images)}张",
                最大重试=f"{self.max_retry_attempts}次",
            )
        )

        # Run provider-specific prechecks when implemented by subclasses.
        pre_result = self._pre_generate(request)
        if pre_result is not None:
            logger.warning(
                f"{prefix} 生图请求预检查未通过: {safe_log_text(pre_result.error)}"
            )
            return pre_result

        last_error = "未配置 API Key"
        for attempt in range(self.max_retry_attempts):
            if request.retry_status_callback:
                request.retry_status_callback(attempt + 1, self.max_retry_attempts)
            if attempt:
                logger.debug(
                    f"{prefix} 重试生图请求: "
                    + format_cn_log_fields(
                        子请求=format_sub_request(
                            request.batch_index,
                            request.batch_count,
                        ),
                        重试=f"{attempt + 1}/{self.max_retry_attempts}",
                    )
                )

            images, err = await self._generate_once(request)
            if images is not None:
                return GenerationResult(images=images, error=None)

            last_error = err or "生成失败"
            logger.debug(
                f"{prefix} 生图请求尝试失败 ({attempt + 1}/{self.max_retry_attempts}): "
                f"{safe_log_text(last_error, 200)}"
            )
            if not self._is_retryable_error(last_error):
                logger.warning(
                    f"{prefix} 生图请求错误不可重试，停止重试: "
                    + format_cn_log_fields(
                        子请求=format_sub_request(
                            request.batch_index,
                            request.batch_count,
                        ),
                        错误=safe_log_error_body(last_error, 200),
                    )
                )
                return GenerationResult(images=None, error=last_error)
            if attempt < self.max_retry_attempts - 1:
                self._rotate_api_key()
                # Apply exponential backoff after rotating through all keys.
                if (attempt + 1) % max(1, len(self.api_keys)) == 0:
                    await asyncio.sleep(
                        min(2 ** ((attempt + 1) // len(self.api_keys)), 10)
                    )

        logger.warning(
            f"{prefix} 生图请求全部重试失败: "
            + format_cn_log_fields(
                子请求=format_sub_request(request.batch_index, request.batch_count),
                重试=f"{self.max_retry_attempts}/{self.max_retry_attempts}",
                错误=safe_log_error_body(last_error, 200),
            )
        )
        return GenerationResult(images=None, error=f"重试失败: {last_error}")

    def _is_retryable_error(self, error: str) -> bool:
        """Return whether an adapter error should be retried."""
        if not error:
            return True

        if status_code := self._extract_status_code(error):
            return status_code not in self.non_retryable_status_codes

        normalized_error = error.lower()
        return not any(
            keyword in normalized_error for keyword in self.non_retryable_error_keywords
        )

    def _extract_status_code(self, error: str) -> int | None:
        """Extract an HTTP status code from a normalized adapter error."""
        match = API_STATUS_ERROR_PATTERN.search(error)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _pre_generate(self, _request: GenerationRequest) -> GenerationResult | None:
        """Run pre-generation validation.

        Returns:
            None when validation passes, otherwise a result to return early.
        """
        return None

    @abc.abstractmethod
    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one provider-specific generation request.

        Returns:
            A tuple of generated images and an error message.
        """
