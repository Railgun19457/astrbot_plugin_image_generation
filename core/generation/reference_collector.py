"""Reference image collection helpers."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from ..shared.logging import log_prefix, mask_sensitive, safe_log_text, safe_log_url
from ..shared.types import ImageCapability, ImageData

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

    from .image_processor import ImageProcessor


LOG = log_prefix("Reference")
MAX_CONTEXT_IMAGE_SCAN_MESSAGES = 20


class ContextReferenceImageNotFoundError(Exception):
    """Raised when an image edit requested chat context but none was usable."""


def ensure_image_data(item: ImageData | tuple[bytes, str]) -> ImageData:
    """Normalize legacy (data, mime) tuples into ImageData."""
    if isinstance(item, ImageData):
        return item
    data, mime = item
    return ImageData(data=data, mime_type=mime)


def normalize_string_items(raw: Any) -> list[str]:
    """Normalize one or many string-like values from tool arguments."""
    if raw is None:
        return []
    if isinstance(raw, str):
        item = raw.strip()
        return [item] if item else []
    if isinstance(raw, dict):
        for key in ("url", "path", "file", "name"):
            if items := normalize_string_items(raw.get(key)):
                return items
        return []
    if isinstance(raw, Iterable):
        items: list[str] = []
        for value in raw:
            items.extend(normalize_string_items(value))
        return items
    item = str(raw).strip()
    return [item] if item else []


def extract_latest_user_context_images(
    messages: Any,
    *,
    message_limit: int = MAX_CONTEXT_IMAGE_SCAN_MESSAGES,
) -> list[str]:
    """Return image references from the latest recent image-bearing user message."""
    if (
        not isinstance(messages, Iterable)
        or isinstance(
            messages,
            (str, bytes, bytearray, dict),
        )
        or message_limit <= 0
    ):
        return []

    recent_messages = deque(messages, maxlen=message_limit)
    for message in reversed(recent_messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(
                message,
                "role",
                None,
            )
        )
        if role != "user":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(
                message,
                "content",
                None,
            )
        )
        if not isinstance(content, Iterable) or isinstance(
            content,
            (str, bytes, bytearray, dict),
        ):
            continue

        references: list[str] = []
        for part in content:
            part_type = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(
                    part,
                    "type",
                    None,
                )
            )
            if part_type != "image_url":
                continue
            image_url = (
                part.get("image_url")
                if isinstance(part, dict)
                else getattr(part, "image_url", None)
            )
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            elif not isinstance(image_url, str):
                image_url = getattr(image_url, "url", None)
            if isinstance(image_url, str) and image_url.strip():
                references.append(image_url.strip())
        if references:
            return references
    return []


def resolve_avatar_user_id(event: Any, ref: str) -> str | None:
    """Resolve an avatar reference into a platform user id."""
    normalized = ref.strip().lower()
    if not normalized:
        return None
    if normalized == "self" and hasattr(event, "get_self_id"):
        return str(event.get_self_id())
    if normalized == "sender" and hasattr(event, "get_sender_id"):
        return str(event.get_sender_id() or event.unified_msg_origin)

    cleaned = normalized.removeprefix("qq:").removeprefix("@").strip()
    if cleaned.isdigit():
        return cleaned
    return None


def deduplicate_reference_images(
    images_data: list[ImageData],
    *,
    task_id: str | None = None,
    log_context: str = "Reference",
) -> list[ImageData]:
    """Remove duplicate reference images by content hash."""
    if len(images_data) < 2:
        return images_data

    unique_images: list[ImageData] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    for image in images_data:
        image = ensure_image_data(image)
        digest = hashlib.sha256(image.data).hexdigest()
        if digest in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(digest)
        unique_images.append(image)

    if duplicate_count:
        task_log = log_prefix(log_context, task_id) if task_id else LOG
        logger.debug(f"{task_log} 已忽略 {duplicate_count} 张重复参考图")
    return unique_images


async def collect_reference_images_from_personas(
    image_processor: ImageProcessor,
    persona_images: list[tuple[str, str]],
    *,
    task_id: str | None = None,
    log_context: str = "Reference",
    workspace_dir: str | None = None,
) -> list[ImageData]:
    """Download all configured persona reference images."""
    images_data: list[ImageData] = []
    task_log = log_prefix(log_context, task_id) if task_id else LOG
    for persona_name, persona_image in persona_images:
        if persona_image_data := await image_processor.download_image(
            persona_image,
            workspace_dir=workspace_dir,
        ):
            images_data.append(persona_image_data)
        else:
            logger.warning(
                f"{task_log} 人设参考图获取失败: {safe_log_text(persona_name)}"
            )
    return images_data


async def download_reference_images(
    image_processor: ImageProcessor,
    references: Any,
    *,
    reference_label: str,
    task_id: str | None = None,
    log_context: str = "Reference",
    workspace_dir: str | None = None,
) -> list[ImageData]:
    """Download explicit reference images from URLs or local file paths."""
    images_data: list[ImageData] = []
    task_log = log_prefix(log_context, task_id) if task_id else LOG
    for reference in normalize_string_items(references):
        if image_data := await image_processor.download_image(
            reference,
            workspace_dir=workspace_dir,
        ):
            images_data.append(image_data)
        else:
            logger.warning(
                f"{task_log} {reference_label}参考图获取失败: {safe_log_url(reference)}"
            )
    return images_data


async def collect_command_reference_images(
    image_processor: ImageProcessor,
    event: AstrMessageEvent,
    persona_images: list[tuple[str, str]],
    *,
    task_id: str,
) -> list[ImageData]:
    """Collect command persona and message reference images."""
    workspace_dir = image_processor.workspace_dir_for_origin(
        getattr(event, "unified_msg_origin", None)
    )
    images_data = await collect_reference_images_from_personas(
        image_processor,
        persona_images,
        task_id=task_id,
        log_context="Task",
        workspace_dir=workspace_dir,
    )
    images_data.extend(await image_processor.fetch_images_from_event(event))
    return deduplicate_reference_images(
        images_data,
        task_id=task_id,
        log_context="Task",
    )


async def collect_tool_reference_images(
    image_processor: ImageProcessor,
    event: Any,
    *,
    capabilities: ImageCapability,
    reference_images: Any = None,
    context_images: Any = None,
    cached_context_images: list[ImageData] | None = None,
    use_context_images: bool = False,
    avatar_references: Any = None,
    persona_images: list[tuple[str, str]] | None = None,
    task_id: str | None = None,
) -> list[ImageData]:
    """Collect LLM tool message, persona, URL/path, and avatar reference images."""
    task_log = log_prefix("LLMTool", task_id) if task_id else LOG
    if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
        if (
            reference_images
            or use_context_images
            or avatar_references
            or persona_images
        ):
            logger.warning(f"{task_log} 当前适配器不支持参考图，已忽略工具参考图参数")
        return []

    images_data: list[ImageData] = []
    avatar_user_ids: set[str] = set()
    workspace_dir = image_processor.workspace_dir_for_origin(
        getattr(event, "unified_msg_origin", None)
    )

    context_source_images: list[ImageData] = []
    if use_context_images:
        context_source_images = await image_processor.fetch_images_from_event(event)
        if context_images and not context_source_images:
            context_source_images = await download_reference_images(
                image_processor,
                context_images,
                reference_label="上下文",
                task_id=task_id,
                log_context="LLMTool",
                workspace_dir=workspace_dir,
            )
        if cached_context_images and not context_source_images:
            context_source_images = [
                ensure_image_data(image) for image in cached_context_images
            ]
        if not context_source_images:
            raise ContextReferenceImageNotFoundError

    if persona_images:
        images_data.extend(
            await collect_reference_images_from_personas(
                image_processor,
                persona_images,
                task_id=task_id,
                log_context="LLMTool",
                workspace_dir=workspace_dir,
            )
        )

    images_data.extend(context_source_images)

    images_data.extend(
        await download_reference_images(
            image_processor,
            reference_images,
            reference_label="显式",
            task_id=task_id,
            log_context="LLMTool",
            workspace_dir=workspace_dir,
        )
    )

    for ref in normalize_string_items(avatar_references):
        user_id = resolve_avatar_user_id(event, ref)
        if not user_id or user_id in avatar_user_ids:
            continue
        avatar_user_ids.add(user_id)
        if avatar_data := await image_processor.get_avatar(user_id):
            images_data.append(ImageData(data=avatar_data, mime_type="image/jpeg"))
            logger.debug(
                f"{task_log} 已添加 {mask_sensitive(user_id)} 的头像作为参考图"
            )

    return deduplicate_reference_images(
        images_data,
        task_id=task_id,
        log_context="LLMTool",
    )
