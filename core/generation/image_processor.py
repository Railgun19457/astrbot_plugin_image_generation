"""Image processing helpers for download, extraction, and temporary storage."""

from __future__ import annotations

import base64
import binascii
import hashlib
import ntpath
import os
import posixpath
import re
import time
from collections.abc import Iterable
from io import BytesIO
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from PIL import Image, UnidentifiedImageError

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_workspaces_path
from astrbot.core.utils.io import download_image_by_url

from ..shared.logging import (
    log_prefix,
    mask_sensitive,
    safe_log_error_body,
    safe_log_url,
)
from ..shared.types import ImageData

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


LOG = log_prefix("ImageProcessor")
ALLOWED_IMAGE_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/heic",
        "image/heif",
    }
)
PIL_VERIFIABLE_IMAGE_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)
GENERATED_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


class ImageProcessor:
    """Download, extract, validate, and store image files."""

    def __init__(
        self,
        temp_dir: str,
        max_image_size_mb: int,
        local_base_dir: str | None = None,
        allowed_local_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
    ) -> None:
        self._temp_dir = os.path.realpath(temp_dir)
        self._max_image_size_mb = max_image_size_mb
        self._local_base_dir = (
            os.path.realpath(local_base_dir) if local_base_dir else ""
        )
        base_dirs: list[str | os.PathLike[str] | None] = [
            self._local_base_dir,
            self._temp_dir,
        ]
        if allowed_local_base_dirs:
            base_dirs.extend(allowed_local_base_dirs)
        self._allowed_local_base_dirs = self._normalize_allowed_base_dirs(base_dirs)
        os.makedirs(self._temp_dir, exist_ok=True)

    def update_settings(self, max_image_size_mb: int | None = None) -> None:
        """Update runtime image processing settings."""
        if max_image_size_mb is not None:
            self._max_image_size_mb = max_image_size_mb

    @property
    def temp_dir(self) -> str:
        """Return the temporary directory path."""
        return self._temp_dir

    def workspace_dir_for_origin(self, unified_msg_origin: str | None) -> str | None:
        """Return the AstrBot local workspace path for a session origin."""
        origin = str(unified_msg_origin or "").strip()
        if not origin:
            return None
        normalized_origin = re.sub(r"[^A-Za-z0-9._-]+", "_", origin)
        normalized_origin = normalized_origin or "unknown"
        return os.path.realpath(
            os.path.join(get_astrbot_workspaces_path(), normalized_origin)
        )

    def _normalize_allowed_base_dirs(
        self,
        paths: Iterable[str | os.PathLike[str] | None],
    ) -> tuple[str, ...]:
        """Normalize and deduplicate allowed local directory roots."""
        result: list[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            if not raw_path:
                continue
            path = os.path.realpath(os.fspath(raw_path))
            normalized = os.path.normcase(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(path)
        return tuple(result)

    def _is_path_within_allowed_dirs(
        self,
        path: str,
        allowed_base_dirs: tuple[str, ...],
    ) -> bool:
        """Return whether path is inside one of the configured safe roots."""
        normalized_path = os.path.normcase(os.path.realpath(path))
        for base_dir in allowed_base_dirs:
            normalized_base = os.path.normcase(os.path.realpath(base_dir))
            try:
                if (
                    os.path.commonpath([normalized_base, normalized_path])
                    == normalized_base
                ):
                    return True
            except ValueError:
                continue
        return False

    def _resolve_local_path(
        self,
        value: str,
        *,
        workspace_dir: str | None = None,
    ) -> str | None:
        """Resolve a safe local image path inside workspace/temp/plugin data dirs."""
        value = self._normalize_local_path_value(value)
        if not value:
            return None

        parsed = urlparse(value)
        if (
            parsed.scheme
            and parsed.scheme.lower() != "file"
            and not self._is_absolute_path(value)
        ):
            return None

        allowed_base_dirs = self._allowed_local_base_dirs
        if workspace_dir:
            allowed_base_dirs = self._normalize_allowed_base_dirs(
                (*allowed_base_dirs, workspace_dir)
            )
        candidates: list[str] = []
        if self._is_absolute_path(value):
            candidates.append(value)
        elif self._local_base_dir and value.replace("\\", "/").startswith("files/"):
            candidates.append(os.path.join(self._local_base_dir, value))
        elif workspace_dir:
            candidates.append(os.path.join(workspace_dir, value))

        for candidate in candidates:
            path = os.path.realpath(candidate)
            if not self._is_path_within_allowed_dirs(path, allowed_base_dirs):
                logger.warning(
                    f"{LOG} 本地参考图路径不在允许目录内: {safe_log_url(path)}"
                )
                continue
            if os.path.exists(path) and os.path.isfile(path):
                return path
            logger.warning(f"{LOG} 本地参考图不存在或不是文件: {safe_log_url(path)}")
        return None

    def _is_absolute_path(self, value: str) -> bool:
        """Return whether a value is a Linux/Windows absolute path."""
        return os.path.isabs(value) or ntpath.isabs(value) or posixpath.isabs(value)

    def _is_network_url(self, value: str) -> bool:
        """Return whether a value is an HTTP(S) image source."""
        scheme = urlparse(value).scheme.lower()
        return scheme in {"http", "https"}

    def _normalize_local_path_value(self, value: str) -> str:
        """Normalize local file paths, including file:// URI values."""
        value = value.strip()
        if not value.lower().startswith("file:"):
            return value

        parsed = urlparse(value)
        if parsed.scheme.lower() != "file":
            return value

        netloc = unquote(parsed.netloc)
        path = unquote(parsed.path)
        if netloc and netloc.lower() != "localhost" and path:
            path = f"//{netloc}{path}"
        elif netloc and netloc.lower() != "localhost":
            path = netloc

        # AstrBot or platform adapters may pass file:///E:\path or file:///E:/path.
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return os.path.normpath(path)

    async def download_image(
        self,
        url: str,
        *,
        workspace_dir: str | None = None,
    ) -> ImageData | None:
        """Download or read an image and return normalized image data."""
        try:
            url = url.strip()
            if not url:
                return None

            data: bytes | None = None
            source_url = url if self._is_network_url(url) else None
            if url.lower().startswith("data:image/"):
                header, separator, payload = url.partition(",")
                declared_mime = header[5:].split(";", 1)[0].lower()
                if (
                    not separator
                    or ";base64" not in header.lower()
                    or declared_mime not in ALLOWED_IMAGE_MIME_TYPES
                ):
                    logger.warning(f"{LOG} 不支持的图片 Data URL")
                    return None
                max_bytes = self._max_image_size_mb * 1024 * 1024
                max_encoded_length = ((max_bytes + 2) // 3) * 4 + 8
                if len(payload) > max_encoded_length:
                    logger.warning(
                        f"{LOG} 图片超过大小限制 ({self._max_image_size_mb}MB)"
                    )
                    return None
                try:
                    data = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    logger.warning(f"{LOG} 图片 Data URL Base64 无效")
                    return None
                source_url = None
            elif local_path := self._resolve_local_path(
                url,
                workspace_dir=workspace_dir,
            ):
                source_url = None
                with open(local_path, "rb") as f:
                    data = f.read()
            else:
                if not self._is_network_url(url):
                    logger.warning(f"{LOG} 不支持的图片来源: {safe_log_url(url)}")
                    return None
                # Use the plugin temporary directory for downloaded references.
                file_name = f"ref_{hashlib.md5(url.encode()).hexdigest()[:10]}"
                path = os.path.join(self._temp_dir, file_name)
                path = await download_image_by_url(url, path=path)
                if path:
                    with open(path, "rb") as f:
                        data = f.read()

            if not data:
                return None

            if len(data) > self._max_image_size_mb * 1024 * 1024:
                logger.warning(f"{LOG} 图片超过大小限制 ({self._max_image_size_mb}MB)")
                return None

            return self.validate_image_data(
                data,
                source_url=source_url,
                log_source=url,
            )
        except Exception as exc:
            logger.error(
                f"{LOG} 获取图片失败: {safe_log_url(url)} ({safe_log_error_body(exc)})"
            )
        return None

    def validate_image_data(
        self,
        data: bytes,
        *,
        source_url: str | None = None,
        log_source: str | None = None,
    ) -> ImageData | None:
        """Validate bytes as a supported image and return normalized image data."""
        mime = self._detect_mime_type(data)
        if not self._is_valid_image_data(data, mime):
            logger.warning(
                f"{LOG} 参考图文件类型不支持或内容不是有效图片: {safe_log_url(log_source or source_url or '<bytes>')}"
            )
            return None
        return ImageData(data=data, mime_type=mime, source_url=source_url)

    def _detect_mime_type(self, data: bytes) -> str:
        """Detect the image MIME type from magic bytes."""
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        elif data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif data.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        elif len(data) > 12 and data[4:8] == b"ftyp":
            brand = data[8:12]
            if brand in (b"heic", b"heix", b"heim", b"heis"):
                return "image/heic"
            if brand in (b"mif1", b"msf1", b"heif"):
                return "image/heif"
        return "application/octet-stream"

    def _is_valid_image_data(self, data: bytes, mime: str) -> bool:
        """Return whether bytes are a supported, real image payload."""
        if mime not in ALLOWED_IMAGE_MIME_TYPES:
            return False
        if mime not in PIL_VERIFIABLE_IMAGE_MIME_TYPES:
            return True
        try:
            with Image.open(BytesIO(data)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            return False
        return True

    async def get_avatar(self, user_id: str) -> bytes | None:
        """Fetch a user's avatar image bytes."""
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            file_name = f"avatar_{user_id}.jpg"
            path = os.path.join(self._temp_dir, file_name)
            path = await download_image_by_url(url, path=path)
            if path:
                with open(path, "rb") as f:
                    data = f.read()
                if self.validate_image_data(data, log_source=url):
                    return data
        except Exception as e:
            logger.debug(
                f"{LOG} 获取头像失败 (user_id={mask_sensitive(user_id)}): {safe_log_error_body(e)}"
            )
        return None

    def _message_body_leading_component(self, event: AstrMessageEvent):
        """Return the first non-reply, non-empty message component."""
        if not event.message_obj or not event.message_obj.message:
            return None

        for component in event.message_obj.message:
            if isinstance(component, Comp.Reply):
                continue
            if isinstance(component, Comp.Plain) and not component.text.strip():
                continue
            return component
        return None

    def _has_reply_from_bot(self, event: AstrMessageEvent, bot_self_id: str) -> bool:
        """Return whether the current event replies to a bot-sent message."""
        if not bot_self_id or not event.message_obj or not event.message_obj.message:
            return False

        for component in event.message_obj.message:
            if not isinstance(component, Comp.Reply):
                continue
            for value in (component.sender_id, component.qq):
                if value is not None and str(value).strip() == bot_self_id:
                    return True
        return False

    def _should_skip_leading_bot_at(
        self,
        event: AstrMessageEvent,
        bot_self_id: str,
    ) -> bool:
        """Return whether the leading bot mention is only the command trigger."""
        if not bot_self_id or self._has_reply_from_bot(event, bot_self_id):
            return False

        leading_component = self._message_body_leading_component(event)
        if not isinstance(leading_component, Comp.At):
            return False

        return str(leading_component.qq).strip() == bot_self_id

    def collect_event_image_urls(self, event: AstrMessageEvent) -> list[str]:
        """Return deduplicated image URLs from the current and replied message chain."""
        references: list[str] = []
        if not event.message_obj or not event.message_obj.message:
            return references

        components = list(event.message_obj.message)
        for component in components:
            image_components = []
            if isinstance(component, Comp.Image):
                image_components.append(component)
            elif isinstance(component, Comp.Reply) and component.chain:
                image_components.extend(
                    sub_comp
                    for sub_comp in component.chain
                    if isinstance(sub_comp, Comp.Image)
                )
            for image_component in image_components:
                url = image_component.url or image_component.file
                if url and str(url).strip():
                    references.append(str(url).strip())
        return list(dict.fromkeys(references))

    async def fetch_images_from_event(
        self,
        event: AstrMessageEvent,
        avatar_user_ids: set[str] | None = None,
    ) -> list[ImageData]:
        """Extract direct, replied, and mentioned-user images from an event."""
        images_data: list[ImageData] = []
        if avatar_user_ids is None:
            avatar_user_ids = set()

        if not event.message_obj or not event.message_obj.message:
            return images_data

        workspace_dir = self.workspace_dir_for_origin(
            getattr(event, "unified_msg_origin", None)
        )
        bot_self_id = str(event.get_self_id()) if hasattr(event, "get_self_id") else ""
        should_skip_leading_bot_at = self._should_skip_leading_bot_at(
            event,
            bot_self_id,
        )
        leading_bot_at_skipped = False

        for component in event.message_obj.message:
            try:
                if isinstance(component, Comp.Image):
                    # Handle directly sent images.
                    url = component.url or component.file
                    if url and (
                        data := await self.download_image(
                            url,
                            workspace_dir=workspace_dir,
                        )
                    ):
                        images_data.append(data)
                elif isinstance(component, Comp.Reply):
                    # Handle images inside replied messages.
                    if component.chain:
                        for sub_comp in component.chain:
                            if isinstance(sub_comp, Comp.Image):
                                url = sub_comp.url or sub_comp.file
                                if url and (
                                    data := await self.download_image(
                                        url,
                                        workspace_dir=workspace_dir,
                                    )
                                ):
                                    images_data.append(data)
                elif isinstance(component, Comp.At):
                    # Handle mentioned-user avatars.
                    if hasattr(component, "qq") and component.qq != "all":
                        uid = str(component.qq).strip()
                        if (
                            should_skip_leading_bot_at
                            and not leading_bot_at_skipped
                            and uid == bot_self_id
                        ):
                            leading_bot_at_skipped = True
                            continue
                        if uid in avatar_user_ids:
                            continue
                        avatar_user_ids.add(uid)
                        if avatar_data := await self.get_avatar(uid):
                            images_data.append(
                                ImageData(data=avatar_data, mime_type="image/jpeg")
                            )
            except Exception as e:
                logger.error(
                    f"{LOG} 提取消息组件图片失败: {safe_log_error_body(e)}",
                    exc_info=True,
                )
                continue
        return images_data

    def save_generated_image(self, task_id: str, img_bytes: bytes) -> str | None:
        """Save generated image bytes to the temporary directory."""
        try:
            mime = self._detect_mime_type(img_bytes)
            extension = GENERATED_IMAGE_EXTENSIONS.get(mime, ".png")
            file_name = f"gen_{task_id}_{int(time.time())}_{hashlib.md5(img_bytes).hexdigest()[:6]}{extension}"
            file_path = os.path.join(self._temp_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(img_bytes)
            return file_path
        except Exception as exc:
            logger.error(
                f"{LOG} 保存图片失败: {safe_log_error_body(exc)}", exc_info=True
            )
            return None
