"""Bounded in-memory cache for recent conversation images."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..shared.types import ImageData

DEFAULT_CONTEXT_IMAGE_TTL_SECONDS = 30 * 60
DEFAULT_CONTEXT_IMAGE_MAX_SESSIONS = 32
DEFAULT_CONTEXT_IMAGE_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    expires_at: float
    images: tuple[ImageData, ...]
    size_bytes: int


class RecentContextImageCache:
    """Keep the latest image-bearing request for a bounded set of sessions."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_CONTEXT_IMAGE_TTL_SECONDS,
        max_sessions: int = DEFAULT_CONTEXT_IMAGE_MAX_SESSIONS,
        max_bytes: int = DEFAULT_CONTEXT_IMAGE_MAX_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = max(0.0, ttl_seconds)
        self._max_sessions = max(0, max_sessions)
        self._max_bytes = max(0, max_bytes)
        self._clock = clock
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._size_bytes = 0

    def put(self, session_id: str, images: Iterable[ImageData]) -> None:
        """Replace one session's cached image set and enforce cache bounds."""
        session_id = session_id.strip()
        image_items = tuple(images)
        size_bytes = sum(len(image.data) for image in image_items)
        self._remove(session_id)
        self._prune_expired()

        if (
            not session_id
            or not image_items
            or self._ttl_seconds <= 0
            or self._max_sessions <= 0
            or size_bytes > self._max_bytes
        ):
            return

        self._entries[session_id] = _CacheEntry(
            expires_at=self._clock() + self._ttl_seconds,
            images=image_items,
            size_bytes=size_bytes,
        )
        self._size_bytes += size_bytes
        while (
            len(self._entries) > self._max_sessions
            or self._size_bytes > self._max_bytes
        ):
            oldest_session_id = next(iter(self._entries))
            self._remove(oldest_session_id)

    def get(self, session_id: str) -> list[ImageData]:
        """Return one session's cached images and refresh its LRU position."""
        self._prune_expired()
        entry = self._entries.get(session_id.strip())
        if not entry:
            return []
        self._entries.move_to_end(session_id.strip())
        return list(entry.images)

    def clear(self) -> None:
        """Drop all cached images."""
        self._entries.clear()
        self._size_bytes = 0

    def _remove(self, session_id: str) -> None:
        entry = self._entries.pop(session_id, None)
        if entry:
            self._size_bytes -= entry.size_bytes

    def _prune_expired(self) -> None:
        now = self._clock()
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for session_id in expired:
            self._remove(session_id)
