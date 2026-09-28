from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class GenerationTaskStatus(str, Enum):
    """Lifecycle states for image generation tasks."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class GenerationTaskItemStatus(str, Enum):
    """Lifecycle states for one generation sub-request."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationTaskCreationError(Exception):
    """Error raised when a generation task cannot be accepted."""

    def __init__(self, code: str, message: str):
        """Initialize a generation task creation error.

        Args:
            code: Stable machine-readable error code.
            message: User-facing error message.
        """
        super().__init__(message)
        self.code = code
        self.message = message


ACTIVE_GENERATION_STATUSES = {
    GenerationTaskStatus.QUEUED,
    GenerationTaskStatus.RUNNING,
    GenerationTaskStatus.CANCELLING,
}

TERMINAL_GENERATION_STATUSES = {
    GenerationTaskStatus.SUCCEEDED,
    GenerationTaskStatus.FAILED,
    GenerationTaskStatus.CANCELLED,
}

ACTIVE_GENERATION_ITEM_STATUSES = {
    GenerationTaskItemStatus.PENDING,
    GenerationTaskItemStatus.RUNNING,
}

GENERATION_TASK_STATUS_LABELS = {
    GenerationTaskStatus.QUEUED: "排队中",
    GenerationTaskStatus.RUNNING: "运行中",
    GenerationTaskStatus.SUCCEEDED: "已完成",
    GenerationTaskStatus.FAILED: "失败",
    GenerationTaskStatus.CANCELLING: "取消中",
    GenerationTaskStatus.CANCELLED: "已取消",
}


@dataclass
class GenerationTaskItem:
    """Per-request generation result metadata."""

    index: int
    status: GenerationTaskItemStatus = GenerationTaskItemStatus.PENDING
    result_count: int = 0
    error: str = ""
    retry_attempts: int = 0
    max_retry_attempts: int = 0


@dataclass(frozen=True)
class GenerationQueueItem:
    """Queued generation task entry."""

    task_id: str
    coro_factory: Callable[[], Coroutine[Any, Any, Any]]


@dataclass
class GenerationTaskRecord:
    """In-memory metadata for one image generation task."""

    task_id: str
    source: str
    unified_msg_origin: str
    prompt_summary: str
    reference_image_count: int
    requested_count: int
    aspect_ratio: str
    resolution: str
    model: str = ""
    prompt: str = ""
    preset: str | None = None
    preset_label: str = "预设"
    status: GenerationTaskStatus = GenerationTaskStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = "任务已提交"
    error: str = ""
    result_count: int = 0
    result_paths: list[str] = field(default_factory=list)
    current_index: int = 0
    retry_attempt: int = 0
    max_retry_attempts: int = 0
    items: dict[int, GenerationTaskItem] = field(default_factory=dict)
    usage_scope: str = ""
    reserved_count: int = 0
    quota_date: str = ""
    is_usage_limit_admin: bool = False
    quota_released: bool = False
    quota_settled: bool = False
    task: asyncio.Task | None = field(default=None, repr=False, compare=False)

    @property
    def is_active(self) -> bool:
        """Return whether the task can still change state."""
        return self.status in ACTIVE_GENERATION_STATUSES

    @property
    def status_label(self) -> str:
        """Return a user-facing status label."""
        return GENERATION_TASK_STATUS_LABELS.get(self.status, self.status.value)

    @property
    def duration_seconds(self) -> float | None:
        """Return active execution duration, excluding queued time."""
        if not self.started_at:
            return None
        end_time = self.finished_at or datetime.now()
        return max(0.0, (end_time - self.started_at).total_seconds())

    @property
    def queued_seconds(self) -> float:
        """Return time spent since creation."""
        end_time = self.started_at or self.finished_at or datetime.now()
        return max(0.0, (end_time - self.created_at).total_seconds())

    @property
    def request_stats(self) -> dict[str, int]:
        """Return normalized sub-request progress statistics."""
        statuses = [item.status for item in self.items.values()]
        succeeded = statuses.count(GenerationTaskItemStatus.SUCCEEDED)
        failed = statuses.count(GenerationTaskItemStatus.FAILED)
        cancelled = statuses.count(GenerationTaskItemStatus.CANCELLED)
        running = statuses.count(GenerationTaskItemStatus.RUNNING)
        pending = statuses.count(GenerationTaskItemStatus.PENDING)
        finished = succeeded + failed + cancelled
        total = max(self.requested_count, len(statuses))
        return {
            "total": total,
            "finished": finished,
            "succeeded": succeeded,
            "failed": failed,
            "cancelled": cancelled,
            "running": running,
            "pending": pending,
            "result_count": self.result_count or len(self.result_paths),
        }


def coerce_generation_item_status(value: Any) -> GenerationTaskItemStatus:
    """Coerce a raw item status value to a known enum value.

    Args:
        value: Raw item status from runtime or persisted history.

    Returns:
        A valid generation item status. Unknown values fall back to pending so
        unfinished legacy records remain visible.
    """
    try:
        return GenerationTaskItemStatus(str(value or "pending"))
    except ValueError:
        return GenerationTaskItemStatus.PENDING
