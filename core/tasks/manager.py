from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from astrbot.api import logger

from ..shared.constants import (
    DEFAULT_ENABLE_GENERATION_TASK_HISTORY,
    DEFAULT_GENERATION_TASK_HISTORY_LIMIT,
    DEFAULT_GENERATION_TASK_HISTORY_RETENTION_DAYS,
)
from .models import (
    ACTIVE_GENERATION_ITEM_STATUSES,
    ACTIVE_GENERATION_STATUSES,
    TERMINAL_GENERATION_STATUSES,
    GenerationQueueItem,
    GenerationTaskCreationError,
    GenerationTaskItem,
    GenerationTaskItemStatus,
    GenerationTaskRecord,
    GenerationTaskStatus,
    coerce_generation_item_status,
)
from .store import GenerationTaskStore
from .runtime import (
    BackgroundTaskMixin,
    GenerationQueueRunnerMixin,
    GenerationTaskNotifierMixin,
    TaskSchedulerMixin,
)
from ..shared.logging import (
    format_log_event,
    format_optional,
    format_seconds,
    log_prefix,
    mask_sensitive,
    safe_log_error_body,
    safe_log_text,
)

LOG = log_prefix("TaskManager")


def _task_elapsed(record: GenerationTaskRecord) -> str:
    """Return a compact task timing summary for logs."""
    queued = record.queued_seconds
    duration = record.duration_seconds
    if duration is None:
        return f"排队={format_seconds(queued)}"
    return f"排队={format_seconds(queued)}，耗时={format_seconds(duration)}"


def _task_creation_summary(record: GenerationTaskRecord) -> str:
    """Return a compact creation summary for generation task logs."""
    return (
        f"来源={safe_log_text(record.source)}，"
        f"用户={mask_sensitive(record.unified_msg_origin)}，"
        f"数量={record.requested_count}张，"
        f"参考图={record.reference_image_count}张，"
        f"{record.preset_label}={format_optional(record.preset)}，"
        f"宽高比={safe_log_text(record.aspect_ratio)}，"
        f"分辨率={safe_log_text(record.resolution)}"
    )


def _task_failure_summary(record: GenerationTaskRecord) -> str:
    """Return a compact failure summary for generation task logs."""
    return (
        f"来源={safe_log_text(record.source)}，"
        f"用户={mask_sensitive(record.unified_msg_origin)}，"
        f"数量={record.requested_count}张，"
        f"结果={record.result_count}张，"
        f"{_task_elapsed(record)}，"
        f"错误={record.error}"
    )


GenerationTaskCallback = Callable[[GenerationTaskRecord], Any]
GenerationCoroutineFactory = Callable[[], Coroutine[Any, Any, Any]]


class TaskManager(
    BackgroundTaskMixin,
    GenerationQueueRunnerMixin,
    GenerationTaskNotifierMixin,
    TaskSchedulerMixin,
):
    """Unified task manager for background, scheduled, and generation tasks."""

    def __init__(
        self,
        generation_history_limit: int = DEFAULT_GENERATION_TASK_HISTORY_LIMIT,
        generation_history_retention_days: int = DEFAULT_GENERATION_TASK_HISTORY_RETENTION_DAYS,
        enable_generation_task_history: bool = DEFAULT_ENABLE_GENERATION_TASK_HISTORY,
        max_queued_generation_tasks: int = 20,
        persistence_file: str | Path | None = None,
    ):
        self.background_tasks: set[asyncio.Task] = set()
        self._loop_tasks: dict[str, asyncio.Task] = {}
        self._daily_tasks: dict[str, asyncio.Task] = {}
        self._last_run_dates: dict[str, str] = {}
        self._startup_tasks: list[
            tuple[str, Callable[[], Coroutine[Any, Any, Any]]]
        ] = []
        self._startup_completed: bool = False
        self._generation_tasks: dict[str, GenerationTaskRecord] = {}
        self._generation_terminal_callbacks: dict[
            str, list[GenerationTaskCallback]
        ] = {}
        self._generation_done_callbacks: dict[str, list[GenerationTaskCallback]] = {}
        self._generation_done_events: dict[str, asyncio.Event] = {}
        self._generation_terminal_notifying: set[str] = set()
        self._generation_terminal_notified: set[str] = set()
        self._generation_notification_tasks: dict[str, asyncio.Task] = {}
        self._generation_history_limit = max(1, generation_history_limit)
        self._generation_history_retention_days = max(
            0,
            generation_history_retention_days,
        )
        self._enable_generation_task_history = bool(enable_generation_task_history)
        self._generation_queue: asyncio.Queue[GenerationQueueItem | None] = (
            asyncio.Queue()
        )
        self._max_queued_generation_tasks = max(1, max_queued_generation_tasks)
        self._generation_workers: set[asyncio.Task] = set()
        self._running_generation_tasks: set[asyncio.Task] = set()
        self._generation_worker_target_count = 0
        self._generation_worker_sequence = 0
        self._accepting_generation_tasks = False
        self._generation_shutdown = False
        self._generation_store = GenerationTaskStore(persistence_file)

    def get_generation_queue_rejection(self) -> tuple[str, str] | None:
        """Return the current reason for rejecting a new generation task.

        Returns:
            ``None`` if a new generation task can be accepted, otherwise a
            stable error code and user-facing message.
        """
        if not self._accepting_generation_tasks or self._generation_shutdown:
            logger.debug(
                f"{LOG} "
                + format_log_event(
                    "任务提交拒绝",
                    原因="任务队列未开放或正在关闭",
                )
            )
            return "task_manager_closed", "生图任务队列暂不可用，请稍后再试"
        if self._queued_generation_task_count() >= self._max_queued_generation_tasks:
            logger.warning(
                f"{LOG} "
                + format_log_event(
                    "任务提交拒绝",
                    原因="队列已满",
                    排队=f"{self._queued_generation_task_count()}/{self._max_queued_generation_tasks}",
                )
            )
            return "queue_full", "生图任务队列已满，请稍后再试"
        return None

    def can_accept_generation_task(self) -> bool:
        """Return whether a new generation task can currently be accepted."""
        return self.get_generation_queue_rejection() is None

    def create_generation_task(
        self,
        coro_factory: GenerationCoroutineFactory,
        *,
        task_id: str,
        source: str,
        unified_msg_origin: str,
        prompt: str,
        reference_image_count: int,
        requested_count: int,
        aspect_ratio: str,
        resolution: str,
        model: str = "",
        preset: str | None = None,
        preset_label: str = "预设",
        usage_scope: str = "",
        reserved_count: int = 0,
        quota_date: str = "",
        is_usage_limit_admin: bool = False,
        terminal_callback: GenerationTaskCallback | None = None,
    ) -> GenerationTaskRecord:
        """Create and enqueue an image generation task."""
        if rejection := self.get_generation_queue_rejection():
            code, message = rejection
            raise GenerationTaskCreationError(code, message)
        if task_id in self._generation_tasks:
            logger.warning(
                f"{log_prefix('Task', task_id)} "
                + format_log_event("任务提交拒绝", 原因="任务 ID 冲突")
            )
            raise GenerationTaskCreationError(
                "task_id_conflict",
                f"生图任务 ID 冲突: {task_id}",
            )

        safe_requested_count = max(1, requested_count)
        full_prompt = str(prompt or "")
        record = GenerationTaskRecord(
            task_id=task_id,
            source=source,
            unified_msg_origin=unified_msg_origin,
            prompt_summary=safe_log_text(full_prompt, 80),
            prompt=full_prompt,
            reference_image_count=reference_image_count,
            requested_count=safe_requested_count,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            model=str(model or "").strip(),
            preset=preset,
            preset_label=preset_label,
            items={
                index: GenerationTaskItem(index=index)
                for index in range(1, safe_requested_count + 1)
            },
            usage_scope=usage_scope,
            quota_date=str(quota_date or "").strip(),
            reserved_count=max(0, reserved_count),
            is_usage_limit_admin=bool(is_usage_limit_admin),
        )
        self._generation_tasks[task_id] = record
        self._generation_terminal_notified.discard(task_id)
        if terminal_callback:
            self.add_generation_task_terminal_callback(task_id, terminal_callback)
        self._trim_generation_history()

        self._generation_queue.put_nowait(
            GenerationQueueItem(task_id=task_id, coro_factory=coro_factory)
        )
        self._save_generation_tasks()
        logger.info(
            f"{log_prefix('Task', task_id)} "
            + format_log_event("任务提交", 摘要=_task_creation_summary(record))
        )
        logger.debug(
            f"{log_prefix('Task', task_id)} 生图任务提示词摘要: "
            f"提示词={safe_log_text(prompt, 80)}"
        )
        return record

    @property
    def generation_tasks(self) -> dict[str, GenerationTaskRecord]:
        """Return tracked generation tasks keyed by task id."""
        return self._generation_tasks

    def load_generation_history(self, usage_manager: Any | None = None) -> None:
        """Load persisted generation task history from disk.

        Args:
            usage_manager: Optional quota manager. When provided, unsettled
                reservations are restored before interrupted tasks are closed.
        """
        if not self._enable_generation_task_history:
            logger.debug(f"{LOG} 生图任务历史持久化已关闭，跳过加载")
            return
        if not self._generation_store.has_history_file:
            return

        restored_tasks: dict[str, GenerationTaskRecord] = {}
        history_changed = False
        now = datetime.now()
        loaded_records = self._generation_store.load()
        if usage_manager is not None:
            history_changed = bool(usage_manager.restore_reservations(loaded_records))
        for record in loaded_records:
            if record.status in ACTIVE_GENERATION_STATUSES:
                record.status = GenerationTaskStatus.CANCELLED
                record.message = "插件重启导致任务中断"
                record.error = "插件重启导致任务中断"
                record.finished_at = record.finished_at or now
                self._mark_unfinished_generation_items(
                    record,
                    status=GenerationTaskItemStatus.CANCELLED,
                    error=record.error,
                )
                history_changed = True
            restored_tasks[record.task_id] = record

        self._generation_tasks = restored_tasks
        self._generation_terminal_callbacks.clear()
        self._generation_done_callbacks.clear()
        self._generation_done_events.clear()
        self._generation_terminal_notifying.clear()
        self._generation_terminal_notified = {
            task_id
            for task_id, record in restored_tasks.items()
            if record.status in TERMINAL_GENERATION_STATUSES
        }
        self._generation_notification_tasks.clear()
        history_changed = self._trim_generation_history() or history_changed
        if history_changed:
            self._save_generation_tasks()
        logger.info(f"{LOG} 已加载生图任务历史: {len(self._generation_tasks)} 条")

    def flush_generation_history(self) -> None:
        """Persist the current generation task history immediately."""
        self._save_generation_tasks()

    def configure_generation_history(
        self,
        *,
        generation_history_limit: int,
        generation_history_retention_days: int,
        enable_generation_task_history: bool,
    ) -> None:
        """Update generation task history persistence and retention settings.

        Args:
            generation_history_limit: Maximum retained generation task records.
            generation_history_retention_days: Maximum age for finished records;
                ``0`` disables age-based trimming.
            enable_generation_task_history: Whether task history is persisted.
        """
        self._generation_history_limit = max(1, generation_history_limit)
        self._generation_history_retention_days = max(
            0,
            generation_history_retention_days,
        )
        self._enable_generation_task_history = bool(enable_generation_task_history)
        self._trim_generation_history()
        self._save_generation_tasks()

    def configure_generation_queue(self, *, max_queued_generation_tasks: int) -> None:
        """Update generation queue capacity for newly submitted tasks.

        Args:
            max_queued_generation_tasks: Maximum number of queued, not-running tasks.
        """
        self._max_queued_generation_tasks = max(1, max_queued_generation_tasks)

    def _queued_generation_task_count(self) -> int:
        """Return the number of queued generation tasks that are still active."""
        return sum(
            1
            for record in self._generation_tasks.values()
            if record.status == GenerationTaskStatus.QUEUED
        )

    def _save_generation_tasks(self) -> None:
        """Persist serializable generation task metadata to disk."""
        if not self._enable_generation_task_history:
            return
        self._generation_store.save(list(self._generation_tasks.values()))

    def _safe_generation_item_status(self, value: Any) -> GenerationTaskItemStatus:
        """Coerce raw persisted item status to a known enum value.

        Args:
            value: Raw persisted item status value.

        Returns:
            A valid generation item status. Unknown values fall back to
            ``pending`` so unfinished legacy records remain visible.
        """
        return coerce_generation_item_status(value)

    def _mark_unfinished_generation_items(
        self,
        record: GenerationTaskRecord,
        *,
        status: GenerationTaskItemStatus,
        error: str = "",
    ) -> None:
        """Mark pending or running sub-requests with a final item status."""
        safe_error = safe_log_error_body(error, 200) if error else ""
        for item in record.items.values():
            if item.status not in ACTIVE_GENERATION_ITEM_STATUSES:
                continue
            item.status = status
            if safe_error and not item.error:
                item.error = safe_error

    def _enter_generation_terminal_status(
        self,
        task_id: str,
        status: GenerationTaskStatus,
        *,
        message: str,
        error: str = "",
        result_count: int = 0,
        result_paths: list[str] | None = None,
    ) -> GenerationTaskRecord | None:
        """Move a task into terminal state exactly once."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status in TERMINAL_GENERATION_STATUSES:
            return None

        record.status = status
        record.finished_at = record.finished_at or datetime.now()
        record.message = safe_log_error_body(message, 300)
        record.error = safe_log_error_body(error, 300) if error else ""
        if result_paths is not None:
            record.result_paths = list(result_paths)
        final_result_count = (
            result_count or len(record.result_paths) or record.result_count
        )
        record.result_count = max(0, final_result_count)

        if status == GenerationTaskStatus.CANCELLED:
            self._mark_unfinished_generation_items(
                record,
                status=GenerationTaskItemStatus.CANCELLED,
                error=message,
            )
        elif status == GenerationTaskStatus.FAILED:
            self._mark_unfinished_generation_items(
                record,
                status=GenerationTaskItemStatus.FAILED,
                error=record.error or message,
            )
        return record

    def mark_generation_task_running(self, task_id: str) -> None:
        """Mark a generation task as actively running."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status == GenerationTaskStatus.CANCELLING:
            return
        if record.status in TERMINAL_GENERATION_STATUSES:
            return
        record.status = GenerationTaskStatus.RUNNING
        record.started_at = record.started_at or datetime.now()
        record.message = "任务运行中"
        logger.debug(
            f"{log_prefix('Task', task_id)} 生图任务开始运行: "
            f"排队={format_seconds(record.queued_seconds)}"
        )
        self._save_generation_tasks()

    def update_generation_task_references(
        self,
        task_id: str,
        *,
        reference_image_count: int,
    ) -> None:
        """Update prepared reference image metadata for a generation task."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status in TERMINAL_GENERATION_STATUSES:
            return
        record.reference_image_count = max(0, reference_image_count)

    def update_generation_task_retry_status(
        self,
        task_id: str,
        *,
        current_index: int,
        retry_attempt: int,
        max_retry_attempts: int,
    ) -> None:
        """Update the currently running generation retry attempt."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status in TERMINAL_GENERATION_STATUSES:
            return
        record.current_index = max(1, current_index)
        record.retry_attempt = max(0, retry_attempt)
        record.max_retry_attempts = max(0, max_retry_attempts)
        item = record.items.setdefault(
            record.current_index,
            GenerationTaskItem(
                index=record.current_index,
                status=GenerationTaskItemStatus.RUNNING,
            ),
        )
        item.status = GenerationTaskItemStatus.RUNNING
        item.retry_attempts = max(item.retry_attempts, record.retry_attempt)
        item.max_retry_attempts = max(
            item.max_retry_attempts, record.max_retry_attempts
        )

    def update_generation_task_item_result(
        self,
        task_id: str,
        *,
        index: int,
        status: str,
        result_count: int = 0,
        error: str = "",
    ) -> None:
        """Record per-request generation result details."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status in TERMINAL_GENERATION_STATUSES:
            return
        safe_index = max(1, index)
        item = record.items.setdefault(
            safe_index,
            GenerationTaskItem(index=safe_index),
        )
        item.status = self._safe_generation_item_status(status)
        item.result_count = max(0, result_count)
        item.error = safe_log_error_body(error, 200) if error else ""

    def update_generation_task_progress(
        self,
        task_id: str,
        *,
        current_index: int,
        result_count: int,
        message: str,
    ) -> None:
        """Update image count progress for one generation task."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status in TERMINAL_GENERATION_STATUSES:
            return
        record.current_index = max(1, current_index)
        record.result_count = max(0, result_count)
        record.message = message

    def mark_generation_task_cancelling(
        self,
        task_id: str,
        message: str = "正在取消任务",
    ) -> bool:
        """Move an active task into cancelling state.

        Args:
            task_id: Generation task ID to cancel.
            message: User-facing cancellation progress message.

        Returns:
            Whether the task was changed to cancelling state.
        """
        record = self._generation_tasks.get(task_id)
        if not record or record.status in TERMINAL_GENERATION_STATUSES:
            return False
        if record.status == GenerationTaskStatus.CANCELLING:
            return True
        record.status = GenerationTaskStatus.CANCELLING
        record.message = message
        return True

    def mark_generation_task_succeeded(
        self,
        task_id: str,
        *,
        result_count: int = 0,
        result_paths: list[str] | None = None,
        message: str = "任务已完成",
    ) -> None:
        """Mark a generation task as successful."""
        record = self._enter_generation_terminal_status(
            task_id,
            GenerationTaskStatus.SUCCEEDED,
            message=message,
            result_count=result_count,
            result_paths=result_paths,
        )
        if not record:
            return
        logger.info(
            f"{log_prefix('Task', task_id)} "
            + format_log_event(
                "任务完成",
                来源=safe_log_text(record.source),
                用时=_task_elapsed(record),
                结果=f"{record.result_count}张",
            )
        )
        self._save_generation_tasks()
        self._notify_generation_task_terminal(task_id)

    def mark_generation_task_failed(self, task_id: str, error: str) -> None:
        """Mark a generation task as failed."""
        record = self._enter_generation_terminal_status(
            task_id,
            GenerationTaskStatus.FAILED,
            message="任务失败",
            error=error,
        )
        if not record:
            return
        logger.warning(
            f"{log_prefix('Task', task_id)} "
            + format_log_event("任务失败", 摘要=_task_failure_summary(record))
        )
        self._save_generation_tasks()
        self._notify_generation_task_terminal(task_id)

    def mark_generation_task_cancelled(
        self, task_id: str, reason: str = "任务已取消"
    ) -> None:
        """Mark a generation task as cancelled."""
        record = self._enter_generation_terminal_status(
            task_id,
            GenerationTaskStatus.CANCELLED,
            message=reason,
        )
        if not record:
            return
        logger.info(
            f"{log_prefix('Task', task_id)} "
            + format_log_event(
                "任务取消",
                用时=_task_elapsed(record),
                原因=format_optional(reason),
            )
        )
        self._save_generation_tasks()
        self._notify_generation_task_terminal(task_id)

    def mark_generation_task_quota_released(self, task_id: str) -> None:
        """Mark that the task's reserved quota has been released."""
        if record := self._generation_tasks.get(task_id):
            record.quota_released = True
            self._save_generation_tasks()

    def mark_generation_task_quota_settled(self, task_id: str) -> None:
        """Mark that the task's reserved quota has been settled."""
        if record := self._generation_tasks.get(task_id):
            record.quota_settled = True
            record.quota_released = True
            self._save_generation_tasks()

    def mark_unfinished_generation_task_items_cancelled(
        self,
        task_id: str,
        reason: str = "任务已取消",
    ) -> None:
        """Mark pending or running sub-requests as cancelled."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return
        self._mark_unfinished_generation_items(
            record,
            status=GenerationTaskItemStatus.CANCELLED,
            error=reason,
        )

    def get_generation_task(self, task_id: str) -> GenerationTaskRecord | None:
        """Return a tracked image generation task by id."""
        return self._generation_tasks.get(task_id)

    def list_generation_tasks(
        self,
        *,
        unified_msg_origin: str | None = None,
        include_finished: bool = True,
        limit: int = 10,
    ) -> list[GenerationTaskRecord]:
        """List tracked generation tasks from newest to oldest."""
        tasks = list(reversed(self._generation_tasks.values()))
        if unified_msg_origin is not None:
            tasks = [t for t in tasks if t.unified_msg_origin == unified_msg_origin]
        if not include_finished:
            tasks = [t for t in tasks if t.is_active]
        return tasks[: max(1, limit)]

    def cancel_generation_task(
        self,
        task_id: str,
        *,
        unified_msg_origin: str | None = None,
    ) -> tuple[bool, str]:
        """Request cancellation for one generation task."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return False, f"❌ 任务不存在: {task_id}"
        if (
            unified_msg_origin is not None
            and record.unified_msg_origin != unified_msg_origin
        ):
            return False, "❌ 不能取消其他会话的生图任务"
        if record.status == GenerationTaskStatus.CANCELLING:
            return True, f"⏳ 任务正在取消中: {task_id}"
        if not record.is_active:
            return False, f"❌ 任务已结束，当前状态: {record.status_label}"

        self.mark_generation_task_cancelling(task_id)
        logger.debug(f"{log_prefix('Task', task_id)} 收到取消生图任务请求")
        if record.task and not record.task.done():
            record.task.cancel()
            return True, f"✅ 已请求取消任务: {task_id}"

        self.mark_generation_task_cancelled(task_id)
        return True, f"✅ 任务已取消: {task_id}"

    def cleanup_generation_tasks(self, *, unified_msg_origin: str | None = None) -> int:
        """Remove finished generation task records."""
        removed = 0
        for task_id, record in list(self._generation_tasks.items()):
            if record.is_active:
                continue
            if (
                unified_msg_origin is not None
                and record.unified_msg_origin != unified_msg_origin
            ):
                continue
            del self._generation_tasks[task_id]
            self._generation_terminal_callbacks.pop(task_id, None)
            self._generation_done_callbacks.pop(task_id, None)
            self._generation_done_events.pop(task_id, None)
            self._generation_terminal_notifying.discard(task_id)
            self._generation_terminal_notified.discard(task_id)
            self._generation_notification_tasks.pop(task_id, None)
            removed += 1
        if removed:
            self._save_generation_tasks()
        return removed

    def _trim_generation_history(self) -> bool:
        """Keep finished task history bounded while preserving active tasks."""
        changed = False
        if self._generation_history_retention_days > 0:
            cutoff = datetime.now() - timedelta(
                days=self._generation_history_retention_days
            )
            for task_id, record in list(self._generation_tasks.items()):
                if record.is_active:
                    continue
                finished_at = (
                    record.finished_at or record.started_at or record.created_at
                )
                if finished_at >= cutoff:
                    continue
                del self._generation_tasks[task_id]
                self._discard_generation_task_bookkeeping(task_id)
                changed = True

        overflow = len(self._generation_tasks) - self._generation_history_limit
        if overflow <= 0:
            return changed

        for task_id, record in list(self._generation_tasks.items()):
            if overflow <= 0:
                break
            if record.is_active:
                continue
            del self._generation_tasks[task_id]
            self._discard_generation_task_bookkeeping(task_id)
            overflow -= 1
            changed = True
        return changed

    def _discard_generation_task_bookkeeping(self, task_id: str) -> None:
        """Discard callbacks, waiters, and notification records for one task."""
        self._generation_terminal_callbacks.pop(task_id, None)
        self._generation_done_callbacks.pop(task_id, None)
        self._generation_done_events.pop(task_id, None)
        self._generation_terminal_notifying.discard(task_id)
        self._generation_terminal_notified.discard(task_id)
        self._generation_notification_tasks.pop(task_id, None)
