from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from astrbot.api import logger

from .models import (
    GenerationQueueItem,
    GenerationTaskRecord,
    GenerationTaskStatus,
    TERMINAL_GENERATION_STATUSES,
)
from ..shared.logging import (
    format_log_event,
    format_seconds,
    log_prefix,
    safe_log_error_body,
    safe_log_text,
)

LOG = log_prefix("TaskManager")
GENERATION_TERMINAL_CALLBACK_TIMEOUT_SECONDS = 10.0


def _task_name(name: str) -> str:
    """Return a compact task name for logs."""
    return safe_log_text(name, 80)


class BackgroundTaskMixin:
    """Mixin for generic background task lifecycle management."""

    def create_task(
        self, coro: Coroutine[Any, Any, Any], name: str | None = None
    ) -> asyncio.Task:
        """Create a generic background task."""
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        self.background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        """Remove a background task and log unhandled exceptions."""
        self.background_tasks.discard(task)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.error(
                f"{LOG} 后台任务异常退出: {_task_name(task.get_name())}: {exc}",
                exc_info=(type(exc), exc, exc.__traceback__),
            )


class TaskSchedulerMixin:
    """Mixin for startup, daily, and shutdown task management."""

    def register_startup_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task to run during startup.

        Args:
            name: Task name used for logs.
            coro_func: Function that returns the task coroutine.
        """
        self._startup_tasks.append((name, coro_func))
        logger.debug(f"{LOG} 已注册启动任务: {_task_name(name)}")

    async def run_startup_tasks(self) -> None:
        """Run all registered startup tasks.

        This should be called once after plugin initialization is complete.
        """
        if self._startup_completed:
            logger.warning(f"{LOG} 启动任务已执行过，跳过重复执行")
            return

        if not self._startup_tasks:
            self._startup_completed = True
            return

        logger.debug(f"{LOG} 开始执行 {len(self._startup_tasks)} 个启动任务")

        for name, coro_func in self._startup_tasks:
            log_name = _task_name(name)
            try:
                logger.debug(f"{LOG} 执行启动任务: {log_name}")
                await coro_func()
                logger.debug(f"{LOG} 启动任务 {log_name} 执行完成")
            except Exception as e:
                logger.error(
                    f"{LOG} 启动任务 {log_name} 执行失败: {e}",
                    exc_info=True,
                )

        self._startup_completed = True

    def start_daily_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        check_interval_seconds: float = 60.0,
        run_immediately: bool = False,
    ) -> None:
        """Start a daily task that runs when the date changes.

        Args:
            name: Unique task name used for tracking and logs.
            coro_func: Function that returns the task coroutine.
            check_interval_seconds: Date-change check interval in seconds.
            run_immediately: Whether to run once immediately regardless of date.
        """
        if name in self._daily_tasks:
            self.stop_daily_task(name)

        log_name = _task_name(name)

        async def _daily_loop():
            # Initialize the last run date.
            if run_immediately:
                try:
                    await coro_func()
                    self._last_run_dates[name] = datetime.now().strftime("%Y-%m-%d")
                    logger.debug(f"{LOG} 每日任务 {log_name} 初始执行完成")
                except Exception as e:
                    logger.error(
                        f"{LOG} 每日任务 {log_name} 初始执行失败: {e}",
                        exc_info=True,
                    )
            else:
                # Store today's date to avoid duplicate runs on startup day.
                self._last_run_dates[name] = datetime.now().strftime("%Y-%m-%d")

            while True:
                try:
                    await asyncio.sleep(check_interval_seconds)
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    last_run_date = self._last_run_dates.get(name)

                    if current_date != last_run_date:
                        logger.info(
                            f"{LOG} 检测到日期变更 ({last_run_date} -> {current_date})，执行每日任务 {log_name}"
                        )
                        try:
                            await coro_func()
                            self._last_run_dates[name] = current_date
                            logger.info(f"{LOG} 每日任务 {log_name} 执行完成")
                        except Exception as e:
                            logger.error(
                                f"{LOG} 每日任务 {log_name} 执行出错: {e}",
                                exc_info=True,
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        f"{LOG} 每日任务 {log_name} 循环出错: {e}",
                        exc_info=True,
                    )

        task = asyncio.create_task(_daily_loop(), name=f"daily_{name}")
        self._daily_tasks[name] = task
        self.background_tasks.add(task)
        task.add_done_callback(functools.partial(self._on_daily_task_done, name))
        logger.debug(
            f"{LOG} 每日任务 {log_name} 已启动 (检查间隔: {check_interval_seconds}s)"
        )

    def stop_daily_task(self, name: str) -> None:
        """Stop one daily task."""
        if task := self._daily_tasks.pop(name, None):
            if not task.done():
                task.cancel()
            self._last_run_dates.pop(name, None)
            logger.debug(f"{LOG} 每日任务 {_task_name(name)} 已停止")

    def _on_daily_task_done(self, name: str, task: asyncio.Task) -> None:
        """Handle daily task completion."""
        self.background_tasks.discard(task)
        self._daily_tasks.pop(name, None)
        self._last_run_dates.pop(name, None)

    async def cancel_all(self):
        """Cancel all running tasks."""
        self._accepting_generation_tasks = False
        self._generation_shutdown = True

        queued_task_ids: list[str] = []
        while True:
            try:
                queue_item = self._generation_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if queue_item is not None:
                    queued_task_ids.append(queue_item.task_id)
            finally:
                self._generation_queue.task_done()

        for task_id in queued_task_ids:
            self.mark_generation_task_cancelled(
                task_id,
                "插件卸载/重启导致任务中断",
            )

        await self._wait_generation_notifications()

        for task in list(self._running_generation_tasks):
            if not task.done():
                task.cancel()

        if self._running_generation_tasks:
            await asyncio.gather(
                *self._running_generation_tasks,
                return_exceptions=True,
            )

        await self._wait_generation_notifications()

        for _ in list(self._generation_workers):
            self._generation_queue.put_nowait(None)

        for worker in list(self._generation_workers):
            if not worker.done():
                worker.cancel()

        if self._generation_workers:
            await asyncio.gather(*self._generation_workers, return_exceptions=True)

        for task in list(self.background_tasks):
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()
        self._daily_tasks.clear()
        self._last_run_dates.clear()
        self._generation_workers.clear()
        self._running_generation_tasks.clear()
        self._generation_terminal_notifying.clear()
        self._generation_notification_tasks.clear()
        self._generation_worker_target_count = 0
        self._save_generation_tasks()
        logger.debug(f"{LOG} 所有后台任务已取消")


class GenerationQueueRunnerMixin:
    """Mixin for generation worker and queue execution."""

    def start_generation_workers(self, worker_count: int) -> None:
        """Start generation workers and allow new generation tasks.

        Args:
            worker_count: Target number of complete generation tasks to run at once.
        """
        self._generation_shutdown = False
        self._accepting_generation_tasks = True
        self.update_generation_worker_count(worker_count)

    def update_generation_worker_count(self, worker_count: int) -> None:
        """Resize generation workers without cancelling running generation tasks.

        Args:
            worker_count: New target worker count.
        """
        target_count = max(1, worker_count)
        self._generation_worker_target_count = target_count

        live_workers = {task for task in self._generation_workers if not task.done()}
        self._generation_workers = live_workers
        current_count = len(self._generation_workers)
        if current_count != target_count:
            logger.debug(
                f"{LOG} "
                + format_log_event(
                    "worker数量调整",
                    当前=current_count,
                    目标=target_count,
                )
            )
        while len(self._generation_workers) < target_count:
            self._generation_worker_sequence += 1
            worker_index = self._generation_worker_sequence
            task = asyncio.create_task(
                self._generation_worker_loop(worker_index),
                name=f"image_generation_worker:{worker_index}",
            )
            self._generation_workers.add(task)
            task.add_done_callback(self._on_generation_worker_done)

        extra_count = len(self._generation_workers) - target_count
        for _ in range(max(0, extra_count)):
            self._generation_queue.put_nowait(None)

    async def _generation_worker_loop(self, worker_index: int) -> None:
        """Run queued generation tasks one at a time."""
        try:
            while True:
                queue_item = await self._generation_queue.get()
                try:
                    if queue_item is None:
                        if (
                            self._generation_shutdown
                            or len(self._generation_workers)
                            > self._generation_worker_target_count
                        ):
                            break
                        continue
                    await self._run_generation_queue_item(queue_item)
                finally:
                    self._generation_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"{LOG} 生图任务 worker {worker_index} 异常退出: {safe_log_error_body(exc)}",
                exc_info=True,
            )
            raise

    def _on_generation_worker_done(self, task: asyncio.Task) -> None:
        """Cleanup a generation worker and log unexpected failures."""
        self._generation_workers.discard(task)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.error(
                f"{LOG} 生图任务 worker 异常结束: {_task_name(task.get_name())}: {safe_log_error_body(exc)}",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _run_generation_queue_item(
        self,
        queue_item: GenerationQueueItem,
    ) -> None:
        """Execute one queued generation task if it is still active."""
        record = self._generation_tasks.get(queue_item.task_id)
        if not record:
            logger.debug(
                f"{log_prefix('Task', queue_item.task_id)} 队列项对应任务不存在，已跳过"
            )
            return
        if record.status == GenerationTaskStatus.CANCELLED:
            logger.debug(
                f"{log_prefix('Task', queue_item.task_id)} 已取消的排队任务由 worker 惰性跳过"
            )
            return
        if record.status == GenerationTaskStatus.CANCELLING:
            self.mark_generation_task_cancelled(queue_item.task_id)
            return
        if record.status in TERMINAL_GENERATION_STATUSES:
            logger.debug(
                f"{log_prefix('Task', queue_item.task_id)} 已结束的排队任务由 worker 惰性跳过: {record.status_label}"
            )
            return

        self.mark_generation_task_running(queue_item.task_id)
        try:
            coro = queue_item.coro_factory()
        except Exception as exc:
            self.mark_generation_task_failed(
                queue_item.task_id,
                f"任务创建执行协程失败: {exc}",
            )
            logger.error(
                f"{log_prefix('Task', queue_item.task_id)} 生图任务创建执行协程失败: {safe_log_error_body(exc)}",
                exc_info=True,
            )
            return

        task = asyncio.create_task(
            self._run_generation_task(queue_item.task_id, coro),
            name=f"image_generation:{queue_item.task_id}",
        )
        record.task = task
        self._running_generation_tasks.add(task)
        task.add_done_callback(self._running_generation_tasks.discard)
        task.add_done_callback(
            functools.partial(self._on_generation_task_done, queue_item.task_id)
        )
        try:
            await task
        finally:
            if record.task is task:
                record.task = None

    async def _run_generation_task(
        self, task_id: str, coro: Coroutine[Any, Any, Any]
    ) -> None:
        """Run a tracked generation coroutine and close unhandled states."""
        try:
            await coro
            record = self.get_generation_task(task_id)
            if record and record.is_active:
                self.mark_generation_task_succeeded(task_id, message="任务已完成")
        except asyncio.CancelledError:
            self.mark_generation_task_cancelled(task_id, "任务已取消")
            raise
        except Exception as exc:
            self.mark_generation_task_failed(task_id, f"任务执行异常: {exc}")
            logger.error(
                f"{log_prefix('Task', task_id)} 生图任务执行异常: {safe_log_error_body(exc)}",
                exc_info=True,
            )

    def _on_generation_task_done(self, task_id: str, _task: asyncio.Task) -> None:
        """Detach asyncio task references when a generation task finishes."""
        if record := self._generation_tasks.get(task_id):
            record.task = None


GenerationTaskCallback = Callable[[GenerationTaskRecord], Any]


class GenerationTaskNotifierMixin:
    """Mixin for generation task callback and waiter notifications."""

    def add_generation_task_terminal_callback(
        self,
        task_id: str,
        callback: GenerationTaskCallback,
    ) -> None:
        """Register a callback that runs before public completion notification.

        Args:
            task_id: Generation task ID to observe.
            callback: Callback receiving the terminal task record.
        """
        record = self._generation_tasks.get(task_id)
        if record and record.status in TERMINAL_GENERATION_STATUSES:
            if task_id in self._generation_terminal_notified:
                self._dispatch_generation_callback(callback, record, "终态业务回调")
                return
            self._generation_terminal_callbacks.setdefault(task_id, []).append(callback)
            self._notify_generation_task_terminal(task_id)
            return
        self._generation_terminal_callbacks.setdefault(task_id, []).append(callback)

    def add_generation_task_done_callback(
        self,
        task_id: str,
        callback: GenerationTaskCallback,
    ) -> None:
        """Register a callback that runs once after a task reaches terminal state.

        Args:
            task_id: Generation task ID to observe.
            callback: Callback receiving the terminal task record.
        """
        record = self._generation_tasks.get(task_id)
        if record and record.status in TERMINAL_GENERATION_STATUSES:
            if task_id in self._generation_terminal_notified:
                self._dispatch_generation_callback(callback, record, "完成回调")
                return
            self._generation_done_callbacks.setdefault(task_id, []).append(callback)
            self._notify_generation_task_terminal(task_id)
            return
        self._generation_done_callbacks.setdefault(task_id, []).append(callback)

    async def wait_generation_task_done(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> GenerationTaskRecord | None:
        """Wait for a generation task to reach terminal state.

        Args:
            task_id: Generation task ID to wait for.
            timeout_seconds: Optional maximum wait time in seconds.

        Returns:
            The terminal task record, or ``None`` when the task disappears or times out.
        """
        record = self._generation_tasks.get(task_id)
        if not record:
            return None
        if (
            record.status in TERMINAL_GENERATION_STATUSES
            and task_id in self._generation_terminal_notified
        ):
            return record

        event = self._generation_done_events.setdefault(task_id, asyncio.Event())
        if record.status in TERMINAL_GENERATION_STATUSES:
            self._notify_generation_task_terminal(task_id)
        try:
            if timeout_seconds is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=max(0.0, timeout_seconds))
        except asyncio.TimeoutError:
            return None
        return self._generation_tasks.get(task_id)

    def _dispatch_generation_callback(
        self,
        callback: GenerationTaskCallback,
        record: GenerationTaskRecord,
        label: str,
    ) -> None:
        """Run a generation callback and isolate callback failures."""
        try:
            result = callback(record)
        except Exception as exc:
            logger.error(
                f"{log_prefix('Task', record.task_id)} 生图任务{label}异常: {exc}",
                exc_info=True,
            )
            return

        if inspect.isawaitable(result):
            self.create_task(
                self._await_generation_callback(record.task_id, result, label),
                name=f"image_generation_{label}:{record.task_id}",
            )

    async def _await_generation_callback(
        self,
        task_id: str,
        awaitable: Any,
        label: str,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        """Await an asynchronous generation callback with error isolation."""
        try:
            if timeout_seconds is None:
                await awaitable
            else:
                await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                f"{log_prefix('Task', task_id)} 生图任务{label}超时: "
                f"超过 {format_seconds(timeout_seconds)}"
            )
        except Exception as exc:
            logger.error(
                f"{log_prefix('Task', task_id)} 生图任务{label}异步异常: {exc}",
                exc_info=True,
            )

    async def _run_generation_callback_ordered(
        self,
        callback: GenerationTaskCallback,
        record: GenerationTaskRecord,
        label: str,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        """Run and await one generation callback with error isolation."""
        try:
            result = callback(record)
            if inspect.isawaitable(result):
                await self._await_generation_callback(
                    record.task_id,
                    result,
                    label,
                    timeout_seconds=timeout_seconds,
                )
        except Exception as exc:
            logger.error(
                f"{log_prefix('Task', record.task_id)} 生图任务{label}异常: {exc}",
                exc_info=True,
            )

    def _run_generation_callback_inline(
        self,
        callback: GenerationTaskCallback,
        record: GenerationTaskRecord,
        label: str,
    ) -> Any | None:
        """Run one callback synchronously and return its awaitable part if any."""
        try:
            result = callback(record)
        except Exception as exc:
            logger.error(
                f"{log_prefix('Task', record.task_id)} 生图任务{label}异常: {exc}",
                exc_info=True,
            )
            return None
        return result if inspect.isawaitable(result) else None

    def _notify_generation_task_terminal(self, task_id: str) -> None:
        """Schedule terminal callbacks, public callbacks, and waiters once."""
        if (
            task_id in self._generation_terminal_notified
            or task_id in self._generation_terminal_notifying
        ):
            return
        record = self._generation_tasks.get(task_id)
        if not record or record.status not in TERMINAL_GENERATION_STATUSES:
            return

        self._generation_terminal_notifying.add(task_id)
        pending_awaitables: list[Any] = []
        while terminal_callbacks := self._generation_terminal_callbacks.pop(
            task_id,
            [],
        ):
            for callback in terminal_callbacks:
                awaitable = self._run_generation_callback_inline(
                    callback,
                    record,
                    "终态业务回调",
                )
                if awaitable is not None:
                    pending_awaitables.append(awaitable)

        if pending_awaitables:
            task = self.create_task(
                self._notify_generation_task_terminal_async(
                    task_id,
                    pending_awaitables,
                ),
                name=f"image_generation_terminal_notify:{task_id}",
            )
            self._generation_notification_tasks[task_id] = task
            task.add_done_callback(
                functools.partial(self._on_generation_notification_done, task_id)
            )
            return

        self._complete_generation_terminal_notification(task_id)

    async def _notify_generation_task_terminal_async(
        self,
        task_id: str,
        pending_awaitables: list[Any],
    ) -> None:
        """Await asynchronous terminal callbacks before public notifications."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status not in TERMINAL_GENERATION_STATUSES:
            return

        for awaitable in pending_awaitables:
            await self._await_generation_callback(
                task_id,
                awaitable,
                "终态业务回调",
                timeout_seconds=GENERATION_TERMINAL_CALLBACK_TIMEOUT_SECONDS,
            )

        while terminal_callbacks := self._generation_terminal_callbacks.pop(
            task_id,
            [],
        ):
            for callback in terminal_callbacks:
                await self._run_generation_callback_ordered(
                    callback,
                    record,
                    "终态业务回调",
                    timeout_seconds=GENERATION_TERMINAL_CALLBACK_TIMEOUT_SECONDS,
                )

        self._complete_generation_terminal_notification(task_id)

    def _complete_generation_terminal_notification(self, task_id: str) -> None:
        """Run public completion callbacks and release waiters."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status not in TERMINAL_GENERATION_STATUSES:
            self._generation_terminal_notifying.discard(task_id)
            return

        # Terminal callbacks may update quota settlement flags.
        self._save_generation_tasks()

        while done_callbacks := self._generation_done_callbacks.pop(task_id, []):
            for callback in done_callbacks:
                self._dispatch_generation_callback(callback, record, "完成回调")

        self._generation_terminal_notified.add(task_id)
        self._generation_terminal_notifying.discard(task_id)
        if event := self._generation_done_events.get(task_id):
            event.set()

        self._trim_generation_history()
        self._save_generation_tasks()

    def _on_generation_notification_done(
        self,
        task_id: str,
        task: asyncio.Task,
    ) -> None:
        """Cleanup notification task bookkeeping and log unexpected failures."""
        self.background_tasks.discard(task)
        self._generation_terminal_notifying.discard(task_id)
        if self._generation_notification_tasks.get(task_id) is task:
            self._generation_notification_tasks.pop(task_id, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            logger.error(
                f"{log_prefix('Task', task_id)} 生图任务终态通知异常: {exc}",
                exc_info=True,
            )

    async def _wait_generation_notifications(self) -> None:
        """Wait for in-flight terminal notifications before shutdown cleanup."""
        tasks = [
            task
            for task in self._generation_notification_tasks.values()
            if not task.done()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
