"""Dashboard Page statistics aggregation.

These helpers only read generation task records and the safe result-image
resolver, so they live outside the page route handlers.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from ..tasks.models import GenerationTaskRecord


class PageStatsMixin:
    """Aggregate generation statistics for the Dashboard Page."""

    def _count_top(
        self, counter: Counter[str], *, limit: int = 8
    ) -> list[dict[str, Any]]:
        """Convert a counter into a sorted top-N list for charts."""
        return [
            {"name": name, "count": count}
            for name, count in counter.most_common(max(1, limit))
            if name
        ]

    def _record_stamp(self, record: GenerationTaskRecord) -> datetime | None:
        """Pick the best timestamp for statistics aggregation."""
        return record.finished_at or record.started_at or record.created_at

    def _build_trend_series(
        self,
        stamps: list[datetime],
        *,
        days: int,
        now: datetime | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build a continuous trend series for the selected stats range.

        Args:
            stamps: Task timestamps inside the filtered window.
            days: Selected range in days. ``1`` uses hourly buckets; other
                positive values use daily buckets; ``0`` spans all history.
            now: Optional clock override for deterministic tests.

        Returns:
            A tuple of ``(granularity, points)`` where each point contains
            ``key``, ``label``, ``count``, and optional ``date``/``hour``.
        """
        current = now or datetime.now()
        if days == 1:
            end_hour = current.replace(minute=0, second=0, microsecond=0)
            start_hour = end_hour - timedelta(hours=23)
            bucket_counts: Counter[str] = Counter()
            for stamp in stamps:
                hour_key = stamp.replace(minute=0, second=0, microsecond=0)
                if hour_key < start_hour or hour_key > end_hour:
                    continue
                bucket_counts[hour_key.strftime("%Y-%m-%d %H:00")] += 1
            points: list[dict[str, Any]] = []
            cursor = start_hour
            while cursor <= end_hour:
                key = cursor.strftime("%Y-%m-%d %H:00")
                points.append(
                    {
                        "key": key,
                        "date": cursor.date().isoformat(),
                        "hour": cursor.hour,
                        "label": cursor.strftime("%H:00"),
                        "count": bucket_counts.get(key, 0),
                    }
                )
                cursor += timedelta(hours=1)
            return "hour", points

        if days > 0:
            end_day = current.date()
            start_day = end_day - timedelta(days=days - 1)
        else:
            if not stamps:
                end_day = current.date()
                start_day = end_day
            else:
                start_day = min(stamp.date() for stamp in stamps)
                end_day = max(stamp.date() for stamp in stamps)
                if end_day < current.date():
                    end_day = current.date()

        day_counts: Counter[str] = Counter()
        for stamp in stamps:
            day_key = stamp.date().isoformat()
            if stamp.date() < start_day or stamp.date() > end_day:
                continue
            day_counts[day_key] += 1

        points = []
        cursor_day = start_day
        while cursor_day <= end_day:
            key = cursor_day.isoformat()
            points.append(
                {
                    "key": key,
                    "date": key,
                    "label": cursor_day.strftime("%m-%d"),
                    "count": day_counts.get(key, 0),
                }
            )
            cursor_day += timedelta(days=1)
        return "day", points

    def _page_stats_payload(self, *, days: int = 7) -> dict[str, Any]:
        """Aggregate generation statistics from tracked task history."""
        safe_days = max(0, int(days or 0))
        now = datetime.now()
        since = now - timedelta(days=safe_days) if safe_days > 0 else None
        if safe_days == 1:
            since = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=23)
        records = self.plugin.task_manager.list_generation_tasks(
            include_finished=True,
            limit=1000,
        )
        filtered: list[GenerationTaskRecord] = []
        for record in records:
            stamp = self._record_stamp(record)
            if since and stamp and stamp < since:
                continue
            filtered.append(record)

        status_counter: Counter[str] = Counter()
        source_counter: Counter[str] = Counter()
        model_counter: Counter[str] = Counter()
        model_success_counter: Counter[str] = Counter()
        model_terminal_counter: Counter[str] = Counter()
        user_counter: Counter[str] = Counter()
        stamps: list[datetime] = []
        total_images = 0
        available_images = 0
        total_duration = 0.0
        duration_samples = 0
        active_count = 0
        success_count = 0
        failed_count = 0
        cancelled_count = 0

        for record in filtered:
            model_name = record.model or "unknown"
            status_counter[record.status.value] += 1
            source_counter[record.source or "unknown"] += 1
            model_counter[model_name] += 1
            user_counter[
                str(record.unified_msg_origin or "").strip() or "anonymous"
            ] += 1
            stamp = self._record_stamp(record)
            if stamp:
                stamps.append(stamp)
            if record.is_active:
                active_count += 1
            if record.status.value == "succeeded":
                success_count += 1
                model_success_counter[model_name] += 1
                model_terminal_counter[model_name] += 1
            elif record.status.value == "failed":
                failed_count += 1
                model_terminal_counter[model_name] += 1
            elif record.status.value == "cancelled":
                cancelled_count += 1
                model_terminal_counter[model_name] += 1
            total_images += record.result_count or len(record.result_paths)
            for index, path_value in enumerate(record.result_paths, 1):
                if self._result_image_path(record, index):
                    available_images += 1
            if record.duration_seconds is not None:
                total_duration += float(record.duration_seconds)
                duration_samples += 1

        terminal_count = success_count + failed_count + cancelled_count
        success_rate = (
            round((success_count / terminal_count) * 100, 1) if terminal_count else 0.0
        )
        avg_duration = (
            round(total_duration / duration_samples, 2) if duration_samples else 0.0
        )
        granularity, trend_points = self._build_trend_series(
            stamps,
            days=safe_days,
            now=now,
        )
        return {
            "days": safe_days,
            "trend_granularity": granularity,
            "summary": {
                "total_tasks": len(filtered),
                "active_tasks": active_count,
                "success_tasks": success_count,
                "failed_tasks": failed_count,
                "cancelled_tasks": cancelled_count,
                "success_rate": success_rate,
                "total_images": total_images,
                "available_images": available_images,
                "avg_duration_seconds": avg_duration,
                "unique_users": len(
                    [name for name in user_counter if name != "anonymous"]
                ),
                "unique_models": len(
                    [name for name in model_counter if name != "unknown"]
                ),
            },
            "status_distribution": self._count_top(status_counter, limit=10),
            "source_distribution": self._count_top(source_counter, limit=10),
            "model_distribution": [
                {
                    "name": name,
                    "count": count,
                    "success_count": model_success_counter.get(name, 0),
                    "terminal_count": model_terminal_counter.get(name, 0),
                    "success_rate": (
                        round(
                            (
                                model_success_counter.get(name, 0)
                                / model_terminal_counter[name]
                            )
                            * 100,
                            1,
                        )
                        if model_terminal_counter.get(name, 0)
                        else 0.0
                    ),
                }
                for name, count in model_counter.most_common(10)
                if name
            ],
            "user_distribution": self._count_top(user_counter, limit=10),
            "trend": trend_points,
        }
