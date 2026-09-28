from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .models import (
    GenerationTaskItem,
    GenerationTaskRecord,
    GenerationTaskStatus,
    coerce_generation_item_status,
)
from ..shared.logging import log_prefix, safe_log_error_body, safe_log_text

LOG = log_prefix("GenerationTaskStore")


class GenerationTaskStore:
    """Persist and restore generation task records."""

    def __init__(self, persistence_file: str | Path | None = None):
        """Initialize the generation task store.

        Args:
            persistence_file: JSON file used for persisted task metadata.
        """
        self._persistence_file = Path(persistence_file) if persistence_file else None

    @property
    def has_history_file(self) -> bool:
        """Return whether a persisted history file currently exists."""
        return bool(self._persistence_file and self._persistence_file.exists())

    def load(self) -> list[GenerationTaskRecord]:
        """Load persisted generation task history from disk.

        Returns:
            Restored task records. Invalid records are skipped.
        """
        if not self._persistence_file:
            return []
        persistence_file = self._persistence_file
        if not persistence_file.exists():
            return []

        try:
            with persistence_file.open(encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.error(f"{LOG} 加载生图任务历史失败: {exc}", exc_info=True)
            corrupt_path = persistence_file.with_name(
                f"{persistence_file.name}.{datetime.now().strftime('%Y%m%d%H%M%S')}.corrupt"
            )
            try:
                os.replace(persistence_file, corrupt_path)
            except Exception as rename_exc:
                logger.error(
                    f"{LOG} 保留损坏生图任务历史失败: {rename_exc}",
                    exc_info=True,
                )
            return []

        raw_tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
        if isinstance(raw_tasks, dict):
            raw_tasks = list(raw_tasks.values())
        if not isinstance(raw_tasks, list):
            raw_tasks = []

        records: list[GenerationTaskRecord] = []
        for raw_record in raw_tasks:
            if not isinstance(raw_record, dict):
                continue
            record = self.record_from_dict(raw_record)
            if record:
                records.append(record)
        return records

    def save(self, records: list[GenerationTaskRecord]) -> None:
        """Persist generation task metadata to disk.

        Args:
            records: Task records to persist.
        """
        if not self._persistence_file:
            return
        payload = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "tasks": [self.record_to_dict(record) for record in records],
        }
        target_file = self._persistence_file
        temp_file = target_file.with_name(f"{target_file.name}.tmp")
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, target_file)
        except Exception as exc:
            logger.error(f"{LOG} 保存生图任务历史失败: {exc}", exc_info=True)
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass

    def record_to_dict(self, record: GenerationTaskRecord) -> dict[str, Any]:
        """Convert a generation task record to JSON-safe metadata.

        Args:
            record: Task record to serialize.

        Returns:
            JSON-safe metadata dictionary.
        """
        return {
            "task_id": record.task_id,
            "source": record.source,
            "unified_msg_origin": record.unified_msg_origin,
            "prompt_summary": record.prompt_summary,
            "prompt": record.prompt or "",
            "reference_image_count": record.reference_image_count,
            "requested_count": record.requested_count,
            "result_count": record.result_count,
            "aspect_ratio": record.aspect_ratio,
            "resolution": record.resolution,
            "model": record.model,
            "preset": record.preset,
            "preset_label": record.preset_label,
            "status": record.status.value,
            "message": record.message,
            "error": record.error,
            "created_at": self._datetime_to_str(record.created_at),
            "started_at": self._datetime_to_str(record.started_at),
            "finished_at": self._datetime_to_str(record.finished_at),
            "result_paths": list(record.result_paths),
            "current_index": record.current_index,
            "retry_attempt": record.retry_attempt,
            "max_retry_attempts": record.max_retry_attempts,
            "items": [
                self.item_to_dict(item)
                for item in sorted(record.items.values(), key=lambda item: item.index)
            ],
            "usage_scope": record.usage_scope,
            "reserved_count": record.reserved_count,
            "quota_date": record.quota_date,
            "quota_released": record.quota_released,
            "quota_settled": record.quota_settled,
        }

    def item_to_dict(self, item: GenerationTaskItem) -> dict[str, Any]:
        """Convert one generation sub-request item to JSON-safe metadata.

        Args:
            item: Sub-request item to serialize.

        Returns:
            JSON-safe metadata dictionary.
        """
        return {
            "index": item.index,
            "status": item.status.value,
            "result_count": item.result_count,
            "error": item.error,
            "retry_attempts": item.retry_attempts,
            "max_retry_attempts": item.max_retry_attempts,
        }

    def record_from_dict(
        self,
        raw_record: dict[str, Any],
    ) -> GenerationTaskRecord | None:
        """Restore one generation task record from persisted metadata.

        Args:
            raw_record: Raw persisted record dictionary.

        Returns:
            Restored record, or ``None`` when the input is invalid.
        """
        task_id = str(raw_record.get("task_id") or "").strip()
        if not task_id:
            return None
        requested_count = self._safe_int(raw_record.get("requested_count"), 1, 1)
        try:
            status = GenerationTaskStatus(str(raw_record.get("status") or "failed"))
        except ValueError:
            status = GenerationTaskStatus.FAILED

        return GenerationTaskRecord(
            task_id=task_id,
            source=str(raw_record.get("source") or "历史记录"),
            unified_msg_origin=str(raw_record.get("unified_msg_origin") or ""),
            prompt_summary=safe_log_text(raw_record.get("prompt_summary") or "", 80),
            prompt=str(raw_record.get("prompt") or ""),
            reference_image_count=self._safe_int(
                raw_record.get("reference_image_count"),
                0,
                0,
            ),
            requested_count=requested_count,
            aspect_ratio=str(raw_record.get("aspect_ratio") or ""),
            resolution=str(raw_record.get("resolution") or ""),
            model=str(raw_record.get("model") or ""),
            preset=(
                str(raw_record.get("preset")) if raw_record.get("preset") else None
            ),
            preset_label=str(raw_record.get("preset_label") or "预设"),
            status=status,
            created_at=self._str_to_datetime(raw_record.get("created_at"))
            or datetime.now(),
            started_at=self._str_to_datetime(raw_record.get("started_at")),
            finished_at=self._str_to_datetime(raw_record.get("finished_at")),
            message=safe_log_error_body(raw_record.get("message") or "", 300),
            error=safe_log_error_body(raw_record.get("error") or "", 300),
            result_count=self._safe_int(raw_record.get("result_count"), 0, 0),
            result_paths=self._safe_str_list(raw_record.get("result_paths")),
            current_index=self._safe_int(raw_record.get("current_index"), 0, 0),
            retry_attempt=self._safe_int(raw_record.get("retry_attempt"), 0, 0),
            max_retry_attempts=self._safe_int(
                raw_record.get("max_retry_attempts"),
                0,
                0,
            ),
            items=self.items_from_raw(
                raw_record.get("items"),
                requested_count,
            ),
            usage_scope=str(raw_record.get("usage_scope") or ""),
            reserved_count=self._safe_int(raw_record.get("reserved_count"), 0, 0),
            quota_date=str(raw_record.get("quota_date") or "").strip(),
            quota_released=bool(raw_record.get("quota_released", False)),
            quota_settled=bool(raw_record.get("quota_settled", False)),
        )

    def items_from_raw(
        self,
        raw_items: Any,
        requested_count: int,
    ) -> dict[int, GenerationTaskItem]:
        """Restore sub-request items from list or legacy dict forms.

        Args:
            raw_items: Raw persisted item list or legacy dictionary.
            requested_count: Requested sub-request count used to fill missing items.

        Returns:
            Mapping from sub-request index to restored item metadata.
        """
        items: dict[int, GenerationTaskItem] = {}
        iterable: list[Any]
        if isinstance(raw_items, dict):
            iterable = list(raw_items.values())
        elif isinstance(raw_items, list):
            iterable = raw_items
        else:
            iterable = []

        for raw_item in iterable:
            if not isinstance(raw_item, dict):
                continue
            index = self._safe_int(raw_item.get("index"), 0, 1)
            if index <= 0:
                continue
            items[index] = GenerationTaskItem(
                index=index,
                status=coerce_generation_item_status(raw_item.get("status")),
                result_count=self._safe_int(raw_item.get("result_count"), 0, 0),
                error=safe_log_error_body(raw_item.get("error") or "", 200),
                retry_attempts=self._safe_int(raw_item.get("retry_attempts"), 0, 0),
                max_retry_attempts=self._safe_int(
                    raw_item.get("max_retry_attempts"),
                    0,
                    0,
                ),
            )

        for index in range(1, max(1, requested_count) + 1):
            items.setdefault(index, GenerationTaskItem(index=index))
        return items

    def _datetime_to_str(self, value: datetime | None) -> str | None:
        """Serialize a datetime to ISO text."""
        return value.isoformat() if value else None

    def _str_to_datetime(self, value: Any) -> datetime | None:
        """Parse an ISO datetime string defensively."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    def _safe_int(self, value: Any, default: int, minimum: int) -> int:
        """Coerce a value to int and clamp it to a minimum."""
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, parsed)

    def _safe_str_list(self, value: Any) -> list[str]:
        """Return only string entries from a persisted list."""
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item]
