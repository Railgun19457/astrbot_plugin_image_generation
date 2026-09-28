"""User usage accounting and quota management."""

from __future__ import annotations

import datetime
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from ..shared.constants import USAGE_DATA_RETENTION_DAYS
from ..shared.logging import log_prefix

if TYPE_CHECKING:
    from ..config.manager import UsageSettings


LOG = log_prefix("Usage")


class UsageManager:
    """Manage user usage counts, reservations, and cooldowns."""

    def __init__(self, data_dir: str, settings: UsageSettings):
        self._data_dir = Path(data_dir)
        self._settings = settings
        self._usage_file = self._data_dir / "usage.json"
        self._usage_data: dict[str, dict[str, int]] = {}  # {date: {user_id: count}}
        self._usage_reservations: dict[str, dict[str, int]] = {}
        self._user_request_timestamps: dict[str, float] = {}  # Used for cooldowns.
        self._load_usage_data()

    def update_settings(self, settings: UsageSettings) -> None:
        """Update usage settings."""
        self._settings = settings

    def _load_usage_data(self) -> None:
        """Load persisted usage data."""
        if self._usage_file.exists():
            try:
                with self._usage_file.open(encoding="utf-8") as f:
                    self._usage_data = json.load(f)

                # Keep only recent usage data according to the retention policy.
                today = datetime.date.today()
                keys_to_delete = []
                for date_str in self._usage_data:
                    try:
                        date_obj = datetime.date.fromisoformat(date_str)
                        if (today - date_obj).days > USAGE_DATA_RETENTION_DAYS:
                            keys_to_delete.append(date_str)
                    except ValueError:
                        keys_to_delete.append(date_str)

                if keys_to_delete:
                    for key in keys_to_delete:
                        del self._usage_data[key]
                    self._save_usage_data()
            except Exception as exc:
                logger.error(f"{LOG} 加载使用数据失败: {exc}", exc_info=True)
                self._usage_data = {}

    def _save_usage_data(self) -> None:
        """Persist usage data."""
        try:
            self._usage_file.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_file.open("w", encoding="utf-8") as f:
                json.dump(self._usage_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"{LOG} 保存使用数据失败: {exc}", exc_info=True)

    def _today(self) -> str:
        """Return today's date key for usage accounting."""
        return datetime.date.today().isoformat()

    def _get_reserved_usage(self, date: str, user_id: str) -> int:
        """Return pending reserved usage for one user on one quota date."""
        return self._usage_reservations.get(date, {}).get(user_id, 0)

    def _reserve_usage(self, date: str, user_id: str, count: int) -> None:
        """Reserve quota before an async generation task starts."""
        count = max(0, count)
        user_id = str(user_id or "").strip()
        if count <= 0 or not user_id:
            return
        self._usage_reservations.setdefault(date, {})[user_id] = (
            self._get_reserved_usage(date, user_id) + count
        )

    def reserve_usage(
        self,
        user_id: str,
        count: int,
        *,
        is_admin: bool = False,
        quota_date: str = "",
    ) -> str:
        """Reserve quota and return the date key that owns the reservation.

        Args:
            user_id: Usage scope being charged.
            count: Number of images to reserve.
            is_admin: Whether the caller bypasses configured limits.
            quota_date: Existing quota date to restore. Empty uses today.

        Returns:
            Date key for a real reservation, otherwise an empty string.
        """
        if not self._settings.enable_daily_limit:
            return ""
        if self.is_limit_exempt(user_id, is_admin=is_admin):
            return ""
        user_id = str(user_id or "").strip()
        count = max(0, count)
        if not user_id or count <= 0:
            return ""
        date = quota_date.strip() or self._today()
        self._reserve_usage(date, user_id, count)
        return date

    def is_session_blocked(self, user_id: str) -> bool:
        """Check whether the current session UMO is blocked."""
        uid = user_id.strip()
        if not uid:
            return False
        return uid in self._settings.umo_blacklist

    def is_limit_exempt(self, user_id: str, *, is_admin: bool = False) -> bool:
        """Check whether the current request should bypass usage limits."""
        uid = user_id.strip()
        if not uid:
            return False
        if self._settings.admin_bypass_limits and is_admin:
            return True
        return uid in self._settings.umo_whitelist

    def check_rate_limit(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        requested_count: int = 1,
        update_timestamp: bool = True,
    ) -> bool | str:
        """Check per-user cooldowns and daily quota limits.

        Returns:
            True when checks pass, otherwise a user-facing error message.

        Args:
            update_timestamp: Whether to reserve the cooldown when checks pass.
        """
        user_id = str(user_id or "").strip()

        # Check cooldown limits first.
        if self.is_limit_exempt(user_id, is_admin=is_admin):
            return True

        if self.is_session_blocked(user_id):
            return self._settings.blacklist_block_message

        if self._settings.rate_limit_seconds > 0:
            now = time.time()
            last_ts = self._user_request_timestamps.get(user_id, 0)
            if now - last_ts < self._settings.rate_limit_seconds:
                remaining = int(self._settings.rate_limit_seconds - (now - last_ts))
                return f"❌ 请求过于频繁，请在 {remaining} 秒后再试"
            if update_timestamp:
                self._user_request_timestamps[user_id] = now

        # Check daily quota limits.
        if self._settings.enable_daily_limit:
            requested_count = max(1, requested_count)
            today = self._today()
            if today not in self._usage_data:
                self._usage_data[today] = {}

            count = self._usage_data[today].get(user_id, 0)
            reserved_count = self._get_reserved_usage(today, user_id)
            if (
                count + reserved_count + requested_count
                > self._settings.daily_limit_count
            ):
                remaining = max(
                    0,
                    self._settings.daily_limit_count - count - reserved_count,
                )
                return (
                    f"❌ 今日剩余生图额度不足，剩余 {remaining} 张，"
                    f"本次请求 {requested_count} 张"
                )
            if update_timestamp:
                self.reserve_usage(
                    user_id,
                    requested_count,
                    is_admin=is_admin,
                    quota_date=today,
                )

        return True

    def record_usage(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        count: int = 1,
    ) -> None:
        """Record generated image usage for one user."""
        if not self._settings.enable_daily_limit:
            return

        count = max(1, count)
        today = self._today()
        if today not in self._usage_data:
            self._usage_data[today] = {}
        self._usage_data[today][user_id] = (
            self._usage_data[today].get(user_id, 0) + count
        )
        self._save_usage_data()

    def release_reserved_usage(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        count: int = 1,
        quota_date: str = "",
    ) -> None:
        """Release previously reserved quota without recording usage.

        Args:
            user_id: Usage scope whose reservation should shrink.
            is_admin: Whether the original reservation bypassed limits.
            count: Number of reserved images to release.
            quota_date: Date key returned when the quota was reserved.
        """
        if not self._settings.enable_daily_limit:
            return
        if self.is_limit_exempt(user_id, is_admin=is_admin):
            return

        user_id = str(user_id or "").strip()
        if not user_id:
            return

        date = quota_date.strip() or self._today()
        reservations = self._usage_reservations.get(date)
        if not reservations or user_id not in reservations:
            return

        remaining = max(0, reservations[user_id] - max(0, count))
        if remaining:
            reservations[user_id] = remaining
            return

        del reservations[user_id]
        if not reservations:
            del self._usage_reservations[date]

    def settle_usage(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        reserved_count: int = 0,
        actual_count: int = 0,
        quota_date: str = "",
    ) -> None:
        """Record actual usage and release the matching reservation.

        Args:
            user_id: Usage scope being charged.
            is_admin: Whether the caller bypasses configured limits.
            reserved_count: Number of images originally reserved.
            actual_count: Number of images that should count as used.
            quota_date: Date key that owns both the usage and reservation.
                Empty uses today, including legacy tasks restored without one.
        """
        if not self._settings.enable_daily_limit:
            return

        user_id = str(user_id or "").strip()
        if not user_id:
            return

        date = quota_date.strip() or self._today()
        actual_count = max(0, actual_count)
        if actual_count:
            if date not in self._usage_data:
                self._usage_data[date] = {}
            self._usage_data[date][user_id] = (
                self._usage_data[date].get(user_id, 0) + actual_count
            )
            self._save_usage_data()

        self.release_reserved_usage(
            user_id,
            is_admin=is_admin,
            count=max(0, reserved_count),
            quota_date=date,
        )

    def restore_reservations(self, records: Iterable[Any]) -> bool:
        """Rebuild in-memory reservations from tasks that were not settled.

        Args:
            records: Restored generation task records. Active tasks recover
                their original reservation. Interrupted tasks are charged for
                images that were already produced.
        """
        if not self._settings.enable_daily_limit:
            return False

        restored = 0
        settled = 0
        changed = False
        for record in records:
            if getattr(record, "quota_settled", False) or getattr(
                record, "quota_released", False
            ):
                continue
            scope = str(getattr(record, "usage_scope", "") or "").strip()
            reserved_count = max(0, int(getattr(record, "reserved_count", 0) or 0))
            if not scope or reserved_count <= 0:
                continue
            if getattr(record, "is_usage_limit_admin", False) and self.is_limit_exempt(
                scope,
                is_admin=True,
            ):
                continue

            quota_date = str(getattr(record, "quota_date", "") or "").strip()
            if not quota_date:
                created_at = getattr(record, "created_at", None)
                quota_date = (
                    created_at.date().isoformat()
                    if isinstance(created_at, datetime.datetime)
                    else self._today()
                )
                changed = True
                record.quota_date = quota_date

            if getattr(record, "is_active", False):
                self._reserve_usage(quota_date, scope, reserved_count)
                restored += 1
                continue

            actual_count = max(0, int(getattr(record, "result_count", 0) or 0))
            self.settle_usage(
                scope,
                is_admin=bool(getattr(record, "is_usage_limit_admin", False)),
                reserved_count=0,
                actual_count=actual_count,
                quota_date=quota_date,
            )
            record.quota_settled = True
            record.quota_released = True
            changed = True
            settled += 1

        if restored or settled:
            logger.info(
                f"{LOG} 已恢复生图额度预留: 进行中={restored}，已结算中断任务={settled}"
            )
        return changed

    def get_usage_count(self, user_id: str) -> int:
        """Return today's usage count for one user."""
        today = self._today()
        return self._usage_data.get(today, {}).get(user_id, 0)

    def get_daily_limit(self) -> int:
        """Return the configured daily limit."""
        return self._settings.daily_limit_count

    def is_daily_limit_enabled(self) -> bool:
        """Return whether the daily limit is enabled."""
        return self._settings.enable_daily_limit
