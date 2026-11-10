"""Shared conversation permissions, cooldowns and atomic daily quota accounting."""

from __future__ import annotations

import datetime
import json
import math
import os
import time
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType

from ..shared.constants import USAGE_DATA_RETENTION_DAYS
from ..shared.logging import log_prefix
from .models import GenerationTaskCreationError

if TYPE_CHECKING:
    from ..config.manager import UsageSettings

LOG = log_prefix("Usage")
_MAX_COOLDOWN_ENTRIES = 4096


class UsageManager:
    """One accounting service for every generation entry point.

    Legacy date/UMO counts stay at the root of usage.json. Group counts, personal
    counts and task settlement receipts have separate namespaces.
    A receipt and its increments are committed together, closing the crash gap
    between usage.json and generation task history.
    """

    def __init__(self, data_dir: str, settings: UsageSettings):
        self._settings = settings
        self._usage_file = Path(data_dir) / "usage.json"
        self._lock = RLock()
        self._usage_data: dict[str, dict[str, int]] = {}
        self._group_usage: dict[str, dict[str, int]] = {}
        self._user_usage: dict[str, dict[str, int]] = {}
        self._receipts: dict[str, str] = {}
        self._reservations: dict[str, tuple[str, dict[str, str], int]] = {}
        self._pending_settlements: dict[str, int] = {}
        self._user_request_timestamps: dict[str, float] = {}
        self._load_usage_data()

    def update_settings(self, settings: UsageSettings) -> None:
        with self._lock:
            self._settings = settings

    def _load_usage_data(self) -> None:
        if not self._usage_file.exists():
            return
        # Never replace unreadable quota data with an empty ledger.
        with self._usage_file.open(encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("usage data must be a JSON object")

        def normalize(raw: Any) -> dict[str, dict[str, int]]:
            result = {}
            if not isinstance(raw, dict):
                return result
            for day, counts in raw.items():
                try:
                    date = datetime.date.fromisoformat(day)
                except (ValueError, TypeError):
                    continue
                if (datetime.date.today() - date).days > USAGE_DATA_RETENTION_DAYS:
                    continue
                if isinstance(counts, dict):
                    result[day] = {}
                    for scope, count in counts.items():
                        try:
                            result[day][str(scope)] = max(0, int(count))
                        except (ValueError, TypeError):
                            continue
            return result

        self._usage_data = normalize(loaded)
        self._group_usage = normalize(loaded.get("_group_usage", {}))
        self._user_usage = normalize(loaded.get("_user_usage", {}))
        self._receipts = {
            str(k): str(v) for k, v in loaded.get("_quota_receipts", {}).items()
        }

    def _save_usage_data(self) -> None:
        self._usage_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self._usage_file.with_name(self._usage_file.name + ".tmp")
        payload = dict(self._usage_data)
        payload["_group_usage"] = self._group_usage
        payload["_user_usage"] = self._user_usage
        payload["_quota_receipts"] = self._receipts
        with temp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self._usage_file)

    def _today(self) -> str:
        return datetime.date.today().isoformat()

    @staticmethod
    def user_quota_key(scope: str, sender_id: str) -> str:
        # JSON tuple encoding prevents delimiter collisions. UMO includes the
        # adapter instance ID, message type and conversation ID (AstrBot >=4.0).
        return json.dumps([scope, sender_id], ensure_ascii=False, separators=(",", ":"))

    def quota_scopes(
        self,
        scope: str,
        sender_id: str = "",
        message_type: MessageType | None = None,
        group_id: str = "",
    ) -> dict[str, str]:
        if not scope:
            return {}  # Historical system API call, no user/conversation quotas.
        if message_type is None:
            # Opaque legacy API UMO keeps its session limit.
            with suppress(ValueError, TypeError):
                message_type = MessageSession.from_str(scope).message_type
        scopes = {}
        if self._settings.enable_group_daily_limit:
            layer = (
                "group"
                if message_type == MessageType.GROUP_MESSAGE
                else (
                    "private"
                    if message_type == MessageType.FRIEND_MESSAGE
                    else "session"
                )
            )
            scopes[layer] = (
                self._group_scope(scope, group_id) if layer == "group" else scope
            )
        if self._settings.enable_user_daily_limit and sender_id:
            scopes["user"] = self.user_quota_key(scope, sender_id)
        return scopes

    def _counts(self, layer: str) -> dict[str, dict[str, int]]:
        if layer == "group":
            return self._group_usage
        return self._user_usage if layer == "user" else self._usage_data

    @staticmethod
    def _group_scope(scope: str, group_id: str = "") -> str:
        platform, separator, session_id = scope.partition(
            f":{MessageType.GROUP_MESSAGE.value}:"
        )
        if not platform or not session_id:
            return scope
        if not group_id:
            # Older OneBot, Slack and Lark UMO formats embed the group ID.
            sender, underscore, group = session_id.rpartition("_")
            if underscore and (
                (sender.isdecimal() and group.isdecimal())
                or (
                    sender.startswith(("U", "W"))
                    and group.startswith(("C", "G"))
                    and sender.isalnum()
                    and group.isalnum()
                )
            ):
                group_id = group
            elif "%" in session_id:
                sender, _, group = session_id.rpartition("%")
                if sender.startswith("ou_") and group.startswith("oc_"):
                    group_id = group
        return f"{platform}{separator}{group_id}" if group_id else scope

    @staticmethod
    def _user_bucket(scope: str) -> str | tuple[str, str]:
        try:
            umo, sender = json.loads(scope)
            session = MessageSession.from_str(umo)
        except (ValueError, TypeError, AttributeError):
            return scope
        return session.platform_id, sender

    def _quota_bucket(self, layer: str, scope: str) -> str | tuple[str, str]:
        if layer == "user" and self._settings.share_user_daily_limit_across_sessions:
            return self._user_bucket(scope)
        return scope

    def _used(self, day: str, layer: str, scope: str) -> int:
        with self._lock:
            counts = self._counts(layer).get(day, {})
            if layer == "group":
                return counts.get(scope, 0) + sum(
                    count
                    for key, count in self._usage_data.get(day, {}).items()
                    if self._group_scope(key) == scope
                )
            if (
                layer == "user"
                and self._settings.share_user_daily_limit_across_sessions
            ):
                bucket = self._user_bucket(scope)
                return sum(
                    count
                    for key, count in counts.items()
                    if self._user_bucket(key) == bucket
                )
            return counts.get(scope, 0)

    def _reserved(self, day: str, layer: str, scope: str) -> int:
        bucket = self._quota_bucket(layer, scope)
        total = 0
        for date, scopes, count in self._reservations.values():
            if date != day:
                continue
            reserved_scope = scopes.get(layer, "")
            if layer == "group" and not reserved_scope:
                reserved_scope = self._group_scope(scopes.get("session", ""))
            if reserved_scope and self._quota_bucket(layer, reserved_scope) == bucket:
                total += count
        return total

    def _check_quota(self, scopes: dict[str, str], count: int, day: str) -> bool | str:
        for layer, scope in scopes.items():
            limit = (
                self._settings.user_daily_limit_count
                if layer == "user"
                else self._settings.group_daily_limit_count
            )
            used = self._used(day, layer, scope)
            if used + self._reserved(day, layer, scope) + count > limit:
                if layer == "group":
                    return "❌ 本群今日生图额度已用完，请明天再试。"
                if layer in {"user", "private"}:
                    return "❌ 你今天的生图额度已用完，请明天再试。"
                return "❌ 当前会话今日生图额度已用完，请明天再试。"
        return True

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

    def _check_access_and_cooldown(self, scope: str) -> bool | str:
        if self.is_session_blocked(scope):
            return self._settings.blacklist_block_message
        if self._settings.rate_limit_seconds > 0:
            elapsed = time.time() - self._user_request_timestamps.get(scope, 0)
            if elapsed < self._settings.rate_limit_seconds:
                remaining = math.ceil(self._settings.rate_limit_seconds - elapsed)
                return f"❌ 请求过于频繁，请在 {remaining} 秒后再试"
        return True

    def check_rate_limit(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        requested_count: int = 1,
        sender_id: str = "",
        message_type: MessageType | None = None,
        group_id: str = "",
    ) -> bool | str:
        """Read-only preview; admission checks again after asynchronous preparation."""
        user_id = str(user_id or "").strip()
        with self._lock:
            if self.is_limit_exempt(user_id, is_admin=is_admin):
                return True
            check = self._check_access_and_cooldown(user_id)
            if isinstance(check, str):
                return check
            return self._check_quota(
                self.quota_scopes(user_id, sender_id, message_type, group_id),
                max(1, requested_count),
                self._today(),
            )

    def reserve_task(
        self,
        task_id: str,
        scope: str,
        count: int,
        *,
        sender_id: str = "",
        message_type: MessageType | None = None,
        group_id: str = "",
        is_admin: bool = False,
    ) -> tuple[str, dict[str, str], bool]:
        """Check BOTH layers and reserve all or nothing, without any await."""
        with self._lock:
            if task_id in self._reservations or task_id in self._receipts:
                raise GenerationTaskCreationError(
                    "task_id_conflict", "生图任务 ID 冲突"
                )
            exempt = self.is_limit_exempt(scope, is_admin=is_admin)
            if not exempt and isinstance(
                check := self._check_access_and_cooldown(scope), str
            ):
                raise GenerationTaskCreationError("rate_limited", check)
            scopes = self.quota_scopes(scope, sender_id, message_type, group_id)
            day = self._today()
            if not exempt:
                check = self._check_quota(scopes, count, day)
                if isinstance(check, str):
                    raise GenerationTaskCreationError("rate_limited", check)
            # Exempt requests remain chargeable, but never hold reservations.
            if scopes and not exempt:
                self._reservations[task_id] = (day, dict(scopes), count)
            return day, scopes, exempt

    def rollback_task(self, task_id: str) -> None:
        """Undo admission after task creation failed; repeated calls are safe."""
        with self._lock:
            self._reservations.pop(task_id, None)

    def _record_scopes(self, record: Any) -> dict[str, str]:
        if record.quota_version >= 2:
            scopes = dict(record.quota_scopes)
            if record.quota_version == 2 and "group" in scopes:
                scopes["group"] = self._group_scope(scopes["group"])
            return scopes
        # No reconstructed sender: pre-upgrade tasks keep ONLY their old UMO.
        if record.usage_scope and (
            record.reserved_count or self._settings.enable_group_daily_limit
        ):
            return {"session": record.usage_scope}
        return {}

    def pending_settlements(self) -> tuple[tuple[str, int], ...]:
        with self._lock:
            return tuple(self._pending_settlements.items())

    def settle_task(self, record: Any, *, actual_count: int) -> bool:
        """Commit counts and the idempotency receipt in one atomic replacement."""
        with self._lock:
            actual_count = self._pending_settlements.get(record.task_id, actual_count)
            scopes = self._record_scopes(record)
            if (
                record.task_id in self._receipts
                or record.quota_settled
                or record.quota_released
                or not scopes
                or actual_count <= 0
            ):
                self._reservations.pop(record.task_id, None)
                self._pending_settlements.pop(record.task_id, None)
                return True
            day = record.quota_date or record.created_at.date().isoformat()
            # Preserve actual output accounting, including exempt requests.
            previous = []
            for layer, scope in scopes.items():
                counts = self._counts(layer).setdefault(day, {})
                before = counts.get(scope, 0)
                previous.append((counts, scope, before))
                counts[scope] = before + actual_count
            self._receipts[record.task_id] = day
            try:
                self._save_usage_data()
            except OSError:
                self._receipts.pop(record.task_id, None)
                for counts, scope, before in previous:
                    counts[scope] = before
                if record.task_id not in self._pending_settlements:
                    logger.warning(
                        "%s 生图额度写入失败，将重试: %s",
                        LOG,
                        record.task_id,
                        exc_info=True,
                    )
                self._reservations[record.task_id] = (day, scopes, actual_count)
                self._pending_settlements[record.task_id] = actual_count
                return False
            self._reservations.pop(record.task_id, None)
            self._pending_settlements.pop(record.task_id, None)
            return True

    def restore_reservations(self, records: Iterable[Any]) -> bool:
        """Restore scopes from task records; old tasks never gain personal usage."""
        changed = False
        with self._lock:
            self._reservations.clear()
            for record in records:
                if record.task_id in self._receipts:
                    record.quota_settled = record.quota_released = True
                    changed = True
                    continue
                if record.quota_settled or record.quota_released:
                    continue
                if not record.quota_date:
                    record.quota_date = record.created_at.date().isoformat()
                    changed = True
                scopes = self._record_scopes(record)
                exempt = (
                    record.quota_exempt
                    if record.quota_version >= 2
                    else self.is_limit_exempt(
                        record.usage_scope, is_admin=record.is_usage_limit_admin
                    )
                )
                if record.is_active:
                    if scopes and not exempt:
                        self._reservations[record.task_id] = (
                            record.quota_date,
                            scopes,
                            record.reserved_count,
                        )
                else:
                    settled = self.settle_task(
                        record,
                        actual_count=record.result_count or len(record.result_paths),
                    )
                    record.quota_settled = record.quota_released = settled
                    changed |= settled
        return changed

    def is_daily_limit_enabled(self) -> bool:
        return bool(
            self._settings.enable_group_daily_limit
            or self._settings.enable_user_daily_limit
        )

    def format_usage(self, record: Any) -> str:
        """Display today's real counts using the task's snapshotted identities."""
        parts = []
        scopes = self._record_scopes(record)
        day = self._today()
        exempt = (
            record.quota_exempt
            if record.quota_version >= 2
            else self.is_limit_exempt(
                record.usage_scope, is_admin=record.is_usage_limit_admin
            )
        )
        labels = {
            "user": "个人",
            "group": "群聊",
            "private": "私聊兼容",
            "session": "会话",
        }
        for layer, label in labels.items():
            scope = scopes.get(layer)
            if not scope:
                continue
            used = self._used(day, layer, scope)
            limit = (
                self._settings.user_daily_limit_count
                if layer == "user"
                else self._settings.group_daily_limit_count
            )
            parts.append(f"{label} {used}/{'∞' if exempt else limit}")
        return "📅 今日生图：" + "，".join(parts) if parts else ""

    def record_request_timestamp(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
    ) -> None:
        """Start a new cooldown window for one accepted request.

        Args:
            user_id: Usage scope whose cooldown should be refreshed.
            is_admin: Whether the caller bypasses configured limits.
        """
        if self._settings.rate_limit_seconds <= 0:
            return
        user_id = str(user_id or "").strip()
        if not user_id or self.is_limit_exempt(user_id, is_admin=is_admin):
            return

        now = time.time()
        if len(self._user_request_timestamps) >= _MAX_COOLDOWN_ENTRIES:
            self._prune_cooldown_timestamps(now)
        self._user_request_timestamps[user_id] = now

    def _prune_cooldown_timestamps(self, now: float) -> None:
        """Keep the cooldown tracker bounded.

        Expired windows are dropped first. When every tracked window is still
        active, a short ``rate_limit_seconds`` would otherwise let the map grow
        without bound, so the oldest half is evicted as well.

        Args:
            now: Current monotonic reference used for expiry comparison.
        """
        cutoff = now - self._settings.rate_limit_seconds
        for tracked_user_id in [
            tracked
            for tracked, timestamp in self._user_request_timestamps.items()
            if timestamp < cutoff
        ]:
            del self._user_request_timestamps[tracked_user_id]

        # Evict in batches so sorting does not run on every single write.
        target_size = _MAX_COOLDOWN_ENTRIES // 2
        overflow = len(self._user_request_timestamps) - target_size
        if overflow <= 0:
            return
        oldest_user_ids = sorted(
            self._user_request_timestamps,
            key=self._user_request_timestamps.__getitem__,
        )[:overflow]
        for tracked_user_id in oldest_user_ids:
            del self._user_request_timestamps[tracked_user_id]
