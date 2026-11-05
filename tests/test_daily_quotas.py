"""Dual daily quota regressions. No network calls or real image generation."""

import asyncio
import copy
import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from conftest import ROOT, AstrBotConfig, MessageType
from image_plugin.core.api.public import ImageGenerationPublicAPI
from image_plugin.core.config.manager import ConfigManager
from image_plugin.core.config.models import UsageSettings
from image_plugin.core.formatting.result import build_result_info_message
from image_plugin.core.generation.executor import GenerationExecutor
from image_plugin.core.llm.tools import _start_generation_task
from image_plugin.core.shared.types import GenerationResult, ImageCapability
from image_plugin.core.tasks.manager import TaskManager
from image_plugin.core.tasks.models import (
    GenerationTaskCreationError,
    GenerationTaskStatus,
)
from image_plugin.core.tasks.store import GenerationTaskStore
from image_plugin.core.tasks.usage import UsageManager
from image_plugin.main import ImageGenerationPlugin

GROUP = "qq-bot-a:GroupMessage:100"
OTHER_GROUP = "qq-bot-a:GroupMessage:200"
PRIVATE = "qq-bot-a:FriendMessage:alice"


@pytest.fixture
def plugin(tmp_path):
    obj = ImageGenerationPlugin.__new__(ImageGenerationPlugin)
    config = AstrBotConfig(
        user_limits={
            "daily_limits_migrated": True,
            "enable_group_daily_limit": True,
            "group_daily_limit_count": 30,
            "enable_user_daily_limit": True,
            "user_daily_limit_count": 3,
        }
    )
    obj.config_manager = ConfigManager(config)
    obj.usage_manager = UsageManager(str(tmp_path), obj.config_manager.usage_settings)
    obj.task_manager = TaskManager(
        persistence_file=tmp_path / "tasks.json", max_queued_generation_tasks=300
    )
    obj.task_manager._accepting_generation_tasks = True
    obj.safety_auditor = SimpleNamespace(
        audit_prompt=AsyncMock(return_value=(True, "")),
        audit_generated_images=AsyncMock(return_value=(True, "")),
    )
    obj.image_processor = SimpleNamespace(
        save_generated_image=Mock(side_effect=lambda task_id, _: f"/{task_id}.png")
    )
    obj.generator = SimpleNamespace(
        adapter=SimpleNamespace(get_capabilities=lambda: ImageCapability.TEXT_TO_IMAGE),
        convert_reference_images=AsyncMock(return_value=[]),
        generate_preconverted=AsyncMock(
            return_value=GenerationResult(images=[b"image"])
        ),
    )
    obj.context = SimpleNamespace(send_message=AsyncMock(return_value=True))
    obj.has_required_api_key = lambda: True
    obj.generation_executor = GenerationExecutor(
        context=obj.context,
        config_manager=obj.config_manager,
        image_processor=obj.image_processor,
        task_manager=obj.task_manager,
        usage_manager=obj.usage_manager,
        safety_auditor=obj.safety_auditor,
    )
    obj.generation_executor.update_generator(obj.generator)
    obj.llm_result_handler = SimpleNamespace(
        attach_task_wakeup=Mock(),
        format_tool_start_result=Mock(return_value="submitted"),
    )
    obj.public_api = ImageGenerationPublicAPI(obj)
    return obj


def event(sender="alice", scope=GROUP, admin=False):
    return SimpleNamespace(
        unified_msg_origin=scope,
        get_sender_id=lambda: sender,
        get_message_type=lambda: (
            MessageType.FRIEND_MESSAGE
            if scope == PRIVATE
            else MessageType.GROUP_MESSAGE
        ),
        is_admin=lambda: admin,
        stop_event=Mock(),
        message_str="/生图 cat",
        get_messages=lambda: [],
        plain_result=lambda text: text,
    )


def create(
    plugin, task="task", sender="alice", scope=GROUP, count=1, admin=False, **kw
):
    return plugin.create_generation_task(
        task_id=task,
        source="test",
        prompt="cat",
        images_data=None,
        unified_msg_origin=scope,
        aspect_ratio="1:1",
        resolution="1K",
        image_count=count,
        is_usage_limit_admin=admin,
        sender_id=sender,
        **kw,
    )


def settle(plugin, record, count):
    plugin.generation_executor.settle_generation_task_quota_once(
        record.task_id, actual_count=count
    )


def set_limits(plugin, group=True, user=True, group_count=30, user_count=3, **kw):
    plugin.usage_manager.update_settings(
        UsageSettings(
            enable_group_daily_limit=group,
            group_daily_limit_count=group_count,
            enable_user_daily_limit=user,
            user_daily_limit_count=user_count,
            **kw,
        )
    )


def test_users_independent_group_shared_and_cross_group(plugin):
    a = create(plugin, "a", count=3)
    settle(plugin, a, 3)
    with pytest.raises(GenerationTaskCreationError, match="你今天"):
        create(plugin, "a2")
    b = create(plugin, "b", sender="bob", count=3)
    settle(plugin, b, 3)
    c = create(plugin, "c", scope=OTHER_GROUP, count=3)
    settle(plugin, c, 3)
    assert plugin.usage_manager.get_usage_count(GROUP) == 6
    assert plugin.usage_manager.get_usage_count(OTHER_GROUP) == 3
    assert "个人 3/3，群聊 6/30" in plugin.usage_manager.format_usage(a)


@pytest.mark.parametrize(
    "group,user", [(False, False), (True, False), (False, True), (True, True)]
)
def test_all_switch_combinations(plugin, group, user):
    set_limits(plugin, group, user, group_count=2, user_count=2)
    a = create(plugin, "a", count=2)
    assert set(a.quota_scopes) == ({"group"} if group else set()) | (
        {"user"} if user else set()
    )
    settle(plugin, a, 2)
    if group or user:
        with pytest.raises(GenerationTaskCreationError):
            create(plugin, "b")
    else:
        create(plugin, "b")


@pytest.mark.parametrize("which", ["group", "user"])
def test_atomic_denial_no_partial_reservation(plugin, which):
    set_limits(
        plugin,
        group_count=1 if which == "group" else 30,
        user_count=1 if which == "user" else 30,
    )
    a = create(plugin, "a")
    settle(plugin, a, 1)
    with pytest.raises(
        GenerationTaskCreationError, match="本群" if which == "group" else "你今天"
    ):
        create(plugin, "denied")
    assert not plugin.usage_manager._reservations
    if which == "user":
        create(plugin, "bob", sender="bob")


def test_requested_count_and_pending_reservations(plugin):
    set_limits(plugin, group_count=4, user_count=3)
    a = create(plugin, "a", count=2)
    with pytest.raises(GenerationTaskCreationError):
        create(plugin, "b", count=2)
    with pytest.raises(GenerationTaskCreationError):
        create(plugin, "c", sender="bob", count=3)
    settle(plugin, a, 1)
    create(plugin, "d", count=2)


def test_multithread_atomic_quota_admission(plugin):
    set_limits(plugin, group_count=9, user_count=2)

    def submit(i):
        try:
            plugin.usage_manager.reserve_task(f"t{i}", GROUP, 1, sender_id=f"u{i % 7}")
            return True
        except GenerationTaskCreationError:
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        assert sum(pool.map(submit, range(100))) == 9
    for i in range(7):
        key = plugin.usage_manager.user_quota_key(GROUP, f"u{i}")
        assert (
            plugin.usage_manager._reserved(plugin.usage_manager._today(), "user", key)
            <= 2
        )


def test_multibot_and_platform_identity(plugin):
    set_limits(plugin, group_count=1, user_count=1)
    for i, scope in enumerate(
        [GROUP, "qq-bot-b:GroupMessage:100", "discord:GroupMessage:100"]
    ):
        settle(plugin, create(plugin, str(i), scope=scope), 1)
    assert len(plugin.usage_manager._user_usage[plugin.usage_manager._today()]) == 3
    key = plugin.usage_manager.user_quota_key
    assert key("a", "b:c") != key("a:b", "c")


@pytest.mark.parametrize(
    "exception",
    [
        GenerationTaskCreationError("closed", "closed"),
        RuntimeError("disk failed"),
        asyncio.CancelledError(),
    ],
)
def test_all_creation_errors_release_both_layers(plugin, monkeypatch, exception):
    monkeypatch.setattr(
        plugin.task_manager, "create_generation_task", Mock(side_effect=exception)
    )
    with pytest.raises(type(exception)):
        create(plugin)
    assert not plugin.usage_manager._reservations
    assert not plugin.usage_manager._user_request_timestamps


def test_task_persistence_failure_is_not_admitted(plugin, monkeypatch):
    monkeypatch.setattr(
        plugin.task_manager._generation_store, "save", Mock(side_effect=OSError("disk"))
    )
    with pytest.raises(OSError):
        create(plugin)
    assert not plugin.task_manager._generation_tasks
    assert plugin.task_manager._generation_queue.empty()
    assert not plugin.usage_manager._reservations


@pytest.mark.parametrize(
    "status,actual",
    [
        ("failed", 0),
        ("cancelled", 0),
        ("failed", 1),
        ("cancelled", 1),
        ("succeeded", 2),
    ],
)
def test_terminal_settlement_partial_and_duplicate_callbacks(plugin, status, actual):
    r = create(plugin, count=3)
    r.result_count = actual
    r.status = GenerationTaskStatus(status)
    plugin._handle_generation_task_terminal(r)
    plugin._handle_generation_task_terminal(r)
    plugin.generation_executor.release_generation_task_quota_once(r.task_id)
    assert plugin.usage_manager.get_usage_count(GROUP) == actual
    assert not plugin.usage_manager._reservations
    assert r.quota_settled


def test_double_rollback_does_not_release_other_task(plugin):
    a = create(plugin, "a")
    create(plugin, "b")
    plugin.usage_manager.rollback_task(a.task_id)
    plugin.usage_manager.rollback_task(a.task_id)
    assert list(plugin.usage_manager._reservations) == ["b"]


def test_midnight_settlement_stays_on_submission_date(plugin, monkeypatch):
    a = create(plugin, count=3)
    before = a.quota_date
    next_day = (
        datetime.date.fromisoformat(before) + datetime.timedelta(days=1)
    ).isoformat()
    monkeypatch.setattr(plugin.usage_manager, "_today", lambda: next_day)
    settle(plugin, a, 2)
    assert plugin.usage_manager.get_usage_count(GROUP) == 0
    assert plugin.usage_manager._usage_data[before][GROUP] == 2
    assert plugin.usage_manager._user_usage[before][a.quota_scopes["user"]] == 2
    create(plugin, "next", count=3)


def test_settings_changes_do_not_change_inflight_scopes(plugin):
    a = create(plugin, count=3)
    set_limits(plugin, False, False)
    settle(plugin, a, 2)
    assert plugin.usage_manager.get_usage_count(GROUP) == 2
    assert plugin.usage_manager._user_usage[a.quota_date][a.quota_scopes["user"]] == 2
    assert not plugin.usage_manager._reservations


def test_crash_after_usage_commit_before_history_marker(plugin, tmp_path):
    a = create(plugin, count=3)
    plugin.task_manager.record_generation_task_output(a.task_id, "/image.png")
    # usage.json was committed, but tasks.json still says running/unsettled.
    plugin.usage_manager.settle_task(a, actual_count=1)
    for _ in range(2):
        usage = UsageManager(str(tmp_path), plugin.usage_manager._settings)
        tasks = TaskManager(persistence_file=tmp_path / "tasks.json")
        tasks.load_generation_history(usage)
        assert usage.get_usage_count(GROUP) == 1
        assert not usage._reservations
        assert tasks.get_generation_task(a.task_id).quota_settled


def test_restart_recovers_partial_output_and_original_date(plugin, tmp_path):
    a = create(plugin, count=3)
    plugin.task_manager.record_generation_task_output(a.task_id, "/image.png")
    usage = UsageManager(str(tmp_path), plugin.usage_manager._settings)
    tasks = TaskManager(persistence_file=tmp_path / "tasks.json")
    tasks.load_generation_history(usage)
    r = tasks.get_generation_task(a.task_id)
    assert r.status == GenerationTaskStatus.CANCELLED
    assert r.quota_date == a.quota_date
    assert usage.get_usage_count(GROUP) == 1
    assert usage._user_usage[a.quota_date][a.quota_scopes["user"]] == 1
    assert not usage._reservations


def test_history_disabled_keeps_minimal_quota_recovery(plugin, tmp_path):
    plugin.task_manager._enable_generation_task_history = False
    a = create(plugin, count=3)
    plugin.task_manager.record_generation_task_output(a.task_id, "/image.png")
    raw = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))["tasks"][0]
    assert raw["prompt"] == "" and raw["result_paths"] == []
    usage = UsageManager(str(tmp_path), plugin.usage_manager._settings)
    tasks = TaskManager(
        persistence_file=tmp_path / "tasks.json", enable_generation_task_history=False
    )
    tasks.load_generation_history(usage)
    assert usage.get_usage_count(GROUP) == 1
    assert (
        json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))["tasks"] == []
    )


def test_legacy_usage_and_incomplete_task_recovery(plugin, tmp_path):
    day = datetime.date.today().isoformat()
    (tmp_path / "usage.json").write_text(json.dumps({day: {GROUP: 7}}))
    old = {
        "task_id": "old",
        "source": "指令",
        "unified_msg_origin": GROUP,
        "usage_scope": GROUP,
        "requested_count": 3,
        "reserved_count": 3,
        "status": "running",
        "result_count": 2,
        "created_at": datetime.datetime.now().isoformat(),
    }
    (tmp_path / "tasks.json").write_text(json.dumps({"version": 1, "tasks": [old]}))
    for _ in range(2):
        usage = UsageManager(str(tmp_path), plugin.usage_manager._settings)
        tasks = TaskManager(persistence_file=tmp_path / "tasks.json")
        tasks.load_generation_history(usage)
        assert usage.get_usage_count(GROUP) == 9
        assert not usage._user_usage
        assert tasks.get_generation_task("old").quota_scopes == {}


def schema_defaults(schema):
    return {
        k: schema_defaults(v["items"])
        if v.get("type") == "object"
        else copy.deepcopy(v.get("default"))
        for k, v in schema.items()
    }


@pytest.mark.parametrize("migration_marker", [False, "false"])
def test_old_config_migrates_after_astrbot_inserts_defaults_idempotently(
    migration_marker,
):
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    cfg = AstrBotConfig(schema_defaults(schema))
    cfg.schema = schema
    cfg["user_limits"]["daily_limits_migrated"] = migration_marker
    cfg["user_limits"].update(enable_daily_limit=True, daily_limit_count=7)
    manager = ConfigManager(cfg)
    assert manager.usage_settings.enable_group_daily_limit
    assert manager.usage_settings.group_daily_limit_count == 7
    assert not manager.usage_settings.enable_user_daily_limit
    cfg["user_limits"].update(
        enable_group_daily_limit=False, group_daily_limit_count=12
    )
    manager.reload()
    assert not manager.usage_settings.enable_group_daily_limit
    assert manager.usage_settings.group_daily_limit_count == 12
    assert cfg["user_limits"]["daily_limit_count"] == 7


@pytest.mark.parametrize("user", [False, True])
def test_private_chat_legacy_limit_not_lost(plugin, user):
    set_limits(plugin, user=user, group_count=1, user_count=3)
    a = create(plugin, scope=PRIVATE)
    assert "group" not in a.quota_scopes
    assert a.quota_scopes["private"] == PRIVATE
    settle(plugin, a, 1)
    with pytest.raises(GenerationTaskCreationError, match="你今天"):
        create(plugin, "b", scope=PRIVATE)


def test_private_user_only_and_explicit_disable(plugin):
    set_limits(plugin, group=False, user=True, user_count=1)
    settle(plugin, create(plugin, scope=PRIVATE), 1)
    with pytest.raises(GenerationTaskCreationError):
        create(plugin, "b", scope=PRIVATE)
    set_limits(plugin, group=False, user=False)
    create(plugin, "c", scope=PRIVATE)


@pytest.mark.parametrize("admin,whitelist", [(True, False), (False, True)])
def test_admin_whitelist_bypass_but_actual_usage_still_recorded(
    plugin, admin, whitelist
):
    set_limits(
        plugin,
        group_count=1,
        user_count=1,
        rate_limit_seconds=600,
        umo_blacklist=[GROUP],
        umo_whitelist=[GROUP] if whitelist else [],
    )
    a = create(plugin, "a", count=3, admin=admin)
    b = create(plugin, "b", count=3, admin=admin)
    assert a.quota_exempt and not plugin.usage_manager._reservations
    settle(plugin, a, 3)
    settle(plugin, b, 3)
    assert plugin.usage_manager.get_usage_count(GROUP) == 6
    assert "6/∞（豁免）" in plugin.usage_manager.format_usage(a)


def test_blacklist_cooldown_and_real_admin_semantics(plugin):
    set_limits(plugin, umo_blacklist=[GROUP])
    with pytest.raises(GenerationTaskCreationError):
        create(plugin)
    set_limits(plugin, rate_limit_seconds=600)
    a = create(plugin, "accepted")
    with pytest.raises(GenerationTaskCreationError, match="请求过于频繁"):
        create(plugin, "bob", sender="bob")
    assert not plugin.is_usage_limit_admin(event())
    assert plugin.is_usage_limit_admin(event(admin=True))
    assert a.usage_scope == GROUP


def test_public_api_identity_compatibility_and_wait_forwarding(plugin, monkeypatch):
    async def run():
        old = await plugin.public_api.submit_generation_task(
            prompt="cat", unified_msg_origin=GROUP
        )
        r = plugin.task_manager.get_generation_task(old.task_id)
        assert old.ok and r.quota_caller_type == "legacy"
        assert set(r.quota_scopes) == {"group"} and not r.sender_id
        system = await plugin.public_api.submit_generation_task(
            prompt="cat", caller_type="system"
        )
        assert system.ok
        assert not plugin.task_manager.get_generation_task(system.task_id).quota_scopes
        user = await plugin.public_api.submit_generation_task(
            prompt="cat", unified_msg_origin=GROUP, sender_id="bob"
        )
        assert user.ok
        assert (
            "user" in plugin.task_manager.get_generation_task(user.task_id).quota_scopes
        )
        missing = await plugin.public_api.submit_generation_task(
            prompt="cat", unified_msg_origin=GROUP, caller_type="user"
        )
        assert not missing.ok
        missing_scope = await plugin.public_api.submit_generation_task(
            prompt="cat", sender_id="alice"
        )
        assert not missing_scope.ok
        wait = AsyncMock(return_value="files")
        monkeypatch.setattr(plugin.public_api, "wait_generation_result", wait)
        assert (
            await plugin.public_api.generate_image_files(
                prompt="cat",
                unified_msg_origin=GROUP,
                sender_id="carol",
                caller_type="user",
            )
            == "files"
        )
        record = plugin.task_manager.get_generation_task(wait.call_args.args[0])
        assert record.sender_id == "carol" and record.quota_caller_type == "user"

    asyncio.run(run())


def test_legacy_api_user_only_does_not_invent_sender(plugin):
    set_limits(plugin, group=False, user=True, user_count=1)

    async def run():
        for _ in range(3):
            result = await plugin.public_api.submit_generation_task(
                prompt="cat", unified_msg_origin=GROUP
            )
            assert result.ok
            assert not plugin.task_manager.get_generation_task(
                result.task_id
            ).quota_scopes

    asyncio.run(run())


def test_command_and_llm_use_same_real_sender_scopes(plugin, monkeypatch):
    import image_plugin.core.llm.tools as tools

    monkeypatch.setattr(
        tools, "collect_tool_reference_images", AsyncMock(return_value=[])
    )

    async def run():
        responses = [r async for r in plugin.generate_image_command(event())]
        assert responses
        command = list(plugin.task_manager._generation_tasks.values())[-1]
        assert command.source == "指令"
        result = await _start_generation_task(
            plugin, event(), prompt="cat", aspect_ratio="1:1", resolution="1K"
        )
        assert result == "submitted"
        tool = list(plugin.task_manager._generation_tasks.values())[-1]
        assert command.quota_scopes == tool.quota_scopes
        assert command.sender_id == tool.sender_id == "alice"
        settle(plugin, command, 1)
        settle(plugin, tool, 1)
        assert plugin.usage_manager.get_usage_count(GROUP) == 2
        result = await _start_generation_task(
            plugin,
            event(),
            prompt="cat",
            aspect_ratio="1:1",
            resolution="1K",
            image_count=2,
        )
        assert "你今天" in result

    asyncio.run(run())


def test_concurrent_api_preparation_cannot_overbook(plugin, monkeypatch):
    set_limits(plugin, group_count=5, user_count=2)

    async def run():
        ready = asyncio.Event()
        arrived = 0

        async def prepare(**kw):
            nonlocal arrived
            arrived += 1
            if arrived == 30:
                ready.set()
            await ready.wait()
            return []

        monkeypatch.setattr(plugin.public_api, "_collect_reference_images", prepare)
        results = await asyncio.gather(
            *[
                plugin.public_api.submit_generation_task(
                    prompt="cat", unified_msg_origin=GROUP, sender_id=f"u{i % 4}"
                )
                for i in range(30)
            ]
        )
        assert sum(r.ok for r in results) == 5
        assert all(r.code in {"accepted", "rate_limited"} for r in results)

    asyncio.run(run())


def test_executor_partial_success_and_actual_output_limit(plugin):
    async def run():
        a = create(plugin, count=3)
        plugin.generator.generate_preconverted.side_effect = [
            GenerationResult(images=[b"a"]),
            GenerationResult(error="failed"),
            GenerationResult(images=[b"c"]),
        ]
        await plugin.generation_executor.generate_and_send_image_async(
            "cat", GROUP, task_id=a.task_id, image_count=3, auto_send=False
        )
        await plugin.task_manager._wait_generation_notifications()
        assert a.result_count == 2 and a.quota_settled
        assert plugin.usage_manager.get_usage_count(GROUP) == 2
        b = create(plugin, "overflow", sender="bob")
        plugin.generator.generate_preconverted.side_effect = None
        plugin.generator.generate_preconverted.return_value = GenerationResult(
            images=[b"a", b"extra"]
        )
        await plugin.generation_executor.generate_and_send_image_async(
            "cat", GROUP, task_id=b.task_id, image_count=1, auto_send=False
        )
        await plugin.task_manager._wait_generation_notifications()
        assert b.result_count == 1
        assert plugin.usage_manager.get_usage_count(GROUP) == 3

    asyncio.run(run())


def test_executor_cancellation_after_one_valid_image(plugin):
    async def run():
        a = create(plugin, count=3)
        plugin.generation_executor.config_manager._plugin_config.generation_settings.max_concurrent_tasks = 1
        saved = asyncio.Event()
        calls = 0

        async def generate(*args, **kw):
            nonlocal calls
            calls += 1
            if calls == 1:
                return GenerationResult(images=[b"ok"])
            saved.set()
            await asyncio.Event().wait()

        plugin.generator.generate_preconverted.side_effect = generate
        coro = plugin.generation_executor.generate_and_send_image_async(
            "cat", GROUP, task_id=a.task_id, image_count=3, auto_send=False
        )
        task = asyncio.create_task(
            plugin.task_manager._run_generation_task(a.task_id, coro)
        )
        await saved.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await plugin.task_manager._wait_generation_notifications()
        assert a.status == GenerationTaskStatus.CANCELLED
        assert a.result_count == plugin.usage_manager.get_usage_count(GROUP) == 1
        assert not plugin.usage_manager._reservations

    asyncio.run(run())


def test_usage_file_error_keeps_reservations_and_retry_idempotent(plugin, monkeypatch):
    a = create(plugin)
    save = plugin.usage_manager._save_usage_data
    monkeypatch.setattr(
        plugin.usage_manager, "_save_usage_data", Mock(side_effect=OSError("disk"))
    )
    with pytest.raises(OSError):
        settle(plugin, a, 1)
    assert plugin.usage_manager.get_usage_count(GROUP) == 0
    assert a.task_id in plugin.usage_manager._reservations
    assert not a.quota_settled
    monkeypatch.setattr(plugin.usage_manager, "_save_usage_data", save)
    settle(plugin, a, 1)
    settle(plugin, a, 1)
    assert plugin.usage_manager.get_usage_count(GROUP) == 1


def test_corrupt_usage_is_not_silently_reset(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("broken")
    with pytest.raises(ValueError):
        UsageManager(str(tmp_path), UsageSettings())
    assert path.read_text(encoding="utf-8") == "broken"


def test_usage_display_can_be_disabled(plugin):
    a = create(plugin)
    settle(plugin, a, 1)
    info = build_result_info_message(
        plugin.config_manager,
        plugin.usage_manager,
        unified_msg_origin=GROUP,
        is_usage_limit_admin=False,
        duration=1,
        result_count=1,
        task_id=a.task_id,
        quota_record=a,
    )
    assert "个人 1/3，群聊 1/30" in info
    plugin.config_manager._plugin_config.generation_settings.result_info_items.clear()
    assert (
        build_result_info_message(
            plugin.config_manager,
            plugin.usage_manager,
            unified_msg_origin=GROUP,
            is_usage_limit_admin=False,
            duration=1,
            result_count=1,
            task_id=a.task_id,
            quota_record=a,
        )
        == ""
    )


def test_store_roundtrip_preserves_quota_identities(plugin):
    a = create(plugin)
    store = GenerationTaskStore()
    b = store.record_from_dict(store.record_to_dict(a))
    assert (b.sender_id, b.quota_scopes, b.quota_date, b.quota_caller_type) == (
        a.sender_id,
        a.quota_scopes,
        a.quota_date,
        a.quota_caller_type,
    )


def test_disabled_legacy_config_preserves_configured_count():
    cfg = AstrBotConfig(
        user_limits={"enable_daily_limit": False, "daily_limit_count": 23}
    )
    settings = ConfigManager(cfg).usage_settings
    assert not settings.enable_group_daily_limit
    assert settings.group_daily_limit_count == 23


def test_missing_native_sender_never_bypasses_personal_limit(plugin):
    with pytest.raises(GenerationTaskCreationError, match="发送者"):
        create(plugin, source_event=event(sender=""))
    set_limits(plugin, user=False)
    plugin.config_manager._plugin_config.usage_settings.enable_user_daily_limit = False
    r = create(plugin, source_event=event(sender=""))
    assert not r.sender_id and set(r.quota_scopes) == {"group"}


def test_sender_comes_from_real_event_not_supplied_override(plugin):
    r = create(plugin, sender="wrong", source_event=event(sender="real"))
    assert r.sender_id == "real"
    with pytest.raises(GenerationTaskCreationError, match="会话不一致"):
        create(plugin, "mismatch", scope=OTHER_GROUP, source_event=event())


def test_old_private_data_survives_upgrade(plugin, tmp_path):
    day = plugin.usage_manager._today()
    (tmp_path / "usage.json").write_text(json.dumps({day: {PRIVATE: 3, GROUP: 4}}))
    plugin.usage_manager = UsageManager(
        str(tmp_path), plugin.config_manager.usage_settings
    )
    plugin.generation_executor.usage_manager = plugin.usage_manager
    set_limits(plugin, group_count=3, user_count=3)
    with pytest.raises(GenerationTaskCreationError):
        create(plugin, scope=PRIVATE)
    assert plugin.usage_manager.get_usage_count(GROUP) == 4
    assert not plugin.usage_manager._user_usage


@pytest.mark.parametrize("admin", [False, True])
def test_old_terminal_task_recovery_only_charges_legacy_scope(plugin, tmp_path, admin):
    store = GenerationTaskStore(tmp_path / "tasks.json")
    r = store.record_from_dict(
        {
            "task_id": "legacy-done",
            "status": "succeeded",
            "usage_scope": GROUP,
            "requested_count": 2,
            "reserved_count": 0 if admin else 2,
            "is_usage_limit_admin": admin,
            "result_count": 1,
        }
    )
    store.save([r])
    usage = UsageManager(str(tmp_path), plugin.config_manager.usage_settings)
    tasks = TaskManager(persistence_file=tmp_path / "tasks.json")
    tasks.load_generation_history(usage)
    assert usage.get_usage_count(GROUP) == 1
    assert not usage._user_usage
    tasks.load_generation_history(usage)
    assert usage.get_usage_count(GROUP) == 1


def test_history_trimming_never_loses_unsettled_terminal_task(plugin):
    a = create(plugin)
    a.status = GenerationTaskStatus.FAILED
    plugin.task_manager._generation_history_limit = 1
    create(plugin, "b", sender="bob")
    assert plugin.task_manager.get_generation_task(a.task_id) is a
    settle(plugin, a, 0)
    plugin.task_manager._trim_generation_history()
    assert plugin.task_manager.get_generation_task(a.task_id) is None


def test_admin_bypass_disabled_obeys_both_limits(plugin):
    set_limits(plugin, group_count=1, user_count=1, admin_bypass_limits=False)
    settle(plugin, create(plugin, admin=True), 1)
    with pytest.raises(GenerationTaskCreationError):
        create(plugin, "again", admin=True)


def test_blank_blacklist_message_still_rejects_task(plugin):
    set_limits(plugin, umo_blacklist=[GROUP], blacklist_block_message="")
    with pytest.raises(GenerationTaskCreationError) as exc:
        create(plugin)
    assert exc.value.code == "rate_limited" and exc.value.message == ""
    assert not plugin.usage_manager._reservations


def test_opaque_api_scope_preserves_old_session_limit(plugin):
    set_limits(plugin, group_count=1, user_count=3)
    a = create(plugin, sender=None, scope="opaque-legacy-scope")
    assert a.quota_scopes == {"session": "opaque-legacy-scope"}
    settle(plugin, a, 1)
    with pytest.raises(GenerationTaskCreationError, match="会话"):
        create(plugin, "again", sender=None, scope="opaque-legacy-scope")


def test_provider_timeout_settles_and_releases(plugin):
    async def run():
        a = create(plugin)
        plugin.generator.generate_preconverted.side_effect = TimeoutError(
            "provider timeout"
        )
        await plugin.task_manager._run_generation_task(
            a.task_id,
            plugin.generation_executor.generate_and_send_image_async(
                "cat", GROUP, task_id=a.task_id, auto_send=False
            ),
        )
        await plugin.task_manager._wait_generation_notifications()
        assert a.status == GenerationTaskStatus.FAILED and a.quota_settled
        assert plugin.usage_manager.get_usage_count(GROUP) == 0
        assert not plugin.usage_manager._reservations

    asyncio.run(run())


def test_queued_cancel_and_api_wait_timeout(plugin):
    async def run():
        a = create(plugin)
        result = await plugin.public_api.wait_generation_result(
            a.task_id, timeout_seconds=0.001
        )
        assert result.code == "timeout"
        assert a.task_id in plugin.usage_manager._reservations
        ok, _ = plugin.task_manager.cancel_generation_task(
            a.task_id, unified_msg_origin=GROUP
        )
        assert ok
        await plugin.task_manager._wait_generation_notifications()
        assert a.quota_settled and not plugin.usage_manager._reservations
        assert plugin.usage_manager.get_usage_count(GROUP) == 0

    asyncio.run(run())


def test_llm_usage_payload_uses_real_counts_and_display_setting(plugin):
    from image_plugin.core.llm.result_handler import LLMResultHandler

    a = create(plugin)
    settle(plugin, a, 1)
    handler = LLMResultHandler.__new__(LLMResultHandler)
    handler.config_manager = plugin.config_manager
    payload = handler._build_task_result_payload(
        a, supports_image_input=False, delivery_paths=[]
    )
    assert payload["usage_summary"] == "📅 今日生图：个人 1/3，群聊 1/30"
    plugin.config_manager._plugin_config.generation_settings.result_info_items.clear()
    assert "usage_summary" not in handler._build_task_result_payload(
        a, supports_image_input=False, delivery_paths=[]
    )


def test_recovery_still_charges_original_scopes_if_limits_disabled(plugin, tmp_path):
    a = create(plugin)
    plugin.task_manager.record_generation_task_output(a.task_id, "/saved.png")
    usage = UsageManager(str(tmp_path), UsageSettings())
    tasks = TaskManager(persistence_file=tmp_path / "tasks.json")
    tasks.load_generation_history(usage)
    assert usage.get_usage_count(GROUP) == 1
    assert usage._user_usage[a.quota_date][a.quota_scopes["user"]] == 1


def test_llm_wakeup_preserves_recorded_human_identity(plugin, monkeypatch):
    import sys
    import types

    from image_plugin.core.llm import result_handler

    a = create(plugin)
    handler = result_handler.LLMResultHandler.__new__(result_handler.LLMResultHandler)
    handler.context = plugin.context
    core = types.ModuleType("astrbot.core.astr_main_agent")
    core.MainAgentBuildConfig = Mock()
    core._get_session_conv = AsyncMock()
    core.build_main_agent = Mock()
    monkeypatch.setitem(sys.modules, "astrbot.core.astr_main_agent", core)

    class Captured(Exception):
        pass

    factory = Mock(side_effect=Captured)
    monkeypatch.setattr(result_handler, "CronMessageEvent", factory)

    async def run():
        with pytest.raises(Captured):
            await handler._run_generation_result_agent(
                a, source_event=event(sender="later-sender")
            )

    asyncio.run(run())
    assert factory.call_args.kwargs["sender_id"] == "alice"
    assert str(factory.call_args.kwargs["session"]) == GROUP
