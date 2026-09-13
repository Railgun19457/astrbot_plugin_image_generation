"""Regression tests for LLM tool context reference images."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from core.generation.context_image_cache import RecentContextImageCache
from core.generation.image_processor import ImageProcessor
from core.generation.reference_collector import (
    ContextReferenceImageNotFoundError,
    collect_tool_reference_images,
    extract_latest_user_context_images,
)
from core.llm.tools import _start_generation_task
from core.shared.types import ImageCapability, ImageData


class FakeImageProcessor:
    def __init__(
        self,
        *,
        event_images: list[ImageData] | None = None,
        downloads: dict[str, ImageData] | None = None,
        fail_on_fetch: bool = False,
        fail_on_download: bool = False,
    ) -> None:
        self.event_images = event_images or []
        self.downloads = downloads or {}
        self.fail_on_fetch = fail_on_fetch
        self.fail_on_download = fail_on_download

    def workspace_dir_for_origin(self, _origin):
        return None

    async def fetch_images_from_event(self, _event):
        if self.fail_on_fetch:
            raise AssertionError("纯文生图不应读取消息图片")
        return self.event_images

    async def download_image(self, reference, *, workspace_dir=None):
        if self.fail_on_download:
            raise AssertionError("当前事件已有图片时不应重复读取上下文副本")
        return self.downloads.get(reference)

    async def get_avatar(self, _user_id):
        return None


def test_extracts_images_from_latest_image_bearing_user_message() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,older"},
                }
            ],
        },
        {"role": "assistant", "content": "看到了"},
        {"role": "user", "content": "把上一张图的衣服改成绿色"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,assistant"},
                }
            ],
        },
    ]

    assert extract_latest_user_context_images(messages) == [
        "data:image/png;base64,older"
    ]


def test_extracts_image_url_from_astrbot_content_part() -> None:
    image_part = SimpleNamespace(
        type="image_url",
        image_url=SimpleNamespace(url="https://example.com/latest.png"),
    )
    messages = [SimpleNamespace(role="user", content=[image_part])]

    assert extract_latest_user_context_images(messages) == [
        "https://example.com/latest.png"
    ]


def test_context_image_search_is_limited_to_recent_messages() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/stale.png"},
                }
            ],
        },
        *({"role": "user", "content": f"message {index}"} for index in range(20)),
    ]

    assert extract_latest_user_context_images(messages) == []


def test_image_processor_accepts_astrbot_history_data_url(tmp_path) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_bytes, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(
        image_bytes.getvalue()
    ).decode("ascii")
    processor = ImageProcessor(str(tmp_path), max_image_size_mb=1)

    result = asyncio.run(processor.download_image(data_url))

    assert result is not None
    assert result.mime_type == "image/png"
    assert result.data == image_bytes.getvalue()


def test_llm_tool_collects_current_event_images_when_requested() -> None:
    event_image = ImageData(data=b"current-event-image", mime_type="image/png")
    event = SimpleNamespace(unified_msg_origin="Sora:FriendMessage:1")
    images = asyncio.run(
        collect_tool_reference_images(
            FakeImageProcessor(
                event_images=[event_image],
                fail_on_download=True,
            ),
            event,
            capabilities=ImageCapability.IMAGE_TO_IMAGE,
            context_images=["data:image/png;base64,duplicate"],
            use_context_images=True,
        )
    )

    assert images == [event_image]


def test_llm_tool_uses_context_image_when_current_event_has_none() -> None:
    context_image = ImageData(data=b"context-image", mime_type="image/jpeg")
    event = SimpleNamespace(unified_msg_origin="Sora:FriendMessage:1")
    images = asyncio.run(
        collect_tool_reference_images(
            FakeImageProcessor(
                downloads={"data:image/jpeg;base64,context": context_image}
            ),
            event,
            capabilities=ImageCapability.IMAGE_TO_IMAGE,
            context_images=["data:image/jpeg;base64,context"],
            use_context_images=True,
        )
    )

    assert images == [context_image]


def test_llm_tool_falls_back_to_cached_session_image() -> None:
    cached_image = ImageData(data=b"cached-image", mime_type="image/jpeg")
    event = SimpleNamespace(unified_msg_origin="Sora:FriendMessage:1")

    images = asyncio.run(
        collect_tool_reference_images(
            FakeImageProcessor(),
            event,
            capabilities=ImageCapability.IMAGE_TO_IMAGE,
            cached_context_images=[cached_image],
            use_context_images=True,
        )
    )

    assert images == [cached_image]


def test_recent_context_image_cache_expires_and_evicts_lru_sessions() -> None:
    now = [100.0]
    cache = RecentContextImageCache(
        ttl_seconds=10,
        max_sessions=2,
        max_bytes=10,
        clock=lambda: now[0],
    )
    first = ImageData(data=b"111", mime_type="image/png")
    second = ImageData(data=b"22", mime_type="image/png")
    third = ImageData(data=b"3", mime_type="image/png")

    cache.put("first", [first])
    cache.put("second", [second])
    assert cache.get("first") == [first]
    cache.put("third", [third])

    assert cache.get("second") == []
    assert cache.get("first") == [first]
    now[0] = 111.0
    assert cache.get("first") == []
    assert cache.get("third") == []


def test_recent_context_image_cache_rejects_oversized_entry() -> None:
    cache = RecentContextImageCache(max_bytes=2)
    oversized = ImageData(data=b"123", mime_type="image/png")

    cache.put("session", [oversized])

    assert cache.get("session") == []


def test_llm_tool_does_not_collect_event_images_for_text_to_image() -> None:
    event = SimpleNamespace(unified_msg_origin="Sora:FriendMessage:1")
    images = asyncio.run(
        collect_tool_reference_images(
            FakeImageProcessor(fail_on_fetch=True),
            event,
            capabilities=ImageCapability.IMAGE_TO_IMAGE,
        )
    )

    assert images == []


def test_llm_tool_rejects_image_edit_without_a_usable_context_image() -> None:
    event = SimpleNamespace(unified_msg_origin="Sora:FriendMessage:1")

    with pytest.raises(ContextReferenceImageNotFoundError):
        asyncio.run(
            collect_tool_reference_images(
                FakeImageProcessor(),
                event,
                capabilities=ImageCapability.IMAGE_TO_IMAGE,
                use_context_images=True,
            )
        )


def test_missing_context_image_does_not_create_text_to_image_task() -> None:
    class FakeUsageManager:
        def __init__(self):
            self.released = 0

        def check_rate_limit(self, *_args, **_kwargs):
            return None

        def release_reserved_usage(self, *_args, count, **_kwargs):
            self.released += count

    class FakeSafetyAuditor:
        async def audit_prompt(self, *_args, **_kwargs):
            return True, ""

    class FakeTaskManager:
        def get_generation_queue_rejection(self):
            return None

    usage_manager = FakeUsageManager()
    plugin = SimpleNamespace(
        generator=SimpleNamespace(
            adapter=SimpleNamespace(
                get_capabilities=lambda: ImageCapability.IMAGE_TO_IMAGE
            )
        ),
        image_processor=FakeImageProcessor(),
        usage_manager=usage_manager,
        safety_auditor=FakeSafetyAuditor(),
        task_manager=FakeTaskManager(),
        normalize_image_count=lambda count: count,
        is_usage_limit_admin=lambda _event: False,
        has_required_api_key=lambda: True,
        create_generation_task=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("缺少上下文图片时不应创建文生图任务")
        ),
    )
    event = SimpleNamespace(unified_msg_origin="Sora:FriendMessage:1")

    result = asyncio.run(
        _start_generation_task(
            plugin,
            event,
            prompt="把图片背景改成夜晚",
            aspect_ratio="不指定",
            resolution="不指定",
            use_context_images=True,
        )
    )

    assert "未找到可用的聊天图片" in result
    assert usage_manager.released == 1
