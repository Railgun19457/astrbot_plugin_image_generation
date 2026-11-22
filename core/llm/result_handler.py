"""LLM tool task result handling for image generation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from contextlib import aclosing, suppress
from copy import deepcopy
from dataclasses import replace
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.star import Context
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.tool import ToolSet
from astrbot.core.db.po import Conversation
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider

from ..config.manager import RESULT_INFO_USAGE, ConfigManager
from ..formatting.result import format_generation_failure_message
from ..generation.image_processor import ImageProcessor
from ..shared.logging import log_prefix, safe_log_text
from ..tasks.manager import TaskManager
from ..tasks.models import GenerationTaskRecord, GenerationTaskStatus
from .tools import DeliverySendMessageToUserTool, delivery_reference_key

IMAGE_GENERATION_TASK_WOKE_SYSTEM_PROMPT = (
    "You are an autonomous proactive agent.\n\n"
    "You are awakened because an image generation task you initiated earlier has "
    "completed. The completed task payload is provided in the user message.\n"
    "# CONTEXT\n"
    "1. The images already exist and their paths are listed in the payload. Never "
    "call `generate_image` again for this completed task.\n"
    "2. `send_message_to_user` is the only channel to the user here: it is the tool "
    "description's 'proactively message the user' case, so your final reply text is "
    "not delivered.\n"
    "3. Decide from the conversation whether the user should see this result now, "
    "later, or not at all. Staying silent is a valid choice when the images are "
    "working material for your own next steps, while a failed task is usually worth "
    "telling the user about.\n"
    "4. When you do report, send everything the user should see in one call: images "
    "with type='image' and the given path, wording with type='plain'. Send each "
    "image at most once.\n"
    "# IMAGE GENERATION TASK CONTEXT\n"
    "The completed task payload is provided in the user message."
)


class LLMResultHandler:
    """Handle LLM tool task submission text and task-result AI delivery."""

    def __init__(
        self,
        *,
        context: Context,
        config_manager: ConfigManager,
        task_manager: TaskManager,
        image_processor: ImageProcessor,
        create_background_task: Callable[
            [Coroutine[Any, Any, Any], str | None], asyncio.Task
        ],
    ) -> None:
        self.context = context
        self.config_manager = config_manager
        self.task_manager = task_manager
        self.image_processor = image_processor
        self._create_background_task = create_background_task

    def attach_task_wakeup(
        self,
        record: GenerationTaskRecord,
        *,
        source_event: AstrMessageEvent | None,
    ) -> None:
        """Attach an AI wakeup task to a completed LLM image generation task."""
        if not source_event:
            return

        def _on_done(current_record: GenerationTaskRecord) -> None:
            if current_record.status in {
                GenerationTaskStatus.CANCELLING,
                GenerationTaskStatus.CANCELLED,
            }:
                logger.debug(
                    f"{log_prefix('Task', record.task_id)} 生图任务已取消，跳过 AI 结果唤醒"
                )
                return
            self._create_background_task(
                self.wake_ai_for_generation_task_result(
                    task_id=record.task_id,
                    source_event=source_event,
                ),
                f"image_generation_ai_wakeup:{record.task_id}",
            )

        self.task_manager.add_generation_task_done_callback(record.task_id, _on_done)

    def format_tool_start_result(
        self,
        *,
        prompt: str,
        reference_image_count: int,
        preset: str | None,
        preset_label: str = "预设",
        presets: list[str] | None = None,
        personas: list[str] | None = None,
        aspect_ratio: str,
        resolution: str,
        image_count: int,
        task_id: str,
    ) -> str:
        """Format the successful LLM tool result for the agent."""
        model = ""
        if self.config_manager.adapter_config:
            model = (
                f"{self.config_manager.adapter_config.name}/"
                f"{self.config_manager.adapter_config.model}"
            )

        template_summary_parts = []
        preset_names = "、".join(presets or [])
        persona_names = "、".join(personas or [])
        if preset_names:
            template_summary_parts.append(f"预设={preset_names}")
        if persona_names:
            template_summary_parts.append(f"人设={persona_names}")
        if not template_summary_parts and preset:
            template_summary_parts.append(f"{preset_label}={preset}")

        lines = [
            "✅ 生图任务已提交，正在后台执行。",
            f"任务ID: {task_id}",
            "状态: 排队/运行中（尚未完成）",
            f"请求数量: {image_count}；结果图片数以任务完成为准",
            f"模式: {'图生图' if reference_image_count else '文生图'}",
            f"参考图: {reference_image_count}张",
            f"宽高比: {aspect_ratio}，分辨率: {resolution}",
        ]
        if model:
            lines.append(f"模型: {model}")
        if template_summary_parts:
            lines.append("模板: " + "，".join(template_summary_parts))
        if prompt:
            lines.append(f"提示词摘要: {safe_log_text(prompt, 120)}")

        lines.extend(
            [
                "",
                "内部指引:",
                "- 当前仅表示任务已提交，图片尚未完成，不要向用户声明已完成。",
                "- 结果就绪后会唤醒你并提供图片路径，届时由你判断是否需要发送给用户。",
                "- 不要重复调用 generate_image。",
            ]
        )
        return "\n".join(lines)

    def _build_task_result_payload(
        self,
        record: GenerationTaskRecord,
        *,
        supports_image_input: bool,
        delivery_paths: list[str],
    ) -> dict[str, Any]:
        """Build structured task result context for the awakened AI."""
        request_stats = record.request_stats
        payload: dict[str, Any] = {
            "task_id": record.task_id,
            "source": record.source,
            "status": record.status.value,
            "status_label": record.status_label,
            "prompt": record.prompt_summary,
            "reference_image_count": record.reference_image_count,
            "requested_count": record.requested_count,
            "current_index": record.current_index,
            "aspect_ratio": record.aspect_ratio,
            "resolution": record.resolution,
            "preset_label": record.preset_label,
            "preset": record.preset or "",
            "result_count": record.result_count,
            "result_paths": delivery_paths,
            "request_stats": request_stats,
            "items": [
                {
                    "index": item.index,
                    "status": item.status.value,
                    "result_count": item.result_count,
                    "error": safe_log_text(item.error, 160) if item.error else "",
                    "retry_attempts": item.retry_attempts,
                    "max_retry_attempts": item.max_retry_attempts,
                }
                for item in sorted(
                    record.items.values(), key=lambda task_item: task_item.index
                )
            ],
            "error": record.error,
            "message": record.message,
            "images_attached_to_model": supports_image_input
            and bool(record.result_paths),
        }
        if (
            self.config_manager.should_show_result_info(RESULT_INFO_USAGE)
            and record.quota_usage_summary
        ):
            payload["usage_summary"] = record.quota_usage_summary
            payload["usage_instruction"] = (
                "用量和豁免状态以 usage_summary 为准，不要推断发送者身份或额度。"
            )
        if record.duration_seconds is not None:
            payload["duration_seconds"] = round(record.duration_seconds, 2)
        return payload

    def _build_fallback_chain(
        self, record: GenerationTaskRecord, pending_paths: list[str]
    ) -> MessageChain:
        """Build a direct user-facing fallback message when AI handling fails."""
        chain = MessageChain()
        if record.error:
            chain.message(format_generation_failure_message(record.error))
            return chain

        if not pending_paths:
            chain.message("❌ 生成失败: 未能获取生成图片")
            return chain

        for file_path in pending_paths:
            chain.file_image(file_path)
        info_parts = [f"🧾 任务ID: {record.task_id}"]
        info_parts.append(f"🖼️ 数量: {len(pending_paths)}张")
        chain.message("\n" + "\n".join(info_parts))
        return chain

    async def _send_fallback(
        self,
        record: GenerationTaskRecord,
        *,
        reason: str,
    ) -> str:
        """Directly send task result when the wake-up turn could not run.

        Args:
            record: Completed generation task record.
            reason: Why the fallback delivery was needed.

        Returns:
            Plain text of the sent fallback message, or an empty string when
            nothing was sent.

        """
        sent_keys = {delivery_reference_key(path) for path in record.delivered_paths}
        pending_paths = [
            path
            for path in record.result_paths
            if delivery_reference_key(path) not in sent_keys
        ]
        if not pending_paths and (record.result_paths or record.delivered):
            return ""

        logger.warning(
            f"{log_prefix('Task', record.task_id)} 唤醒轮次未能完成，改为直接发送: {safe_log_text(reason, 200)}"
        )
        chain = self._build_fallback_chain(record, pending_paths)
        sent = await self.context.send_message(record.unified_msg_origin, chain)
        if not sent:
            return ""
        # 失败兜底只发文字，此时不能把结果图片记成已送达。
        self.task_manager.mark_generation_task_delivered(
            record.task_id,
            paths=[] if record.error else pending_paths,
        )
        return chain.get_plain_text()

    async def _save_task_note_to_history(
        self,
        *,
        record: GenerationTaskRecord,
        conversation: Conversation | None,
        note: str,
        token_usage: int | None = None,
    ) -> None:
        """Append the proactive turn note to the session conversation history.

        The history is re-read right before appending so a user turn that landed
        while the result was being delivered is not overwritten.

        Args:
            record: Completed generation task record.
            conversation: Conversation bound to the proactive turn.
            note: Assistant note describing the delivered task result.
            token_usage: Optional total token usage of the proactive turn.

        """
        if not conversation:
            return
        conversation_manager = self.context.conversation_manager
        latest = await conversation_manager.get_conversation(
            record.unified_msg_origin,
            conversation.cid,
        )
        try:
            history = json.loads(getattr(latest, "history", None) or "[]")
        except (json.JSONDecodeError, TypeError):
            history = []
        if not isinstance(history, list):
            history = []
        history.append({"role": "assistant", "content": note})
        await conversation_manager.update_conversation(
            record.unified_msg_origin,
            conversation.cid,
            history=history,
            token_usage=token_usage,
        )

    async def wake_ai_for_generation_task_result(
        self,
        *,
        task_id: str,
        source_event: AstrMessageEvent,
    ) -> None:
        """Wake the main agent to process a completed image generation task."""
        record = self.task_manager.get_generation_task(task_id)
        if not record:
            logger.warning(
                f"{log_prefix('Task', task_id)} 生图任务记录不存在，无法唤醒 AI"
            )
            return
        if record.status == GenerationTaskStatus.CANCELLED:
            logger.debug(f"{log_prefix('Task', task_id)} 生图任务已取消，不唤醒 AI")
            return
        if record.unified_msg_origin != source_event.unified_msg_origin:
            await self._send_fallback(
                record,
                reason="任务会话与来源事件不一致",
            )
            return

        try:
            await self._run_generation_result_agent(
                record,
                source_event=source_event,
            )
        except Exception as exc:
            logger.error(
                f"{log_prefix('Task', task_id)} 唤醒 AI 处理生图结果异常: {exc}",
                exc_info=True,
            )
            await self._send_fallback(record, reason=f"唤醒轮次异常: {exc}")

    async def _run_generation_result_agent(
        self,
        record: GenerationTaskRecord,
        *,
        source_event: AstrMessageEvent,
    ) -> None:
        """Run a proactive agent turn with the completed task result."""
        if callable(getattr(getattr(self.context, "agent", None), "run_turn", None)):
            await self._run_generation_result_agent_v430(record, source_event)
            return

        from astrbot.core.astr_main_agent import (
            MainAgentBuildConfig,
            _get_session_conv,
            build_main_agent,
        )
        from astrbot.core.cron.events import CronMessageEvent

        origin = record.unified_msg_origin
        session = MessageSession.from_str(origin)
        sender_id = record.sender_id or source_event.get_sender_id()
        event = CronMessageEvent(
            context=self.context,
            session=session,
            message=f"图像生成任务 {record.task_id} 已完成",
            sender_id=sender_id,
            sender_name="ImageGeneration",
            message_type=session.message_type,
        )
        # CronMessageEvent uses sender_id for self_id, so restore the caller.
        event.message_obj.sender.user_id = sender_id
        event.message_obj.group_id = source_event.get_group_id()
        event.role = source_event.role
        event.plugins_name = source_event.plugins_name
        event.set_extra(
            "_api_key_allow_admin_role",
            source_event.get_extra("_api_key_allow_admin_role"),
        )

        conversation = await _get_session_conv(event=event, plugin_context=self.context)
        provider = self.context.get_using_provider(origin)
        req, tool, delivery_paths = self._build_result_request(
            record, provider, conversation
        )
        settings = self.context.get_config(umo=origin).get("provider_settings", {})
        config = MainAgentBuildConfig(
            tool_call_timeout=settings.get("tool_call_timeout", 120),
            llm_safety_mode=False,
            streaming_response=False,
            provider_settings=settings,
            computer_use_runtime=settings.get("computer_use_runtime", "none"),
            sandbox_cfg=settings.get("sandbox", {}),
            add_cron_tools=settings.get("proactive_capability", {}).get(
                "add_cron_tools", True
            ),
        )
        result = await build_main_agent(
            event=event,
            plugin_context=self.context,
            config=config,
            provider=provider,
            req=req,
            apply_reset=False,
        )
        if result is None:
            raise RuntimeError("Could not build the image task result agent")

        result.provider_request.func_tool.add_tool(tool)
        if result.reset_coro:
            await result.reset_coro
        response = await self._deliver_result(record, result.agent_runner, tool)
        if event.is_stopped() or result.agent_runner.was_aborted():
            return
        with suppress(Exception):
            await self._save_task_note_to_history(
                record=record,
                conversation=conversation,
                note=self._task_note(record, tool, delivery_paths),
                token_usage=response.usage.total
                if response and response.usage
                else None,
            )

    async def _run_generation_result_agent_v430(
        self, record: GenerationTaskRecord, source_event: AstrMessageEvent
    ) -> None:
        """Run a native image result turn with read-only conversation context."""
        from astrbot.core.agent_input import prepare_agent_input
        from astrbot.core.astr_agent_runtime.main_agent.builder import build_main_agent
        from astrbot.core.astr_agent_runtime.main_agent.config import (
            MainAgentBuildConfig,
        )
        from astrbot.core.astr_agent_runtime.main_agent.persona import apply_persona
        from astrbot.core.platform.agent_wakeup_event import AgentWakeupEvent
        from astrbot.core.utils.active_event_registry import active_event_registry
        from astrbot.core.utils.session_lock import session_lock_manager

        origin = record.unified_msg_origin
        session = MessageSession.from_str(origin)
        platform = self.context.get_platform_inst(session.platform_id)
        if platform is None:
            raise RuntimeError("Image result platform is unavailable")
        event = AgentWakeupEvent(
            context=self.context,
            session=session,
            platform_meta=platform.meta(),
            prompt=f"图像生成任务 {record.task_id} 已完成",
            source="plugin",
            source_id=record.task_id,
            role=source_event.role,
        )
        event.message_obj.sender.user_id = source_event.get_sender_id()
        event.message_obj.sender.nickname = source_event.get_sender_name()
        event.message_obj.self_id = source_event.get_self_id()
        event.message_obj.group_id = source_event.get_group_id()
        event.set_extra(
            "_api_key_allow_admin_role",
            source_event.get_extra("_api_key_allow_admin_role"),
        )
        admitted = await prepare_agent_input(self.context, event, streaming=False)
        if admitted is None:
            return

        manager = self.context.conversation_manager
        session_id = getattr(
            getattr(source_event, "agent_session", None), "session_id", None
        )
        if session_id is None:
            source_request = source_event.get_extra("provider_request")
            session_id = getattr(
                getattr(source_request, "conversation", None), "cid", None
            )
        conversation = None
        # The source turn may still be committing its tool calls and results.
        async with session_lock_manager.acquire_lock(origin):
            if session_id is None:
                session_id = await manager.get_curr_conversation_id(origin)
            if session_id is not None:
                conversation = await manager.get_conversation(origin, session_id)
                if conversation is None or conversation.user_id != origin:
                    raise RuntimeError("Image result conversation is unavailable")

        provider = await self.context.get_using_provider_async(origin)
        if provider is None:
            raise RuntimeError("No chat provider available for the image result")
        req, tool, delivery_paths = self._build_result_request(
            record, provider, conversation
        )
        config = replace(
            MainAgentBuildConfig.from_config(
                deepcopy(dict(self.context.get_config(umo=origin)))
            ),
            streaming_response=False,
            enable_message_delivery=True,
            validate_provider_endpoint=True,
        )
        # Apply the conversation persona, then build without a conversation writer.
        # Only the delivery summary should become part of the user's next turn.
        await apply_persona(
            req,
            config.provider_settings,
            self.context,
            event,
            plugin_tools=admitted.plugin_tools,
            plugin_skills=admitted.plugin_skills,
        )
        req.conversation = None
        result = await build_main_agent(
            event=event,
            plugin_context=self.context,
            config=config,
            provider=provider,
            req=req,
            plugin_tools=admitted.plugin_tools,
            plugin_skills=admitted.plugin_skills,
        )
        if result is None:
            raise RuntimeError("Could not build the image task result agent")

        result.provider_request.func_tool.add_tool(tool)
        runner = result.agent_runner
        runner.run_context.context.permissions = admitted.permissions
        active_event_registry.register_agent_stop_callback(event, runner.request_stop)
        try:
            await self._deliver_result(record, runner, tool)
        finally:
            active_event_registry.unregister_agent_stop_callback(
                event, runner.request_stop
            )
        if event.is_stopped() or runner.was_aborted() or conversation is None:
            return
        try:
            async with session_lock_manager.acquire_lock(origin):
                writer = await manager.create_event_writer(
                    conversation.cid, unified_msg_origin=origin, link_display=False
                )
                await writer.append_message(
                    {
                        "role": "assistant",
                        "content": self._task_note(record, tool, delivery_paths),
                    }
                )
        except Exception:
            logger.warning(
                f"{log_prefix('Task', record.task_id)} 保存生图结果摘要失败",
                exc_info=True,
            )

    def _build_result_request(
        self,
        record: GenerationTaskRecord,
        provider: Provider | None,
        conversation: Conversation | None,
    ) -> tuple[ProviderRequest, DeliverySendMessageToUserTool, list[str]]:
        """Build the same result prompt and tracking tool for both agent versions."""
        result_paths = [path for path in record.result_paths if path.strip()]
        delivery_paths = self.image_processor.ensure_delivery_paths(result_paths)
        references = {
            delivery_reference_key(reference): original
            for original, delivery in zip(result_paths, delivery_paths, strict=True)
            for reference in (original, delivery)
        }
        tool = DeliverySendMessageToUserTool(result_paths_by_reference=references)
        supports_images = bool(
            provider
            and "image" in provider.provider_config.get("modalities", [])
            and delivery_paths
        )
        payload = self._build_task_result_payload(
            record,
            supports_image_input=supports_images,
            delivery_paths=delivery_paths,
        )
        req = ProviderRequest(
            conversation=conversation,
            contexts=json.loads(conversation.history or "[]") if conversation else [],
            system_prompt=IMAGE_GENERATION_TASK_WOKE_SYSTEM_PROMPT,
            prompt=(
                "Proceed according to your system instructions. "
                "Output using same language as previous conversation. "
                "Review the completed image generation task and decide whether the user "
                "should see the result; deliver it with `send_message_to_user` when they "
                "should, otherwise end this turn without sending anything."
            ),
            func_tool=ToolSet(tools=[tool]),
        )
        req.extra_user_content_parts.append(
            TextPart(
                text=(
                    "<image_generation_task_result>\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
                    "</image_generation_task_result>"
                )
            )
        )
        if supports_images:
            req.image_urls = list(delivery_paths)
        return req, tool, delivery_paths

    async def _deliver_result(
        self,
        record: GenerationTaskRecord,
        runner: Any,
        tool: DeliverySendMessageToUserTool,
    ) -> LLMResponse | None:
        """Run the agent and supplement failed delivery with any unsent results."""
        try:
            async with aclosing(runner.step_until_done(30)) as steps:
                async for _ in steps:
                    pass
        finally:
            if tool.delivered:
                self.task_manager.mark_generation_task_delivered(
                    record.task_id, paths=tool.delivered_paths
                )
        response = runner.get_final_llm_resp()
        if not runner.was_aborted() and (
            tool.delivery_failed or (response is not None and response.role == "err")
        ):
            await self._send_fallback(record, reason="Agent 未能完成结果投递")
        return response

    @staticmethod
    def _task_note(
        record: GenerationTaskRecord,
        tool: DeliverySendMessageToUserTool,
        delivery_paths: list[str],
    ) -> str:
        """Summarize confirmed delivery using references the agent can send."""
        delivered = {delivery_reference_key(path) for path in record.delivered_paths}
        sent, remaining = [], []
        for path in delivery_paths:
            original = tool.result_paths_by_reference[delivery_reference_key(path)]
            (
                sent if delivery_reference_key(original) in delivered else remaining
            ).append(path)
        note = (
            f"[ImageGenerationTask] task_id={record.task_id}, "
            f"status={record.status.value}, result_count={record.result_count}, "
            f"delivered={str(record.delivered).lower()}, error={record.error or ''}"
        )
        if sent:
            note += (
                f", delivered_paths={json.dumps(sent, ensure_ascii=False)}. "
                "These images were already sent to the user; do not send them again."
            )
        if remaining:
            note += (
                f", unsent_paths={json.dumps(remaining, ensure_ascii=False)}. "
                "Send these only when the conversation calls for it."
            )
        return note
