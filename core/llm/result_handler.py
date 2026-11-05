"""LLM tool task result handling for image generation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Coroutine
from contextlib import suppress
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.star import Context
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.tool import ToolSet
from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.db.po import Conversation
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.provider.entities import ProviderRequest

from ..config.manager import RESULT_INFO_USAGE, ConfigManager
from ..formatting.result import format_generation_failure_message
from ..generation.image_processor import ImageProcessor
from ..shared.logging import log_prefix, safe_log_text
from ..tasks.manager import TaskManager
from ..tasks.models import GenerationTaskRecord, GenerationTaskStatus
from .tools import DeliverySendMessageToUserTool

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

    def _build_fallback_chain(self, record: GenerationTaskRecord) -> MessageChain:
        """Build a direct user-facing fallback message when AI handling fails."""
        chain = MessageChain()
        if record.error:
            chain.message(format_generation_failure_message(record.error))
            return chain

        if not record.result_paths:
            chain.message("❌ 生成失败: 未能获取生成图片")
            return chain

        for file_path in record.result_paths:
            chain.file_image(file_path)
        info_parts = [f"🧾 任务ID: {record.task_id}"]
        if record.result_count:
            info_parts.append(f"🖼️ 数量: {record.result_count}张")
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
        logger.warning(
            f"{log_prefix('Task', record.task_id)} 唤醒轮次未能完成，改为直接发送: {safe_log_text(reason, 200)}"
        )
        chain = self._build_fallback_chain(record)
        sent = await self.context.send_message(record.unified_msg_origin, chain)
        if not sent:
            return ""
        # 失败兜底只发文字，此时不能把结果图片记成已送达。
        self.task_manager.mark_generation_task_delivered(
            record.task_id,
            paths=[] if record.error else record.result_paths,
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
            if record.delivered:
                # The AI already delivered the result before the failure.
                logger.warning(
                    f"{log_prefix('Task', task_id)} 结果已发送给用户，跳过兜底发送"
                )
                return
            # 只有唤醒轮次本身异常（AI 没能做出判断）时才兜底直发；
            # AI 正常结束但选择不发送属于它的判断，不会被兜底。
            await self._send_fallback(record, reason=f"唤醒轮次异常: {exc}")

    async def _run_generation_result_agent(
        self,
        record: GenerationTaskRecord,
        *,
        source_event: AstrMessageEvent,
    ) -> None:
        """Run a proactive agent turn with the completed task result."""
        from astrbot.core.astr_main_agent import (
            MainAgentBuildConfig,
            _get_session_conv,
            build_main_agent,
        )

        session = MessageSession.from_str(record.unified_msg_origin)
        cron_event = CronMessageEvent(
            context=self.context,
            session=session,
            message=f"图像生成任务 {record.task_id} 已完成",
            # Preserve the original human identity for any further tool calls.
            sender_id=record.sender_id or source_event.get_sender_id(),
            sender_name="ImageGeneration",
            message_type=session.message_type,
        )
        cron_event.role = source_event.role
        cron_event.plugins_name = source_event.plugins_name

        cfg = self.context.get_config(umo=record.unified_msg_origin)
        provider_settings = cfg.get("provider_settings", {})
        tool_call_timeout = provider_settings.get("tool_call_timeout", 120)
        provider = self.context.get_using_provider(record.unified_msg_origin)
        supports_image_input = bool(
            provider
            and "image" in provider.provider_config.get("modalities", [])
            and record.result_paths
        )
        # 核心发送工具只允许发送会话工作区与 AstrBot 临时目录下的文件，
        # 持久化目录中的图片需要先映射到临时目录再交给唤醒后的 AI 投递。
        delivery_paths = self.image_processor.ensure_delivery_paths(record.result_paths)
        payload = self._build_task_result_payload(
            record,
            supports_image_input=supports_image_input,
            delivery_paths=delivery_paths,
        )

        req = ProviderRequest()
        req.conversation = await _get_session_conv(
            event=cron_event,
            plugin_context=self.context,
        )
        # 直接使用真实会话消息作为上下文：把历史压成文本塞进 system prompt 会让
        # system prompt 每轮都变，provider 的前缀缓存无法复用。
        history_context = json.loads(req.conversation.history or "[]")
        if history_context:
            req.contexts = history_context

        req.system_prompt += IMAGE_GENERATION_TASK_WOKE_SYSTEM_PROMPT
        req.prompt = (
            "Proceed according to your system instructions. "
            "Output using same language as previous conversation. "
            "Review the completed image generation task and decide whether the user "
            "should see the result; deliver it with `send_message_to_user` when they "
            "should, otherwise end this turn without sending anything."
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
        if supports_image_input:
            req.image_urls = list(delivery_paths)
        else:
            for path in delivery_paths:
                req.extra_user_content_parts.append(
                    TextPart(text=f"[Generated image path: {path}]")
                )

        delivery_tool = DeliverySendMessageToUserTool()
        # 用插件自己的工具集作为载体，让 build_main_agent 把全量工具合并到这个对象上；
        # 若这里留空，core 会直接把它的工具集对象赋给 req.func_tool。
        req.func_tool = ToolSet()
        req.func_tool.add_tool(delivery_tool)

        # 与普通轮一致地继承用户配置：Computer Use 运行时与主动型 Agent 的定时工具。
        config = MainAgentBuildConfig(
            tool_call_timeout=tool_call_timeout,
            llm_safety_mode=False,
            streaming_response=False,
            provider_settings=provider_settings,
            computer_use_runtime=provider_settings.get("computer_use_runtime", "none"),
            sandbox_cfg=provider_settings.get("sandbox", {}),
            add_cron_tools=provider_settings.get("proactive_capability", {}).get(
                "add_cron_tools", True
            ),
        )
        result = await build_main_agent(
            event=cron_event,
            plugin_context=self.context,
            config=config,
            provider=provider,
            req=req,
            apply_reset=False,
        )
        if not result:
            await self._send_fallback(record, reason="无法构建主 Agent")
            return

        # 唤醒轮保留完整工具集，只把 send_message_to_user 换回带投递记账的变体：
        # build_main_agent 合并的 persona/插件工具集里有核心的同名工具，会把变体覆盖掉，
        # 换掉后插件才能确认结果是否真的送达。必须在 await reset_coro 之前完成，
        # 因为 reset 持有的是同一个 request 对象。
        result.provider_request.func_tool.add_tool(delivery_tool)
        if result.reset_coro:
            await result.reset_coro

        runner = result.agent_runner
        async for _resp in runner.step_until_done(30):
            # 是否发送、发送什么由 AI 决定，本轮不代它发送任何内容：这里只负责把
            # 这一轮完整跑完，投递记录由 send_message_to_user 变体自己写入。
            pass

        llm_resp = runner.get_final_llm_resp()
        # 投递是否发生以投递工具自身的记录为准（它只在核心工具真正发送成功后写入）。
        # 注意 delivered 只说明有内容发出，AI 也可能只发了文字，所以图片是否送达要看
        # delivered_paths。
        if delivery_tool.delivered:
            self.task_manager.mark_generation_task_delivered(
                record.task_id,
                paths=delivery_tool.delivered_paths,
            )
        else:
            logger.debug(
                f"{log_prefix('Task', record.task_id)} 本轮未通过 send_message_to_user "
                f"投递结果（会话发送标记={cron_event._has_send_oper}）"
            )

        summary_note = (
            f"[ImageGenerationTask] task_id={record.task_id}, "
            f"status={record.status.value}, result_count={record.result_count}, "
            f"delivered={str(record.delivered).lower()}, "
            f"error={record.error or ''}"
        )
        # 历史 note 统一使用投递路径（AstrBot 临时目录）：这是核心发送工具唯一允许
        # 发送的位置，持久化目录的路径模型即使拿到也用不了；投递路径是持久化结果的
        # 同名链接，因此按文件名判断哪些图已经发过。
        delivered_names = {os.path.basename(path) for path in record.delivered_paths}
        unsent_paths = [
            path
            for path in delivery_paths
            if os.path.basename(path) not in delivered_names
        ]
        if record.delivered_paths:
            summary_note += (
                ", delivered_paths="
                f"{json.dumps(record.delivered_paths, ensure_ascii=False)}. "
                "These images were already sent to the user; do not send them again."
            )
            if unsent_paths:
                summary_note += (
                    ", unsent_paths="
                    f"{json.dumps(unsent_paths, ensure_ascii=False)}. "
                    "These images were generated but not sent to the user; send them "
                    "only when the conversation calls for it."
                )
        elif cron_event._has_send_oper:
            # 唤醒轮保留完整工具集，结果也可能由本插件之外的发送工具发出，
            # 此时记账不完整，不要断言"未发送"以免后续重复发送。
            summary_note += (
                ", delivered_paths=unknown. A message already reached the user in this "
                "turn, but the delivered images could not be recorded; check the "
                "conversation before sending them again."
            )
        elif unsent_paths:
            summary_note += (
                ", unsent_paths="
                f"{json.dumps(unsent_paths, ensure_ascii=False)}. "
                "These images were generated but not sent to the user; send them only "
                "when the conversation calls for it."
            )
        if llm_resp and llm_resp.completion_text:
            summary_note += f" AI response: {llm_resp.completion_text}"
        with suppress(Exception):
            await self._save_task_note_to_history(
                record=record,
                conversation=req.conversation,
                note=summary_note,
                token_usage=llm_resp.usage.total
                if llm_resp and llm_resp.usage
                else None,
            )
