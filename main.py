"""AstrBot image generation plugin entrypoint."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.star_tools import StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .core.config.manager import (
    LLM_TOOL_IMAGE_GENERATION,
    LLM_TOOL_PRESET_EDIT,
    LLM_TOOL_PRESET_QUERY,
    LLM_TOOL_TASK_MANAGEMENT,
    ConfigManager,
)
from .core.generation.executor import GenerationExecutor
from .core.tasks.models import (
    GenerationTaskCreationError,
    GenerationTaskRecord,
    GenerationTaskStatus,
)
from .core.adapters.generator import ImageGenerator
from .core.generation.image_processor import ImageProcessor
from .core.llm.command_handoff import try_request_llm_handoff
from .core.llm.result_handler import LLMResultHandler
from .core.llm.tools import (
    ImageGenerationTool,
    ImageTaskTool,
    PresetEditTool,
    PresetQueryTool,
    adjust_tool_parameters,
)
from .core.shared.constants import LLM_HANDLED_COMMAND_GENERATE
from .core.shared.logging import (
    log_prefix,
    mask_sensitive,
    safe_log_error_body,
    safe_log_text,
)
from .core.api.public import ImageGenerationPublicAPI
from .core.generation.reference_collector import collect_command_reference_images
from .core.formatting.result import (
    format_image_command_help as render_image_command_help,
)
from .core.formatting.result import (
    format_start_task_message as render_start_task_message,
)
from .core.formatting.result import (
    format_task_detail as render_task_detail,
)
from .core.formatting.result import (
    format_task_list as render_task_list,
)
from .core.audit.safety import SafetyAuditor
from .core.page import ImageGenerationPageAPI
from .core.tasks.ids import new_task_id
from .core.tasks.manager import TaskManager
from .core.config.templates import (
    build_generation_prompt,
    find_named_entry,
    format_template_summary,
    parse_preset_prompt,
)
from .core.shared.types import ImageCapability, ImageData
from .core.tasks.usage import UsageManager

LOG = log_prefix("Plugin")


class ImageGenerationPlugin(Star):
    """Main image generation plugin class."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context

        # Store persistent data in the plugin directory and images in AstrBot temp.
        self.data_dir = StarTools.get_data_dir()
        self.astrbot_temp_dir = Path(get_astrbot_temp_path())
        self.image_temp_dir = self.astrbot_temp_dir / "astrbot_plugin_image_generation"
        self.image_temp_dir.mkdir(parents=True, exist_ok=True)
        self.page_upload_dir = self.image_temp_dir / "page_uploads"
        self.page_upload_dir.mkdir(parents=True, exist_ok=True)

        # Initialize configuration management.
        self.config_manager = ConfigManager(config)

        # Initialize usage accounting.
        self.usage_manager = UsageManager(
            str(self.data_dir), self.config_manager.usage_settings
        )

        # Initialize image processing.
        self.image_processor = ImageProcessor(
            str(self.image_temp_dir),
            self.config_manager.usage_settings.max_image_size_mb,
            str(self.data_dir),
            allowed_local_base_dirs=[str(self.astrbot_temp_dir)],
        )

        # Initialize task management.
        self.task_manager = TaskManager(
            generation_history_limit=self.config_manager.generation_task_history_limit,
            generation_history_retention_days=self.config_manager.generation_task_history_retention_days,
            enable_generation_task_history=self.config_manager.enable_generation_task_history,
            max_queued_generation_tasks=self.config_manager.max_queued_generation_tasks,
            persistence_file=self.data_dir / "generation_tasks.json",
        )

        # Initialize LLM tool result handling.
        self.llm_result_handler = LLMResultHandler(
            context=self.context,
            config_manager=self.config_manager,
            task_manager=self.task_manager,
            create_background_task=self.create_background_task,
        )

        # Initialize safety auditing.
        self.safety_auditor = SafetyAuditor(self.context, self.config_manager)

        # Initialize generation execution.
        self.generation_executor = GenerationExecutor(
            context=self.context,
            config_manager=self.config_manager,
            image_processor=self.image_processor,
            task_manager=self.task_manager,
            usage_manager=self.usage_manager,
            safety_auditor=self.safety_auditor,
        )

        # Initialize the public API exposed to other plugins.
        self.public_api = ImageGenerationPublicAPI(self)

        # Initialize the generator lazily after configuration is loaded.
        self.generator: ImageGenerator | None = None

        # Register WebUI Plugin Page APIs.
        self.page_api = ImageGenerationPageAPI(self)
        self.page_api.register()

    # Lifecycle.

    async def initialize(self):
        """Run when the plugin is loaded."""
        if self.config_manager.adapter_config:
            self.generator = ImageGenerator(self.config_manager.adapter_config)
            self.generation_executor.update_generator(self.generator)
            self.generation_executor.refresh_request_semaphore()
            logger.info(
                f"{LOG} 初始化生图生成器: "
                f"供应商={safe_log_text(self.config_manager.adapter_config.name)}，"
                f"模型={safe_log_text(self.config_manager.adapter_config.model)}，"
                f"最大并发生图请求={self.config_manager.max_concurrent_tasks}，"
                f"最大并发完整任务={self.config_manager.max_running_generation_tasks}，"
                f"最大排队任务={self.config_manager.max_queued_generation_tasks}"
            )
        else:
            logger.error(f"{LOG} 适配器配置加载失败，插件未初始化")

        self.task_manager.load_generation_history()
        self.task_manager.configure_generation_queue(
            max_queued_generation_tasks=self.config_manager.max_queued_generation_tasks
        )
        self.task_manager.start_generation_workers(
            self.config_manager.max_running_generation_tasks
        )

        # Register LLM tools.
        self._register_llm_tools()

        # Configure scheduled tasks.
        self._setup_jimeng_token_task()

        # Run startup tasks in the background.
        self.task_manager.create_task(self.task_manager.run_startup_tasks())

        logger.info(
            f"{LOG} 插件加载完成，模型: {safe_log_text(self.config_manager.adapter_config.model if self.config_manager.adapter_config else '未知')}"
        )

    async def terminate(self):
        """Run when the plugin is unloaded."""
        try:
            await self.task_manager.cancel_all()
            if self.generator:
                await self.generator.close()
            logger.info(f"{LOG} 插件已卸载")
        except Exception as exc:
            logger.error(f"{LOG} 卸载清理出错: {exc}", exc_info=True)

    # Internal helpers.

    def _register_llm_tools(self) -> None:
        """Register enabled LLM tools."""
        tools = []
        if self.config_manager.is_llm_tool_enabled(LLM_TOOL_IMAGE_GENERATION):
            if self.generator:
                image_tool = ImageGenerationTool(plugin=self)
                self._adjust_tool_parameters(image_tool)
                tools.append(image_tool)
            else:
                logger.warning(f"{LOG} 生图工具已启用，但生成器未初始化")

        if self.config_manager.is_llm_tool_enabled(LLM_TOOL_PRESET_QUERY):
            tools.append(PresetQueryTool(plugin=self))

        if self.config_manager.is_llm_tool_enabled(LLM_TOOL_TASK_MANAGEMENT):
            tools.append(ImageTaskTool(plugin=self))

        if self.config_manager.is_llm_tool_enabled(LLM_TOOL_PRESET_EDIT):
            tools.append(PresetEditTool(plugin=self))

        if tools:
            self.context.add_llm_tools(*tools)
            logger.info(
                f"{LOG} 已注册 LLM 工具: " + ", ".join(tool.name for tool in tools)
            )

    def _setup_jimeng_token_task(self) -> None:
        """Configure the Jimeng2API automatic credit receiving task.

        The task runs once during plugin startup and then when the date changes.

        It is enabled whenever the config contains a Jimeng2API provider,
        regardless of the currently active provider.
        """
        from .adapter.jimeng2api_adapter import Jimeng2APIAdapter
        from .core.shared.types import AdapterType

        # Check configured providers instead of the currently active adapter.
        jimeng_config = self.config_manager.get_provider_config(AdapterType.JIMENG2API)
        if not jimeng_config:
            return

        # Create a dedicated Jimeng2API adapter instance for the scheduled task.
        jimeng_adapter = Jimeng2APIAdapter(jimeng_config)

        # Register as a startup task so it runs once when the plugin starts.
        self.task_manager.register_startup_task(
            name="jimeng_token_receive",
            coro_func=jimeng_adapter.receive_token,
        )

        # Register as a daily task so it runs when the date changes.
        self.task_manager.start_daily_task(
            name="jimeng_token_receive",
            coro_func=jimeng_adapter.receive_token,
            check_interval_seconds=300,  # Check date changes every 5 minutes.
            run_immediately=False,  # Startup task already handles the first run.
        )
        logger.info(f"{LOG} 已配置即梦2API自动领积分任务（启动时+每日）")

    def _adjust_tool_parameters(self, tool: ImageGenerationTool) -> None:
        """Adjust LLM tool parameters based on adapter capabilities."""
        if not self.generator or not self.generator.adapter:
            return
        capabilities = self.generator.adapter.get_capabilities()
        adjust_tool_parameters(tool, capabilities)
        props = tool.parameters.get("properties", {})
        if not self.config_manager.personas:
            props.pop("persona", None)
            return
        persona_props = props.get("persona")
        if not isinstance(persona_props, dict):
            return
        persona_props.pop("enum", None)
        if persona_names := "、".join(self.config_manager.personas):
            persona_props["description"] = (
                str(persona_props.get("description", "")).rstrip("。")
                + f"。可用人设: {persona_names}；多个名称可用空格分隔。"
            )

    def create_background_task(
        self, coro: Coroutine[Any, Any, Any], name: str | None = None
    ) -> asyncio.Task:
        """Create a background task and register it with the task manager."""
        return self.task_manager.create_task(coro, name=name)

    def current_adapter_requires_api_key(self) -> bool:
        """Return whether the active adapter requires an API key."""
        adapter = self.generator.adapter if self.generator else None
        return bool(getattr(adapter, "requires_api_key", True))

    def has_required_api_key(self) -> bool:
        """Return whether the active adapter has all required credentials."""
        if not self.current_adapter_requires_api_key():
            return True
        return bool(
            self.config_manager.adapter_config
            and self.config_manager.adapter_config.api_keys
        )

    def reload_runtime_settings(self) -> None:
        """Refresh runtime helpers after config values are reloaded."""
        self.usage_manager.update_settings(self.config_manager.usage_settings)
        self.image_processor.update_settings(
            self.config_manager.usage_settings.max_image_size_mb
        )
        self.generation_executor.update_generator(self.generator)
        self.generation_executor.refresh_request_semaphore()
        self.task_manager.configure_generation_queue(
            max_queued_generation_tasks=self.config_manager.max_queued_generation_tasks
        )
        self.task_manager.configure_generation_history(
            generation_history_limit=self.config_manager.generation_task_history_limit,
            generation_history_retention_days=self.config_manager.generation_task_history_retention_days,
            enable_generation_task_history=self.config_manager.enable_generation_task_history,
        )
        self.task_manager.update_generation_worker_count(
            self.config_manager.max_running_generation_tasks
        )

    def create_generation_task(
        self,
        *,
        task_id: str,
        source: str,
        prompt: str,
        images_data: list[ImageData | tuple[bytes, str]] | None,
        unified_msg_origin: str,
        aspect_ratio: str,
        resolution: str,
        image_count: int,
        is_usage_limit_admin: bool,
        preset: str | None = None,
        preset_label: str = "预设",
        presets: list[str] | None = None,
        personas: list[str] | None = None,
        source_event: AstrMessageEvent | None = None,
        auto_send: bool = True,
    ) -> GenerationTaskRecord:
        """Create and track an image generation task in the unified task manager."""
        if preset is None:
            preset, preset_label = format_template_summary(
                presets or [],
                personas or [],
            )
        image_count = self.normalize_image_count(image_count)

        def _generation_coro_factory():
            return self.generation_executor.generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                image_count=image_count,
                task_id=task_id,
                is_usage_limit_admin=is_usage_limit_admin,
                deliver_via_ai=source == "LLM工具",
                auto_send=auto_send,
            )

        try:
            current_model = ""
            if self.config_manager.adapter_config:
                current_model = (
                    f"{self.config_manager.adapter_config.name}/"
                    f"{self.config_manager.adapter_config.model}"
                )
            record = self.task_manager.create_generation_task(
                _generation_coro_factory,
                task_id=task_id,
                source=source,
                unified_msg_origin=unified_msg_origin,
                prompt=prompt,
                reference_image_count=len(images_data or []),
                requested_count=image_count,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                model=current_model,
                preset=preset,
                preset_label=preset_label,
                usage_scope=unified_msg_origin,
                reserved_count=image_count if unified_msg_origin else 0,
                is_usage_limit_admin=is_usage_limit_admin,
                terminal_callback=self._handle_generation_task_terminal,
            )
        except GenerationTaskCreationError:
            self.usage_manager.release_reserved_usage(
                unified_msg_origin,
                is_admin=is_usage_limit_admin,
                count=image_count,
            )
            raise
        if source == "LLM工具":
            self.llm_result_handler.attach_task_wakeup(
                record,
                source_event=source_event,
            )
        return record

    def _handle_generation_task_terminal(self, record: GenerationTaskRecord) -> None:
        """Handle quota side effects when a task reaches terminal state.

        Args:
            record: Terminal generation task record.
        """
        if not record.usage_scope or record.quota_released or record.quota_settled:
            return
        if record.status == GenerationTaskStatus.SUCCEEDED:
            self.generation_executor.settle_generation_task_quota_once(
                record.task_id,
                actual_count=record.result_count,
            )
            return
        if record.status in {
            GenerationTaskStatus.FAILED,
            GenerationTaskStatus.CANCELLED,
        }:
            self.generation_executor.release_generation_task_quota_once(record.task_id)

    def is_usage_limit_admin(self, event: AstrMessageEvent) -> bool:
        """Return whether an event sender is an AstrBot admin for usage limits."""
        try:
            return bool(event.is_admin())
        except Exception as exc:
            logger.debug(f"{LOG} 获取管理员状态失败: {exc}")
            return False

    def normalize_image_count(self, value: Any) -> int:
        """Normalize requested image count using configured bounds."""
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = self.config_manager.default_image_count
        return max(1, min(count, self.config_manager.max_image_count))

    def _parse_command_image_count(self, prompt: str) -> tuple[int, str]:
        """Parse optional image count from command prompt suffix."""
        raw_prompt = prompt.strip()
        default_count = self.config_manager.default_image_count
        if not raw_prompt:
            return default_count, ""

        tokens = raw_prompt.split()
        if tokens[-1].isdecimal():
            return self.normalize_image_count(tokens[-1]), " ".join(tokens[:-1]).strip()

        return default_count, raw_prompt

    def _parse_command_prompt_templates(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
    ) -> tuple[str, str, str, list[str], list[str], list[tuple[str, str]]]:
        """Apply leading space-separated preset/persona names to a command prompt."""
        raw_prompt = prompt.strip()
        if not raw_prompt:
            return "", aspect_ratio, resolution, [], [], []

        tokens = raw_prompt.split()
        preset_prompts: list[str] = []
        persona_prompts: list[str] = []
        matched_presets: list[str] = []
        matched_personas: list[str] = []
        persona_images: list[tuple[str, str]] = []
        extra_content = ""

        for index, token in enumerate(tokens):
            matched_preset = find_named_entry(self.config_manager.presets, token)
            if matched_preset:
                preset_prompt, aspect_ratio, resolution = parse_preset_prompt(
                    self.config_manager.presets[matched_preset],
                    aspect_ratio,
                    resolution,
                )
                if preset_prompt:
                    preset_prompts.append(preset_prompt)
                matched_presets.append(matched_preset)
                continue

            matched_persona = find_named_entry(self.config_manager.personas, token)
            if matched_persona:
                persona = self.config_manager.personas[matched_persona]
                persona_prompt = persona.prompt.strip()
                if persona_prompt:
                    persona_prompts.append(persona_prompt)
                if persona.image:
                    persona_images.append((matched_persona, persona.image))
                matched_personas.append(matched_persona)
                continue

            extra_content = " ".join(tokens[index:]).strip()
            break

        if not matched_presets and not matched_personas:
            return raw_prompt, aspect_ratio, resolution, [], [], []

        return (
            build_generation_prompt(
                preset_prompts=preset_prompts,
                persona_prompts=persona_prompts,
                extra_prompt=extra_content,
            ),
            aspect_ratio,
            resolution,
            matched_presets,
            matched_personas,
            persona_images,
        )

    def format_start_task_message(
        self,
        *,
        prompt: str,
        reference_image_count: int,
        image_count: int,
        preset: str | None,
        preset_label: str = "预设",
        presets: list[str] | None = None,
        personas: list[str] | None = None,
        aspect_ratio: str,
        resolution: str,
        task_id: str,
    ) -> str:
        """Render start-task message from configured template."""
        return render_start_task_message(
            self.config_manager,
            prompt=prompt,
            reference_image_count=reference_image_count,
            image_count=image_count,
            preset=preset,
            preset_label=preset_label,
            presets=presets,
            personas=personas,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
        )

    def format_task_detail(self, record: GenerationTaskRecord) -> str:
        """Format one task record for command output."""
        return render_task_detail(record)

    def format_task_list(self, records: list[GenerationTaskRecord]) -> str:
        """Format a compact task list for command output."""
        return render_task_list(records)

    def format_image_command_help(self) -> str:
        """Format help text for the image generation command."""
        return render_image_command_help(self.config_manager)

    def resolve_task_reference(
        self,
        unified_msg_origin: str,
        task_ref: str,
        *,
        include_finished: bool = False,
    ) -> GenerationTaskRecord | None:
        """Resolve a task id or active list number into a task for one session."""
        task_ref = task_ref.strip()
        if not task_ref:
            return None

        active_records = self.task_manager.list_generation_tasks(
            unified_msg_origin=unified_msg_origin,
            include_finished=False,
            limit=10,
        )
        if task_ref.isdigit():
            index = int(task_ref) - 1
            if 0 <= index < len(active_records):
                return active_records[index]

        for record in active_records:
            if record.task_id == task_ref:
                return record

        if include_finished:
            record = self.task_manager.get_generation_task(task_ref)
            if record and record.unified_msg_origin == unified_msg_origin:
                return record
        return None

    def resolve_active_task_reference(
        self, unified_msg_origin: str, task_ref: str
    ) -> GenerationTaskRecord | None:
        """Resolve a task id or list number into an active task for one session."""
        return self.resolve_task_reference(
            unified_msg_origin,
            task_ref,
            include_finished=False,
        )

    # Command handlers.

    @filter.command("生图任务")
    async def image_task_command(self, event: AstrMessageEvent, task_id: str = ""):
        """Show image generation tasks or one task detail."""
        user_id = event.unified_msg_origin
        task_id = (task_id or "").strip()

        if task_id:
            record = self.resolve_task_reference(
                user_id,
                task_id,
                include_finished=True,
            )
            if not record:
                if task_id.isdigit():
                    yield event.plain_result(
                        "❌ 未找到编号对应的进行中任务；已结束任务请使用完整任务ID查看"
                    )
                    return
                yield event.plain_result(f"❌ 任务不存在或已被清理: {task_id}")
                return
            if record.unified_msg_origin != user_id and not self.is_usage_limit_admin(
                event
            ):
                yield event.plain_result("❌ 不能查看其他会话的生图任务")
                return
            yield event.plain_result(self.format_task_detail(record))
            return

        records = self.task_manager.list_generation_tasks(
            unified_msg_origin=user_id,
            include_finished=False,
            limit=10,
        )
        yield event.plain_result(self.format_task_list(records))

    @filter.command("生图取消")
    async def cancel_image_task_command(
        self, event: AstrMessageEvent, task_id: str = ""
    ):
        """Cancel one image generation task."""
        task_id = (task_id or "").strip()
        if not task_id:
            active_records = self.task_manager.list_generation_tasks(
                unified_msg_origin=event.unified_msg_origin,
                include_finished=False,
                limit=5,
            )
            if active_records:
                yield event.plain_result(
                    "❌ 请提供要取消的编号或任务ID\n"
                    + self.format_task_list(active_records)
                )
            else:
                yield event.plain_result("📭 当前没有可取消的生图任务")
            return

        record = self.resolve_active_task_reference(event.unified_msg_origin, task_id)
        if not record:
            yield event.plain_result(
                f"❌ 未找到对应的进行中任务，请检查编号或任务ID: {task_id}"
            )
            return

        _, message = self.task_manager.cancel_generation_task(
            record.task_id,
            unified_msg_origin=event.unified_msg_origin,
        )
        logger.debug(
            f"{log_prefix('Task', record.task_id)} 用户请求取消任务: "
            f"用户={mask_sensitive(event.unified_msg_origin)}，结果={safe_log_text(message)}"
        )
        yield event.plain_result(message)

    @filter.command("生图")
    async def generate_image_command(self, event: AstrMessageEvent):
        """Handle the image generation command."""
        user_id = event.unified_msg_origin
        is_usage_limit_admin = self.is_usage_limit_admin(event)

        user_input = (event.message_str or "").strip()
        masked_uid = mask_sensitive(user_id)
        logger.debug(
            f"{LOG} 收到生图指令: 用户={masked_uid}，输入长度={len(user_input)}"
        )

        cmd_parts = user_input.split(maxsplit=1)
        if not cmd_parts:
            return

        raw_prompt = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
        if not raw_prompt:
            yield event.plain_result(self.format_image_command_help())
            return

        if self.config_manager.should_llm_handle_command(LLM_HANDLED_COMMAND_GENERATE):
            tokens = raw_prompt.split()
            if tokens and tokens[-1].isdecimal():
                image_count, handoff_demand = self._parse_command_image_count(
                    raw_prompt
                )
                handoff_image_count: int | None = image_count
            else:
                image_count = self.config_manager.default_image_count
                handoff_demand = raw_prompt
                handoff_image_count = None

            if not handoff_demand.strip():
                yield event.plain_result("❌ 请提供图片生成的提示词或预设名称！")
                return

            if not self.generator or not self.generator.adapter:
                logger.debug(
                    f"{LOG} 生图指令失败: 生成器未初始化，用户={masked_uid}"
                )
                yield event.plain_result("❌ 生图生成器未初始化")
                return

            if not self.has_required_api_key():
                logger.debug(
                    f"{LOG} 生图指令失败: 未配置 API Key，用户={masked_uid}"
                )
                yield event.plain_result("❌ 未配置 API Key，无法生成图片")
                return

            check_result = self.usage_manager.check_rate_limit(
                user_id,
                is_admin=is_usage_limit_admin,
                requested_count=image_count,
                update_timestamp=False,
            )
            if isinstance(check_result, str):
                if check_result:
                    yield event.plain_result(check_result)
                return

            if rejection := self.task_manager.get_generation_queue_rejection():
                _code, message = rejection
                yield event.plain_result(f"❌ 生图任务提交失败: {message}")
                return

            provider_request = await try_request_llm_handoff(
                self,
                event,
                raw_demand=handoff_demand,
                image_count=handoff_image_count,
            )
            if provider_request is not None:
                yield provider_request
                return
            logger.info(
                f"{LOG} LLM 接管 /生图 不可用，回退为直接执行指令: 用户={masked_uid}"
            )

        if not self.generator or not self.generator.adapter:
            logger.debug(f"{LOG} 生图指令失败: 生成器未初始化，用户={masked_uid}")
            yield event.plain_result("❌ 生图生成器未初始化")
            return

        image_count, prompt = self._parse_command_image_count(raw_prompt)

        aspect_ratio = self.config_manager.default_aspect_ratio
        resolution = self.config_manager.default_resolution
        (
            prompt,
            aspect_ratio,
            resolution,
            matched_presets,
            matched_personas,
            persona_images,
        ) = self._parse_command_prompt_templates(prompt, aspect_ratio, resolution)
        preset_or_persona, preset_label = format_template_summary(
            matched_presets,
            matched_personas,
        )

        if not prompt:
            yield event.plain_result("❌ 请提供图片生成的提示词或预设名称！")
            return

        if not self.has_required_api_key():
            logger.debug(f"{LOG} 生图指令失败: 未配置 API Key，用户={masked_uid}")
            yield event.plain_result("❌ 未配置 API Key，无法生成图片")
            return

        check_result = self.usage_manager.check_rate_limit(
            user_id,
            is_admin=is_usage_limit_admin,
            requested_count=image_count,
            update_timestamp=False,
        )
        if isinstance(check_result, str):
            if check_result:
                yield event.plain_result(check_result)
            return

        prompt_allowed, prompt_reason = await self.safety_auditor.audit_prompt(
            prompt, event.unified_msg_origin
        )
        if not prompt_allowed:
            logger.warning(
                f"{LOG} 提示词审核未通过: 用户={masked_uid}, 原因={safe_log_text(prompt_reason, 160)}"
            )
            yield event.plain_result(f"❌ 提示词审核未通过: {prompt_reason}")
            return

        if rejection := self.task_manager.get_generation_queue_rejection():
            _code, message = rejection
            yield event.plain_result(f"❌ 生图任务提交失败: {message}")
            return

        check_result = self.usage_manager.check_rate_limit(
            user_id,
            is_admin=is_usage_limit_admin,
            requested_count=image_count,
        )
        if isinstance(check_result, str):
            if check_result:
                yield event.plain_result(check_result)
            return
        usage_reserved = True
        task_created = False

        task_id = new_task_id()
        try:
            images_data: list[ImageData] | None = None
            if (
                self.generator.adapter.get_capabilities()
                & ImageCapability.IMAGE_TO_IMAGE
            ):
                try:
                    images_data = await collect_command_reference_images(
                        self.image_processor,
                        event,
                        persona_images,
                        task_id=task_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        f"{log_prefix('Task', task_id)} 参考图准备失败: {safe_log_text(exc, 200)}",
                        exc_info=True,
                    )
                    images_data = []

            reference_image_count = len(images_data or [])

            try:
                self.create_generation_task(
                    task_id=task_id,
                    source="指令",
                    prompt=prompt,
                    images_data=images_data,
                    unified_msg_origin=event.unified_msg_origin,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    image_count=image_count,
                    is_usage_limit_admin=is_usage_limit_admin,
                    preset=preset_or_persona,
                    preset_label=preset_label,
                    presets=matched_presets,
                    personas=matched_personas,
                )
                task_created = True
            except GenerationTaskCreationError:
                # create_generation_task() rolls back quota reservation on creation failure.
                usage_reserved = False
                raise
        except asyncio.CancelledError:
            if usage_reserved and not task_created:
                self.usage_manager.release_reserved_usage(
                    user_id,
                    is_admin=is_usage_limit_admin,
                    count=image_count,
                )
            raise
        except GenerationTaskCreationError as exc:
            yield event.plain_result(f"❌ 生图任务提交失败: {exc.message}")
            return
        except Exception as exc:
            if usage_reserved and not task_created:
                self.usage_manager.release_reserved_usage(
                    user_id,
                    is_admin=is_usage_limit_admin,
                    count=image_count,
                )
            logger.error(
                f"{log_prefix('Task', task_id)} 生图任务提交前处理失败: {safe_log_error_body(exc, 200)}",
                exc_info=True,
            )
            yield event.plain_result(
                f"❌ 生图任务提交失败: {safe_log_error_body(exc, 160)}"
            )
            return

        msg = self.format_start_task_message(
            prompt=prompt,
            reference_image_count=reference_image_count,
            image_count=image_count,
            preset=preset_or_persona,
            preset_label=preset_label,
            presets=matched_presets,
            personas=matched_personas,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
        )
        if msg:
            yield event.plain_result(msg)

    @filter.command("生图模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """Switch the active image generation model."""
        if not self.config_manager.adapter_config:
            yield event.plain_result("❌ 适配器未初始化")
            return

        models = self.config_manager.adapter_config.available_models or []
        current_model_full = f"{self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"

        if not model_index:
            if not models:
                yield event.plain_result(
                    f"📋 当前没有配置可用模型\n\n当前使用: {current_model_full}"
                )
                return

            lines = ["📋 可用模型列表:"]
            for idx, model in enumerate(models, 1):
                marker = " ✓" if model == current_model_full else ""
                lines.append(f"{idx}. {model}{marker}")
            lines.append(f"\n当前使用: {current_model_full}")
            yield event.plain_result("\n".join(lines))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(models):
                raw_model = models[index]  # "provider/model"

                # Save config and reload runtime helpers.
                self.config_manager.save_model_setting(raw_model)
                self.config_manager.reload()
                self.reload_runtime_settings()

                if self.generator:
                    await self.generator.update_adapter(
                        self.config_manager.adapter_config
                    )
                    self.generation_executor.update_generator(self.generator)

                yield event.plain_result(f"✅ 模型已切换: {raw_model}")
            else:
                yield event.plain_result("❌ 无效的序号")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字序号")

    @filter.command("预设")
    async def preset_command(self, event: AstrMessageEvent):
        """Manage image generation presets."""
        user_id = event.unified_msg_origin
        masked_uid = mask_sensitive(user_id)
        message_str = (event.message_str or "").strip()
        logger.debug(
            f"{LOG} 收到预设指令: 用户={masked_uid}，输入={safe_log_text(message_str)}"
        )

        parts = message_str.split(maxsplit=1)
        cmd_text = parts[1].strip() if len(parts) > 1 else ""

        if not cmd_text:
            if not self.config_manager.presets and not self.config_manager.personas:
                yield event.plain_result("📋 当前没有预设或人设")
                return
            preset_list = []
            if self.config_manager.presets:
                preset_list.append("📋 预设列表:")
            for idx, (name, prompt) in enumerate(
                self.config_manager.presets.items(), 1
            ):
                display = prompt[:20] + "..." if len(prompt) > 20 else prompt
                preset_list.append(f"{idx}. {name}: {display}")

            if self.config_manager.personas:
                if preset_list:
                    preset_list.append("")
                preset_list.append("👤 人设列表:")
                for idx, (name, persona) in enumerate(
                    self.config_manager.personas.items(), 1
                ):
                    display = (
                        persona.prompt[:20] + "..."
                        if len(persona.prompt) > 20
                        else persona.prompt
                    )
                    image_mark = "有参考图" if persona.image else "无参考图"
                    preset_list.append(f"{idx}. {name}: {display} [{image_mark}]")
            yield event.plain_result("\n".join(preset_list))
            return

        if cmd_text.startswith("添加 "):
            preset_text = cmd_text[3:].strip()
            delimiter_positions = [
                (index, delimiter)
                for index, delimiter in (
                    (preset_text.find(":"), ":"),
                    (preset_text.find("："), "："),
                )
                if index >= 0
            ]
            if delimiter_positions:
                split_index, _ = min(delimiter_positions, key=lambda item: item[0])
                name = preset_text[:split_index].strip()
                prompt = preset_text[split_index + 1 :].strip()
                self.config_manager.save_preset(name, prompt)
                yield event.plain_result(f"✅ 预设已添加: {name}")
            else:
                yield event.plain_result("❌ 格式错误: /预设 添加 名称:内容")
        elif cmd_text.startswith("删除 "):
            name = cmd_text[3:].strip()
            if self.config_manager.delete_preset(name):
                yield event.plain_result(f"✅ 预设已删除: {name}")
            else:
                yield event.plain_result(f"❌ 预设不存在: {name}")
