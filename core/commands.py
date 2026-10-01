"""Command-side helpers shared by the image generation plugin.

Decorated command handlers must stay in ``main.py`` so the framework can map
their module back to the plugin. These helpers carry no decorators and are
mixed into the plugin class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from .config.templates import (
    build_generation_prompt,
    extract_templates_from_prompt,
    find_named_entry,
    resolve_named_templates,
)
from .formatting.result import format_image_command_help as render_image_command_help
from .formatting.result import format_start_task_message as render_start_task_message
from .formatting.result import format_task_detail as render_task_detail
from .formatting.result import format_task_list as render_task_list
from .generation.options import split_trailing_count_token
from .shared.logging import log_prefix
from .tasks.models import GenerationTaskRecord

if TYPE_CHECKING:
    from .config.manager import ConfigManager
    from .tasks.manager import TaskManager

LOG = log_prefix("Plugin")


class CommandHandlerMixin:
    """Prompt parsing, task lookup, and result formatting for commands."""

    if TYPE_CHECKING:
        config_manager: ConfigManager
        task_manager: TaskManager

    def is_usage_limit_admin(self, event: Any) -> bool:
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

        remainder, count_token = split_trailing_count_token(raw_prompt)
        if count_token is None:
            return default_count, raw_prompt
        return self.normalize_image_count(count_token), remainder

    def _parse_command_prompt_templates(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
    ) -> tuple[str, str, str, list[str], list[str], list[tuple[str, str]]]:
        """Apply command prompt templates by leading tokens or optional body match.

        Args:
            prompt: Command prompt after image-count suffix parsing.
            aspect_ratio: Current aspect ratio, updated by JSON presets.
            resolution: Current resolution, updated by JSON presets.

        Returns:
            Final prompt, aspect ratio, resolution, matched preset names,
            matched persona names, and persona reference images.
        """
        raw_prompt = prompt.strip()
        if not raw_prompt:
            return "", aspect_ratio, resolution, [], [], []

        if self.config_manager.match_templates_in_prompt_body:
            (
                extra_content,
                aspect_ratio,
                resolution,
                preset_prompts,
                persona_prompts,
                matched_presets,
                matched_personas,
                persona_images,
            ) = extract_templates_from_prompt(
                raw_prompt,
                self.config_manager.prompt_body_presets(),
                self.config_manager.prompt_body_personas(),
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
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

        tokens = raw_prompt.split()
        preset_names: list[str] = []
        persona_names: list[str] = []
        extra_content = ""

        for index, token in enumerate(tokens):
            if find_named_entry(self.config_manager.presets, token):
                preset_names.append(token)
                continue
            if find_named_entry(self.config_manager.personas, token):
                persona_names.append(token)
                continue
            extra_content = " ".join(tokens[index:]).strip()
            break

        preset_prompts, aspect_ratio, resolution, matched_presets, _, _ = (
            resolve_named_templates(
                preset_names,
                self.config_manager.presets,
                kind="preset",
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
        )
        persona_prompts, _, _, matched_personas, persona_images, _ = (
            resolve_named_templates(
                persona_names,
                self.config_manager.personas,
                kind="persona",
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
        )

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
