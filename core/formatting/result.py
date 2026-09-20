"""User-facing result formatting helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from astrbot.api import logger

from ..config.manager import (
    RESULT_INFO_COUNT,
    RESULT_INFO_DURATION,
    RESULT_INFO_MODEL,
    RESULT_INFO_TASK_ID,
    RESULT_INFO_USAGE,
)
from ..shared.constants import UNSPECIFIED_OPTION
from ..tasks.models import GenerationTaskItemStatus, GenerationTaskStatus
from ..shared.logging import log_prefix, safe_log_text

if TYPE_CHECKING:
    from ..config.manager import ConfigManager
    from ..tasks.models import GenerationTaskRecord
    from ..tasks.usage import UsageManager


LOG = log_prefix("Formatter")
GENERATION_ITEM_ERROR_RE = re.compile(r"^第\s*(\d+)\s*张生成失败\s*[:：]\s*(.*)$")


class SafeFormatDict(dict[str, str]):
    """Keep unknown template placeholders unchanged."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_start_template_values(
    *,
    preset: str | None,
    presets: list[str] | None,
    personas: list[str] | None,
) -> dict[str, str]:
    """Build preset/persona placeholder values for the start-task template."""
    preset_names = "、".join(presets or [])
    persona_names = "、".join(personas or [])
    return {
        "preset": preset_names or (preset or ""),
        "presets": preset_names,
        "persona": persona_names,
        "personas": persona_names,
        "preset_block": f"[预设: {preset_names}]" if preset_names else "",
        "persona_block": f"[人设: {persona_names}]" if persona_names else "",
    }


def format_start_task_message(
    config_manager: ConfigManager,
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
    template = config_manager.start_task_message_template
    if not template.strip():
        return ""

    model = ""
    if config_manager.adapter_config:
        model = (
            f"{config_manager.adapter_config.name}/"
            f"{config_manager.adapter_config.model}"
        )

    values = SafeFormatDict(
        reference_image_count=str(reference_image_count),
        image_count=str(image_count),
        count=str(image_count),
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        task_id=task_id,
        model=model,
        mode="图生图" if reference_image_count else "文生图",
        preset_label=preset_label,
        image_count_block=f"[数量: {image_count}张]" if image_count > 1 else "",
        count_block=f"[数量: {image_count}张]" if image_count > 1 else "",
        reference_images_block=(
            f"[{reference_image_count}张参考图]" if reference_image_count else ""
        ),
        **format_start_template_values(
            preset=preset,
            presets=presets,
            personas=personas,
        ),
    )

    try:
        return template.format_map(values)
    except Exception as exc:
        logger.warning(f"{LOG} 开始任务提示模板格式化失败: {exc}")
        return (
            "已开始生图任务{reference_images_block}{preset_block}"
            "{persona_block}{image_count_block} [任务ID: {task_id}]"
        ).format_map(values)


def format_generation_failure_message(error: object) -> str:
    """Build a concise user-facing generation failure message.

    Args:
        error: Raw error collected from the generation pipeline.

    Returns:
        A multi-line user-facing failure message with repeated technical prefixes
        removed where possible.
    """
    raw_text = safe_log_text(error, 500) or "模型未返回图片"
    raw_items = [
        item.strip() for item in re.split(r"\s*;\s*", raw_text) if item.strip()
    ]
    if not raw_items:
        raw_items = ["模型未返回图片"]

    entries: list[tuple[str, str]] = []
    for raw_item in raw_items[:3]:
        item_text = raw_item
        image_label = ""

        for _ in range(4):
            item_match = GENERATION_ITEM_ERROR_RE.match(item_text)
            if item_match:
                image_label = f"第 {item_match.group(1)} 张"
                item_text = item_match.group(2).strip()
                continue

            stripped = ""
            for prefix in ("生成失败", "重试失败"):
                for separator in (":", "："):
                    marker = f"{prefix}{separator}"
                    if item_text.startswith(marker):
                        stripped = item_text[len(marker) :].strip()
                        break
                if stripped:
                    break

            if not stripped:
                break
            item_text = stripped

        entries.append((image_label, safe_log_text(item_text or "模型未返回图片", 360)))

    lines = ["❌ 生成失败", "原因："]
    show_item_label = len(raw_items) > 1
    for image_label, reason in entries:
        label = image_label if show_item_label and image_label else ""
        prefix = f"{label}：" if label else ""
        api_error_match = re.match(r"^(API 错误\s*\(\d{3}\))\s*[:：]\s*(.+)$", reason)
        if api_error_match:
            lines.append(f"- {prefix}{api_error_match.group(1)}")
            lines.append(f"  详情：{api_error_match.group(2)}")
        else:
            lines.append(f"- {prefix}{reason}")
    if len(raw_items) > len(entries):
        lines.append(f"- 另有 {len(raw_items) - len(entries)} 个失败未展示")
    return "\n".join(lines)


def format_task_detail(record: GenerationTaskRecord) -> str:
    """Format one task record for command output."""
    stats = record.request_stats
    status_icons = {
        GenerationTaskStatus.QUEUED: "⏳",
        GenerationTaskStatus.RUNNING: "🔄",
        GenerationTaskStatus.SUCCEEDED: "✅",
        GenerationTaskStatus.FAILED: "❌",
        GenerationTaskStatus.CANCELLING: "🛑",
        GenerationTaskStatus.CANCELLED: "🚫",
    }
    aspect_ratio = (
        record.aspect_ratio if record.aspect_ratio != UNSPECIFIED_OPTION else "模型默认"
    )
    resolution = (
        record.resolution if record.resolution != UNSPECIFIED_OPTION else "模型默认"
    )
    prompt_text = record.prompt_summary or "无"
    if len(prompt_text) > 64:
        prompt_text = f"{prompt_text[:64]}..."

    total_requests = max(1, stats["total"])
    finished_requests = min(stats["finished"], total_requests)
    progress_percent = int(finished_requests / total_requests * 100)
    progress_blocks = int(finished_requests / total_requests * 10)
    progress_bar = "█" * progress_blocks + "░" * (10 - progress_blocks)
    distribution_parts = []
    if stats["succeeded"]:
        distribution_parts.append(f"✅ 成功 {stats['succeeded']}")
    if stats["failed"]:
        distribution_parts.append(f"❌ 失败 {stats['failed']}")
    if stats["cancelled"]:
        distribution_parts.append(f"🚫 取消 {stats['cancelled']}")
    if stats["running"]:
        distribution_parts.append(f"🔄 运行中 {stats['running']}")
    if stats["pending"]:
        distribution_parts.append(f"⏳ 等待 {stats['pending']}")

    lines = [
        f"🧾 生图任务 {record.task_id}",
        f"状态：{status_icons.get(record.status, '📌')} {record.status_label} ｜ 来源：{record.source}",
        "",
        "📌 请求",
        f"- 模式：{'图生图' if record.reference_image_count else '文生图'} ｜ 参考图：{record.reference_image_count}张",
        f"- 尺寸：宽高比 {aspect_ratio} ｜ 分辨率 {resolution}",
    ]
    if record.preset:
        lines.append(f"- 模板：{record.preset_label}「{record.preset}」")
    lines.append(f"- 提示词摘要：{prompt_text}")

    lines.extend(
        [
            "",
            "📊 进度",
            f"- 完成：{finished_requests}/{total_requests}（{progress_percent}%） {progress_bar}",
            "- 状态：" + "，".join(distribution_parts or ["暂无子请求状态"]),
            f"- 结果：{stats['result_count']}张图片",
        ]
    )

    time_parts = [f"排队 {record.queued_seconds:.2f}s"]
    if record.started_at:
        duration = record.duration_seconds
        if duration is not None:
            time_parts.append(f"执行 {duration:.2f}s")
    lines.append("- 时间：" + " ｜ ".join(time_parts))

    if record.message and record.message not in {"任务已提交", "任务运行中"}:
        lines.append(f"- 当前：{safe_log_text(record.message, 80)}")
    if record.error:
        for error_line in format_generation_failure_message(record.error).splitlines()[
            1:
        ]:
            lines.append(
                error_line if error_line.startswith("- ") else f"- {error_line}"
            )
    if record.items:
        lines.extend(["", "🧩 明细"])
        for item in sorted(
            record.items.values(), key=lambda task_item: task_item.index
        ):
            if item.status == GenerationTaskItemStatus.SUCCEEDED:
                item_line = f"- #{item.index} ✅ 成功，结果 {item.result_count} 张"
            elif item.status == GenerationTaskItemStatus.FAILED:
                item_line = f"- #{item.index} ❌ 失败"
                if item.error:
                    item_line += f"，错误: {safe_log_text(item.error, 120)}"
            elif item.status == GenerationTaskItemStatus.CANCELLED:
                item_line = f"- #{item.index} 🚫 已取消"
            elif item.status == GenerationTaskItemStatus.PENDING:
                item_line = f"- #{item.index} ⏳ 等待中"
            else:
                item_line = f"- #{item.index} 🔄 运行中"
            if item.max_retry_attempts:
                item_line += f"，重试 {item.retry_attempts}/{item.max_retry_attempts}"
            lines.append(item_line)
    lines.append("")
    if record.is_active:
        lines.append(f"💡 可取消：/生图取消 {record.task_id}")
    else:
        lines.append("💡 已结束任务只保留最近历史记录，之后可能被自动清理")
    return "\n".join(lines)


def format_task_list(records: list[GenerationTaskRecord]) -> str:
    """Format a compact task list for command output."""
    if not records:
        return "📭 当前没有正在进行的生图任务"

    lines = ["📋 正在进行的生图任务:"]
    for index, record in enumerate(records, 1):
        stats = record.request_stats
        parts = [
            f"{record.task_id}",
            record.status_label,
            record.source,
            f"子请求{stats['finished']}/{stats['total']}",
            f"结果{stats['result_count']}张",
            f"参考图{record.reference_image_count}张",
        ]
        lines.append(f"{index}. " + " | ".join(parts))
    lines.append(
        "\n用法: \n/生图任务 <编号或任务ID> 查看详情\n/生图取消 <编号或任务ID> 取消任务"
    )
    return "\n".join(lines)


def format_image_command_help(config_manager: ConfigManager) -> str:
    """Format help text for the image generation command."""
    adapter_config = config_manager.adapter_config
    current_model = (
        f"{adapter_config.name}/{adapter_config.model}" if adapter_config else "未配置"
    )
    lines = [
        "🎨 生图帮助",
        f"当前模型: {current_model}",
        "",
        "指令列表:",
        "/生图 [预设/人设] [提示词] [数量]",
        "尺寸示例: /生图 猫，比例 16:9，分辨率 2K 3",
        "末尾简写: /生图 猫 16:9 2K 3（数量放最后）",
        "尺寸优先级: 本次指定 > 高级预设 > 默认配置",
        "无效或冲突尺寸会提示修改；粘连词和无法识别的末尾简写留在提示词中。",
        "/生图模型 - 查看或切换模型",
        "/生图任务 [编号或任务ID] - 查看正在进行的任务",
        "/生图取消 <编号或任务ID> - 取消指定任务",
        "/预设 [添加/删除] - 查看或管理预设/人设",
    ]
    return "\n".join(lines)


def build_result_info_message(
    config_manager: ConfigManager,
    usage_manager: UsageManager,
    *,
    unified_msg_origin: str,
    is_usage_limit_admin: bool,
    duration: float,
    result_count: int,
    task_id: str,
) -> str:
    """Build the optional metadata appended to generated image messages."""
    info_parts: list[str] = []
    if config_manager.should_show_result_info(RESULT_INFO_DURATION):
        info_parts.append(f"📊 耗时: {duration:.2f}s")

    if (
        config_manager.should_show_result_info(RESULT_INFO_MODEL)
        and config_manager.adapter_config
    ):
        info_parts.append(
            f"🤖 模型: {config_manager.adapter_config.name}/{config_manager.adapter_config.model}"
        )

    if config_manager.should_show_result_info(RESULT_INFO_COUNT):
        info_parts.append(f"🖼️ 数量: {result_count}张")

    if config_manager.should_show_result_info(RESULT_INFO_TASK_ID):
        info_parts.append(f"🧾 任务ID: {task_id}")

    if (
        config_manager.should_show_result_info(RESULT_INFO_USAGE)
        and usage_manager.is_daily_limit_enabled()
    ):
        count = usage_manager.get_usage_count(unified_msg_origin)
        daily_limit = (
            "∞"
            if usage_manager.is_limit_exempt(
                unified_msg_origin,
                is_admin=is_usage_limit_admin,
            )
            else str(usage_manager.get_daily_limit())
        )
        info_parts.append(f"📅 今日用量: {count}/{daily_limit}")

    return "\n".join(info_parts)
