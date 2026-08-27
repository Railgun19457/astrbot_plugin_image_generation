"""Hand slash image commands off to the session chat LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.message_components import Image, Reply

from ..shared.logging import log_prefix, mask_sensitive, safe_log_text

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import ProviderRequest
    from astrbot.api.star import Context

LOG = log_prefix("CommandHandoff")

HANDOFF_SYSTEM_PROMPT = """\
用户通过 /生图 命令请求生成或修改图片。你必须调用 generate_image 工具完成出图，不要只用文字空谈或假装已画图。

结合当前人设与对话上下文理解用户意图，编写具体、可独立理解的视觉 prompt；涉及自身形象、角色或人设时，优先填写 persona（若工具可用）并写清外观，不要把用户的模糊原话原样塞进 prompt。

工具返回任务已提交后，用符合人设的简短语气确认即可，不要复述技术细节或重复调用 generate_image。
""".strip()


def build_handoff_prompt(*, raw_demand: str, image_count: int | None = None) -> str:
    """Build the user-side prompt passed to request_llm for /生图 handoff."""
    demand = str(raw_demand or "").strip()
    lines = [
        "用户通过命令要求生成图片。",
        "命令原文中的需求：",
        demand or "（空）",
        "",
        "请结合当前人设与对话上下文理解意图，调用 generate_image 工具完成出图。",
        "编写具体、可画的 prompt；若涉及自身形象/人设角色，优先使用 persona 参数（如已配置）并写清外观。",
        "不要只把上面那句原话原样塞进 prompt。",
        "同一轮只调用一次 generate_image（除非用户明确要多张且通过 image_count 表达）。",
    ]
    if image_count is not None and image_count > 0:
        lines.extend(
            [
                "",
                f"用户请求生成数量：{image_count}。请在调用 generate_image 时传入 image_count={image_count}。",
            ]
        )
    return "\n".join(lines).strip()


async def collect_handoff_image_urls(event: AstrMessageEvent) -> list[str]:
    """Collect message and replied image paths/URLs for request_llm."""
    urls: list[str] = []
    seen: set[str] = set()

    async def _append_from_image(component: Any) -> None:
        path_or_url = ""
        convert = getattr(component, "convert_to_file_path", None)
        if callable(convert):
            try:
                path_or_url = str(await convert() or "").strip()
            except Exception:
                logger.debug(f"{LOG} convert_to_file_path 失败", exc_info=True)
                path_or_url = ""
        if not path_or_url:
            path_or_url = str(
                getattr(component, "url", None) or getattr(component, "file", None) or ""
            ).strip()
        if not path_or_url or path_or_url in seen:
            return
        seen.add(path_or_url)
        urls.append(path_or_url)

    message = getattr(getattr(event, "message_obj", None), "message", None) or []
    for component in message:
        try:
            if isinstance(component, Image):
                await _append_from_image(component)
            elif isinstance(component, Reply):
                chain = getattr(component, "chain", None) or []
                for sub in chain:
                    if isinstance(sub, Image):
                        await _append_from_image(sub)
        except Exception:
            logger.debug(f"{LOG} 收集 handoff 图片失败", exc_info=True)
    return urls


async def resolve_session_conversation(
    context: Context,
    event: AstrMessageEvent,
) -> Any | None:
    """Load or create the current session conversation for persona and history."""
    conversation_manager = getattr(context, "conversation_manager", None)
    if conversation_manager is None:
        return None

    umo = event.unified_msg_origin
    try:
        curr_cid = await conversation_manager.get_curr_conversation_id(umo)
        if curr_cid:
            return await conversation_manager.get_conversation(umo, curr_cid)

        platform_id = ""
        if hasattr(event, "get_platform_id"):
            platform_id = str(event.get_platform_id() or "")
        curr_cid = await conversation_manager.new_conversation(
            umo,
            platform_id=platform_id,
        )
        if not curr_cid:
            return None
        return await conversation_manager.get_conversation(umo, curr_cid)
    except Exception as exc:
        logger.warning(
            f"{LOG} 获取会话 conversation 失败: 用户={mask_sensitive(umo)}，"
            f"错误={safe_log_text(exc, 160)}"
        )
        return None


async def try_request_llm_handoff(
    plugin: Any,
    event: AstrMessageEvent,
    *,
    raw_demand: str,
    image_count: int | None = None,
) -> ProviderRequest | None:
    """Build a ProviderRequest for command handoff, or None to fall back."""
    context: Context = plugin.context
    umo = event.unified_msg_origin
    masked_uid = mask_sensitive(umo)

    provider = None
    get_provider = getattr(context, "get_using_provider", None)
    if callable(get_provider):
        try:
            provider = get_provider(umo)
        except Exception:
            logger.warning(
                f"{LOG} 查询聊天模型失败: 用户={masked_uid}",
                exc_info=True,
            )
            provider = None
    if not provider:
        logger.warning(
            f"{LOG} 无可用聊天模型，/生图 回退为直接执行指令: 用户={masked_uid}"
        )
        return None

    conversation = await resolve_session_conversation(context, event)
    image_urls = await collect_handoff_image_urls(event)
    prompt = build_handoff_prompt(raw_demand=raw_demand, image_count=image_count)

    logger.info(
        f"{LOG} 将 /生图 交给会话 LLM: 用户={masked_uid}，"
        f"需求长度={len(str(raw_demand or '').strip())}，"
        f"数量={image_count if image_count is not None else '默认'}，"
        f"参考图={len(image_urls)}，"
        f"conversation={'有' if conversation is not None else '无'}"
    )
    return event.request_llm(
        prompt=prompt,
        system_prompt=HANDOFF_SYSTEM_PROMPT,
        conversation=conversation,
        image_urls=image_urls,
    )


__all__ = (
    "HANDOFF_SYSTEM_PROMPT",
    "build_handoff_prompt",
    "collect_handoff_image_urls",
    "resolve_session_conversation",
    "try_request_llm_handoff",
)
