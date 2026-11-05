"""Offline AstrBot contract doubles; all plugin quota/task code remains real."""

import importlib
import logging
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, **attrs):
    obj = types.ModuleType(name)
    obj.__path__ = []
    obj.__dict__.update(attrs)
    sys.modules[name] = obj
    if "." in name:
        parent, attr = name.rsplit(".", 1)
        if parent not in sys.modules:
            module(parent)
        setattr(sys.modules[parent], attr, obj)
    return obj


class MessageType(Enum):
    GROUP_MESSAGE = "GroupMessage"
    FRIEND_MESSAGE = "FriendMessage"
    OTHER_MESSAGE = "OtherMessage"


@dataclass
class MessageSession:
    # Matches AstrBot v4.25.3's documented parsing and serialization.
    platform_name: str
    message_type: MessageType
    session_id: str

    def __str__(self):
        return f"{self.platform_name}:{self.message_type.value}:{self.session_id}"

    @staticmethod
    def from_str(value):
        platform, kind, session = value.split(":", 2)
        return MessageSession(platform, MessageType(kind), session)


class Dummy:
    def __class_getitem__(cls, _item):
        return cls


class Filters:
    def __getattr__(self, name):
        return lambda *a, **kw: lambda fn: fn


class AstrBotConfig(dict):
    schema = None
    saves = 0

    def save_config(self):
        self.saves += 1


class MessageChain:
    def __init__(self):
        self.parts = []

    def message(self, text):
        self.parts.append(text)
        return self

    def file_image(self, path):
        self.parts.append(path)
        return self


module("astrbot.api", logger=logging.getLogger("astrbot-test"))
module(
    "astrbot.api.event",
    AstrMessageEvent=Dummy,
    MessageChain=MessageChain,
    filter=Filters(),
)
module("astrbot.api.star", Context=Dummy, Star=Dummy)
module("astrbot.api.provider", ProviderRequest=Dummy)
module(
    "astrbot.api.message_components", Plain=Dummy, Image=Dummy, Reply=Dummy, At=Dummy
)
module("astrbot.core.config.astrbot_config", AstrBotConfig=AstrBotConfig)
module("astrbot.core.platform.message_type", MessageType=MessageType)
module("astrbot.core.platform.message_session", MessageSession=MessageSession)
module("astrbot.core.star.star_tools", StarTools=Dummy)
module(
    "astrbot.core.utils.astrbot_path",
    get_astrbot_temp_path=lambda: str(ROOT),
    get_astrbot_workspaces_path=lambda: str(ROOT),
)
module("astrbot.core.utils.io", download_image_by_url=Dummy)
module("astrbot.core.agent.run_context", ContextWrapper=Dummy)
module(
    "astrbot.core.agent.tool", FunctionTool=Dummy, ToolExecResult=Dummy, ToolSet=Dummy
)
module("astrbot.core.astr_agent_context", AstrAgentContext=Dummy)
module("astrbot.core.tools.message_tools", SendMessageToUserTool=Dummy)
module("astrbot.core.agent.message", TextPart=Dummy)
module("astrbot.core.cron.events", CronMessageEvent=Dummy)
module("astrbot.core.db.po", Conversation=Dummy)
module("astrbot.core.provider.entities", ProviderRequest=Dummy)
package = module("image_plugin")
package.__path__ = [str(ROOT)]
module("image_plugin.core").__path__ = [str(ROOT / "core")]
# Dashboard host plumbing is outside this task and requires a running AstrBot.
module("image_plugin.core.page", ImageGenerationPageAPI=Dummy)
plugin_module = importlib.import_module("image_plugin.main")
