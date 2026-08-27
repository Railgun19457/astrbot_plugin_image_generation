from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


class _Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass


ROOT = Path(__file__).resolve().parents[1]
astrbot_module = types.ModuleType("astrbot")
astrbot_api_module = types.ModuleType("astrbot.api")
astrbot_api_module.logger = _Logger()
sys.modules.setdefault("astrbot", astrbot_module)
sys.modules.setdefault("astrbot.api", astrbot_api_module)

message_components = types.ModuleType("astrbot.api.message_components")


class _Image:
    pass


class _Reply:
    pass


message_components.Image = _Image
message_components.Reply = _Reply
sys.modules.setdefault("astrbot.api.message_components", message_components)

plugin_module = types.ModuleType("astrbot_plugin_image_generation")
plugin_module.__path__ = [str(ROOT)]
core_module = types.ModuleType("astrbot_plugin_image_generation.core")
core_module.__path__ = [str(ROOT / "core")]
core_shared_module = types.ModuleType("astrbot_plugin_image_generation.core.shared")
core_shared_module.__path__ = [str(ROOT / "core" / "shared")]
core_llm_module = types.ModuleType("astrbot_plugin_image_generation.core.llm")
core_llm_module.__path__ = [str(ROOT / "core" / "llm")]
core_config_module = types.ModuleType("astrbot_plugin_image_generation.core.config")
core_config_module.__path__ = [str(ROOT / "core" / "config")]

for name, module in (
    ("astrbot_plugin_image_generation", plugin_module),
    ("astrbot_plugin_image_generation.core", core_module),
    ("astrbot_plugin_image_generation.core.shared", core_shared_module),
    ("astrbot_plugin_image_generation.core.llm", core_llm_module),
    ("astrbot_plugin_image_generation.core.config", core_config_module),
):
    sys.modules.setdefault(name, module)

constants = importlib.import_module(
    "astrbot_plugin_image_generation.core.shared.constants"
)
models = importlib.import_module("astrbot_plugin_image_generation.core.config.models")
command_handoff = importlib.import_module(
    "astrbot_plugin_image_generation.core.llm.command_handoff"
)


def _should_llm_handle_command(plugin_config, command_name: str) -> bool:
    """Mirror ConfigManager.should_llm_handle_command without importing manager."""
    if not plugin_config.llm_handles_commands:
        return False
    if constants.LLM_TOOL_IMAGE_GENERATION not in plugin_config.enabled_llm_tools:
        return False
    return command_name in plugin_config.llm_handled_commands


def _parse_llm_handled_commands(raw) -> list[str]:
    """Mirror ConfigManager._parse_llm_handled_commands."""
    if isinstance(raw, bool):
        return list(constants.ALL_LLM_HANDLED_COMMANDS) if raw else []
    if not isinstance(raw, list):
        return list(constants.ALL_LLM_HANDLED_COMMANDS)
    selected: list[str] = []
    for item in raw:
        command_name = str(item).strip()
        if (
            command_name in constants.ALL_LLM_HANDLED_COMMANDS
            and command_name not in selected
        ):
            selected.append(command_name)
    return selected


class BuildHandoffPromptTests(unittest.TestCase):
    def test_includes_raw_demand(self):
        text = command_handoff.build_handoff_prompt(raw_demand="画出你自己的形象")
        self.assertIn("画出你自己的形象", text)
        self.assertIn("generate_image", text)
        self.assertNotIn("image_count=", text)

    def test_includes_image_count_when_set(self):
        text = command_handoff.build_handoff_prompt(
            raw_demand="一只猫",
            image_count=3,
        )
        self.assertIn("一只猫", text)
        self.assertIn("image_count=3", text)
        self.assertIn("用户请求生成数量：3", text)


class ShouldLlmHandleCommandTests(unittest.TestCase):
    def _config(
        self,
        *,
        handles: bool,
        commands: set[str] | None = None,
        tools: set[str] | None = None,
    ):
        return models.PluginConfig(
            llm_handles_commands=handles,
            llm_handled_commands=set(
                commands
                if commands is not None
                else constants.ALL_LLM_HANDLED_COMMANDS
            ),
            enabled_llm_tools=set(
                tools if tools is not None else constants.ALL_LLM_TOOLS
            ),
        )

    def test_default_off(self):
        cfg = self._config(handles=False)
        self.assertFalse(
            _should_llm_handle_command(cfg, constants.LLM_HANDLED_COMMAND_GENERATE)
        )

    def test_on_with_tool_and_command(self):
        cfg = self._config(handles=True)
        self.assertTrue(
            _should_llm_handle_command(cfg, constants.LLM_HANDLED_COMMAND_GENERATE)
        )

    def test_on_but_image_tool_disabled(self):
        cfg = self._config(
            handles=True,
            tools={constants.LLM_TOOL_PRESET_QUERY},
        )
        self.assertFalse(
            _should_llm_handle_command(cfg, constants.LLM_HANDLED_COMMAND_GENERATE)
        )

    def test_on_but_command_not_selected(self):
        cfg = self._config(handles=True, commands=set())
        self.assertFalse(
            _should_llm_handle_command(cfg, constants.LLM_HANDLED_COMMAND_GENERATE)
        )


class ParseLlmHandledCommandsTests(unittest.TestCase):
    def test_filters_unknown_and_keeps_generate(self):
        parsed = _parse_llm_handled_commands(["生图", "未知", "生图"])
        self.assertEqual(parsed, ["生图"])

    def test_invalid_type_falls_back_to_default(self):
        parsed = _parse_llm_handled_commands("not-a-list")
        self.assertEqual(parsed, list(constants.ALL_LLM_HANDLED_COMMANDS))


if __name__ == "__main__":
    unittest.main()
