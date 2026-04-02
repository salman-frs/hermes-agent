#!/usr/bin/env python3
"""Benchmark intelligence-preserving auto-expansion for deferred tool surfaces."""

import json
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


TOOLSET_MAP = {
    "terminal": "terminal",
    "read_file": "file",
    "todo": "todo",
    "memory": "memory",
    "session_search": "session_search",
    "clarify": "clarify",
    "skills_list": "skills",
    "send_message": "messaging",
    "browser_navigate": "browser",
}


def defs_for_toolsets(enabled_toolsets=None, **_kwargs):
    if enabled_toolsets is None:
        enabled_toolsets = sorted(set(TOOLSET_MAP.values()))
    names = [name for name, toolset in TOOLSET_MAP.items() if toolset in set(enabled_toolsets)]
    return _make_tool_defs(*names)


with (
    patch("run_agent.get_tool_definitions", side_effect=defs_for_toolsets),
    patch("run_agent.get_toolset_for_tool", side_effect=TOOLSET_MAP.get),
    patch("run_agent.get_all_tool_names", return_value=list(TOOLSET_MAP)),
    patch("run_agent.check_toolset_requirements", return_value={}),
    patch("run_agent.OpenAI"),
    patch("hermes_cli.config.load_config", return_value={"tool_routing": {"enabled": True}}),
):
    agent = AIAgent(
        api_key="bench",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=sorted(set(TOOLSET_MAP.values())),
    )
    agent.client = MagicMock()

    def schema_chars() -> int:
        return len(json.dumps(agent.tools or [], ensure_ascii=False))

    full_tools = len(agent.tools or [])
    full_chars = schema_chars()
    agent._maybe_route_toolsets_for_turn("please explain this code simply")
    lean_tools = len(agent.tools or [])
    lean_chars = schema_chars()
    expanded = agent._maybe_expand_for_tool_call("send_message")
    expanded_tools = len(agent.tools or [])
    expanded_chars = schema_chars()

    print(f"Full surface   : {full_tools} tools, {full_chars} chars")
    print(f"Lean surface   : {lean_tools} tools, {lean_chars} chars")
    print(f"Auto-expanded  : {expanded_tools} tools, {expanded_chars} chars")
    print(f"Recovered tool : {expanded}")
    print(f"Lean reduction : {(full_chars - lean_chars) / full_chars * 100:.1f}%")
    print(f"Expansion delta: {expanded_chars - lean_chars} chars")
