#!/usr/bin/env python3
"""Benchmark tool-result context hygiene by comparing raw vs compacted payload sizes."""

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


def make_agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("terminal", "read_file", "search_files")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="bench",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent.context_compressor.threshold_tokens = 20_000
        return agent


SCENARIOS = [
    (
        "large_search_json",
        "search_files",
        json.dumps({
            "success": True,
            "message": "Search completed",
            "results": [{"path": f"src/file_{i}.py", "content": "A" * 5000} for i in range(12)],
        }),
    ),
    (
        "terminal_log_dump",
        "terminal",
        "\n".join([f"[{i:04d}] log line " + ("B" * 350) for i in range(180)]),
    ),
]


if __name__ == "__main__":
    agent = make_agent()
    messages = [{"role": "user", "content": "analyze output"}] * 18
    system_prompt = "system prompt " * 120

    reductions = []
    for name, function_name, payload in SCENARIOS:
        compacted = agent._compact_tool_result_for_context(
            function_name=function_name,
            function_result=payload,
            messages=messages,
            active_system_prompt=system_prompt,
            tool_call_id=f"tc-{name}",
        )
        raw_chars = len(payload)
        compacted_chars = len(compacted)
        reduction = (raw_chars - compacted_chars) / raw_chars * 100 if raw_chars else 0.0
        reductions.append(reduction)
        print("=" * 80)
        print(f"Scenario       : {name}")
        print(f"Tool           : {function_name}")
        print(f"Raw chars      : {raw_chars}")
        print(f"Compacted chars: {compacted_chars}")
        print(f"Reduction      : {reduction:.1f}%")

    avg = sum(reductions) / len(reductions)
    print("=" * 80)
    print(f"Average compaction reduction across {len(SCENARIOS)} scenarios: {avg:.1f}%")
