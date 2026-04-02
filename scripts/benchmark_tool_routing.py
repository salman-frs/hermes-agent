#!/usr/bin/env python3
"""Benchmark adaptive tool routing by comparing tool-schema payload sizes."""

import json
from statistics import mean

from run_agent import AIAgent


SCENARIOS = [
    "please explain this code simply",
    "tolong riset dokumentasi terbaru terraform lalu ringkas",
    "open this website URL in browser and click login",
    "tolong analisis gambar https://example.com/diagram.png",
    "jadwalkan report ini setiap hari jam 9 pagi",
]


def tool_schema_chars(agent: AIAgent) -> int:
    return len(json.dumps(agent.tools or [], ensure_ascii=False))


def bench_prompt(prompt: str) -> dict:
    agent = AIAgent(
        api_key="benchmark",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["hermes-cli"],
    )
    full_toolsets = sorted(agent._loaded_toolsets())
    full_tools = len(agent.tools or [])
    full_chars = tool_schema_chars(agent)

    agent._maybe_route_toolsets_for_turn(prompt)
    routed_toolsets = sorted(agent._loaded_toolsets())
    routed_tools = len(agent.tools or [])
    routed_chars = tool_schema_chars(agent)

    reduction = 0.0
    if full_chars:
        reduction = (full_chars - routed_chars) / full_chars * 100.0

    return {
        "prompt": prompt,
        "full_toolsets": full_toolsets,
        "routed_toolsets": routed_toolsets,
        "full_tools": full_tools,
        "routed_tools": routed_tools,
        "full_chars": full_chars,
        "routed_chars": routed_chars,
        "reduction_percent": reduction,
    }


if __name__ == "__main__":
    results = [bench_prompt(prompt) for prompt in SCENARIOS]
    for item in results:
        print("=" * 80)
        print(f"Prompt: {item['prompt']}")
        print(f"Full toolsets  : {', '.join(item['full_toolsets'])}")
        print(f"Routed toolsets: {', '.join(item['routed_toolsets'])}")
        print(f"Tools          : {item['full_tools']} -> {item['routed_tools']}")
        print(f"Schema chars   : {item['full_chars']} -> {item['routed_chars']}")
        print(f"Reduction      : {item['reduction_percent']:.1f}%")

    avg = mean(item["reduction_percent"] for item in results)
    print("=" * 80)
    print(f"Average schema reduction across {len(results)} scenarios: {avg:.1f}%")
