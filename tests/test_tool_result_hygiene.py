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


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("terminal", "read_file", "todo")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


class TestToolResultHygiene:
    def test_plaintext_compaction_keeps_head_and_tail(self):
        text = "\n".join([f"line-{i}: " + ("x" * 80) for i in range(40)])
        compacted = AIAgent._compact_plaintext_for_budget(text, 400)
        assert "line-0" in compacted
        assert "line-39" in compacted
        assert "omitted" in compacted
        assert len(compacted) < len(text)

    def test_json_compaction_preserves_high_signal_keys(self):
        payload = {
            "success": True,
            "message": "done",
            "results": [{"title": f"item-{i}", "content": "x" * 600} for i in range(10)],
            "extra": "y" * 2000,
        }
        compacted = AIAgent._compact_json_for_budget(payload, 700)
        assert compacted["success"] is True
        assert compacted["message"] == "done"
        assert "results" in compacted
        assert len(compacted["results"]) <= 5

    def test_small_tool_result_is_unchanged(self):
        agent = _make_agent()
        result = json.dumps({"success": True, "message": "ok"})
        compacted = agent._compact_tool_result_for_context(
            function_name="read_file",
            function_result=result,
            messages=[{"role": "user", "content": "hello"}],
            active_system_prompt="system",
            tool_call_id="tc1",
        )
        assert compacted == result

    def test_large_json_tool_result_is_compacted(self):
        agent = _make_agent()
        agent.context_compressor.threshold_tokens = 20000
        payload = {
            "success": True,
            "results": [{"path": f"file-{i}.py", "content": "z" * 5000} for i in range(12)],
            "message": "large payload",
        }
        result = json.dumps(payload)
        compacted = agent._compact_tool_result_for_context(
            function_name="search_files",
            function_result=result,
            messages=[{"role": "user", "content": "find things"}] * 20,
            active_system_prompt="system prompt" * 100,
            tool_call_id="tc1",
        )
        assert len(compacted) < len(result)
        assert "Context-optimized search_files result" in compacted
        assert "large payload" in compacted

    def test_large_plaintext_tool_result_is_compacted(self):
        agent = _make_agent()
        agent.context_compressor.threshold_tokens = 15000
        result = "\n".join([f"row-{i}: " + ("x" * 400) for i in range(120)])
        compacted = agent._compact_tool_result_for_context(
            function_name="terminal",
            function_result=result,
            messages=[{"role": "user", "content": "show logs"}] * 15,
            active_system_prompt="system prompt" * 80,
            tool_call_id="tc2",
        )
        assert len(compacted) < len(result)
        assert "row-0" in compacted
        assert "row-119" in compacted
        assert "Context-optimized terminal result" in compacted
