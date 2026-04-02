from contextlib import ExitStack, contextmanager
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


class TestAutomaticToolRouting:
    _TOOLSET_MAP = {
        "terminal": "terminal",
        "read_file": "file",
        "todo": "todo",
        "memory": "memory",
        "session_search": "session_search",
        "clarify": "clarify",
        "skills_list": "skills",
        "browser_navigate": "browser",
        "vision_analyze": "vision",
        "text_to_speech": "tts",
        "execute_code": "code_execution",
        "delegate_task": "delegation",
        "cronjob": "cronjob",
        "web_search": "web",
        "send_message": "messaging",
        "honcho_context": "honcho",
    }

    def _defs_for_toolsets(self, enabled_toolsets=None, **_kwargs):
        if enabled_toolsets is None:
            enabled_toolsets = sorted(set(self._TOOLSET_MAP.values()))
        names = [
            name for name, toolset in self._TOOLSET_MAP.items()
            if toolset in set(enabled_toolsets)
        ]
        return _make_tool_defs(*names)

    @contextmanager
    def _agent_ctx(self):
        with ExitStack() as stack:
            stack.enter_context(patch("run_agent.get_tool_definitions", side_effect=self._defs_for_toolsets))
            stack.enter_context(patch("run_agent.get_toolset_for_tool", side_effect=self._TOOLSET_MAP.get))
            stack.enter_context(patch("run_agent.check_toolset_requirements", return_value={}))
            stack.enter_context(patch("run_agent.OpenAI"))
            stack.enter_context(
                patch(
                    "hermes_cli.config.load_config",
                    return_value={"tool_routing": {"enabled": True}},
                )
            )
            agent = AIAgent(
                api_key="***",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                enabled_toolsets=sorted(set(self._TOOLSET_MAP.values())),
            )
            agent.client = MagicMock()
            yield agent

    def test_simple_chat_routes_to_core_only(self):
        with self._agent_ctx() as agent:
            agent._maybe_route_toolsets_for_turn("please explain this code simply")

            assert set(agent.enabled_toolsets) == {
                "terminal", "file", "todo", "memory", "session_search", "clarify", "skills"
            }
            assert "browser_navigate" not in agent.valid_tool_names
            assert "delegate_task" not in agent.valid_tool_names
            assert "cronjob" not in agent.valid_tool_names
            assert "execute_code" not in agent.valid_tool_names

    def test_browser_request_keeps_browser_tools(self):
        with self._agent_ctx() as agent:
            agent._maybe_route_toolsets_for_turn("open this website URL in browser and click login")

            assert "browser" in set(agent.enabled_toolsets)
            assert "browser_navigate" in agent.valid_tool_names

    def test_indonesian_research_and_schedule_route_correct_toolsets(self):
        with self._agent_ctx() as agent:
            agent._maybe_route_toolsets_for_turn("tolong riset dokumentasi terbaru lalu jadwalkan report setiap hari")

            active = set(agent.enabled_toolsets)
            assert "web" in active
            assert "cronjob" in active

    def test_later_turn_can_expand_missing_toolsets(self):
        with self._agent_ctx() as agent:
            agent._maybe_route_toolsets_for_turn("please explain this code simply")
            assert "messaging" not in set(agent.enabled_toolsets)

            agent._maybe_route_toolsets_for_turn(
                "sekarang kirim pesan ke whatsapp",
                conversation_history=[{"role": "user", "content": "please explain this code simply"}],
            )

            assert "messaging" in set(agent.enabled_toolsets)
            assert "send_message" in agent.valid_tool_names

    def test_image_request_routes_to_vision(self):
        with self._agent_ctx() as agent:
            agent._maybe_route_toolsets_for_turn("tolong analisis gambar https://example.com/diagram.png")

            assert "vision" in set(agent.enabled_toolsets)
            assert "vision_analyze" in agent.valid_tool_names

    def test_existing_history_noop_when_no_new_capability_needed(self):
        with self._agent_ctx() as agent:
            before = set(agent.valid_tool_names)
            agent._tool_route_locked = True
            agent._maybe_route_toolsets_for_turn(
                "please explain this code simply",
                conversation_history=[{"role": "user", "content": "hi"}],
            )

            assert set(agent.valid_tool_names) == before
