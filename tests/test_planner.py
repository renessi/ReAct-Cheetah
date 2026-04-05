from agent.state import AgentState
from planner.planner import Planner


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate_with_tools(self, messages, tools, system=None):
        return self.response


def _tool_call(name, **arguments):
    """Helper to build a tool-call response dict."""
    return {
        "tool_call": True,
        "name": name,
        "arguments": arguments,
    }


def test_parse_valid_tool_call():
    llm = FakeLLM(_tool_call(
        "SEARCH_WIKIPEDIA",
        thought="I need to search for the bridge length.",
        query="Golden Gate Bridge",
        language="en",
    ))
    planner = Planner(llm)
    state = AgentState()
    state.add_user_message("How long is the Golden Gate Bridge?")

    thought, action, action_input = planner.decide(state)

    assert thought == "I need to search for the bridge length."
    assert action == "SEARCH_WIKIPEDIA"
    assert action_input == {"query": "Golden Gate Bridge", "language": "en"}


def test_no_tool_call_defaults_to_finish():
    llm = FakeLLM({
        "tool_call": False,
        "content": "Some random text without proper format",
    })
    planner = Planner(llm)
    state = AgentState()
    state.add_user_message("test")

    thought, action, action_input = planner.decide(state)

    assert action == "FINISH"
    assert "answer" in action_input


def test_invalid_function_name_defaults_to_finish():
    llm = FakeLLM(_tool_call(
        "NONEXISTENT",
        thought="Trying something weird.",
    ))
    planner = Planner(llm)
    state = AgentState()
    state.add_user_message("test")

    thought, action, action_input = planner.decide(state)

    assert action == "FINISH"
    assert thought == "Trying something weird."


def test_prompt_includes_dialogue_history():
    llm = FakeLLM(_tool_call(
        "FINISH",
        thought="User provided speed.",
        answer="done",
    ))
    planner = Planner(llm)
    state = AgentState()
    state.add_user_message("How long to cross 60 km?")
    state.add_agent_message("What speed?")
    state.add_user_message("100 km/h")

    prompt = planner._build_prompt(state)

    assert "How long to cross 60 km?" in prompt
    assert "What speed?" in prompt
    assert "100 km/h" in prompt


def test_prompt_includes_react_steps():
    llm = FakeLLM(_tool_call(
        "COMPUTE_SVT",
        thought="Now I can compute.",
        distance={"value": 60, "unit": "km"},
        speed={"value": 100, "unit": "km/h"},
        time=None,
    ))
    planner = Planner(llm)
    state = AgentState()
    state.add_user_message("test")

    from agent.step import ReActStep
    state.react_steps.append(ReActStep(
        thought="Need to look up distance",
        action="SEARCH_WIKIPEDIA",
        action_input={"query": "test", "language": "en"},
        observation="Wikipedia [en] 'Test': some info",
    ))

    prompt = planner._build_prompt(state)

    assert "Need to look up distance" in prompt
    assert "SEARCH_WIKIPEDIA" in prompt
    assert "Wikipedia [en] 'Test': some info" in prompt


def test_prompt_includes_cached_facts_via_react_steps():
    """Cached facts appear in the prompt through react step observations."""
    llm = FakeLLM(_tool_call(
        "FINISH",
        thought="done.",
        answer="done",
    ))
    planner = Planner(llm)
    state = AgentState()
    state.add_user_message("test")

    from agent.step import ReActStep
    state.react_steps.append(ReActStep(
        thought="Look up cheetah speed",
        action="SEARCH_WIKIPEDIA",
        action_input={"query": "cheetah", "language": "en"},
        observation="Wikipedia [en] 'Cheetah': Max speed: 112 km/h.",
    ))

    prompt = planner._build_prompt(state)

    assert "Cheetah" in prompt
    assert "Max speed: 112 km/h." in prompt
