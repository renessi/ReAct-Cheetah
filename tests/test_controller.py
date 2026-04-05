from agent.actions import ActionType
from agent.controller import AgentController


class FakeLLM:
    """Returns pre-configured responses in sequence.

    Planner calls use generate_with_tools() and expect tool-call dicts.
    Summarization calls use generate() and expect plain strings.
    Both consume from the same response list in order.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def generate(self, messages, system=None, stop=None):
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

    def generate_with_tools(self, messages, tools, system=None):
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


def _tool_call(name, **arguments):
    """Helper to build a tool-call response dict."""
    return {
        "tool_call": True,
        "name": name,
        "arguments": arguments,
    }


class FakeWikipediaTool:
    def run(self, payload):
        return {
            "ok": True,
            "tool": "wikipedia",
            "language": payload.get("language", "ru"),
            "query": payload.get("query", ""),
            "title": "Большой Каменный мост",
            "content": (
                "Большой Каменный мост — мост через Москву-реку. "
                "Длина 168 м. Ширина 21 м."
            ),
            "url": "https://ru.wikipedia.org/wiki/Большой_Каменный_мост",
        }


class FakeSVTTool:
    def run(self, payload):
        return {
            "ok": True,
            "tool": "svt_solver",
            "solved_for": "time",
            "value_si": 1800.0,
            "human_readable": "30.0 min",
            "formula": "time = distance / speed",
            "inputs": {
                "distance": {"value": 60000.0, "unit": "meter"},
                "speed": {"value": 33.33, "unit": "meter/second"},
                "time": None,
            },
        }


def test_search_then_finish():
    """Agent searches Wikipedia, LLM summarizes, then agent finishes."""
    llm = FakeLLM([
        # Step 1: planner decides SEARCH_WIKIPEDIA
        _tool_call(
            "SEARCH_WIKIPEDIA",
            thought="I need to find the length of this bridge.",
            query="Большой Каменный мост",
            language="ru",
        ),
        # Summarization LLM call (plain string via generate())
        "Bridge length: 168 m. Width: 21 m.",
        # Step 2: planner decides FINISH
        _tool_call(
            "FINISH",
            thought="I found the length is 168 m.",
            answer="Длина Большого Каменного моста — 168 метров.",
        ),
    ])
    tools = {
        ActionType.SEARCH_WIKIPEDIA: FakeWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)

    response = controller.run("Какая длина Большого Каменного моста?")

    assert response == "Длина Большого Каменного моста — 168 метров."
    assert len(controller.state.react_steps) == 2
    assert controller.state.sources[0]["title"] == "Большой Каменный мост"
    # Fact cache should contain the summarized content
    cached = controller.state.fact_cache.get("Большой Каменный мост")
    assert cached is not None
    assert cached["summary"] == "Bridge length: 168 m. Width: 21 m."


def test_compute_svt_then_finish():
    """Agent uses COMPUTE_SVT tool, then finishes — no summarization call."""
    llm = FakeLLM([
        # Step 1: compute
        _tool_call(
            "COMPUTE_SVT",
            thought="I have distance and speed, I can compute time.",
            distance={"value": 60, "unit": "km"},
            speed={"value": 120, "unit": "km/h"},
            time=None,
        ),
        # Step 2: finish
        _tool_call(
            "FINISH",
            thought="The result is 30 minutes.",
            answer="It takes 30 minutes.",
        ),
    ])
    tools = {
        ActionType.SEARCH_WIKIPEDIA: FakeWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)

    response = controller.run("How long to travel 60 km at 120 km/h?")

    assert response == "It takes 30 minutes."
    assert len(controller.state.react_steps) == 2
    assert controller.state.react_steps[0].action == "COMPUTE_SVT"


def test_ask_user_then_resume():
    """Agent asks a clarifying question, user responds, agent finishes."""
    llm = FakeLLM([
        # Turn 1: ask for missing info
        _tool_call(
            "ASK_USER",
            thought="I need to know the speed.",
            question="What speed should I use?",
        ),
        # Turn 2: now finish
        _tool_call(
            "FINISH",
            thought="User said 100 km/h, I can answer now.",
            answer="At 100 km/h it takes 36 minutes.",
        ),
    ])
    tools = {
        ActionType.SEARCH_WIKIPEDIA: FakeWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)

    # Turn 1
    response = controller.run("How long to cross 60 km?")
    assert response == "What speed should I use?"

    # Turn 2 — user responds, agent should see dialogue history
    response = controller.run("100 km/h")
    assert response == "At 100 km/h it takes 36 minutes."
    assert len(controller.state.dialogue_history) == 4


def test_max_steps_reached():
    """Agent hits max_steps and returns fallback message."""
    compute_call = _tool_call(
        "COMPUTE_SVT",
        thought="I need to compute again.",
        distance={"value": 60, "unit": "km"},
        speed={"value": 120, "unit": "km/h"},
        time=None,
    )
    llm = FakeLLM([compute_call] * 5)
    tools = {
        ActionType.SEARCH_WIKIPEDIA: FakeWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)
    controller.max_steps = 3

    response = controller.run("test")

    assert "maximum number of reasoning steps" in response


def test_fact_cache_hit_skips_wikipedia():
    """Second search for same query uses cache — no Wikipedia call."""
    llm = FakeLLM([
        # Step 1: planner decides SEARCH_WIKIPEDIA
        _tool_call(
            "SEARCH_WIKIPEDIA",
            thought="Look up the bridge.",
            query="мост",
            language="ru",
        ),
        # Summarization call (plain string)
        "Bridge length: 168 m.",
        # Step 2: planner decides to search same query again
        _tool_call(
            "SEARCH_WIKIPEDIA",
            thought="Let me check the bridge again.",
            query="мост",
            language="ru",
        ),
        # No summarization call — cache hit
        # Step 3: finish
        _tool_call(
            "FINISH",
            thought="I have the data.",
            answer="168 m.",
        ),
    ])

    call_count = 0
    original_run = FakeWikipediaTool.run

    class CountingWikipediaTool(FakeWikipediaTool):
        def run(self, payload):
            nonlocal call_count
            call_count += 1
            return original_run(self, payload)

    tools = {
        ActionType.SEARCH_WIKIPEDIA: CountingWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)

    response = controller.run("test")

    assert response == "168 m."
    assert call_count == 1  # Wikipedia called only once, second was cached
    assert "(cached)" in controller.state.react_steps[1].observation


def test_irrelevant_wikipedia_article_not_cached():
    """Article with no SVT facts is not stored in fact cache."""
    llm = FakeLLM([
        # Step 1: planner searches Wikipedia
        _tool_call(
            "SEARCH_WIKIPEDIA",
            thought="Look up this topic.",
            query="French Revolution",
            language="en",
        ),
        # Summarization returns the no-facts sentinel
        "NO_RELEVANT_FACTS",
        # Step 2: planner finishes
        _tool_call(
            "FINISH",
            thought="No useful data found.",
            answer="Could not find relevant data.",
        ),
    ])
    tools = {
        ActionType.SEARCH_WIKIPEDIA: FakeWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)

    controller.run("test")

    # Article should NOT be in the fact cache
    assert controller.state.fact_cache.get("French Revolution") is None
    # Observation should tell the planner the article was unhelpful
    obs = controller.state.react_steps[0].observation
    assert "no speed, distance, or time facts" in obs


def test_out_of_domain():
    """Agent recognizes non-SVT question and politely declines."""
    llm = FakeLLM([
        _tool_call(
            "FINISH",
            thought="This is not an SVT problem.",
            answer="I can only solve speed-velocity-time problems.",
        ),
    ])
    tools = {
        ActionType.SEARCH_WIKIPEDIA: FakeWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)

    response = controller.run("What is the capital of France?")

    assert "speed-velocity-time" in response.lower() or "SVT" in response


def test_unknown_tool_handled_gracefully():
    """If planner returns invalid function name, controller doesn't crash."""
    llm = FakeLLM([
        _tool_call(
            "NONEXISTENT_TOOL",
            thought="Hmm.",
        ),
    ])
    tools = {
        ActionType.SEARCH_WIKIPEDIA: FakeWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)

    response = controller.run("test")

    assert isinstance(response, str)
    assert len(response) > 0


def test_wikipedia_language_passed_to_tool():
    """The language from planner's action input reaches the Wikipedia tool."""
    received_payloads = []

    class SpyWikipediaTool:
        def run(self, payload):
            received_payloads.append(payload)
            return {
                "ok": True,
                "tool": "wikipedia",
                "language": payload.get("language", "ru"),
                "query": payload.get("query", ""),
                "title": "Cheetah",
                "content": "The cheetah can reach speeds of 112 km/h.",
                "url": "https://en.wikipedia.org/wiki/Cheetah",
            }

    llm = FakeLLM([
        _tool_call(
            "SEARCH_WIKIPEDIA",
            thought="Cheetah is international, use English Wikipedia.",
            query="cheetah",
            language="en",
        ),
        # Summarization call
        "Maximum speed: 112 km/h.",
        # Finish
        _tool_call(
            "FINISH",
            thought="Done.",
            answer="done",
        ),
    ])
    tools = {
        ActionType.SEARCH_WIKIPEDIA: SpyWikipediaTool(),
        ActionType.COMPUTE_SVT: FakeSVTTool(),
    }
    controller = AgentController(llm=llm, tools=tools)

    controller.run("test")

    assert received_payloads[0]["language"] == "en"
    assert "[en]" in controller.state.react_steps[0].observation
