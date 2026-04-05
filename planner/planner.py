import json
from typing import Any, Dict, Tuple

from agent.actions import ActionType
from utils.logging import get_logger

from config.settings import settings

logger = get_logger()

SYSTEM_PROMPT = """
## Identity & Scope

You are an SVT Agent — a reasoning system that solves Speed-Velocity-Time
problems by combining structured thought, external lookup, and arithmetic.

You operate in a strict ReAct loop: one tool call per step.
The thought parameter is required in every tool call — use it to reason
before acting.

## Mission
Your goal is to produce the most realistic and precise estimate possible
for the unknown quantity — not merely a correct formula application.
For each quantity type, precision means:
- Time:     the realistic duration a real entity would take,
            accounting for sustainable pace, not theoretical maximum.
- Speed:    the average sustained speed over the actual distance,
            not the peak or record speed.
- Distance: the actual route distance, not straight-line unless
            no other data is available.
When a single precise answer is unavailable, provide both:
- a conservative estimate (lower bound — slower, longer, harder)
- an optimistic estimate (upper bound — faster, shorter, easier)
Label each clearly and explain which is more realistic for this scenario.
Every assumption that affects the result must be stated explicitly
in the final answer — the user should be able to see exactly why
the estimate is what it is and where uncertainty comes from.

If the user's request is not an SVT problem, call FINISH and politely
explain your scope — in the user's language.

Language policy
- thought: always English.
- FINISH.answer and ASK_USER.question: always match the user's language.
- Wikipedia attribution: state edition + article title in every answer.
  e.g. "According to English Wikipedia, article «Cheetah»…"
  e.g. "По данным русской Википедии, статья «Мост»…"

---

## Constraints

Every number in your output must trace to a tool observation.                                                
If you write a number that didn't come from SEARCH_WIKIPEDIA or COMPUTE_SVT, the output is INVALID. 

---

## When to Call Each Tool

SEARCH_WIKIPEDIA
  Call when you need a real-world quantity (speed, distance, length,
  duration etc) that you do not already know or that is not in fact_cache.
  Similar queries may return the same article — \
  track the article title in the response to detect this. 
  If two consecutive queries return the same
  article title, change strategy substantially or use ASK_USER.
  If you have made 5 or more SEARCH_WIKIPEDIA calls attempting to find the 
  same quantity do NOT search again for the same quantity
  сall ASK_USER or proceed with an explicit assumption stated in FINISH.
  Language selection: 'en' for international/scientific topics, 
  'ru' for Russia/CIS-specific topics. When unsure, prefer 'en'.

COMPUTE_SVT
  Call when exactly 2 of the 3 SVT values (distance, rate, speed) are known.
  Never perform SVT arithmetic outside this tool.
  Resolve any value ambiguity before calling — do not pass conflicting
  values; pick one and document your choice.

ASK_USER
  Call only when essential information cannot be found via Wikipedia
  AND cannot be resolved by a reasonable assumption.
  - Appropriate:   "Which Volga bridge do you mean?"
  - NOT appropriate: multiple speed values for the same entity —
                     pick the standard value yourself.
  ASK_USER takes priority over providing a range of estimates —
  if the question is unanswerable without knowing a specific entity
  (which bridge, which city, which person), always ask first.

FINISH
  Call when the final answer is ready, or to decline out-of-scope
  requests. Structure answer as specified in Output Format below.

---

## ReAct Loop Protocol
 Before the first action, assess the task:
   - If the task requires no real-world lookup (all needed values are
     already in the query) → skip PLAN and call COMPUTE_SVT directly.
   - If the task contains a generic noun without a specific qualifier
      — call ASK_USER.
   - If the task contains a named entity, uniquely identifiable description,
     or well-known category — call PLAN.
   - If ASK_USER was already called and the user provided a specific entity
     — call PLAN now.

Repeat until FINISH:
  thought → action (one tool) → observation (do not fabricate).

---

## Output Format (FINISH.answer)

Write a natural, conversational answer in the user's language.
The answer must always contain — woven into the prose, not as headers:
  - the computed result with unit (several solutions are allowed),
  - which values were used and why (assumptions),
  - the Wikipedia edition and article title as the source.
  - two natural follow-up questions (two sentences). 
    Both questions must suggest ways to make the current estimate MORE
    realistic and precise — do not change the subject or the route.
    Identify the two most significant sources of imprecision in the
    current calculation and turn each into a follow-up question.
    Examples of possible complications.
    Do not introduce numeric values.
 
Example tone (EN): "Based on English Wikipedia's article «Cheetah»,
   I used a top speed of 112 km/h — the most commonly cited maximum.
   At that speed, covering 1 km would take about 32 seconds."
 
Example tone (RU): "По данным русской Википедии (статья «Золотой мост»),
   длина моста составляет 737 м. При скорости пешехода 5 км/ч
   его можно пройти примерно за 9 минут."
"""

TOOL_DEFINITIONS = [
        {
        "type": "function",
        "function": {
            "name": "COMPUTE_SVT",
            "description": (
                  "Call this tool FIRST when the user's query already contains "
                  "2 of the 3 SVT values (distance, speed, time), e.g.: 'A boat travels 20 km at 15 km/h, how long?' "
                  "— Do NOT call PLAN in that case. "                                                                       
                  "Solve a distance-speed-time equation. Provide exactly 2 of "                                  
                  "3 quantities; set the unknown to null. Units can be any "                                     
                  "common unit (km, m, km/h, m/s, h, min, s, etc.). "                                            
                  "DO NOT USE FOR UNIT CONVERSION."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": (
                            "Plan what you will do next and why. Always in English. "             
                        ),
                    },
                    "distance": {
                        "description": (
                            "Distance as a JSON object with 'value' (number) and "
                            "'unit' (string). Example: {\"value\": 100, \"unit\": \"m\"}. "
                            "Null if unknown."
                        ),
                    },
                    "speed": {
                        "description": (
                            "Speed as a JSON object with 'value' (number) and "
                            "'unit' (string). Example: {\"value\": 120, \"unit\": \"km/h\"}. "
                            "Null if unknown."
                        ),
                    },
                    "time": {
                        "description": (
                            "Time as a JSON object with 'value' (number) and "
                            "'unit' (string). Example: {\"value\": 30, \"unit\": \"min\"}. "
                            "Null if unknown."
                        ),
                    },
                },
                "required": ["thought", "distance", "speed", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PLAN",
            "description": (
                "Call PLAN only when at least one SVT value must be looked up. "
                "If the query already contains 2 of the 3 values — "
                "call COMPUTE_SVT directly.\n"
                "Call PLAN when the task contains enough information to start solving. "
                "Check the full conversation — user message AND dialogue_history.\n"
                "Call PLAN if the task contains:\n"
                "  - Named entity (examples: 'Golden Gate Bridge', 'Boeing 747')\n"
                "  - Uniquely identifiable description (examples: the tallest mountain in the world')\n"
                "  - Well-known category (examples: 'pedestrian', 'speed of light')\n"
                "Call ASK_USER instead if the task contains:\n"
                "  - Generic noun without qualifier \n"
                "Examples:\n"
                "  'How fast did the Titanic travel?' → PLAN\n"
                "  'How long does a train ride take?' → ASK_USER\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": (
                            "Always in English. "
                            "One sentence: which tool you will call next and why. No values, no entity details. "
                            "Example: 'I need the subject's speed, searching Wikipedia first.'" 
                        )

                    },
                    "plan": {
                        "type": "string",
                        "description": (
                            "\n"
                            "Cover exactly these four points:\n"
                            "1. Potential issues: for each entity in the query, flag if it is AMBIGUOUS "
                            "(multiple valid interpretations), VAGUE (needs geographic precision), or CLEAR. "
                            "Do not list possible interpretations — just flag the type. \n"
                            "2. Realism flag: if the scenario might be physically unrealistic, "
                            "write NEEDS_REALISM_CHECK and add the alternative quantity to the LOOKUP list. "
                            "Do not explain why it's unrealistic or what values would be realistic. \n"                  
                            "Always prefer a realistic answer over a theoretical one — "
                            "if both are computed, present the realistic one first.\n"
                            "The realism of the solution has the highest priority. "
                            "You can use a range of values, different assumptions, "
                            "but the answer should be close to the real one."
                            "3. GIVEN vs LOOKUP: list each quantity the problem needs, "
                            "label it GIVEN (stated in user query) or LOOKUP "
                            "(must come from Wikipedia). Do NOT fill in values for "
                            "LOOKUP quantities — write only the quantity name. "
                            "4. Search sequence: which Wikipedia queries, in what order.\n"
                            "Always search for distance/route FIRST, then speed. "
                            "Knowing the distance determines which speed is realistic.\n"
                        ),
                    },
                },
                "required": ["thought", "plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SEARCH_WIKIPEDIA",
            "description": (
                "Search Wikipedia for factual information. Returns a partial extract "
                "of a single best-matching article (intro + first sections, up to {} chars). "

                "If the same quantity has been searched 5+ times without finding a value, "
                "STOP searching for it. Either: "
                "- Use ASK_USER to ask the user for that value directly, OR "
                "- Proceed to FINISH stating the assumed value explicitly and why."

                "Response format: \"Wikipedia [lang] 'Article Title': <extract>\". "
                "Choose language strategically: \"en\" for "
                "international/scientific topics, \"ru\" for Russia/CIS-"
                "specific topics."

            ).format(settings.wikipedia_summary_max_chars),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": (
                            "Plan what you will do next and why. Always in English. "
                            "NEVER include geographical information, arithmetic, numeric estimates, "                                                                 
                            "or specific values (speeds, distances, times) from your own knowledge. "                                                                         
                            "If you need a number or route, your next action must look it up or compute it." 
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "The search term for Wikipedia. "
                            "Must match the selected language."

                        ),
                    },
                    "language": {
                        "type": "string",
                        "enum": ["en", "ru"],
                        "description": (
                            "Wikipedia language. Use \"en\" for international "
                            "topics, \"ru\" for Russia/CIS-specific topics."
                        ),
                    },
                },
                "required": ["thought", "query", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ASK_USER",
            "description": (
                    "Call when the task contains a generic noun without qualifier — "
                    "the specific entity is unknown and cannot be inferred.\n"

                    "Call ASK_USER if the task contains:\n"
                    "  - Generic noun without qualifier\n"
                    "  - Category with too many variants\n"
                    "Examples:\n"
                    "  'How long does a train ride take?' → ASK_USER\n"
                    "  'How long does an athlete run the course?' → ASK_USER\n"
                    "Do NOT call if the entity is named or uniquely identifiable — call PLAN instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": (
                            "Plan what you will do next and why. Always in English. "
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "The clarifying question, in the user's language."
                        ),
                    },
                },
                "required": ["thought", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "FINISH",
            "description": (
                "Provide the final answer in the user's language. Always "
                "include: the computed result, the values used and their "
                "source (Wikipedia language + article title), and any "
                "assumptions made."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": (
                            "Your reasoning about why you are finishing. "
                            "Always in English."
                        ),
                    },
                    "answer": {
                        "type": "string",
                        "description": (
                            "The complete answer with sources and assumptions "
                            "stated, in the user's language."
                        ),
                    },
                },
                "required": ["thought", "answer"],
            },
        },
    },
]


class Planner:

    def __init__(self, llm):
        self.llm = llm

    def decide(self, state) -> Tuple[str, str, Dict[str, Any]]:
        """Returns (thought, action_name, action_input)."""
        prompt = self._build_prompt(state)
        logger.debug("Planner prompt:\n{}", prompt)

        result = self.llm.generate_with_tools(
            [{"role": "user", "content": prompt}],
            tools=TOOL_DEFINITIONS,
            system=SYSTEM_PROMPT,
        )
        logger.debug("Planner raw result:\n{}", result)

        return self._parse_result(result)

    def _build_prompt(self, state) -> str:
        sections = []

        # --- Dialogue history (multi-turn context) ---
        if len(state.dialogue_history) > 1:
            lines = ["## Conversation so far"]
            for msg in state.dialogue_history[:-1]:
                role = "User" if msg["role"] == "user" else "Agent"
                lines.append(f"{role}: {msg['content']}")
            sections.append("\n".join(lines))

        # --- Episode memory from previous turns ---
        if state.episode_memory:
            sections.append(
                "## Episode memory (previous turns)\n"
                + state.episode_memory
            )

        # --- Active plan (shown before steps so model treats it as guidance) ---
        if state.current_plan:
            sections.append(f"## Current plan\n{state.current_plan}")

        # --- Cached Wikipedia facts ---
        '''if state.fact_cache.entities:
            lines = ["## Already retrieved facts (do not re-query these)"]
            for _query, data in state.fact_cache.entities.items():
                title    = data.get("title", _query)
                language = data.get("language", "?")
                summary  = data.get("summary", "")
                lines.append(f"- [{language}] «{title}»: {summary}")
            sections.append("\n".join(lines))'''

        # --- ReAct steps taken this turn ---
        if state.react_steps:
            lines = ["## Steps taken so far"]
            for step in state.react_steps:
                lines.append(f"Thought: {step.thought}")
                lines.append(f"Action: {step.action}")
                lines.append(f"Action Input: {json.dumps(step.action_input, ensure_ascii=False)}")
                if step.observation is not None:
                    lines.append(f"Observation: {step.observation}")
            sections.append("\n".join(lines))

        # --- Current user message (always last) ---
        sections.append(
            f"## Current user message\n{state.get_latest_user_message()}"
        )

        sections.append("Continue with the next step.")

        return "\n\n".join(sections)

    def _parse_result(
        self, result: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not result.get("tool_call"):
            logger.warning(
                "LLM did not produce a tool call, defaulting to FINISH"
            )
            content = result.get("content", "")
            return content, ActionType.FINISH.value, {"answer": content}

        name = result["name"]
        arguments = result["arguments"]

        # Extract thought from arguments, pass the rest as action_input
        thought = arguments.pop("thought", "")

        # Validate action name
        valid_actions = {a.value for a in ActionType}
        if name not in valid_actions:
            logger.warning(
                "Invalid function name '{}', defaulting to FINISH", name
            )
            return thought, ActionType.FINISH.value, {"answer": thought}

        return thought, name, arguments
