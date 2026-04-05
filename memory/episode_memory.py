import json
import time

from agent.actions import ActionType
from config.settings import settings
from llm.base_client import LLMClient
from utils.logging import get_logger

logger = get_logger()

EPISODE_SUMMARY_PROMPT = """\
You are a memory summarizer for a Speed-Velocity-Time reasoning agent.

You will receive the full ReAct transcript of a completed turn: the plan, \
every Thought → Action → Observation step, and the final answer.

Your job is to distill this into a compact episode summary that the agent \
can use in future turns to avoid redundant work and maintain reasoning \
continuity.

The summary MUST include, in this order:

1. **User query** — one sentence restating what was asked.
2. **Facts found** — each fact with its source (Wikipedia edition + article \
title + language). Include exact numeric values with units. \
Group by entity.
3. **Computation results** — every COMPUTE_SVT call: inputs, formula, result. \
Include intermediate steps if there were multiple computations.
4. **Assumptions made** — every assumption the agent chose (e.g., "used top \
speed of 112 km/h, not sustained 80 km/h") and the reasoning behind \
the choice.
5. **Problems & resolutions** — any failed searches, dead ends, ambiguities \
encountered, and how they were resolved.
6. **Search strategies** — which Wikipedia queries worked (language, query \
string → useful article) and which failed (returned irrelevant or empty \
results), so the agent does not repeat failed searches.
7. **Final answer** — the answer delivered to the user (abbreviated if long).

Formatting rules:
- Use plain text with markdown-style headers (## for sections).
- Be precise with numbers — never round or approximate values from \
the transcript.
- Maximum {max_chars} characters.
- Do not add information not present in the transcript.
- Write in English regardless of the user's language."""


class EpisodeMemory:
    """Handles episode summarization: deciding the strategy (LLM vs snapshot),
    building transcripts, and writing to state.episode_memory."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.summary_elapsed: float = 0

    def needs_llm_summary(self, state) -> bool:
        """Decide whether the current turn needs LLM summarization
        or can use a deterministic snapshot.

        LLM summary when: PLAN step present OR more than 5 steps.
        Deterministic snapshot otherwise (simple COMPUTE_SVT → FINISH turns).
        """
        if len(state.react_steps) > 5:
            return True
        for step in state.react_steps:
            if step.action == ActionType.PLAN.value:
                return True
        return False

    def snapshot(self, state) -> None:
        """Append a compact deterministic snapshot of the current turn
        to state.episode_memory. No LLM call — instant."""
        compact = self._build_compact_snapshot(state)
        previous = state.episode_memory
        if previous:
            state.episode_memory = "{}\n\n---\n\n{}".format(
                previous, compact
            )
        else:
            state.episode_memory = compact

        max_chars = settings.episode_summary_max_chars
        if len(state.episode_memory) > max_chars:
            state.episode_memory = state.episode_memory[:max_chars]

        logger.info(
            "Episode snapshot stored ({} chars)",
            len(state.episode_memory),
        )

    def summarize(self, state) -> None:
        """Run the episode summarization LLM call and store the result
        in state.episode_memory.  Designed to run in a background thread."""
        max_chars = settings.episode_summary_max_chars
        transcript = self._build_episode_transcript(state)

        prompt = EPISODE_SUMMARY_PROMPT.format(max_chars=max_chars)

        previous = state.episode_memory
        user_content = "Summarize this completed turn.\n\n{}".format(
            transcript
        )
        if previous:
            user_content = (
                "## Previous episode memory\n{}\n\n"
                "---\n\n"
                "## New turn to incorporate\n{}".format(
                    previous, transcript
                )
            )

        messages = [{"role": "user", "content": user_content}]

        try:
            t0 = time.time()
            summary = self.llm.generate(messages, system=prompt)
            state.episode_memory = summary[:max_chars]
            elapsed = time.time() - t0
            self.summary_elapsed = elapsed
        except Exception as e:
            logger.error("Episode summarization failed: {}", e)
            # On failure, keep whatever was there before — don't wipe memory

    def _build_episode_transcript(self, state) -> str:
        """Serialize the current turn's plan + react steps into a text
        transcript suitable for the episode summarizer LLM call."""
        lines = []

        lines.append("## User query")
        lines.append(state.get_latest_user_message())

        if state.current_plan:
            lines.append("\n## Plan")
            lines.append(state.current_plan)

        if state.react_steps:
            lines.append("\n## ReAct steps")
            for i, step in enumerate(state.react_steps, 1):
                lines.append("### Step {}".format(i))
                lines.append("Thought: {}".format(step.thought))
                lines.append("Action: {}".format(step.action))
                lines.append("Action Input: {}".format(
                    json.dumps(step.action_input, ensure_ascii=False)
                ))
                if step.observation is not None:
                    lines.append("Observation: {}".format(step.observation))

        return "\n".join(lines)

    def _build_compact_snapshot(self, state) -> str:
        """Build a compact snapshot of observations and answer only —
        no thoughts or action inputs."""
        lines = [
            "## Previous turn (snapshot)",
            "User query: {}".format(state.get_latest_user_message()),
        ]
        for step in state.react_steps:
            if step.action == ActionType.SEARCH_WIKIPEDIA.value:
                lines.append("- {}".format(step.observation))
            elif step.action == ActionType.COMPUTE_SVT.value:
                lines.append("- {}".format(step.observation))
            elif step.action == ActionType.FINISH.value:
                answer = (step.action_input or {}).get("answer", "")
                lines.append("- Answer: {}".format(answer))
        return "\n".join(lines)
