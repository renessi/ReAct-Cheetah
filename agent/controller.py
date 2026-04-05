import threading
import time
from typing import Dict, Optional

from config.settings import settings
from agent.actions import ActionType
from agent.state import AgentState
from agent.step import ReActStep
from llm.base_client import LLMClient
from memory.episode_memory import EpisodeMemory
from memory.wikipedia_summarizer import WikipediaSummarizer, NO_RELEVANT_FACTS
from planner.planner import Planner
from tools.base_tool import Tool
from utils.logging import get_logger

logger = get_logger()


class AgentController:

    def __init__(self, llm: LLMClient, tools: Dict[str, Tool]):
        self.llm = llm
        self.state = AgentState()
        self.planner = Planner(llm)
        self.tools = tools
        self.max_steps = settings.max_steps
        self._summary_thread: Optional[threading.Thread] = None
        self._episode = EpisodeMemory(llm)
        self._wiki_summarizer = WikipediaSummarizer(llm)

    def run(self, user_message: str) -> str:
        self._await_summary_thread()
        self.state.add_user_message(user_message)
        self.state.reset_turn()

        t0 = time.time()

        for step_num in range(1, self.max_steps + 1):
            logger.info("Step {}", step_num)

            try:
                thought, action, action_input = self.planner.decide(
                    self.state
                )
            except Exception as e:
                logger.error("Planner failed: {}", e)
                fallback = "An internal error occurred. Please try again."
                self.state.add_agent_message(fallback)
                return fallback

            logger.info("Thought: {}", thought)
            logger.info("Action: {}", action)
            logger.debug("Action Input: {}", action_input)

            step = ReActStep(
                thought=thought,
                action=action,
                action_input=action_input,
            )

            if action == ActionType.FINISH.value:
                answer = action_input.get("answer", "")
                self.state.add_agent_message(answer)
                step.observation = "Done."
                self.state.react_steps.append(step)
                elapsed = time.time() - t0
                logger.log("RESULT", "Total time: {:.1f}s ({} steps)", elapsed, step_num)
                self._launch_summary_thread()
                return answer

            if action == ActionType.ASK_USER.value:
                question = action_input.get("question", "")
                self.state.add_agent_message(question)
                step.observation = "Question sent to user."
                self.state.react_steps.append(step)
                elapsed = time.time() - t0
                logger.log("RESULT", "Total time: {:.1f}s ({} steps)", elapsed, step_num)
                return question

            '''if action == ActionType.PLAN.value:
                step.observation = None
                self.state.react_steps.append(step)
                continue'''

            if action in (ActionType.PLAN.value):
                plan_text = action_input.get("plan", "")
                self.state.current_plan = plan_text
                step.observation = "Plan recorded."
                self.state.react_steps.append(step)
                logger.info(
                    "{} recorded: {}", action, plan_text
                    )
                continue

            # Execute tool
            observation = self._execute_tool(action, action_input)
            step.observation = observation
            self.state.react_steps.append(step)
            logger.info("Observation: {}", observation)

        fallback = (
            "I stopped because the maximum number of reasoning steps "
            "was reached. "
            "(TBD: ASK_USER fallback)"
        )
        self.state.add_agent_message(fallback)
        return fallback

    def _execute_tool(self, action: str, action_input: dict) -> str:
        try:
            action_type = ActionType(action)
        except ValueError:
            return "Error: unknown action '{}'".format(action)

        if action_type == ActionType.SEARCH_WIKIPEDIA:
            return self._handle_wikipedia(action_input)

        tool = self.tools.get(action_type)
        if tool is None:
            return "Error: no tool registered for '{}'".format(action)

        try:
            result = tool.run(action_input)
        except Exception as e:
            logger.error("Tool {} failed: {}", action, e)
            return "Error: tool '{}' failed: {}".format(action, e)

        if not result.get("ok"):
            return "Tool error: {}".format(
                result.get("error", "unknown error")
            )

        if action_type == ActionType.COMPUTE_SVT:
            return (
                "Solved for {solved_for}: {human_readable} "
                "(formula: {formula}, inputs: {inputs})"
            ).format(
                solved_for=result["solved_for"],
                human_readable=result["human_readable"],
                formula=result["formula"],
                inputs=result["inputs"],
            )

        return str(result)

    def _handle_wikipedia(self, action_input: dict) -> str:
        query = action_input.get("query", "")

        # Check fact cache first
        cached = self.state.fact_cache.get(query)
        if cached is not None:
            return "Wikipedia [{}] '{}' (cached): {}".format(
                cached.get("language", "?"),
                cached.get("title", query),
                cached.get("summary", ""),
            )

        # Fetch from Wikipedia
        tool = self.tools.get(ActionType.SEARCH_WIKIPEDIA)
        if tool is None:
            return "Error: no Wikipedia tool registered"

        try:
            result = tool.run(action_input)
        except Exception as e:
            logger.error("Wikipedia tool failed: {}", e)
            return "Error: Wikipedia tool failed: {}".format(e)

        if not result.get("ok"):
            return "Tool error: {}".format(
                result.get("error", "unknown error")
            )

        # Summarize raw content via LLM
        raw_content = result.get("content", "")
        search_query = action_input.get("query", "")

        summary = self._wiki_summarizer.summarize(raw_content, search_query)

        title = result.get("title", "")
        url = result.get("url", "")
        language = result.get("language", "")

        # Skip caching if the article had no SVT-relevant facts
        if summary.strip() == NO_RELEVANT_FACTS:
            logger.info(
                "No relevant SVT facts in '{}' [{}], skipping cache",
                title, language,
            )
            return (
                "Wikipedia [{}] '{}': no speed, distance, or time facts "
                "found in this article.".format(language, title)
            )

        # Store source and cache the compact summary
        self.state.add_source(
            title=title, url=url, language=language,
        )
        self.state.fact_cache.store(query, {
            "summary": summary,
            "title": title,
            "url": url,
            "language": language,
        })

        return "Wikipedia [{}] '{}': {}".format(language, title, summary)

    # ---- Episode memory thread lifecycle ----

    def _launch_summary_thread(self) -> None:
        """Start episode summarization — LLM in background thread for
        complex turns, deterministic snapshot for simple turns."""
        if self._episode.needs_llm_summary(self.state):
            self._summary_thread = threading.Thread(
                target=self._episode.summarize,
                args=(self.state,),
                daemon=True,
            )
            self._summary_thread.start()
            logger.debug("Episode summary thread launched (LLM)")
        else:
            self._episode.snapshot(self.state)

    def _await_summary_thread(self) -> None:
        """Block until the background summary thread completes (if any)."""
        if self._summary_thread is None:
            return

        if self._summary_thread.is_alive():
            logger.info("Waiting for episode summary thread to finish...")
            self._summary_thread.join(timeout=60)
            if self._summary_thread.is_alive():
                logger.warning("Episode summary thread did not finish within 60s")

        logger.info(
            "Episode summary ready: {:.1f}s, {} chars",
            self._episode.summary_elapsed,
            len(self.state.episode_memory),
        )
        self._summary_thread = None
