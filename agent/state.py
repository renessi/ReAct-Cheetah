from typing import Any, Dict, List

from agent.step import ReActStep
from memory.fact_memory import FactMemory


class AgentState:
    def __init__(self):
        self.dialogue_history: List[Dict[str, str]] = []
        self.react_steps: List[ReActStep] = []
        self.fact_cache = FactMemory()
        self.sources: List[Dict[str, str]] = []
        self.current_plan: str = ""
        self.episode_memory: str = ""

    def add_user_message(self, message: str):
        self.dialogue_history.append({"role": "user", "content": message})

    def add_agent_message(self, message: str):
        self.dialogue_history.append({"role": "assistant", "content": message})

    def get_latest_user_message(self) -> str:
        for message in reversed(self.dialogue_history):
            if message["role"] == "user":
                return message["content"]
        return ""

    def add_source(self, title: str, url: str, language: str):
        self.sources.append({"title": title, "url": url, "language": language})

    def reset_turn(self):
        self.react_steps = []
        self.current_plan = ""
