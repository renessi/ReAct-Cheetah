import json

from openai import OpenAI
from llm.base_client import LLMClient
from utils.logging import get_logger
from config.settings import settings

logger = get_logger()


class OpenAICompatibleClient(LLMClient):
    """LLM client for any OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""

    def __init__(self, api_key, base_url, model, temperature=settings.temperature):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def generate(self, messages, system=None, stop=None):
        full_messages = self._prepend_system(messages, system)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                stop=stop,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM API call failed: {}", e)
            raise

    def generate_with_tools(self, messages, tools, system=None):
        full_messages = self._prepend_system(messages, system)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                tools=tools,
                tool_choice="required",
            )
            message = response.choices[0].message

            if message.tool_calls:
                tc = message.tool_calls[0]
                return {
                    "tool_call": True,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }

            return {
                "tool_call": False,
                "content": message.content or "",
            }
        except Exception as e:
            logger.error("LLM API call (tools) failed: {}", e)
            raise

    @staticmethod
    def _prepend_system(messages, system):
        """Prepend system message to the list if provided."""
        if system:
            return [{"role": "system", "content": system}] + messages
        return messages
