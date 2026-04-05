from anthropic import Anthropic
from llm.base_client import LLMClient
from utils.logging import get_logger
from config.settings import settings

logger = get_logger()


def _openai_to_anthropic(tools):
    """Convert OpenAI-format tool definitions to Anthropic format.

    OpenAI:    {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
    Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
    """
    result = []
    for tool in tools:
        func = tool["function"]
        result.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object"}),
        })
    return result


class AnthropicClient(LLMClient):
    """LLM client for the Anthropic API (Claude models)."""

    def __init__(self, api_key, model, temperature=settings.temperature, max_tokens=4096):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages, system=None, stop=None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        try:
            response = self.client.messages.create(**kwargs)
            return self._extract_text(response)
        except Exception as e:
            logger.error("Anthropic API call failed: {}", e)
            raise

    def generate_with_tools(self, messages, tools, system=None):
        anthropic_tools = _openai_to_anthropic(tools)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": anthropic_tools,
            "tool_choice": {"type": "any"},
        }
        if system:
            kwargs["system"] = system

        try:
            response = self.client.messages.create(**kwargs)

            for block in response.content:
                if block.type == "tool_use":
                    return {
                        "tool_call": True,
                        "name": block.name,
                        "arguments": block.input,  # already a dict
                    }

            # No tool_use block — return text content
            return {
                "tool_call": False,
                "content": self._extract_text(response),
            }
        except Exception as e:
            logger.error("Anthropic API call (tools) failed: {}", e)
            raise

    @staticmethod
    def _extract_text(response):
        """Extract concatenated text from response content blocks."""
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)
