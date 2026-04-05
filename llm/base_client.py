from abc import ABC, abstractmethod


class LLMClient(ABC):

    @abstractmethod
    def generate(self, messages, system=None, stop=None):
        """Call the LLM for plain text generation.

        Args:
            messages: Chat messages (user/assistant only).
            system: Optional system prompt string. Kept separate
                    because some vendors (Anthropic) require it
                    outside the messages list.
            stop: Optional stop sequence(s).

        Returns:
            Generated text as a string.
        """
        pass

    @abstractmethod
    def generate_with_tools(self, messages, tools, system=None):
        """Call the LLM with function calling.

        Args:
            messages: Chat messages (user/assistant only).
            tools: List of tool definitions in OpenAI format.
            system: Optional system prompt string.

        Returns:
            A dict with keys:
                - "tool_call": True if the model invoked a function, False otherwise.
                - "name": Function name (str) when tool_call is True.
                - "arguments": Parsed arguments (dict) when tool_call is True.
                - "content": Text content (str) when tool_call is False.
        """
        pass
