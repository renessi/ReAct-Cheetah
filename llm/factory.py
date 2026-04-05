from config.settings import settings, OPENAI_COMPAT_PROVIDERS
from llm.base_client import LLMClient


def create_llm_client() -> LLMClient:
    """Create an LLM client based on the LLM_PROVIDER setting."""
    provider = settings.llm_provider

    if provider in OPENAI_COMPAT_PROVIDERS:
        from llm.openai_compatible_client import OpenAICompatibleClient
        return OpenAICompatibleClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.model_name,
        )

    if provider == "anthropic":
        from llm.anthropic_client import AnthropicClient
        return AnthropicClient(
            api_key=settings.llm_api_key,
            model=settings.model_name,
        )

    raise ValueError(
        "Unknown LLM_PROVIDER '{}'".format(provider)
    )
