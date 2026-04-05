import os

from dotenv import load_dotenv

load_dotenv()

# Providers that use the OpenAI-compatible API (same SDK, different endpoint).
# Map: provider name -> (api_key env var, base_url env var, base_url default, model default)
OPENAI_COMPAT_PROVIDERS = {
    "deepseek": (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
        "deepseek-chat",
    ),
    "openai": (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
        "gpt-5.4-nano",
    ),
}

ANTHROPIC_PROVIDER = {
    "anthropic": (
        "ANTHROPIC_API_KEY",
        "claude-haiku-4-5-20251001",
    ),
}

ALL_PROVIDERS = set(OPENAI_COMPAT_PROVIDERS) | set(ANTHROPIC_PROVIDER)


class Settings:
    def __init__(self):
        # --- LLM provider selection ---
        self.llm_provider = os.getenv("LLM_PROVIDER", "deepseek")

        if self.llm_provider not in ALL_PROVIDERS:
            raise ValueError(
                "Unknown LLM_PROVIDER='{}'. Supported: {}".format(
                    self.llm_provider, ", ".join(sorted(ALL_PROVIDERS))
                )
            )

        # --- Provider credentials ---
        if self.llm_provider in OPENAI_COMPAT_PROVIDERS:
            key_var, url_var, url_default, model_default = (
                OPENAI_COMPAT_PROVIDERS[self.llm_provider]
            )
            self.llm_api_key = os.getenv(key_var, "")
            self.llm_base_url = os.getenv(url_var, url_default)
        else:
            key_var, model_default = ANTHROPIC_PROVIDER[self.llm_provider]
            self.llm_api_key = os.getenv(key_var, "")
            self.llm_base_url = None  # Anthropic SDK doesn't use base_url

        if not self.llm_api_key:
            raise ValueError(
                "{} is required when LLM_PROVIDER={}".format(
                    key_var, self.llm_provider
                )
            )

        # --- Model name ---
        self.model_name = os.getenv("MODEL_NAME", "") or model_default

        # --- Agent ---
        self.verbose = os.getenv("VERBOSE", "true").lower() == "true"
        self.max_steps = int(os.getenv("MAX_AGENT_STEPS", 15))
        self.episode_summary_max_chars = int(
            os.getenv("EPISODE_SUMMARY_MAX_CHARS", 20000)
        )
        self.temperature = float(os.getenv("TEMPERATURE", 0.3)) # for all providers

        # --- Wikipedia ---
        self.wikipedia_ru_base_url = os.getenv(
            "WIKIPEDIA_RU_BASE_URL",
            "https://ru.wikipedia.org/w/api.php",
        )
        self.wikipedia_en_base_url = os.getenv(
            "WIKIPEDIA_EN_BASE_URL",
            "https://en.wikipedia.org/w/api.php",
        )
        self.wikipedia_timeout_seconds = int(
            os.getenv("WIKIPEDIA_TIMEOUT_SECONDS", 5)
        )
        self.wikipedia_max_content_chars = int(
            os.getenv("WIKIPEDIA_MAX_CONTENT_CHARS", 8000)
        )
        self.wikipedia_summary_max_chars = int(
            os.getenv("WIKIPEDIA_SUMMARY_MAX_CHARS", 2000)
        )


settings = Settings()
