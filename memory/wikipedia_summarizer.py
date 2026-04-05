import time

from config.settings import settings
from llm.base_client import LLMClient
from utils.logging import get_logger

logger = get_logger()

NO_RELEVANT_FACTS = "NO_RELEVANT_FACTS"

SUMMARIZE_SYSTEM_PROMPT = """\
You are a fact extractor for a Speed-Velocity-Time reasoning agent.
Given a Wikipedia article, extract all facts that could be relevant to SVT \
calculations for the user's query.
Include:
- All speed values with context (maximum, average, sustained, hunting speed etc.)
- Acceleration and deceleration data
- Distance and length measurements with context
- Time-related facts
- Any conditions or variations affecting the above (terrain, age, species variant)
- Other facts closely related to the query
Exclude: historical background, taxonomy, cultural references, \
unrelated biographical data.
Be precise with numbers and units. Preserve ranges and variations — do not \
pick one value when multiple exist.
If the article contains NO facts relevant to speed, velocity, time, or \
distance calculations, respond with exactly: NO_RELEVANT_FACTS
Maximum {max_chars} characters."""


class WikipediaSummarizer:

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def summarize(self, raw_content: str, search_query: str) -> str:
        """Summarize raw Wikipedia content into SVT-relevant facts.

        Returns the summary string, or NO_RELEVANT_FACTS sentinel
        if nothing useful was found.
        """
        max_chars = settings.wikipedia_summary_max_chars
        prompt = SUMMARIZE_SYSTEM_PROMPT.format(max_chars=max_chars)

        messages = [
            {
                "role": "user",
                "content": "Extract SVT-relevant facts about '{}' from this "
                "Wikipedia article.\n\n{}".format(
                    search_query, raw_content
                ),
            },
        ]

        try:
            t0 = time.time()
            summary = self.llm.generate(messages, system=prompt)
            elapsed = time.time() - t0
            logger.info(
                "Wikipedia summarization took {:.1f}s for query '{}'",
                elapsed, search_query,
            )
            return summary[:max_chars]
        except Exception as e:
            logger.error("Wikipedia summarization failed: {}", e)
            return raw_content[:max_chars]
