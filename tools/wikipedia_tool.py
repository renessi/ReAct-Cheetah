from typing import Any, Dict, List, Optional

import requests
from urllib.parse import quote

from config.settings import settings
from tools.base_tool import Tool
from utils.logging import get_logger

logger = get_logger()


class WikipediaTool(Tool):
    name = "wikipedia"
    description = (
        "Searches Wikipedia and returns article content "
        "(intro + first body sections)."
    )

    def __init__(self):
        self.session = requests.Session()
        self.ru_base_url = settings.wikipedia_ru_base_url
        self.en_base_url = settings.wikipedia_en_base_url
        self.timeout = settings.wikipedia_timeout_seconds
        self.headers = {"User-Agent": "ReActAgent/1.0 (research project)"}
        self.max_content_chars = settings.wikipedia_max_content_chars

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "").strip()
        if not query:
            return {
                "ok": False,
                "error": "Empty query.",
                "tool": self.name,
            }

        preferred = payload.get("language", "ru")
        fallback = "en" if preferred == "ru" else "ru"

        result = self._search_and_extract(query=query, language=preferred)
        if result["ok"]:
            return result

        return self._search_and_extract(query=query, language=fallback)

    def _search_and_extract(
        self, query: str, language: str
    ) -> Dict[str, Any]:
        base_url = (
            self.ru_base_url if language == "ru" else self.en_base_url
        )

        search_title = self._search_title(base_url=base_url, query=query)
        if search_title is None:
            return {
                "ok": False,
                "tool": self.name,
                "language": language,
                "query": query,
                "error": "No matching Wikipedia page found.",
            }

        content = self._get_content(base_url=base_url, title=search_title)
      
        if content is None:
            return {
                "ok": False,
                "tool": self.name,
                "language": language,
                "query": query,
                "title": search_title,
                "error": "Wikipedia page found but content is unavailable.",
            }

        return {
            "ok": True,
            "tool": self.name,
            "language": language,
            "query": query,
            "title": search_title,
            "content": content,
            "url": self._build_article_url(
                title=search_title, language=language
            ),
        }

    def _search_title(self, base_url: str, query: str) -> Optional[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
        }

        try:
            response = self.session.get(
                base_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error("Wikipedia search request failed: {}", e)
            return None

        data = response.json()
        results: List[Dict[str, Any]] = (
            data.get("query", {}).get("search", [])
        )
        if not results:
            return None

        return results[0].get("title")

    def _get_content(self, base_url: str, title: str) -> Optional[str]:
        #print("exchars =", self.max_content_chars)
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",  # убирает ==заголовки==
            "exlimit": 1,                # одна статья
            "titles": title,
            "format": "json",
            "utf8": 1,
        }

        try:
            response = self.session.get(
                base_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
           
            response.raise_for_status()
        except Exception as e:
            logger.error("Wikipedia content request failed: {}", e)
            return None

        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        first_page = next(iter(pages.values()))
        extract = first_page.get("extract", "").strip()
        #print(extract)

        if not extract:
            return None
        #print("max_content_chars =", self.max_content_chars)
        #print("len(extract) before slice =", len(extract))
        #print("extract[-1000:] =", extract[-1000:])

        return extract[: self.max_content_chars]

    @staticmethod
    def _build_article_url(title: str, language: str) -> str:
        normalized_title = quote(title.replace(" ", "_"))
        return "https://{lang}.wikipedia.org/wiki/{title}".format(
            lang=language,
            title=normalized_title,
        )
