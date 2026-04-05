from tools.wikipedia_tool import WikipediaTool


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_wikipedia_tool_returns_content(monkeypatch):
    tool = WikipediaTool()

    responses = [
        DummyResponse(
            {
                "query": {
                    "search": [
                        {"title": "Большой Каменный мост"},
                    ]
                }
            }
        ),
        DummyResponse(
            {
                "query": {
                    "pages": {
                        "123": {
                            "extract": (
                                "Большой Каменный мост — автодорожный "
                                "мост в Москве. Длина 168 м."
                            )
                        }
                    }
                }
            }
        ),
    ]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(tool.session, "get", fake_get)

    result = tool.run({"query": "Большой Каменный мост"})

    assert result["ok"] is True
    assert result["title"] == "Большой Каменный мост"
    assert "мост" in result["content"].lower()
    assert result["language"] == "ru"


def test_wikipedia_tool_empty_query():
    tool = WikipediaTool()
    result = tool.run({"query": ""})
    assert result["ok"] is False
    assert "Empty" in result["error"]


def test_wikipedia_tool_no_results(monkeypatch):
    tool = WikipediaTool()

    def fake_get(*args, **kwargs):
        return DummyResponse({"query": {"search": []}})

    monkeypatch.setattr(tool.session, "get", fake_get)

    result = tool.run({"query": "xyznonexistent"})
    assert result["ok"] is False


def test_wikipedia_tool_http_error(monkeypatch):
    tool = WikipediaTool()

    def fake_get(*args, **kwargs):
        raise ConnectionError("Network down")

    monkeypatch.setattr(tool.session, "get", fake_get)

    result = tool.run({"query": "test"})
    assert result["ok"] is False


def test_wikipedia_tool_language_from_payload(monkeypatch):
    """Language in payload controls which Wikipedia is tried first."""
    tool = WikipediaTool()
    requested_urls = []

    responses = [
        DummyResponse(
            {"query": {"search": [{"title": "Cheetah"}]}}
        ),
        DummyResponse(
            {
                "query": {
                    "pages": {
                        "1": {"extract": "The cheetah is the fastest land animal."}
                    }
                }
            }
        ),
    ]

    def fake_get(url, *args, **kwargs):
        requested_urls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(tool.session, "get", fake_get)

    result = tool.run({"query": "cheetah", "language": "en"})

    assert result["ok"] is True
    assert result["language"] == "en"
    # Both requests should have gone to English Wikipedia
    assert all("en.wikipedia" in u for u in requested_urls)


def test_wikipedia_tool_fallback_language(monkeypatch):
    """If preferred language finds nothing, falls back to other language."""
    tool = WikipediaTool()

    call_count = 0

    def fake_get(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # en search — no results
            return DummyResponse({"query": {"search": []}})
        if call_count == 2:
            # ru search — found
            return DummyResponse(
                {"query": {"search": [{"title": "Тест"}]}}
            )
        # call 3: ru content
        return DummyResponse(
            {"query": {"pages": {"1": {"extract": "Тестовая статья."}}}}
        )

    monkeypatch.setattr(tool.session, "get", fake_get)

    result = tool.run({"query": "тест", "language": "en"})

    assert result["ok"] is True
    assert result["language"] == "ru"
