"""Tests for advanced layered WebSearch module."""

import json
import urllib.error
import pytest
from unittest.mock import MagicMock, patch

from tools.web_search import WebSearch
from tools.web_search.engines import (
    BingEngine,
    BingHTMLEngine,
    BraveEngine,
    DuckDuckGoEngine,
    TavilyEngine,
    get_default_engine,
)
from tools.web_search.cleaner import HTMLCleaner
from tools.web_search.rankers import BM25Ranker, EmbeddingRanker, LLMReranker


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    return client


@pytest.fixture
def web_search(mock_openai_client):
    return WebSearch(openai_client=mock_openai_client)


class TestSearchEngines:
    @patch.dict(
        "os.environ",
        {
            "TAVILY_API_KEY": "",
            "BRAVE_API_KEY": "",
            "BING_API_KEY": "",
        },
    )
    def test_default_engine_without_api_key_uses_bing_html(self):
        assert isinstance(get_default_engine(), BingHTMLEngine)

    @patch("urllib.request.urlopen")
    def test_duckduckgo_engine(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'<html><body>'
            b'<div class="web-result">'
            b'  <a class="result__a" href="https://example.com/ddg">DDG Site</a>'
            b'  <a class="result__snippet" href="https://example.com/ddg">Snippet test</a>'
            b'</div>'
            b'</body></html>'
        )
        mock_urlopen.return_value.__enter__.return_value = mock_response

        engine = DuckDuckGoEngine()
        results = engine.search("python")

        assert len(results) == 1
        assert results[0]["title"] == "DDG Site"
        assert results[0]["url"] == "https://example.com/ddg"
        assert results[0]["snippet"] == "Snippet test"

    @patch("tools.web_search.engines.BingHTMLEngine.search")
    @patch("urllib.request.urlopen")
    def test_duckduckgo_falls_back_to_bing_html(self, mock_urlopen, mock_bing_search):
        mock_urlopen.side_effect = urllib.error.URLError("TLS connection closed")
        mock_bing_search.return_value = [
            {
                "title": "Fallback result",
                "url": "https://example.com/fallback",
                "snippet": "Fallback snippet",
            }
        ]

        results = DuckDuckGoEngine().search("python", limit=2)

        assert results == mock_bing_search.return_value
        mock_bing_search.assert_called_once_with("python", 2)

    @patch("tools.web_search.engines.httpx.get")
    def test_bing_html_engine_parses_and_unwraps_results(self, mock_get):
        # Base64-url encoding of https://example.com/python with Bing's a1 prefix.
        redirect_url = (
            "https://www.bing.com/ck/a?"
            "u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9weXRob24&ntb=1"
        )
        mock_response = MagicMock()
        mock_response.text = (
            '<ol id="b_results">'
            '<li class="b_algo">'
            f'<h2><a href="{redirect_url}">Python Result</a></h2>'
            '<div class="b_caption"><p>Useful Python snippet.</p></div>'
            '</li>'
            '</ol>'
        )
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        results = BingHTMLEngine().search("python", limit=1)

        assert results == [
            {
                "title": "Python Result",
                "url": "https://example.com/python",
                "snippet": "Useful Python snippet.",
            }
        ]

    @patch("urllib.request.urlopen")
    def test_brave_engine(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "web": {
                "results": [
                    {"title": "Brave Site", "url": "https://example.com/brave", "description": "Brave desc"}
                ]
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        engine = BraveEngine(api_key="test_key")
        results = engine.search("python")

        assert len(results) == 1
        assert results[0]["title"] == "Brave Site"
        assert results[0]["url"] == "https://example.com/brave"
        assert results[0]["snippet"] == "Brave desc"

    @patch("urllib.request.urlopen")
    def test_bing_engine(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "webPages": {
                "value": [
                    {"name": "Bing Site", "url": "https://example.com/bing", "snippet": "Bing snippet"}
                ]
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        engine = BingEngine(api_key="test_key")
        results = engine.search("python")

        assert len(results) == 1
        assert results[0]["title"] == "Bing Site"
        assert results[0]["url"] == "https://example.com/bing"
        assert results[0]["snippet"] == "Bing snippet"

    @patch("urllib.request.urlopen")
    def test_tavily_engine(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {"title": "Tavily Site", "url": "https://example.com/tavily", "content": "Tavily content"}
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        engine = TavilyEngine(api_key="test_key")
        results = engine.search("python")

        assert len(results) == 1
        assert results[0]["title"] == "Tavily Site"
        assert results[0]["url"] == "https://example.com/tavily"
        assert results[0]["snippet"] == "Tavily content"


class TestHTMLCleaner:
    def test_cleaner_removes_noise(self):
        cleaner = HTMLCleaner()
        html_input = (
            "<html>"
            "<head><title>Test</title></head>"
            "<body>"
            "<nav>Menu item</nav>"
            "<h1>Header Title</h1>"
            "<script>alert(1);</script>"
            "<p>Main body content about Python programming.</p>"
            "<footer>Copyright</footer>"
            "</body>"
            "</html>"
        )
        cleaned = cleaner.clean(html_input)
        assert "Menu item" not in cleaned
        assert "alert" not in cleaned
        assert "Copyright" not in cleaned
        assert "Header Title" in cleaned
        assert "Main body content" in cleaned

    def test_cleaner_handles_void_elements_before_body(self):
        cleaner = HTMLCleaner()
        html_input = (
            "<html><head>"
            '<meta charset="utf-8">'
            '<link rel="stylesheet" href="site.css">'
            "</head><body>"
            "<h1>Visible heading</h1>"
            "<p>Visible body<br>with a line break.</p>"
            "</body></html>"
        )

        cleaned = cleaner.clean(html_input)

        assert "Visible heading" in cleaned
        assert "Visible body" in cleaned
        assert "with a line break" in cleaned

    def test_chunking_with_overlap(self):
        cleaner = HTMLCleaner()
        text = "This is a sentence. And here is another one. And a third one."
        # Choose small chunk_size to force multiple chunks
        chunks = cleaner.chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 1


class TestRankers:
    def test_bm25_ranker(self):
        corpus = [
            "Python is a great dynamic programming language.",
            "JavaScript is commonly used for frontend web development.",
            "Rust focuses on safety and extreme performance."
        ]
        ranker = BM25Ranker(corpus)
        results = ranker.rank("python language", top_n=2)
        assert len(results) == 2
        # Index 0 should be ranked higher for python query
        assert results[0][0] == 0
        assert results[0][1] > results[1][1]

    def test_embedding_ranker(self, mock_openai_client):
        # Mock embeddings responses
        mock_emb_res = MagicMock()
        mock_emb_res.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai_client.embeddings.create.return_value = mock_emb_res

        corpus = ["Doc 1", "Doc 2"]
        ranker = EmbeddingRanker(mock_openai_client)
        results = ranker.rank("query", corpus)
        assert len(results) == 2
        assert mock_openai_client.embeddings.create.call_count == 3  # 1 query + 2 corpus docs


class TestWebSearchIntegration:
    @patch("tools.web_search.engines.DuckDuckGoEngine.search")
    @patch("tools.web_search.downloader.Downloader.download_pages")
    def test_web_search_end_to_end_bm25(self, mock_download, mock_search, web_search):
        # 1. Mock Search Engine return list
        mock_search.return_value = [
            {"title": "Result Title", "url": "https://example.com/page1", "snippet": "Snippet content"}
        ]
        # 2. Mock Downloader HTML content
        mock_download.return_value = {
            "https://example.com/page1": "<html><body><p>Deep Python features in Python 3.14.</p></body></html>"
        }

        # 3. Call search method directly
        json_res = web_search.search("Python 3.14", ranking_method="bm25")
        res = json.loads(json_res)

        assert len(res) == 1
        assert res[0]["source"] == "https://example.com/page1"
        assert "Python 3.14" in res[0]["content"]

    def test_web_search_invalid_queries(self, web_search):
        res_empty = web_search.search("")
        assert "Empty search query" in res_empty

