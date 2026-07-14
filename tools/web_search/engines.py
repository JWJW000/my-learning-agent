"""搜索引擎适配层 (engines)。

提供统一的 SearchEngine 抽象基类，并实现四种搜索引擎后端：
1. DuckDuckGoEngine (免 API 密钥，基于 HTML 爬取)
2. BraveEngine (基于 Brave Search API, 需要 BRAVE_API_KEY)
3. BingEngine (基于 Azure Bing Search API, 需要 BING_API_KEY)
4. TavilyEngine (基于 Tavily Search API, 需要 TAVILY_API_KEY)
"""

from __future__ import annotations

import base64
import html
import html.parser
import json
import logging
import os
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


class SearchEngine(ABC):
    """搜索引擎抽象基类。"""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        """执行检索，返回包含 'title', 'url', 'snippet' 的结构化结果列表。"""
        pass


class DDGHTMLParser(html.parser.HTMLParser):
    """用于解析 html.duckduckgo.com 搜索结果页面的 HTML 解析器。"""

    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.current_result: dict[str, str] | None = None
        self.in_title: bool = False
        self.in_snippet: bool = False
        self.temp_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get("class", "") or ""

        if tag == "div" and "web-result" in class_attr:
            if self.current_result:
                self.results.append(self.current_result)
            self.current_result = {"title": "", "url": "", "snippet": ""}

        elif self.current_result:
            if tag == "a" and "result__a" in class_attr:
                self.in_title = True
                self.current_result["url"] = attrs_dict.get("href", "") or ""
                self.temp_text = []
            elif tag == "a" and "result__snippet" in class_attr:
                self.in_snippet = True
                self.temp_text = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_title and tag == "a":
            self.in_title = False
            if self.current_result:
                raw_text = "".join(self.temp_text)
                self.current_result["title"] = html.unescape(raw_text).strip()
            self.temp_text = []
        elif self.in_snippet and tag == "a":
            self.in_snippet = False
            if self.current_result:
                raw_text = "".join(self.temp_text)
                self.current_result["snippet"] = html.unescape(raw_text).strip()
            self.temp_text = []

    def handle_data(self, data: str) -> None:
        if self.in_title or self.in_snippet:
            self.temp_text.append(data)

    def close(self) -> None:
        super().close()
        if self.current_result and self.current_result not in self.results:
            self.results.append(self.current_result)


class BingHTMLParser(html.parser.HTMLParser):
    """解析 Bing 公开 HTML 搜索结果，不依赖 API Key。"""

    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.current_result: dict[str, str] | None = None
        self.in_heading = False
        self.in_title = False
        self.in_caption = False
        self.in_snippet = False
        self.temp_text: list[str] = []

    def _finish_result(self) -> None:
        if self.current_result:
            self.results.append(self.current_result)
        self.current_result = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class", "") or "").split())

        if tag == "li" and "b_algo" in classes:
            self._finish_result()
            self.current_result = {"title": "", "url": "", "snippet": ""}
        elif not self.current_result:
            return
        elif tag == "h2":
            self.in_heading = True
        elif tag == "a" and self.in_heading:
            self.in_title = True
            self.current_result["url"] = attrs_dict.get("href", "") or ""
            self.temp_text = []
        elif tag == "div" and "b_caption" in classes:
            self.in_caption = True
        elif tag == "p" and self.in_caption:
            self.in_snippet = True
            self.temp_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.in_title:
            self.in_title = False
            if self.current_result:
                self.current_result["title"] = html.unescape(
                    "".join(self.temp_text)
                ).strip()
            self.temp_text = []
        elif tag == "h2":
            self.in_heading = False
        elif tag == "p" and self.in_snippet:
            self.in_snippet = False
            if self.current_result:
                self.current_result["snippet"] = html.unescape(
                    "".join(self.temp_text)
                ).strip()
            self.temp_text = []
        elif tag == "li" and self.current_result:
            self._finish_result()
            self.in_caption = False

    def handle_data(self, data: str) -> None:
        if self.in_title or self.in_snippet:
            self.temp_text.append(data)

    def close(self) -> None:
        super().close()
        self._finish_result()


def _unwrap_bing_url(url: str) -> str:
    """将 Bing ``/ck/a`` 跳转链接还原成实际结果地址。"""
    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc.endswith("bing.com") or parsed.path != "/ck/a":
        return url

    encoded_values = urllib.parse.parse_qs(parsed.query).get("u", [])
    if not encoded_values or not encoded_values[0].startswith("a1"):
        return url

    encoded = encoded_values[0][2:]
    encoded += "=" * (-len(encoded) % 4)
    try:
        decoded = base64.urlsafe_b64decode(encoded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return url
    return decoded if decoded.startswith(("http://", "https://")) else url


class BingHTMLEngine(SearchEngine):
    """基于 Bing 搜索结果页的免密钥降级后端。"""

    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        params = urllib.parse.urlencode({"q": query, "count": limit})
        url = f"https://www.bing.com/search?{params}"
        headers = {
            # Bing 会对部分完整浏览器 UA 返回依赖 JavaScript 的页面。简短 UA
            # 会稳定返回服务端渲染的 b_algo 结果列表，便于无头客户端解析。
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            response = httpx.get(
                url,
                headers=headers,
                timeout=10.0,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Bing HTML search request failed: %s", exc)
            return []

        parser = BingHTMLParser()
        parser.feed(response.text)
        parser.close()

        cleaned_results = []
        for result in parser.results:
            if not result["title"] or not result["url"]:
                continue
            result["title"] = " ".join(result["title"].split())
            result["snippet"] = " ".join(result["snippet"].split())
            result["url"] = _unwrap_bing_url(result["url"])
            cleaned_results.append(result)
        return cleaned_results[:limit]


class DuckDuckGoEngine(SearchEngine):
    """基于 DuckDuckGo HTML 页面的免 API 密钥搜索引擎。"""

    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        url = "https://html.duckduckgo.com/html/"
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) "
                "Gecko/20100101 Firefox/115.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://html.duckduckgo.com",
            "Referer": "https://html.duckduckgo.com/",
        }

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                html_content = response.read().decode("utf-8")
        except Exception as e:
            logger.warning(
                "DuckDuckGo HTML search request failed: %s. Falling back to Bing HTML.",
                e,
            )
            return BingHTMLEngine().search(query, limit)

        parser = DDGHTMLParser()
        parser.feed(html_content)
        parser.close()

        cleaned_results = []
        for r in parser.results:
            if not r["title"] or not r["url"]:
                continue

            parsed_url = urllib.parse.urlparse(r["url"])
            if "duckduckgo.com" in parsed_url.netloc and "uddg" in parsed_url.query:
                query_params = urllib.parse.parse_qs(parsed_url.query)
                if "uddg" in query_params:
                    r["url"] = query_params["uddg"][0]

            cleaned_results.append(r)

        if cleaned_results:
            return cleaned_results[:limit]

        logger.warning(
            "DuckDuckGo returned no parseable results. Falling back to Bing HTML."
        )
        return BingHTMLEngine().search(query, limit)


class BraveEngine(SearchEngine):
    """基于 Brave Search API 的搜索引擎。"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")

    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        if not self.api_key:
            logger.warning("Brave Search API Key missing. Falling back to Bing HTML.")
            return BingHTMLEngine().search(query, limit)

        params = urllib.parse.urlencode({"q": query, "count": limit})
        url = f"https://api.search.brave.com/res/v1/web/search?{params}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
            "User-Agent": "my-learning-agent/1.0"
        })

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = []
            web_results = data.get("web", {}).get("results", [])
            for r in web_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("description", "")
                })
            return results[:limit]
        except Exception as e:
            logger.error("Brave search failed: %s. Falling back to Bing HTML.", e)
            return BingHTMLEngine().search(query, limit)


class BingEngine(SearchEngine):
    """基于 Microsoft Bing Web Search API 的搜索引擎。"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("BING_API_KEY")

    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        if not self.api_key:
            logger.warning("Bing Search API Key missing. Falling back to Bing HTML.")
            return BingHTMLEngine().search(query, limit)

        params = urllib.parse.urlencode({"q": query, "count": limit})
        url = f"https://api.bingwebsearch.microsoft.com/v7.0/search?{params}"
        req = urllib.request.Request(url, headers={
            "Ocp-Apim-Subscription-Key": self.api_key,
            "User-Agent": "my-learning-agent/1.0"
        })

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = []
            web_pages = data.get("webPages", {}).get("value", [])
            for r in web_pages:
                results.append({
                    "title": r.get("name", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", "")
                })
            return results[:limit]
        except Exception as e:
            logger.error("Bing search failed: %s. Falling back to Bing HTML.", e)
            return BingHTMLEngine().search(query, limit)


class TavilyEngine(SearchEngine):
    """基于 Tavily API 的搜索引擎 (专为大模型设计)。"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")

    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        if not self.api_key:
            logger.warning("Tavily Search API Key missing. Falling back to Bing HTML.")
            return BingHTMLEngine().search(query, limit)

        url = "https://api.tavily.com/search"
        payload = json.dumps({
            "api_key": self.api_key,
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "my-learning-agent/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = []
            tavily_results = data.get("results", [])
            for r in tavily_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")
                })
            return results[:limit]
        except Exception as e:
            logger.error("Tavily search failed: %s. Falling back to Bing HTML.", e)
            return BingHTMLEngine().search(query, limit)


def get_default_engine() -> SearchEngine:
    """根据环境变量中可用的 API 密钥，获取默认的搜索引擎实现（自动选择并回退）。"""
    if os.environ.get("TAVILY_API_KEY"):
        return TavilyEngine()
    elif os.environ.get("BRAVE_API_KEY"):
        return BraveEngine()
    elif os.environ.get("BING_API_KEY"):
        return BingEngine()
    else:
        return BingHTMLEngine()
