"""联网检索统一入口 (WebSearch)。

串联 SearchEngine、Downloader、HTMLCleaner 和 Ranker (BM25, Embedding, Rerank) 组件，
在大模型请求联网搜索时提供高维度的检索信息增强能力。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tools.web_search.engines import get_default_engine
from tools.web_search.downloader import Downloader
from tools.web_search.cleaner import HTMLCleaner
from tools.web_search.rankers import BM25Ranker, EmbeddingRanker, LLMReranker

logger = logging.getLogger(__name__)


class WebSearch:
    """分层架构的 Web 搜索工具入口。"""

    def __init__(self, openai_client: Any | None = None):
        """
        Args:
            openai_client: 可选的 OpenAI 客户端，当使用向量相似度重排 (EmbeddingRanker) 或 LLMReranker 时会使用此客户端。
        """
        self.openai_client = openai_client
        self.downloader = Downloader(timeout_sec=3.0, max_concurrent=5)
        self.cleaner = HTMLCleaner()

    def set_openai_client(self, client: Any) -> None:
        """动态配置 OpenAI 客户端以激活向量和模型排序能力。"""
        self.openai_client = client

    def search(self, query: str, limit: int = 3, ranking_method: str = "bm25") -> str:
        """执行联网检索，被 registry.dispatch 直接调用。"""
        query = query.strip()
        if not query:
            return "Error: Empty search query."

        try:
            # 1. 调用默认搜索引擎获取网页列表
            engine = get_default_engine()
            logger.info("Executing web search using: %s with query: '%s'", engine.__class__.__name__, query)
            search_results = engine.search(query, limit=limit)
            if not search_results:
                return "No search results found for this query."

            # 情况 A：如果指定为 'snippet'，直接快速返回搜索引擎给出的摘要片段，无需下载正文
            if ranking_method == "snippet":
                return json.dumps(search_results, ensure_ascii=False, indent=2)

            # 2. 从检索出的链接下载正文 HTML
            urls = [r["url"] for r in search_results if r.get("url")]
            if not urls:
                # 兜底：无 URL 时回退返回 snippet
                return json.dumps(search_results, ensure_ascii=False, indent=2)

            logger.info("Downloading content from %d target page(s)...", len(urls))
            pages = self.downloader.download_pages(urls)
            if not pages:
                logger.warning("Failed to download any web page body. Falling back to search snippets.")
                return json.dumps(search_results, ensure_ascii=False, indent=2)

            # 3. 对每个网页正文执行清洗与分块 (Chunking)
            all_chunks = []
            for url, raw_html in pages.items():
                cleaned_text = self.cleaner.clean(raw_html)
                chunks = self.cleaner.chunk_text(cleaned_text, chunk_size=400, overlap=50)
                # 保留片段元数据
                for c in chunks:
                    all_chunks.append({
                        "source_url": url,
                        "text": c
                    })

            if not all_chunks:
                logger.warning("Cleaned text has 0 chunks. Falling back to search snippets.")
                return json.dumps(search_results, ensure_ascii=False, indent=2)

            # 4. 根据指定的检索排序方法过滤出最相关的前 8 个片段
            corpus = [c["text"] for c in all_chunks]
            top_k = min(8, len(corpus))

            ranked_results = []
            if ranking_method == "embedding" and self.openai_client:
                logger.info("Ranking web page chunks using: EmbeddingRanker")
                ranker = EmbeddingRanker(self.openai_client)
                top_indices = ranker.rank(query, corpus, top_n=top_k)
                for idx, score in top_indices:
                    ranked_results.append({
                        "source": all_chunks[idx]["source_url"],
                        "relevance_score": round(score, 4),
                        "content": corpus[idx]
                    })
            elif ranking_method == "rerank" and self.openai_client:
                logger.info("Ranking web page chunks using: LLMReranker")
                # 先用 BM25 粗筛出前 15 个，再用大模型重排，以限制 API 消费和时间延迟
                bm25_ranker = BM25Ranker(corpus)
                coarse_indices = bm25_ranker.rank(query, top_n=min(15, len(corpus)))
                coarse_corpus = [corpus[idx] for idx, _ in coarse_indices]
                coarse_chunks = [all_chunks[idx] for idx, _ in coarse_indices]

                reranker = LLMReranker(self.openai_client)
                rerank_indices = reranker.rank(query, coarse_corpus, top_n=top_k)
                for idx, score in rerank_indices:
                    ranked_results.append({
                        "source": coarse_chunks[idx]["source_url"],
                        "relevance_score": score,
                        "content": coarse_corpus[idx]
                    })
            else:
                # 默认使用 BM25 / TF-IDF
                logger.info("Ranking web page chunks using: BM25Ranker")
                ranker = BM25Ranker(corpus)
                top_indices = ranker.rank(query, top_n=top_k)
                for idx, score in top_indices:
                    ranked_results.append({
                        "source": all_chunks[idx]["source_url"],
                        "relevance_score": round(score, 4),
                        "content": corpus[idx]
                    })

            return json.dumps(ranked_results, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.exception("Error executing layered web search")
            return f"Error executing web search: {e}"


# -- 自动注册至通用工具中心 ----------------------------------------------------
from tools.registry import registry

WEB_SEARCH_SCHEMA = {
    "description": (
        "Search the internet for real-time information, news, current events, "
        "or technical specifications that are outside your knowledge base. "
        "This tool performs high-precision semantic search over matching web pages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "limit": {
                "type": "integer",
                "description": "Max search result links to scrape and read (default: 3).",
            },
            "ranking_method": {
                "type": "string",
                "enum": ["bm25", "embedding", "rerank", "snippet"],
                "description": (
                    "The text ranking method to extract high-relevance paragraphs. "
                    "'bm25': Pure text keyword matching (default). "
                    "'embedding': Cosine semantic matching (needs OpenAI). "
                    "'rerank': LLM-based reranking. "
                    "'snippet': Fast fallback, returns search engine snippets directly."
                ),
            }
        },
        "required": ["query"],
    },
}

registry.register(
    name="web_search",
    toolset="web_search",
    schema=WEB_SEARCH_SCHEMA,
    handler=lambda query, limit=3, ranking_method="bm25", **kwargs: kwargs["agent"].web_search.search(
        query, limit=limit, ranking_method=ranking_method
    ),
)


