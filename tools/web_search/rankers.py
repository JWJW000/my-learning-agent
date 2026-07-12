"""文本检索排序层 (rankers)。

提供三种过滤与文档重新排序策略：
1. BM25Ranker (基于纯 Python 原生实现的词频 / BM25 算法)
2. EmbeddingRanker (使用 Agent 现成的 OpenAI 客户端计算 text-embedding 向量相似度)
3. LLMReranker (调用轻量级辅助大模型对段落进行相关度打分)
"""

from __future__ import annotations

import math
import re
from typing import Any


class BM25Ranker:
    """基于纯 Python 实现的 BM25 (Best Matching 25) 关键字匹配排序算法。

    无需外部第三方计算库即可在终端极速运行。
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            corpus: 切分出的文档/片段语料库。
            k1: 词频饱和度控制参数。
            b: 文档长度惩罚系数。
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = 0.0

        # 分词处理器（简单移除非字母/中文的标点，以小写空格切分）
        self.docs = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.docs]
        if self.corpus_size > 0:
            self.avg_doc_len = sum(self.doc_lens) / self.corpus_size

        # 计算每个 Term 的逆文档频率 (IDF) 还有文档中的 Term Frequencies (TF)
        self.doc_frequencies: dict[str, int] = {}
        self.term_frequencies: list[dict[str, int]] = []

        for doc in self.docs:
            frequencies: dict[str, int] = {}
            for term in doc:
                frequencies[term] = frequencies.get(term, 0) + 1
            self.term_frequencies.append(frequencies)

            for term in frequencies.keys():
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1

        self.idf: dict[str, float] = {}
        for term, freq in self.doc_frequencies.items():
            # 标准 BM25 IDF 计算公式（带平滑防止负值）
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def _tokenize(self, text: str) -> list[str]:
        """对中文与英文词进行基础规则分词与正则净化。"""
        # 转为小写，处理英文字符
        text = text.lower()
        # 简单将中文按字切分，英文按单词切分
        words = []
        for segment in re.findall(r"[一-龥]|[a-z0-9]+", text):
            words.append(segment)
        return words

    def score(self, query: str, doc_index: int) -> float:
        """计算指定查询与对应索引文档的 BM25 匹配分值。"""
        query_terms = self._tokenize(query)
        score = 0.0
        doc_len = self.doc_lens[doc_index]
        tfs = self.term_frequencies[doc_index]

        for term in query_terms:
            if term not in tfs:
                continue
            tf = tfs[term]
            idf = self.idf.get(term, 0.0)

            # BM25 主打分公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
            score += idf * (numerator / denominator)

        return score

    def rank(self, query: str, top_n: int = 5) -> list[tuple[int, float]]:
        """对语料库所有片段打分，返回前 N 个最相关的片段索引及得分 `(index, score)`。"""
        scores = [(idx, self.score(query, idx)) for idx in range(self.corpus_size)]
        # 降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


class EmbeddingRanker:
    """基于 OpenAI Embedding 向量的余弦相似度匹配排序器。"""

    def __init__(self, client: Any, model: str = "text-embedding-3-small"):
        """
        Args:
            client: 外部传入的 openai.OpenAI 客户端对象（共用 Agent 实例）。
            model: 使用的 Embedding 提取模型。
        """
        self.client = client
        self.model = model

    def _get_embedding(self, text: str) -> list[float]:
        """调用接口获取文本的 Embedding 向量。"""
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception:
            # 容错返回空向量
            return []

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """在纯 Python 中实现余弦相似度计算（无需 numpy）。"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def rank(self, query: str, corpus: list[str], top_n: int = 5) -> list[tuple[int, float]]:
        """利用向量模型计算匹配相关性，并对语料库重排。"""
        if not corpus:
            return []

        # 获取 Query 的 Embedding 向量
        query_vec = self._get_embedding(query)
        if not query_vec:
            # 回退：如果网络异常拿不到 embedding，默认返回原始顺序
            return [(idx, 0.0) for idx in range(len(corpus))][:top_n]

        # 获取 Corpus 中每一项的 Embedding 向量并做相似度计算
        scores = []
        for idx, doc in enumerate(corpus):
            doc_vec = self._get_embedding(doc)
            similarity = self._cosine_similarity(query_vec, doc_vec)
            scores.append((idx, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


class LLMReranker:
    """利用大模型 (LLM) 进行细粒度相关性打分并重排。"""

    def __init__(self, client: Any, model: str = "gpt-4o-mini"):
        """
        Args:
            client: 共用的 openai.OpenAI 客户端。
            model: 打分所使用的大模型名。
        """
        self.client = client
        self.model = model

    def score_passage(self, query: str, passage: str) -> float:
        """评估单个 Passage 相对 Query 的相关度得分 (0.0 ~ 10.0)。"""
        prompt = (
            "You are a passage relevance rater. Rate the relevance of the following passage "
            "to the user query. Output ONLY a float number between 0.0 and 10.0 (e.g. 8.5).\n"
            "0.0 means completely irrelevant, 10.0 means perfect answer to the query.\n\n"
            f"Query: {query}\n"
            f"Passage: {passage}\n\n"
            "Relevance Score:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            score_str = response.choices[0].message.content.strip()
            # 尝试解析为 float
            match = re.search(r"\d+(\.\d+)?", score_str)
            if match:
                return float(match.group())
        except Exception:
            pass
        return 0.0

    def rank(self, query: str, corpus: list[str], top_n: int = 5) -> list[tuple[int, float]]:
        """对传入的语料库并行打分并重排。"""
        if not corpus:
            return []

        scores = []
        for idx, doc in enumerate(corpus):
            # 为了限制大模型调用次数与延迟，通常我们会对已经被 BM25 / Embedding 初筛过的前 N 个进行精排打分
            score = self.score_passage(query, doc)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
