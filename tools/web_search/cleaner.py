"""HTML 文本提取与清洗器 (cleaner)。

利用 Python 内置标准库 `html.parser` 过滤不相关标签（如 script, style, nav），
提取并格式化网页的高密度正文文本，并实现分块（Chunking）。
"""

from __future__ import annotations

import html
import html.parser
import re


class HTMLCleanerParser(html.parser.HTMLParser):
    """HTML 剥离与正文内容提取解析器。"""

    def __init__(self):
        super().__init__()
        self.text_parts: list[str] = []
        # 无需读取其内部文本的噪音标签列表
        self.ignore_tags = {
            "script", "style", "nav", "footer", "header", "aside", "noscript", "iframe",
            "select", "option", "button", "form", "head", "meta", "link"
        }
        self.current_tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.current_tag_stack.append(tag.lower())
        # 在遇到常见块级元素或换行元素时，写入换行符以保持格式清晰
        if tag.lower() in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.current_tag_stack:
            self.current_tag_stack.pop()
        if tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        # 如果当前所处的标签栈中包含任何需要忽略的噪音标签，不提取其文字
        if any(ignored in self.current_tag_stack for ignored in self.ignore_tags):
            return

        cleaned_data = data.strip()
        if cleaned_data:
            self.text_parts.append(" " + cleaned_data)


class HTMLCleaner:
    """提供网页 HTML 清洗过滤与分块（Chunking）功能。"""

    def clean(self, html_content: str) -> str:
        """解析 HTML，过滤噪音标签并提取结构化正文文本。"""
        if not html_content:
            return ""

        parser = HTMLCleanerParser()
        try:
            parser.feed(html_content)
            parser.close()
        except Exception:
            # 容错：若解析器发生异常，使用正则进行兜底提取
            stripped = re.sub(r"<script.*?</script>", " ", html_content, flags=re.DOTALL)
            stripped = re.sub(r"<style.*?</style>", " ", stripped, flags=re.DOTALL)
            stripped = re.sub(r"<[^>]*>", " ", stripped)
            return html.unescape(stripped)

        raw_text = "".join(parser.text_parts)

        # 整理空白字符和过多换行符
        lines = []
        for line in raw_text.splitlines():
            cleaned_line = re.sub(r"\s+", " ", line).strip()
            if cleaned_line:
                lines.append(cleaned_line)

        # 用双换行连接，保持段落感
        return "\n\n".join(lines)

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
        """将清洗后的长文本按字符大小进行滑动窗口切分（支持重叠区）。

        Args:
            text: 长正文。
            chunk_size: 每个分块的最大字符长度。
            overlap: 相邻分块之间的重叠字符长度（保证上下文不断层）。
        """
        if not text:
            return []

        # 按双换行段落进行切割，尽量不把一个完整的段落从中间生硬切断
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for p in paragraphs:
            p_len = len(p)
            # 如果单段就超出 chunk_size，需要强制对该段按句切分
            if p_len > chunk_size:
                sentences = re.split(r"(?<=[。！？.!?])\s*", p)
                for s in sentences:
                    s_len = len(s)
                    if current_length + s_len > chunk_size:
                        if current_chunk:
                            chunks.append("\n".join(current_chunk))
                        # 新建分块，保留少量重叠
                        overlap_text = current_chunk[-1][-overlap:] if current_chunk and overlap > 0 else ""
                        current_chunk = [overlap_text, s] if overlap_text else [s]
                        current_length = len(current_chunk[0]) + s_len if overlap_text else s_len
                    else:
                        current_chunk.append(s)
                        current_length += s_len
            else:
                if current_length + p_len > chunk_size:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    # 重叠处理
                    overlap_text = current_chunk[-1][-overlap:] if current_chunk and overlap > 0 else ""
                    current_chunk = [overlap_text, p] if overlap_text else [p]
                    current_length = len(current_chunk[0]) + p_len if overlap_text else p_len
                else:
                    current_chunk.append(p)
                    current_length += p_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]
