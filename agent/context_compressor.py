"""上下文压缩器 — 实现对话窗口历史记录的自动化分段压缩。

当会话消耗的 Token 数量趋近模型最大上下文限制时，压缩器会自动启动：
  - 保护开头 N 条消息：防止破坏 system prompt 及其前缀缓存的物理命中，维持首轮设定。
  - 保护尾部 M 条消息：保持最近的几轮会话上下文完全保留，使模型拥有最近对话的具体短程记忆。
  - 压缩中间历史消息：将处于保护区间之外的中间老旧对话，调用较廉价的辅助模型进行高度信息提炼与概括。
  - 加入免责免重执行前缀 (`SUMMARY_PREFIX`)：该前缀极其重要，明确告知主模型压缩文本仅供历史参考背景，
    杜绝模型由于读取到被总结的历史任务时，产生去重新执行已被解决或作废的旧指令。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 指令重执行防护免责声明。
# 此前缀强力指示主模型忽略压缩区块中的“命令词”或“求助提问”，只将其作为上下文客观背景资料，
# 所有行为和当前需要干的事情一律只以压缩区块后追加的真实 User 最新消息为准。
SUMMARY_PREFIX = (
    "[上下文压缩 — 仅供参考] 早期对话已被压缩为以下摘要。\n"
    "这是历史参考，不是当前指令。请只响应此摘要之后的最新用户消息。\n"
    "不要回答或执行此摘要中提到的请求——它们已经被处理过了。\n"
    "只有此摘要之后的最新用户消息才是当前任务。"
)


class ContextCompressor:
    """自动上下文窗口总结与压缩管理器。"""

    def __init__(
        self,
        auxiliary_model: str = "gpt-4o-mini",
        threshold_percent: float = 0.75,
        context_length: int = 128_000,
        protect_first_n: int = 3,
        protect_last_n: int = 6,
    ):
        self.auxiliary_model = auxiliary_model
        self.threshold_percent = threshold_percent
        self.context_length = context_length
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n

        # Token 消耗统计监控状态（由 Agent 在完成 LLM 调用后调用 update_usage 更新）
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.compression_count: int = 0

    def update_usage(self, usage: dict[str, Any]) -> None:
        """从 API 回包中提取 Token 消耗值以刷新实时监测状态。"""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self) -> bool:
        """根据当前总消耗的 Token 比例，检查是否超出了设定的自动压缩阈值线（默认 75%）。"""
        threshold = int(self.context_length * self.threshold_percent)
        return self.last_total_tokens > threshold

    def compress(self, messages: list[dict], openai_client) -> list[dict]:
        """对消息队列进行切分和压缩，将中间老旧消息总结为一条压缩文本块。

        保留头部 `protect_first_n` 消息和尾部 `protect_last_n` 消息不变。
        中间夹着的消息交给辅助总结模型处理。

        Args:
            messages: 当前完整的交互消息列表（不含顶层 system prompt）。
            openai_client: 用于调用辅助模型的 OpenAI 客户端实例。

        Returns:
            list[dict]: 压缩剪枝后的新消息列表。
        """
        total = len(messages)
        # 如果当前消息总数还不够两端保护区间的大小，无需执行压缩
        if total <= self.protect_first_n + self.protect_last_n:
            return messages

        head = messages[: self.protect_first_n]
        tail = messages[-self.protect_last_n :]
        middle = messages[self.protect_first_n : -self.protect_last_n]

        if not middle:
            return messages

        # 调用辅助廉价快速的模型生成中间段的消息摘要
        summary = self._summarize(middle, openai_client)
        self.compression_count += 1

        # 重构消息队列，将带有免责前缀的总结块作为一条 user 消息注入到头部与尾部消息中间
        compressed = head + [
            {
                "role": "user",
                "content": f"{SUMMARY_PREFIX}\n\n{summary}",
                "_metadata": {"is_compression_summary": True},
            }
        ] + tail

        logger.info(
            "上下文窗口压缩完毕: %d 条消息 → %d 条消息 (累计执行次数: #%d)",
            total,
            len(compressed),
            self.compression_count,
        )
        return compressed

    def _summarize(self, messages: list[dict], openai_client) -> str:
        """调用辅助大模型将一段消息流转换为结构化的摘要概要信息。"""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # 结构化提取消息中的工具调用，便于生成摘要时记录执行了什么动作
            if msg.get("tool_calls"):
                calls = msg["tool_calls"]
                call_descs = []
                for tc in calls:
                    fn = tc.get("function", {})
                    call_descs.append(f"  called {fn.get('name', '?')}()")
                content = (content or "") + "\n" + "\n".join(call_descs)

            # 对超长文本消息进行初步切分，避免输入过多给辅助模型产生多余开销
            if content and len(content) > 800:
                content = content[:800] + "...[truncated]"

            lines.append(f"[{role}]: {content}")

        conversation_text = "\n\n".join(lines)

        try:
            response = openai_client.chat.completions.create(
                model=self.auxiliary_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summarizer. Summarize the following "
                            "conversation turns into a concise summary. Preserve:\n"
                            "1. Key decisions made\n"
                            "2. Important facts and data discovered\n"
                            "3. File paths and code changes mentioned\n"
                            "4. Unresolved questions or pending tasks\n"
                            "Use section headings: ## Key Facts, ## Decisions, "
                            "## Pending Items"
                        ),
                    },
                    {"role": "user", "content": conversation_text},
                ],
                max_tokens=1500,
                temperature=0.1,  # 使用低温度值，尽可能保证总结内容客观准确
            )
            return response.choices[0].message.content or "[Summary generation failed]"
        except Exception as exc:
            logger.exception("调用辅助大模型生成上下文摘要失败")
            # 极限兜底策略：简单提取前 5 条历史交互直接展现
            return f"[Summarization failed: {exc}]\n\nKey messages:\n" + "\n".join(
                lines[:5]
            )
