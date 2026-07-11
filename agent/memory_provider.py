"""可插拔记忆 Provider (Memory Provider) 的抽象基类。

记忆 Provider 赋予 Agent 跨会话持久回忆的本领。
MemoryManager 会对注册的 Provider 进行编排，并严格限制只能拥有一个外部插件类 Provider 运行，
以防各 Provider 注入大量重复工具造成 schema 臃肿或产生记忆回写冲突。

生命周期（由 MemoryManager 统一触发，在 Agent 运行时 run_agent.py 及 main.py 中被调用）:
  initialize(session_id)   — 建立与本地数据库或外部 API 的连接，预热缓存
  system_prompt_block()    — 生成供初始 System Prompt 注入的静态文本块（冻结快照）
  prefetch(query)          — 在发送用户消息前，异步/快速取回当前最相关的上下文记忆
  sync_turn(user, asst)    — 对话轮次结束后，触发持久化或后台线程进行记忆更新
  get_tool_schemas()       — 返回该 Provider 供模型直接调用的工具模式 schema 定义
  handle_tool_call()       — 执行并分发该 Provider 专属的工具调用
  shutdown()               — 进程退出或会话销毁时的收尾释放
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """记忆 Providers 的抽象基类 (Abstract Base Class)。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 的唯一标识简写（例如 'builtin', 'redis', 'mem0'）。"""

    @abstractmethod
    def is_available(self) -> bool:
        """返回此 Provider 在当前系统环境下是否可用。

        该函数应仅进行静态配置与本地依赖包检查，不应在此发起耗时的网络交互请求。
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """为特定会话进行预热和组件初始化。在 Agent 实例启动时被且仅被调用一次。"""

    @abstractmethod
    def system_prompt_block(self) -> str:
        """返回注入系统提示词的静态记忆内容。

        本方法在会话启动时组装 System Prompt 时调用。
        出于 Prefix Cache 性能优化及防止大模型逻辑漂移的考虑，此数据一旦被拼入 System Prompt，
        在此会话中即被“冻结”。后续 turn 写入的记忆不会实时反映在 System Prompt 内。
        """

    @abstractmethod
    def prefetch(self, query: str) -> None:
        """在用户发起对话请求时，根据当前输入，异步或快速地在底层预提取/召回相关的记忆内容。"""

    @abstractmethod
    def sync_turn(self, user_msg: str, assistant_msg: str) -> None:
        """在一轮对话彻底结束后触发。

        Provider 应在此对该轮的交互内容进行分析，提取关键的 Agent 行为纪要或 User 偏好特征，
        并将其写入本地文件或进行远端数据库同步。
        """

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """返回该 Provider 决定向大模型暴露的工具定义 Schema 列表。

        如果不需要让大模型直接通过 `tool_calls` 来读取/删除/更新记忆，可以返回空列表 []。
        """
        return []

    def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """处理并响应大模型触发的属于此 Provider 注册的工具调用。

        此函数通常和 `get_tool_schemas()` 配合重写，用于路由解析。
        """
        return f"Unknown tool: {tool_name}"

    def shutdown(self) -> None:
        """会话销毁或进程安全结束时的生命周期回收，如清理线程池、关闭 DB 连接等。"""

    # -- 可选钩子函数 (子类根据需要进行重写即可) -----------------------------------

    def on_session_end(self, messages: list[dict]) -> None:
        """当整个 Session 即将退场时触发。可用于进行整体对话的二次深度归纳或批量生成 Embedding 向量。"""

    def on_pre_compress(self, messages: list[dict]) -> str | None:
        """在上下文窗口超限、ContextCompressor 启动中间消息总结前触发。

        Provider 可返回一小段总结指示，用于追加压缩摘要的最开头，作为保留背景。
        """
        return None
