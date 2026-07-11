"""记忆管理器 (Memory Manager) — 负责统一编排和管理各类可插拔记忆 Provider 的生命周期。

本模块是 main.py 中唯一的记忆系统调用入口。它在 Agent 交互过程中负责：
  - 注册可用的记忆 Provider（包含内置文件存储与外部插件）
  - 触发多 Provider 的静态提示词注入组装 (`build_system_prompt`)
  - 广播执行前序拉取 (`prefetch`) 和后序同步回写 (`sync_turn`)
  - 聚合各 Provider 暴露出来的工具，并将工具调用请求准确路由分发到对应的 Provider 实例
"""

from __future__ import annotations

import logging
from typing import Any

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆 Providers 编排器。在 main.py 初始化时进行组装并注入 Agent 实例。"""

    def __init__(self):
        # 已激活注册的 Provider 列表
        self._providers: list[MemoryProvider] = []
        # 工具映射路由表，用于快速分发：tool_name → 归属的 MemoryProvider
        self._tool_map: dict[str, MemoryProvider] = {}

    # -- 注册机制 ------------------------------------------------------------

    def add_provider(self, provider: MemoryProvider) -> None:
        """向编排器注册一个新的记忆 Provider。

        注意：由于避免不同记忆 Provider 产生工具命名冲突和数据读写竞争，
        限制尽量只激活必要的 Provider（例如一个内置文件 Provider + 最多一个外部数据库 Provider）。
        """
        if not provider.is_available():
            logger.info("记忆 Provider '%s' 在当前环境下不可用，已跳过注册", provider.name)
            return

        self._providers.append(provider)

        # 扫描该 Provider 暴露的工具列表，将其建立映射关系，供后续 Tool Call 执行时路由分发
        for schema in provider.get_tool_schemas():
            func = schema.get("function", schema)
            name = func.get("name", "")
            if name:
                self._tool_map[name] = provider
        logger.info("记忆 Provider '%s' 成功注册", provider.name)

    # -- 生命周期生命节点 -------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """在会话启动时，初始化所有已注册的 Provider。"""
        for p in self._providers:
            try:
                p.initialize(session_id, **kwargs)
            except Exception:
                logger.exception("初始化记忆 Provider '%s' 失败", p.name)

    def build_system_prompt(self) -> str:
        """向所有已注册的 Provider 请求其需要拼入 System Prompt 的静态信息块，并整合成一个完整的文本块。

        注意：生成的 Prompt 块一旦合并就会被冻结。
        中途大模型写回记忆虽然会实时落盘，但绝不更改已在进行中的 System Prompt。
        这是维持 Prompt Prefix Cache 命中率和减少大模型推理幻觉的关键。
        """
        blocks = []
        for p in self._providers:
            try:
                block = p.system_prompt_block()
                if block and block.strip():
                    blocks.append(block)
            except Exception:
                logger.exception("记忆 Provider '%s' 的 system_prompt_block 运行失败", p.name)
        return "\n\n".join(blocks)

    def prefetch(self, user_message: str) -> None:
        """在向 LLM 提交正式请求前，遍历所有 Provider 异步快速取回与 user_message 相关的短期或长期记忆。"""
        for p in self._providers:
            try:
                p.prefetch(user_message)
            except Exception:
                logger.exception("记忆 Provider '%s' 执行 prefetch 失败", p.name)

    def sync_turn(self, user_msg: str, assistant_msg: str) -> None:
        """当一个对话轮次完整执行并拿到助手回复后，广播通知所有 Provider 提取本轮交互中的知识并执行持久化存储。"""
        for p in self._providers:
            try:
                p.sync_turn(user_msg, assistant_msg)
            except Exception:
                logger.exception("记忆 Provider '%s' 执行 sync_turn 失败", p.name)

    def shutdown(self) -> None:
        """在会话被销毁或应用退出时，安全通知并断开所有 Provider，确保写操作彻底落盘。"""
        for p in self._providers:
            try:
                p.shutdown()
            except Exception:
                logger.exception("记忆 Provider '%s' 执行 shutdown 失败", p.name)

    # -- 工具调用路由器 ------------------------------------------------------

    def get_all_tool_schemas(self) -> list[dict[str, Any]]:
        """从所有激活的 Provider 中收集并汇总所有的工具模式 Schema，用于组装 Agent 可调用的全局工具列表。"""
        schemas = []
        for p in self._providers:
            schemas.extend(p.get_tool_schemas())
        return schemas

    def is_memory_tool(self, tool_name: str) -> bool:
        """通过查表判断某个工具调用请求（tool_name）是否隶属于记忆管理系统。"""
        return tool_name in self._tool_map

    def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """分发工具调用。根据 tool_name 路由至注册它的 Provider 实例并返回工具执行结果。"""
        provider = self._tool_map.get(tool_name)
        if not provider:
            return f"Unknown memory tool: {tool_name}"
        try:
            return provider.handle_tool_call(tool_name, arguments)
        except Exception as exc:
            logger.exception("记忆工具 '%s' 执行发生内部错误", tool_name)
            return f"Error: {exc}"

    # -- 可选钩子扩展 --------------------------------------------------------

    def on_session_end(self, messages: list[dict]) -> None:
        """向所有 Provider 广播会话结束信号，通常用于生成长对话摘要。"""
        for p in self._providers:
            try:
                p.on_session_end(messages)
            except Exception:
                logger.exception("记忆 Provider '%s' 响应 on_session_end 失败", p.name)

    def on_pre_compress(self, messages: list[dict]) -> str | None:
        """在自动上下文窗口压缩触发时，向各 Provider 收集压缩前需要保留的提示附录。"""
        parts = []
        for p in self._providers:
            try:
                result = p.on_pre_compress(messages)
                if result:
                    parts.append(result)
            except Exception:
                logger.exception("记忆 Provider '%s' 响应 on_pre_compress 失败", p.name)
        return "\n".join(parts) if parts else None
