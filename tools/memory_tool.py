"""内置的持久化文件背书记忆工具 (MemoryStore)。

管理两个本地 Markdown 记忆数据库：
  - MEMORY.md: 记录 Agent 自身的经验备忘、工具使用心得、项目规则等。
  - USER.md: 记录 User 的个人画像、特定偏好、本地硬件架构与使用习惯。

设计与核心架构模式：
  1. 冻结快照模式 (Frozen Snapshot)：
     在会话初次启动执行 initialize 时，读取本地 md 文件并解析成内存条目列表，
     然后拼接并冻结生成 `_system_prompt_snapshot` 注入系统提示词。
     后续在会话期间模型如果通过工具（如 `memory_save`）写入新记忆，这些更改会立即持久化落盘，
     并同步更新内存中的 entries 列表，但绝不更改已在进行中的 System Prompt。
     这保证了 prefix cache 的稳定并防止上下文变动引起的推理幻觉。
  2. 漂移检测 (Drift Detection)：
     在回写文件前，会重新计算磁盘上现有文件的 MD5 哈希。
     如果与初次加载时的哈希不匹配，说明有并发会话修改了该文件或人工对其进行了直接修补。
     此时系统会拒绝写入以防止覆盖外部修改，保持数据完整性。
  3. 威胁扫描 (Threat Scanning)：
     对写入记忆的内容进行初级的 Prompt Injection（提示词注入攻击）正则过滤，拒绝记录破坏系统设定的指示。
  4. 容量限制与驱逐 (Eviction)：
     采用“字符数限制”以保持模型中立度（Token 数量受模型分词器差异影响）。
     超出限制时按交互时间线“最老先出”机制强制淘汰旧条目。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# 条目之间的特殊物理分隔符，可支持多行条目解析
ENTRY_DELIMITER = "\n§\n"


class MemoryStore(MemoryProvider):
    """文件背书的记忆 Provider，实现冻结快照和冲突检测。"""

    def __init__(
        self,
        data_dir: str = "./data",
        agent_char_limit: int = 2200,
        user_char_limit: int = 1375,
    ):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._memory_file = self._data_dir / "MEMORY.md"
        self._user_file = self._data_dir / "USER.md"

        self._agent_char_limit = agent_char_limit
        self._user_char_limit = user_char_limit

        # 内存中维护的实时数据条目，写入操作会立即更新此字段并持久化到本地文件
        self._memory_entries: list[str] = []
        self._user_entries: list[str] = []

        # 冻结的系统提示词快照，在会话生命周期内一经 initialize 组装后只读
        self._system_prompt_snapshot: str = ""

        # 漂移检测：记录加载时磁盘文件的 MD5，写入时进行校验
        self._memory_file_hash: str = ""
        self._user_file_hash: str = ""

    # -- 实现 MemoryProvider 生命周期抽象接口 -------------------------------------

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True  # 内置文件记忆无需任何外部鉴权，永远可用

    def initialize(self, session_id: str, **kwargs) -> None:
        """加载磁盘文件条目，并构建冻结的 System Prompt 记忆快照。"""
        self._load_entries()
        self._system_prompt_snapshot = self._format_for_prompt()
        logger.info(
            "成功加载历史记忆: %d 条 Agent 备忘, %d 条用户信息",
            len(self._memory_entries),
            len(self._user_entries),
        )

    def system_prompt_block(self) -> str:
        """获取初始构建的已冻结记忆 System Prompt 块。"""
        return self._system_prompt_snapshot

    def prefetch(self, query: str) -> None:
        """文件数据库无需前置预提取，全部条目均在 system_prompt 中。"""

    def sync_turn(self, user_msg: str, assistant_msg: str) -> None:
        """文件数据库只支持由模型调用工具显式写回，不支持隐式启发式同步。"""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """定义并返回供大模型调用的记忆读写与管理工具 Schema 列表。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_save",
                    "description": (
                        "Save important information to persistent agent memory. "
                        "Use for facts, decisions, patterns you want to remember "
                        "across sessions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The information to remember",
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "user_info_save",
                    "description": (
                        "Record user preferences, habits, environment details, "
                        "or knowledge for personalization."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "User information to record",
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_read",
                    "description": "Read current memory entries (live state, may differ from system prompt snapshot).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "store": {
                                "type": "string",
                                "enum": ["agent", "user", "both"],
                                "description": "Which memory store to read",
                                "default": "both",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_delete",
                    "description": "Delete a memory entry by its index number.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "store": {
                                "type": "string",
                                "enum": ["agent", "user"],
                                "description": "Which store to delete from",
                            },
                            "index": {
                                "type": "integer",
                                "description": "0-based index of the entry to delete",
                            },
                        },
                        "required": ["store", "index"],
                    },
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """路由和派发大模型的工具调用到具体的内部私有逻辑上。"""
        match tool_name:
            case "memory_save":
                return self._save_memory(arguments["content"])
            case "user_info_save":
                return self._save_user_info(arguments["content"])
            case "memory_read":
                return self._read_memory(arguments.get("store", "both"))
            case "memory_delete":
                return self._delete_entry(arguments["store"], arguments["index"])
            case _:
                return f"Unknown memory tool: {tool_name}"

    # -- 内部逻辑组件 ----------------------------------------------------------

    def _save_memory(self, content: str) -> str:
        """将有价值的系统总结写入 MEMORY.md。"""
        content = content.strip()
        if not content:
            return "Error: empty content"

        # 威胁扫描过滤
        if self._looks_suspicious(content):
            return "Error: content rejected by threat scanner"

        self._memory_entries.append(content)
        # 容量检查：若字符数超限，按最老条目驱逐
        self._enforce_limit(self._memory_entries, self._agent_char_limit)
        self._persist()
        return f"Saved to agent memory ({len(self._memory_entries)} entries, {self._total_chars(self._memory_entries)}/{self._agent_char_limit} chars)"

    def _save_user_info(self, content: str) -> str:
        """将用户画像和操作偏好写入 USER.md。"""
        content = content.strip()
        if not content:
            return "Error: empty content"

        if self._looks_suspicious(content):
            return "Error: content rejected by threat scanner"

        self._user_entries.append(content)
        self._enforce_limit(self._user_entries, self._user_char_limit)
        self._persist()
        return f"Saved to user info ({len(self._user_entries)} entries, {self._total_chars(self._user_entries)}/{self._user_char_limit} chars)"

    def _read_memory(self, store: str = "both") -> str:
        """供大模型读取当前内存中的实时记忆条目（返回实时最新的 entries 数据，带 0 起步的序号索引）。"""
        parts = []
        if store in ("agent", "both") and self._memory_entries:
            lines = [f"  [{i}] {e}" for i, e in enumerate(self._memory_entries)]
            parts.append("Agent Memory:\n" + "\n".join(lines))
        if store in ("user", "both") and self._user_entries:
            lines = [f"  [{i}] {e}" for i, e in enumerate(self._user_entries)]
            parts.append("User Info:\n" + "\n".join(lines))
        return "\n\n".join(parts) if parts else "(empty)"

    def _delete_entry(self, store: str, index: int) -> str:
        """根据序号索引物理删除特定的记忆条目。"""
        entries = self._memory_entries if store == "agent" else self._user_entries
        if 0 <= index < len(entries):
            removed = entries.pop(index)
            self._persist()
            return f"Deleted from {store}: {removed[:60]}..."
        return f"Error: index {index} out of range (0-{len(entries) - 1})"

    # -- 本地磁盘 I/O 细节 ----------------------------------------------------

    def _load_entries(self) -> None:
        """从本地磁盘中加载 Markdown 并按特殊分割符切分成条目。"""
        if self._memory_file.exists():
            content = self._memory_file.read_text(encoding="utf-8")
            self._memory_entries = [e.strip() for e in content.split("§") if e.strip()]
            self._memory_file_hash = self._hash(content)

        if self._user_file.exists():
            content = self._user_file.read_text(encoding="utf-8")
            self._user_entries = [e.strip() for e in content.split("§") if e.strip()]
            self._user_file_hash = self._hash(content)

    def _persist(self) -> None:
        """将最新的条目回写入本地磁盘。本步骤不会改动已冻结的 system prompt snapshot。

        写回前执行漂移碰撞检测：如果发现当前的磁盘文件哈希已经改变，则跳过本次写入并报警。
        """
        # MEMORY.md 碰撞校验
        if self._memory_file.exists():
            current_hash = self._hash(self._memory_file.read_text(encoding="utf-8"))
            if current_hash != self._memory_file_hash and self._memory_file_hash:
                logger.warning("MEMORY.md 发生外部修改冲突 — 跳过本次同步回写以免覆盖")
                return

        # 写入文件并更新当前缓存的哈希指纹
        self._memory_file.write_text(
            ENTRY_DELIMITER.join(self._memory_entries), encoding="utf-8"
        )
        self._memory_file_hash = self._hash(
            self._memory_file.read_text(encoding="utf-8")
        )

        # USER.md 同步写入
        self._user_file.write_text(
            ENTRY_DELIMITER.join(self._user_entries), encoding="utf-8"
        )
        if self._user_file.exists():
            self._user_file_hash = self._hash(
                self._user_file.read_text(encoding="utf-8")
            )

    def _format_for_prompt(self) -> str:
        """格式化内部的记忆数组为标准 Markdown 段落，供 System Prompt 拼合注入。"""
        parts = []
        if self._memory_entries:
            entries_text = ENTRY_DELIMITER.join(self._memory_entries)
            parts.append(f"## Agent Memory\n{entries_text}")
        if self._user_entries:
            entries_text = ENTRY_DELIMITER.join(self._user_entries)
            parts.append(f"## User Knowledge\n{entries_text}")
        return "\n\n".join(parts)

    # -- 辅助工具函数 ----------------------------------------------------------

    @staticmethod
    def _total_chars(entries: list[str]) -> int:
        """计算条目集的字符总和。"""
        return sum(len(e) for e in entries)

    @staticmethod
    def _hash(content: str) -> str:
        """生成文本的 MD5 哈希，作为冲突比对的指纹。"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()

    def _enforce_limit(self, entries: list[str], limit: int) -> None:
        """限制字符总容量，如果超限则先进先出（即删除最老的消息）直到符合容量线。"""
        while self._total_chars(entries) > limit and len(entries) > 1:
            removed = entries.pop(0)
            logger.info("记忆容量超限，已自动释放最早的备忘记录: %s...", removed[:40])

    @staticmethod
    def _looks_suspicious(content: str) -> bool:
        """防御式扫描：防止大模型在被提示词注入劫持后，向其自身的长期记忆中注入攻击载荷。"""
        lower = content.lower()
        threats = [
            "ignore previous instructions",
            "ignore all instructions",
            "you are now",
            "system prompt",
            "new instructions:",
            "<system>",
            "</system>",
        ]
        return any(t in lower for t in threats)


# -- 自动注册至通用工具中心 ----------------------------------------------------
from tools.registry import registry

MEMORY_SAVE_SCHEMA = {
    "description": "Save important information to persistent agent memory. Use for facts, decisions, patterns you want to remember across sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember"},
        },
        "required": ["content"],
    },
}

USER_INFO_SAVE_SCHEMA = {
    "description": "Record user preferences, habits, environment details, or knowledge for personalization.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "User information to record"},
        },
        "required": ["content"],
    },
}

MEMORY_READ_SCHEMA = {
    "description": "Read current memory entries (live state, may differ from system prompt snapshot).",
    "parameters": {
        "type": "object",
        "properties": {
            "store": {
                "type": "string",
                "enum": ["agent", "user", "both"],
                "description": "Which memory store to read",
                "default": "both",
            },
        },
    },
}

MEMORY_DELETE_SCHEMA = {
    "description": "Delete a memory entry by its index number.",
    "parameters": {
        "type": "object",
        "properties": {
            "store": {
                "type": "string",
                "enum": ["agent", "user"],
                "description": "Which store to delete from",
            },
            "index": {
                "type": "integer",
                "description": "0-based index of the entry to delete",
            },
        },
        "required": ["store", "index"],
    },
}

registry.register(
    name="memory_save",
    toolset="memory",
    schema=MEMORY_SAVE_SCHEMA,
    handler=lambda content, **kwargs: kwargs["agent"].memory_manager.handle_tool_call("memory_save", {"content": content}),
)

registry.register(
    name="user_info_save",
    toolset="memory",
    schema=USER_INFO_SAVE_SCHEMA,
    handler=lambda content, **kwargs: kwargs["agent"].memory_manager.handle_tool_call("user_info_save", {"content": content}),
)

registry.register(
    name="memory_read",
    toolset="memory",
    schema=MEMORY_READ_SCHEMA,
    handler=lambda store="both", **kwargs: kwargs["agent"].memory_manager.handle_tool_call("memory_read", {"store": store}),
)

registry.register(
    name="memory_delete",
    toolset="memory",
    schema=MEMORY_DELETE_SCHEMA,
    handler=lambda store, index, **kwargs: kwargs["agent"].memory_manager.handle_tool_call("memory_delete", {"store": store, "index": index}),
)

