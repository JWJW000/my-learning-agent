"""基于 SQLite WAL 模式和 FTS5 全文索引虚拟表的会话持久化存储层。

持久化管理的实体对象包含：
  - 会话元数据（Session_id，所用模型，时间戳，会话总结，Token 消耗累计）
  - 对话历史详细消息（Messages）
  - FTS5 全文检索索引数据（借助 SQLite 触发器自动实现增删改的零延迟索引同步）

设计考量：
  - journal_mode = WAL: 采用 SQLite 的 Write-Ahead Logging（预写日志）模式以支持高效的读写并发。
  - synchronous = NORMAL: 降低磁盘强制同步频次，最大化减少频繁磁盘 I/O 带来的对话停顿感。
  - 自动触发器（Triggers）：在常规 messages 表进行写入或变动时自动触发 FTS 映射。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SessionStore:
    """基于 SQLite 的会话与消息持久化存储层，提供 FTS5 全文检索支持。"""

    def __init__(self, db_path: str = "./data/state.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,  # 允许跨线程（如后台同步线程）重用 SQLite 连接
        )
        # 激活 WAL 日志模式并设置同步级别，优化本地 SQLite 在并发读写下的性能表现
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()

    def _init_tables(self) -> None:
        """初始化 sessions、messages 表，建立 FTS5 虚拟索引表以及配置联动触发器。"""
        self.db.executescript(
            """
            -- 1. 会话元数据表。parent_session_id 用于记录由于上下文窗口压缩分裂出的子会话关联链
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL,
                summary TEXT,
                model TEXT,
                source TEXT DEFAULT 'cli',
                parent_session_id TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0
            );

            -- 2. 具体对话消息表。
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );

            -- 为 messages 的 session_id 检索添加索引以优化对话历史的顺序加载效率
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);

            -- 3. 基于 SQLite FTS5 的虚拟全文检索表，关联 messages 表的 content 字段进行倒排索引
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                content='messages',
                content_rowid='id'
            );

            -- 4. 数据库联动触发器 A: 当 messages 表写入新交互时，自动将内容复制同步进 FTS5 虚拟索引表
            CREATE TRIGGER IF NOT EXISTS messages_ai
                AFTER INSERT ON messages
            BEGIN
                INSERT INTO messages_fts(rowid, content)
                VALUES (new.id, new.content);
            END;

            -- 数据库联动触发器 B: 当 messages 表发生数据物理删除时，同步清理 FTS 倒排索引
            CREATE TRIGGER IF NOT EXISTS messages_ad
                AFTER DELETE ON messages
            BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;

            -- 数据库联动触发器 C: 当 messages 发生内容变更（如压缩等）时，自动重构其对应的倒排索引行
            CREATE TRIGGER IF NOT EXISTS messages_au
                AFTER UPDATE ON messages
            BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO messages_fts(rowid, content)
                VALUES (new.id, new.content);
            END;
            """
        )
        self.db.commit()

    # -- 会话级别 DB 操作 ------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        model: str = "",
        source: str = "cli",
        parent_session_id: Optional[str] = None,
    ) -> None:
        """插入或忽略一条新会话的元数据记录。

        Args:
            session_id: 会话的唯一 UUID。
            model: 本次会话主模型的名称。
            source: 区分运行源（默认 'cli', 其他可能为 'subagent', 'cron' 等）。
            parent_session_id: 发生分裂压缩时的父会话 ID。
        """
        now = time.time()
        self.db.execute(
            """
            INSERT OR IGNORE INTO sessions
                (session_id, created_at, updated_at, model, source, parent_session_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, now, now, model, source, parent_session_id),
        )
        self.db.commit()

    def update_session_summary(self, session_id: str, summary: str) -> None:
        """更新会话在交互结束或被销毁时生成的全会话宏观大纲摘要。"""
        self.db.execute(
            "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
            (summary, time.time(), session_id),
        )
        self.db.commit()

    def update_session_tokens(
        self,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """累加该会话轮次的总 Token 消耗，便于后续的账单审计和自动压缩判断。"""
        self.db.execute(
            """
            UPDATE sessions
            SET input_tokens = input_tokens + ?,
                output_tokens = output_tokens + ?,
                updated_at = ?
            WHERE session_id = ?
            """,
            (input_tokens, output_tokens, time.time(), session_id),
        )
        self.db.commit()

    # -- 对话消息级别 DB 操作 --------------------------------------------------

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> int:
        """向消息历史表 `messages` 中归档写入一条新会话记录，此操作会自动关联触发 FTS 索引创建。

        Returns:
            int: 插入记录的自增 row ID。
        """
        cursor = self.db.execute(
            """
            INSERT INTO messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content or "", time.time()),
        )
        self.db.commit()
        return cursor.lastrowid

    def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """按时间顺序拉取指定会话下的消息队列。

        Args:
            session_id: 会话 UUID。
            limit: 拉取的最大限制条数。
            offset: 分页偏移量。
        """
        rows = self.db.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
        )

        return [
            {"role": r[0], "content": r[1], "created_at": r[2]}
            for r in rows
        ]

    # -- 对话检索接口 (中载至 SessionSearch 实现) --------------------------------

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """进行快速的跨会话 FTS5 全文搜索。

        内部会实例化 SessionSearch 并调用其 discover （发现模式）逻辑实现谱系去重。
        """
        from tools.session_search import SessionSearch

        searcher = SessionSearch(self.db)
        return searcher.discover(query, limit)

    # -- 关闭/安全注销 ---------------------------------------------------------

    def close(self) -> None:
        """关闭当前的 SQLite 数据库连接。"""
        self.db.close()
