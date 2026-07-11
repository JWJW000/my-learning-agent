"""会话历史全局搜索工具 (SessionSearch)。

借助于 SQLite FTS5 全文索引能力，本工具提供**零大模型调用开销（Zero LLM Cost）**的跨会话消息检索。
支持三种完全不同的搜索调用模式：
  1. 发现模式 (DISCOVER):
     在大范围历史会话中通过关键词 MATCH 进行全文搜索相关消息片断。
     该模式提供谱系去重（Lineage Deduplication）：基于 `parent_session_id` 链条去重，
     如果多个命中消息来自同一个会话衍生支流，只提取该会话下评分最高的那条，避免返回一大堆重复的历史片断。
  2. 滚屏模式 (SCROLL):
     指定 `session_id` 和分页参数，原样展开目标历史会话的对话细节，供模型“穿越”阅读历史细节。
  3. 概览模式 (BROWSE):
     按时间戳倒序，概览列出最近发生过的会话，自动过滤掉后台子 Agent 运行或自动化工具（如 curator）产生的会话。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SessionSearch:
    """跨会话搜索组件，基于 SQLite FTS5 实现。"""

    # 全文检索在去重前最大扫描的消息记录行数，保证能够在大量自动化记录（如 cron 任务）下，
    # 依然能把被埋没的用户真实 CLI 会话扫描出来进行 Lineage 合并展示
    DISCOVER_SCAN_LIMIT = 300

    def __init__(self, db):
        """
        Args:
            db: 从 SessionStore 传入的 `sqlite3.Connection` 活动数据库连接对象。
        """
        self.db = db

    # -- 核心检索模式实现 ------------------------------------------------------

    def discover(self, query: str, limit: int = 10) -> list[dict]:
        """跨所有历史会话执行全文模糊匹配，使用 sqlite FTS5 进行打分排序，并进行 lineage 去重。

        Returns:
            list[dict]: 命中的消息元数据与关键高亮片段 (highlight) 列表。
        """
        try:
            # JOIN 原始表和 FTS 虚拟表，使用 rank 进行自然语义打分排序，并利用 highlight 标记关键词
            rows = self.db.execute(
                """
                SELECT m.session_id, m.role, m.content, m.created_at,
                       highlight(messages_fts, 0, '**', '**') as snippet,
                       s.summary
                FROM messages_fts
                JOIN messages m ON messages_fts.rowid = m.id
                LEFT JOIN sessions s ON m.session_id = s.session_id
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, self.DISCOVER_SCAN_LIMIT),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # 容错：防止因用户输入的 query 中带有不合规的 FTS 语法符号引起崩溃
            logger.warning("FTS5 检索解析语法失败: %s", exc)
            return []

        # 按会话血统谱系去重（确保相同 lineage 的会话分支仅出现评分最高的一条记录）
        seen_sessions: set[str] = set()
        results = []
        for row in rows:
            sid = row[0]
            if sid in seen_sessions:
                continue
            seen_sessions.add(sid)
            results.append(
                {
                    "session_id": sid,
                    "role": row[1],
                    "snippet": row[4] or row[2][:200],  # 无法生成 snippet 时回退为首部截断
                    "created_at": row[3],
                    "session_summary": row[5],
                }
            )
            if len(results) >= limit:
                break

        return results

    def scroll(self, session_id: str, offset: int = 0, limit: int = 20) -> list[dict]:
        """锚定指定会话 ID，平铺拉取其对话历史细节（支持 offset 分页滚动）。"""
        rows = self.db.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
        ).fetchall()

        return [
            {"role": r[0], "content": r[1], "created_at": r[2]}
            for r in rows
        ]

    def browse(self, limit: int = 10) -> list[dict]:
        """按时间戳倒序浏览会话大纲。屏蔽后台执行的 subagent 或者是外部 tool 触发的影子会话。"""
        rows = self.db.execute(
            """
            SELECT session_id, created_at, summary, model
            FROM sessions
            WHERE source NOT IN ('subagent', 'tool')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "session_id": r[0],
                "created_at": r[1],
                "summary": r[2],
                "model": r[3],
            }
            for r in rows
        ]

    # -- 工具定义 Schemas 组装 --------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """定义供 Agent 使用的 `session_search` 工具 API 接口。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "session_search",
                    "description": (
                        "Search past conversations. Modes: "
                        "'discover' (FTS5 query), 'scroll' (view messages in a session), "
                        "'browse' (recent sessions). Zero LLM cost."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["discover", "scroll", "browse"],
                            },
                            "query": {
                                "type": "string",
                                "description": "Search query (for discover mode)",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session ID (for scroll mode)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (default: 10)",
                            },
                        },
                        "required": ["mode"],
                    },
                },
            }
        ]

    def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """派发和响应 `session_search` 的三种不同模式，并打包为 JSON String 返回给 Agent。"""
        if tool_name != "session_search":
            return f"Unknown tool: {tool_name}"

        mode = arguments.get("mode", "browse")
        limit = arguments.get("limit", 10)

        match mode:
            case "discover":
                query = arguments.get("query", "")
                if not query:
                    return "Error: 'query' required for discover mode"
                results = self.discover(query, limit)
            case "scroll":
                sid = arguments.get("session_id", "")
                if not sid:
                    return "Error: 'session_id' required for scroll mode"
                results = self.scroll(sid, limit=limit)
            case "browse":
                results = self.browse(limit)
            case _:
                return f"Unknown mode: {mode}"

        return json.dumps(results, ensure_ascii=False, indent=2)
