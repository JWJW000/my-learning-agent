"""Hermes-style bounded, curated, file-backed memory.

Two stores are kept as ``§``-delimited Markdown files:

* ``MEMORY.md`` contains durable agent notes about projects and environments.
* ``USER.md`` contains durable user preferences and profile facts.

The files are loaded into a frozen system-prompt snapshot at session start. Tool
writes update live state and disk immediately, but never mutate that snapshot.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agent.memory_provider import MemoryProvider
from tools.registry import registry

logger = logging.getLogger(__name__)

try:  # pragma: no cover - platform-specific branches
    import fcntl
except ImportError:  # Windows
    fcntl = None

try:  # pragma: no cover - platform-specific branches
    import msvcrt
except ImportError:  # Unix
    msvcrt = None

ENTRY_DELIMITER = "\n§\n"
_SEPARATOR = "═" * 46
_INVISIBLE_UNICODE = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
_PROCESS_LOCK = threading.RLock()


class MemoryStore(MemoryProvider):
    """Bounded curated memory with frozen prompt and live file state."""

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
        self._memory_entries: list[str] = []
        self._user_entries: list[str] = []
        self._system_prompt_snapshot = ""

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Load live entries and capture the immutable per-session snapshot."""
        self._memory_entries = self._deduplicate(self._read_file(self._memory_file))
        self._user_entries = self._deduplicate(self._read_file(self._user_file))
        self._system_prompt_snapshot = self._format_for_prompt(
            self._sanitize_for_snapshot(self._memory_entries, "MEMORY.md"),
            self._sanitize_for_snapshot(self._user_entries, "USER.md"),
        )
        logger.info(
            "成功加载历史记忆: %d 条 Agent 备忘, %d 条用户信息",
            len(self._memory_entries),
            len(self._user_entries),
        )

    def system_prompt_block(self) -> str:
        return self._system_prompt_snapshot

    def prefetch(self, query: str) -> None:
        """Built-in Hermes memory is already present in the frozen prompt."""

    def sync_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Built-in memory is curated explicitly through the memory tool."""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Expose Hermes' single action-oriented memory tool."""
        return [{"type": "function", "function": {"name": "memory", **MEMORY_SCHEMA}}]

    def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name == "memory":
            result = self._handle_memory_action(arguments)
        # Compatibility for callers using the repository's previous API. These
        # aliases are intentionally not advertised to the model.
        elif tool_name == "memory_save":
            result = self.add("memory", arguments.get("content", ""))
        elif tool_name == "user_info_save":
            result = self.add("user", arguments.get("content", ""))
        elif tool_name == "memory_read":
            result = self._legacy_read(arguments.get("store", "both"))
        elif tool_name == "memory_delete":
            result = self._legacy_delete(arguments.get("store", "memory"), arguments.get("index"))
        else:
            result = {"success": False, "error": f"Unknown memory tool: {tool_name}"}
        return json.dumps(result, ensure_ascii=False)

    def _handle_memory_action(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action", "")).strip().lower()
        target = self._normalize_target(arguments.get("target", "memory"))
        if target is None:
            return {"success": False, "error": "target must be 'memory' or 'user'."}
        if action == "add":
            return self.add(target, arguments.get("content", ""))
        if action == "replace":
            return self.replace(target, arguments.get("old_text", ""), arguments.get("content", ""))
        if action == "remove":
            return self.remove(target, arguments.get("old_text", ""))
        return {"success": False, "error": "action must be add, replace, or remove."}

    def add(self, target: str, content: str) -> dict[str, Any]:
        content = str(content or "").strip()
        error = self._validate_content(content)
        if error:
            return {"success": False, "error": error}
        with self._locked_target(target):
            entries = self._reload_target(target)
            if content in entries:
                return self._success(target, "Entry already exists; no duplicate added.")
            proposed = [*entries, content]
            overflow = self._overflow_response(target, proposed)
            if overflow:
                return overflow
            self._set_entries(target, proposed)
            self._write_file(self._path_for(target), proposed)
        return self._success(target, "Entry added. This update is complete; do not repeat it.")

    def replace(self, target: str, old_text: str, content: str) -> dict[str, Any]:
        old_text = str(old_text or "").strip()
        content = str(content or "").strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        error = self._validate_content(content)
        if error:
            return {"success": False, "error": error}
        with self._locked_target(target):
            entries = self._reload_target(target)
            match = self._unique_match(entries, old_text)
            if isinstance(match, dict):
                return match
            proposed = entries.copy()
            proposed[match] = content
            proposed = self._deduplicate(proposed)
            overflow = self._overflow_response(target, proposed)
            if overflow:
                return overflow
            self._set_entries(target, proposed)
            self._write_file(self._path_for(target), proposed)
        return self._success(target, "Entry replaced. This update is complete; do not repeat it.")

    def remove(self, target: str, old_text: str) -> dict[str, Any]:
        old_text = str(old_text or "").strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        with self._locked_target(target):
            entries = self._reload_target(target)
            match = self._unique_match(entries, old_text)
            if isinstance(match, dict):
                return match
            proposed = entries.copy()
            proposed.pop(match)
            self._set_entries(target, proposed)
            self._write_file(self._path_for(target), proposed)
        return self._success(target, "Entry removed. This update is complete; do not repeat it.")

    def _legacy_read(self, store: str) -> dict[str, Any]:
        target = self._normalize_target(store)
        if store == "both":
            return {"success": True, "memory": self._memory_entries, "user": self._user_entries}
        if target is None:
            return {"success": False, "error": "store must be agent, memory, user, or both."}
        return {"success": True, "target": target, "entries": self._entries_for(target)}

    def _legacy_delete(self, store: str, index: Any) -> dict[str, Any]:
        target = self._normalize_target(store)
        if target is None or not isinstance(index, int):
            return {"success": False, "error": "valid store and integer index are required."}
        entries = self._entries_for(target)
        if not 0 <= index < len(entries):
            return {"success": False, "error": f"index {index} out of range."}
        return self.remove(target, entries[index])

    def _reload_target(self, target: str) -> list[str]:
        entries = self._deduplicate(self._read_file(self._path_for(target)))
        self._set_entries(target, entries)
        return entries

    def _unique_match(self, entries: list[str], old_text: str) -> int | dict[str, Any]:
        matches = [(index, entry) for index, entry in enumerate(entries) if old_text in entry]
        if not matches:
            return {
                "success": False,
                "error": f"No entry matched '{old_text}'. Use text from current_entries.",
                "current_entries": entries,
            }
        distinct = {entry for _, entry in matches}
        if len(distinct) > 1:
            return {
                "success": False,
                "error": f"Multiple entries matched '{old_text}'. Use a more specific substring.",
                "matches": [entry[:100] for entry in distinct],
            }
        return matches[0][0]

    def _overflow_response(self, target: str, entries: list[str]) -> dict[str, Any] | None:
        total = self._char_count(entries)
        limit = self._limit_for(target)
        if total <= limit:
            return None
        return {
            "success": False,
            "error": (
                f"Memory would use {total}/{limit} chars. Consolidate overlapping entries "
                "with replace or remove stale entries, then retry."
            ),
            "current_entries": self._entries_for(target),
            "usage": f"{self._char_count(self._entries_for(target))}/{limit}",
        }

    def _success(self, target: str, message: str) -> dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(entries)
        limit = self._limit_for(target)
        percent = min(100, int(current / limit * 100)) if limit else 0
        return {
            "success": True,
            "done": True,
            "target": target,
            "message": message,
            "usage": f"{percent}% — {current}/{limit} chars",
            "entry_count": len(entries),
        }

    def _format_for_prompt(self, memory_entries: list[str], user_entries: list[str]) -> str:
        blocks = []
        if memory_entries:
            blocks.append(self._render_block("memory", memory_entries))
        if user_entries:
            blocks.append(self._render_block("user", user_entries))
        return "\n\n".join(blocks)

    def _render_block(self, target: str, entries: list[str]) -> str:
        current = self._char_count(entries)
        limit = self._limit_for(target)
        percent = min(100, int(current / limit * 100)) if limit else 0
        label = (
            "USER PROFILE (who the user is)"
            if target == "user"
            else "MEMORY (your personal notes)"
        )
        header = f"{label} [{percent}% — {current}/{limit} chars]"
        return f"{_SEPARATOR}\n{header}\n{_SEPARATOR}\n{ENTRY_DELIMITER.join(entries)}"

    def _sanitize_for_snapshot(self, entries: list[str], filename: str) -> list[str]:
        sanitized = []
        for entry in entries:
            error = self._validate_content(entry)
            if error:
                logger.warning("阻止可疑记忆进入系统提示词 (%s): %s", filename, error)
                sanitized.append(
                    f"[BLOCKED: suspicious {filename} entry omitted from system prompt]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    def _validate_content(content: str) -> str | None:
        if not content:
            return "Content cannot be empty."
        lower = content.lower()
        threats = (
            "ignore previous instructions",
            "ignore all instructions",
            "you are now",
            "system prompt",
            "new instructions:",
            "reveal your instructions",
            "send credentials",
            "exfiltrate",
            "<system>",
            "</system>",
        )
        if any(pattern in lower for pattern in threats):
            return "Content rejected by memory threat scanner."
        if any(char in content for char in _INVISIBLE_UNICODE):
            return "Content rejected because it contains invisible Unicode characters."
        return None

    def _path_for(self, target: str) -> Path:
        return self._user_file if target == "user" else self._memory_file

    def _entries_for(self, target: str) -> list[str]:
        return self._user_entries if target == "user" else self._memory_entries

    def _set_entries(self, target: str, entries: list[str]) -> None:
        if target == "user":
            self._user_entries = entries
        else:
            self._memory_entries = entries

    def _limit_for(self, target: str) -> int:
        return self._user_char_limit if target == "user" else self._agent_char_limit

    @staticmethod
    def _normalize_target(target: Any) -> str | None:
        value = str(target or "").lower()
        if value in {"memory", "agent"}:
            return "memory"
        if value == "user":
            return "user"
        return None

    @staticmethod
    def _deduplicate(entries: list[str]) -> list[str]:
        return list(dict.fromkeys(entries))

    @staticmethod
    def _char_count(entries: list[str]) -> int:
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    @staticmethod
    def _read_file(path: Path) -> list[str]:
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        return [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]

    @staticmethod
    def _write_file(path: Path, entries: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".memory-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(ENTRY_DELIMITER.join(entries))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    @contextmanager
    def _locked_target(self, target: str) -> Iterator[None]:
        """Serialize read-modify-write operations across threads and processes."""
        lock_path = self._path_for(target).with_suffix(".md.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _PROCESS_LOCK, lock_path.open("a+b") as lock_file:
            if msvcrt is not None:  # Windows requires a byte range to lock.
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            elif fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if msvcrt is not None:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                elif fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


MEMORY_SCHEMA = {
    "description": (
        "Manage bounded persistent memory. Proactively save durable project facts, conventions, "
        "environment details, corrections, and user preferences. Skip trivial, temporary, secret, "
        "or easily rediscovered information. Use target='user' for the user profile and "
        "target='memory' for agent notes. When full, consolidate with replace/remove; never retry "
        "a successful write."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"]},
            "target": {"type": "string", "enum": ["memory", "user"]},
            "content": {"type": "string", "description": "New entry for add/replace."},
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying an entry for replace/remove.",
            },
        },
        "required": ["action", "target"],
    },
}


# Register only the Hermes-style tool. Legacy calls remain accepted directly by
# MemoryStore.handle_tool_call but are no longer exposed to the model.
registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda **kwargs: kwargs["agent"].memory_manager.handle_tool_call(
        "memory", {key: value for key, value in kwargs.items() if key != "agent"}
    ),
)
