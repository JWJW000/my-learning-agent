"""tools — Agent 工具集"""

from __future__ import annotations

from tools.registry import discover_builtin_tools, registry

# Trigger tool discovery immediately on package import
discover_builtin_tools()
