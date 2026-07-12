"""Central registry for all my-learning-agent tools."""

from __future__ import annotations

import ast
import importlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _is_registry_register_call(node: ast.AST) -> bool:
    """Return True when *node* is a ``registry.register(...)`` call expression."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def _module_registers_tools(module_path: Path) -> bool:
    """Return True when the module contains a top-level ``registry.register(...)`` call."""
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        return False

    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """Import built-in self-registering tool modules and return their module names.

    This replaces manual tool imports in main runner paths.
    """
    tools_path = Path(tools_dir) if tools_dir is not None else Path(__file__).resolve().parent
    module_names = []

    # Scan direct modules under tools/
    for path in sorted(tools_path.glob("*.py")):
        if path.name not in {"__init__.py", "registry.py"} and _module_registers_tools(path):
            module_names.append(f"tools.{path.stem}")

    # Scan subdirectories with __init__.py under tools/ (e.g., tools/web_search)
    for path in sorted(tools_path.glob("*/__init__.py")):
        parent_dir = path.parent
        if _module_registers_tools(path):
            module_names.append(f"tools.{parent_dir.name}")

    imported: List[str] = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
    return imported


class ToolEntry:
    """Metadata for a single registered tool."""

    __slots__ = ("name", "toolset", "schema", "handler", "description")

    def __init__(self, name: str, toolset: str, schema: dict, handler: Callable, description: str = ""):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.description = description or schema.get("description", "")


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """Return a registered tool entry by name, or None."""
        return self._tools.get(name)

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        description: str = "",
    ):
        """Register a tool. Called at module-import time by each tool file."""
        self._tools[name] = ToolEntry(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            description=description,
        )

    def get_definitions(self, tool_names: Set[str]) -> List[dict]:
        """Return OpenAI-format tool schemas for the requested tool names."""
        result = []
        for name in sorted(tool_names):
            entry = self.get_entry(name)
            if not entry:
                continue
            schema_with_name = {**entry.schema, "name": entry.name}
            result.append({"type": "function", "function": schema_with_name})
        return result

    def get_all_tool_names(self) -> List[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def get_schema(self, name: str) -> Optional[dict]:
        """Return a tool's raw schema dict."""
        entry = self.get_entry(name)
        return entry.schema if entry else None

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """Execute a tool handler by name."""
        entry = self.get_entry(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = entry.handler(**args, **kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.exception("Tool %s dispatch error: %s", name, e)
            return json.dumps({"error": f"Tool execution failed: {type(e).__name__}: {e}"})


# Module-level singleton
registry = ToolRegistry()
