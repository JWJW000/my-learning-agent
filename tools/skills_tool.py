"""技能工具 — 实现技能目录的管理（CRUD）与大模型的渐进式加载（Progressive Disclosure）。

架构设计要点：
  1. 渐进式加载机制（极低 token 消耗实现海量技能管理）：
     - 初始状态（第 1 层）：仅向大模型的 System Prompt 注入技能列表的索引元数据（技能名及一行简介说明）。
     - 调用状态（第 2 层）：当大模型阅读提示词后，觉得某项技能与其面临的问题高度契合时，
       会主动发起工具调用 `skill_view` 请求查看该技能的具体 Markdown 执行指令（大步骤、代码块与反思预案）。
       这一双层机制避免了一股脑往 prompt 里填充巨量 markdown 导致 token 浪费和上下文失焦。
  2. 基于磁盘文件修改时间(mtime)的缓存失效签名机制：
     为防止每次获取列表都遍历读取磁盘 md 文件头部带来的 IO 惩罚，
     使用基于路径与 `st_mtime` 拼接组合出的 `_cache_signature` 作为校验指纹。
     只有检测到磁盘文件变更时才重新解析，在常规 REPL 多 turn 轮询中实现内存高命中读。
  3. 安全归档机制：
     删除操作（delete_skill）不会物理擦除技能文件，仅在 YAML Frontmatter 中将其 `state` 标记为 `archived`。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class SkillsManager:
    """管理技能文档并对大模型实施渐进式渲染加载的核心类。"""

    def __init__(self, skills_dir: str = "./skills"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存区：用于暂存技能元数据避免重复读盘 IO — 结构为: {skill_name: metadata_dict}
        self._cache: dict[str, dict] = {}
        # 缓存签名，根据所有 SKILL.md 文件的最新 mtime 动态拼接计算出
        self._cache_signature: str = ""

    # -- 供大模型直接调用的只读工具接口 (Exposed Tools) ---------------------------

    def list_skills(self, state: str | None = None) -> list[dict[str, Any]]:
        """获取所有可用技能的元数据列表（不加载完整 Body，用于首层发现）。

        Args:
            state: 可根据状态过滤，如 "active", "stale", "archived"。默认列出所有。
        """
        self._refresh_cache()
        skills = list(self._cache.values())
        if state:
            skills = [s for s in skills if s.get("state") == state]
        return skills

    def view_skill(self, name: str) -> str | None:
        """加载特定技能文件的完整 markdown 说明书（第 2 层按需加载），并递增调用计数。"""
        self._refresh_cache()
        meta = self._cache.get(name)
        if not meta:
            return None

        path = Path(meta["path"])
        if not path.exists():
            return None

        # 更新调用计数，更新后该技能的 last_used 戳被刷新，防退化计时器会被归零
        self._bump_usage(path)
        return path.read_text(encoding="utf-8")

    def get_skills_summary(self) -> str:
        """生成供 Agent 初始 System Prompt 注入的技能摘要索引块。

        这是渐进式加载的基石。仅仅显示可用技能名称和用途描述，告知 Agent “你拥有这些本领，
        当你想用的时候，请调用 skill_view 工具获取详细流程指令”。
        """
        skills = self.list_skills(state="active")
        if not skills:
            return ""

        lines = ["## Available Skills (use skill_view to see full content)"]
        for s in skills:
            lines.append(f"- **{s['name']}**: {s.get('description', 'No description')}")
        return "\n".join(lines)

    # -- 技能的 CRUD 写入管理层 -------------------------------------------------

    def create_skill(
        self,
        name: str,
        description: str,
        content: str,
        category: str = "general",
    ) -> str:
        """从工作流经验中学习，并提炼保存一个全新的可复用技能。

        Args:
            name: 技能小写短横线连接名称，长度限制在 64 字以内。
            description: 核心简短概括，限制在 1024 字以内（系统提示词摘要只提取前 60 字）。
            content: 具体的 Markdown 操作指导正文。
            category: 物理存放的子目录类别名称。
        """
        # 校验名称，转换多余的空格为空格符，转换为全小写
        name = name.strip().lower().replace(" ", "-")
        if len(name) > 64:
            return f"Error: name too long ({len(name)} > 64 chars)"

        skill_dir = self.skills_dir / category / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        # 构造规范的 Frontmatter，Created_by 设为 'agent' 用于策展时的生命周期退化比对
        frontmatter = {
            "name": name,
            "description": description[:1024],
            "version": 1,
            "created_by": "agent",
            "platforms": ["all"],
            "metadata": {
                "use_count": 0,
                "last_used": None,
                "state": "active",
                "pinned": False,
                "created_at": datetime.now().isoformat(),
            },
        }

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            f"---\n{yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False)}---\n\n{content}",
            encoding="utf-8",
        )

        # 清除当前缓存，以保证下一次 list 操作时重新扫描
        self._cache.clear()
        logger.info("创建新技能成功: %s/%s", category, name)
        return f"Skill '{name}' created at {skill_file}"

    def edit_skill(self, name: str, new_content: str) -> str:
        """修改指定名称技能的具体 markdown 操作 Body 正文，原原本本保留原有的 Frontmatter 设置。"""
        self._refresh_cache()
        meta = self._cache.get(name)
        if not meta:
            return f"Error: skill '{name}' not found"

        path = Path(meta["path"])
        raw = path.read_text(encoding="utf-8")

        # 抽离 frontmatter
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            fm_text = parts[1]
            path.write_text(f"---{fm_text}---\n\n{new_content}", encoding="utf-8")
        else:
            path.write_text(new_content, encoding="utf-8")

        self._cache.clear()
        return f"Skill '{name}' updated"

    def delete_skill(self, name: str) -> str:
        """将选定技能的 state 置为 archived 进行逻辑删除归档，规避物理硬删除导致的信息意外丢失。"""
        self._refresh_cache()
        meta = self._cache.get(name)
        if not meta:
            return f"Error: skill '{name}' not found"

        path = Path(meta["path"])
        fm = self._parse_frontmatter(path)
        fm.setdefault("metadata", {})["state"] = "archived"
        self._write_frontmatter(path, fm)
        self._cache.clear()
        return f"Skill '{name}' archived (recoverable)"

    # -- 内存缓存管理逻辑 ------------------------------------------------------

    # -- 内存缓存管理逻辑 ------------------------------------------------------

    def _refresh_cache(self) -> None:
        """检查磁盘上的文件修改签名，如果有文件变动，则重新读取并重构内存缓存。"""
        sig = self._compute_signature()
        if sig == self._cache_signature and self._cache:
            return

        self._cache.clear()
        for skill_file in self.skills_dir.rglob("SKILL.md"):
            try:
                meta = self._parse_frontmatter(skill_file)
                name = meta.get("name", skill_file.parent.name)
                category = skill_file.parent.parent.name
                metadata = meta.get("metadata", {})
                self._cache[name] = {
                    "name": name,
                    "description": meta.get("description", ""),
                    "state": metadata.get("state", "active"),
                    "use_count": metadata.get("use_count", 0),
                    "category": category,
                    "pinned": metadata.get("pinned", False),
                    "path": str(skill_file),
                }
            except Exception:
                logger.exception("解析技能 Frontmatter 失败: %s", skill_file)

        self._cache_signature = sig

    def _compute_signature(self) -> str:
        """组合全部 SKILL.md 文件的修改时间，算出唯一的时效印记。"""
        mtimes = []
        for f in self.skills_dir.rglob("SKILL.md"):
            mtimes.append(f"{f}:{f.stat().st_mtime}")
        return "|".join(sorted(mtimes))

    # -- YAML 头部 Frontmatter 数据存取助手 --------------------------------------

    @staticmethod
    def _parse_frontmatter(path: Path) -> dict:
        content = path.read_text(encoding="utf-8")
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return yaml.safe_load(parts[1]) or {}
        return {}

    @staticmethod
    def _write_frontmatter(path: Path, frontmatter: dict) -> None:
        content = path.read_text(encoding="utf-8")
        if content.startswith("---"):
            parts = content.split("---", 2)
            body = parts[2] if len(parts) >= 3 else ""
        else:
            body = content
        path.write_text(
            f"---\n{yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False)}---{body}",
            encoding="utf-8",
        )

    def _bump_usage(self, path: Path) -> None:
        """更新最后一次调用时间，计数 +1。若处于过期（stale）状态则自动重置回 active 状态。"""
        from datetime import datetime

        fm = self._parse_frontmatter(path)
        fm.setdefault("metadata", {})
        fm["metadata"]["use_count"] = fm["metadata"].get("use_count", 0) + 1
        fm["metadata"]["last_used"] = datetime.now().isoformat()
        fm["metadata"]["state"] = "active"  # 发生调用自动复活回活跃状态
        self._write_frontmatter(path, fm)
        self._cache.clear()


# -- 自动注册至通用工具中心 ----------------------------------------------------
from tools.registry import registry

SKILLS_LIST_SCHEMA = {
    "description": "List available skills (metadata only). Use skill_view to see full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": ["active", "stale", "archived", "all"],
                "description": "Filter by state (default: active)",
            },
        },
    },
}

SKILL_VIEW_SCHEMA = {
    "description": "Load full content of a skill by name.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name"},
        },
        "required": ["name"],
    },
}

SKILL_CREATE_SCHEMA = {
    "description": "Create a new reusable skill from experience. Include clear steps, examples, and notes. Use kebab-case name.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name (kebab-case, ≤64 chars)"},
            "description": {"type": "string", "description": "One-line description"},
            "content": {"type": "string", "description": "Markdown body with steps, examples, notes"},
            "category": {"type": "string", "description": "Category directory (default: general)"},
        },
        "required": ["name", "description", "content"],
    },
}

SKILL_DELETE_SCHEMA = {
    "description": "Archive a skill (recoverable, never hard-deletes).",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name to archive"},
        },
        "required": ["name"],
    },
}

# 内部处理函数逻辑，原先在 handle_tool_call 里
def _handle_skills_list(state=None, **kwargs):
    import json
    agent = kwargs["agent"]
    skills = agent.skill_manager.list_skills(state=state)
    cleaned_skills = []
    for s in skills:
        s_copy = dict(s)
        s_copy.pop("path", None)
        cleaned_skills.append(s_copy)
    return json.dumps(cleaned_skills, ensure_ascii=False, indent=2)

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=_handle_skills_list,
)

registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=lambda name, **kwargs: kwargs["agent"].skill_manager.view_skill(name) or f"Skill '{name}' not found",
)

registry.register(
    name="skill_create",
    toolset="skills",
    schema=SKILL_CREATE_SCHEMA,
    handler=lambda name, description, content, category="general", **kwargs: kwargs["agent"].skill_manager.create_skill(
        name=name, description=description, content=content, category=category
    ),
)

registry.register(
    name="skill_delete",
    toolset="skills",
    schema=SKILL_DELETE_SCHEMA,
    handler=lambda name, **kwargs: kwargs["agent"].skill_manager.delete_skill(name),
)



# -- 自动注册至通用工具中心 ----------------------------------------------------
from tools.registry import registry

SKILLS_LIST_SCHEMA = {
    "description": "List available skills (metadata only). Use skill_view to see full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": ["active", "stale", "archived", "all"],
                "description": "Filter by state (default: active)",
            },
        },
    },
}

SKILL_VIEW_SCHEMA = {
    "description": "Load full content of a skill by name.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name"},
        },
        "required": ["name"],
    },
}

SKILL_CREATE_SCHEMA = {
    "description": "Create a new reusable skill from experience. Include clear steps, examples, and notes. Use kebab-case name.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name (kebab-case, ≤64 chars)"},
            "description": {"type": "string", "description": "One-line description"},
            "content": {"type": "string", "description": "Markdown body with steps, examples, notes"},
            "category": {"type": "string", "description": "Category directory (default: general)"},
        },
        "required": ["name", "description", "content"],
    },
}

SKILL_DELETE_SCHEMA = {
    "description": "Archive a skill (recoverable, never hard-deletes).",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name to archive"},
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=lambda state=None, **kwargs: kwargs["agent"].skill_manager.handle_tool_call("skills_list", {"state": state}),
)

registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=lambda name, **kwargs: kwargs["agent"].skill_manager.handle_tool_call("skill_view", {"name": name}),
)

registry.register(
    name="skill_create",
    toolset="skills",
    schema=SKILL_CREATE_SCHEMA,
    handler=lambda name, description, content, category="general", **kwargs: kwargs["agent"].skill_manager.handle_tool_call(
        "skill_create", {"name": name, "description": description, "content": content, "category": category}
    ),
)

registry.register(
    name="skill_delete",
    toolset="skills",
    schema=SKILL_DELETE_SCHEMA,
    handler=lambda name, **kwargs: kwargs["agent"].skill_manager.handle_tool_call("skill_delete", {"name": name}),
)

