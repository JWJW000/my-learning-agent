"""策展器 (Curator) — 负责管理和退化 Agent 学习到的技能（Skills）的生命周期。

不同于使用系统的 cron 定时守护进程，策展器采用惰性触发机制：
  - 在 Agent 处于空闲（比如每次准备运行新对话轮次前）时进行周期性运行检测；
  - 只有当前时间与上次执行时间差大于 `interval_hours` 才会正式触发。

关键业务逻辑：
  - 范围限定：只对 Agent 自动学习创建的技能（created_by="agent"）生效，严禁干扰系统预装或插件挂载技能。
  - 安全原则：只归档（archived）不硬删除。当技能被标记为 archived 后，它不会再被加载入 system prompt 占用 token，但物理保留，可通过 `restore_skill` 工具随时恢复。
  - 状态流转机制：
    * active 状态技能：若超过 `stale_after_days` 天未曾被模型调用，退化标记为 `stale`。
    * stale 状态技能：若再过 `archive_after_days` 天仍无调用，自动进入 `archived` 归档。
  - 豁免保护：若技能 frontmatter 中的 pinned 字段被设置为 true，则无视上述一切规则。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Curator:
    """技能生命周期后台策展器。"""

    def __init__(
        self,
        skills_dir: str = "./skills",
        state_file: str = "./data/.curator_state",
        interval_hours: int = 168,
        stale_after_days: int = 30,
        archive_after_days: int = 90,
    ):
        self.skills_dir = Path(skills_dir)
        self.state_file = Path(state_file)
        self.interval_hours = interval_hours
        self.stale_after_days = stale_after_days
        self.archive_after_days = archive_after_days

        self._state = self._load_state()

    # -- 外部调用接口 ---------------------------------------------------------

    def should_run_now(self) -> bool:
        """检查策展器是否应该在当前交互中执行。

        如果是系统首次运行（.curator_state 中没有 last_run_at 记录），
        则仅将当前时间写入作为基准，并返回 False（惰性推迟，不立刻加塞耗时动作）。
        后续运行通过比对已消逝的时间差决定是否执行。
        """
        if self._state.get("paused"):
            return False

        last_run = self._state.get("last_run_at")
        if not last_run:
            # 首次：写入当前时间，延迟运行
            self._state["last_run_at"] = datetime.now().isoformat()
            self._save_state()
            return False

        elapsed = datetime.now() - datetime.fromisoformat(last_run)
        return elapsed > timedelta(hours=self.interval_hours)

    def run(self) -> dict[str, Any]:
        """正式执行策展维护。自动扫描技能目录并应用生命周期流转。

        Returns:
            dict[str, Any]: 本次策展操作的统计结果（包含退化、归档的技能列表及技能总数）。
        """
        logger.info("第 #%d 次策展服务启动", self._state.get("run_count", 0) + 1)

        summary = {
            "staled": [],
            "archived": [],
            "total_skills": 0,
        }

        # 遍历技能文件并转换状态
        transitions = self.apply_automatic_transitions()
        summary["staled"] = transitions.get("staled", [])
        summary["archived"] = transitions.get("archived", [])
        summary["total_skills"] = transitions.get("total", 0)

        # 更新运行状态并保存
        self._state["last_run_at"] = datetime.now().isoformat()
        self._state["run_count"] = self._state.get("run_count", 0) + 1
        self._save_state()

        logger.info(
            "策展完成: 总共 %d 个技能中，%d 个被标记为过期，%d 个被归档",
            summary["total_skills"],
            len(summary["staled"]),
            len(summary["archived"]),
        )
        return summary

    def apply_automatic_transitions(self) -> dict[str, Any]:
        """遍历整个技能目录，基于最后使用时间执行生命周期状态转移。

        流转逻辑：
          - active → stale: 最后使用时间(last_used)距离当前超过 stale_after_days
          - stale → archived: 距离当前时间超过 archive_after_days
          - Pinned 的技能不受此影响。
          - 仅对 created_by = "agent"（即 Agent 运行中自动生成的技能）生效，手动书写的技能不在此流转。

        Returns:
            dict[str, Any]: 本次触发转换成功的技能列表。
        """
        now = datetime.now()
        staled = []
        archived = []
        total = 0

        # 深度检索所有的 SKILL.md 技能文件
        for skill_file in self.skills_dir.rglob("SKILL.md"):
            total += 1
            try:
                fm = self._parse_frontmatter(skill_file)
                metadata = fm.get("metadata", {})

                # 豁免规则 A: 用户设定的置顶/钉住技能（pinned）跳过
                if metadata.get("pinned"):
                    continue
                # 豁免规则 B: 非 Agent 自动创建的技能（如手工技能或外部技能）不予退化
                if fm.get("created_by") not in ("agent", None):
                    continue

                current_state = metadata.get("state", "active")
                last_used = metadata.get("last_used")

                # 若从未被调用过，退回以技能的创建时间(created_at)为计算基准
                if not last_used:
                    last_used = metadata.get("created_at")
                if not last_used:
                    continue

                days_inactive = (now - datetime.fromisoformat(last_used)).days

                # 1. 活跃退化为过期状态
                if current_state == "active" and days_inactive > self.stale_after_days:
                    metadata["state"] = "stale"
                    self._write_frontmatter(skill_file, fm)
                    name = fm.get("name", skill_file.parent.name)
                    staled.append(name)
                    logger.info("技能 '%s' → stale (已闲置 %d 天)", name, days_inactive)

                # 2. 过期退化为归档状态
                elif current_state == "stale" and days_inactive > self.archive_after_days:
                    metadata["state"] = "archived"
                    self._write_frontmatter(skill_file, fm)
                    name = fm.get("name", skill_file.parent.name)
                    archived.append(name)
                    logger.info("技能 '%s' → archived (已闲置 %d 天)", name, days_inactive)

            except Exception:
                logger.exception("处理技能生命周期失败: %s", skill_file)

        return {"staled": staled, "archived": archived, "total": total}

    def restore_skill(self, name: str) -> str:
        """提供恢复归档技能的接口。将归档(archived)或过期(stale)的技能强行重置回 active 状态。"""
        for skill_file in self.skills_dir.rglob("SKILL.md"):
            fm = self._parse_frontmatter(skill_file)
            if fm.get("name") == name:
                fm.setdefault("metadata", {})["state"] = "active"
                fm["metadata"]["last_used"] = datetime.now().isoformat()
                self._write_frontmatter(skill_file, fm)
                return f"Skill '{name}' restored to active"
        return f"Skill '{name}' not found"

    def pause(self) -> None:
        """挂起自动策展任务。"""
        self._state["paused"] = True
        self._save_state()

    def resume(self) -> None:
        """恢复自动策展任务。"""
        self._state["paused"] = False
        self._save_state()

    # -- 策展状态加载与保存 ----------------------------------------------------

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("发现损坏的策展器配置文件，重置状态。")
        return {"last_run_at": None, "paused": False, "run_count": 0}

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._state, indent=2))

    # -- Frontmatter YAML 解析与回写助手 ---------------------------------------

    @staticmethod
    def _parse_frontmatter(path: Path) -> dict:
        """解析技能 markdown 文件头部的 YAML frontmatter。"""
        content = path.read_text(encoding="utf-8")
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return yaml.safe_load(parts[1]) or {}
        return {}

    @staticmethod
    def _write_frontmatter(path: Path, frontmatter: dict) -> None:
        """在保留技能 markdown 主体 Body 不变的前提下，更新头部的 YAML frontmatter 数据。"""
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
