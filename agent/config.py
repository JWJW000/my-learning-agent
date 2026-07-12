"""配置加载器。

负责从项目根目录的 config.yaml 文件加载运行时配置，并将其映射为强类型的 Python Dataclass 对象。
支持通过环境变量覆盖默认的配置文件路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """LLM 模型配置。"""
    primary: str = "gemini-3.5-flash-low"           # 用于日常交互与深度推理的主模型
    auxiliary: str = "gemini-3.5-flash-low"     # 用于上下文总结、策展分类等轻量级/异步任务的辅助模型


@dataclass
class MemoryConfig:
    """记忆持久化与限制配置。"""
    agent_char_limit: int = 2200       # MEMORY.md (Agent 记忆) 的字符上限限制，超限则按最老记录驱逐
    user_char_limit: int = 1375        # USER.md (用户信息) 的字符上限限制
    data_dir: str = "./data"           # 记忆持久化数据目录


@dataclass
class SkillsConfig:
    """技能目录配置。"""
    skills_dir: str = "./skills"       # Agent 技能文件（*.md 格式）的根存放目录


@dataclass
class CuratorConfig:
    """策展后台服务配置。"""
    interval_hours: int = 168          # 策展任务执行的周期时间隔（小时），默认 168 小时（7天）
    stale_after_days: int = 30         # 技能无任何调用时退化为过期状态（stale）的闲置阈值（天）
    archive_after_days: int = 90       # stale 状态技能进一步自动移动到归档（archived）的期限（天）


@dataclass
class ContextConfig:
    """上下文窗口大小与自动压缩配置。"""
    threshold_percent: float = 0.75    # 上下文压缩触发阈值（达到上下文窗口大小的 75% 时自动启动压缩）
    context_length: int = 128_000      # 模型的上下文最大 Token 窗口大小（默认 128k）
    protect_first_n: int = 3           # 压缩时，开头必须保留且不能被总结的消息轮数（维持 System Prompt 与核心指示）
    protect_last_n: int = 6            # 压缩时，结尾必须保留且不能被总结的消息轮数（维持最近的会话语境）


@dataclass
class StorageConfig:
    """底层会话持久化存储配置。"""
    db_path: str = "./data/state.db"   # SQLite 数据库文件路径，记录会话与对话消息


@dataclass
class AppConfig:
    """应用级全局配置对象，由各子系统配置 dataclass 组合而成。"""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    curator: CuratorConfig = field(default_factory=CuratorConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """从 YAML 配置文件中加载全局配置信息。

    若指定路径缺失，则尝试读取环境变量 `MYAGENT_CONFIG` 指向的路径，
    若依然没有指定，默认读取当前目录下的 `config.yaml`。若文件均不存在，则全部降级回退到 dataclass 的默认值。

    Args:
        config_path: 显式指定的 YAML 文件路径。

    Returns:
        AppConfig: 解析并组装完毕的强类型配置实例。
    """
    if config_path is None:
        config_path = os.environ.get("MYAGENT_CONFIG", "config.yaml")

    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    raw = yaml.safe_load(path.read_text()) or {}

    return AppConfig(
        model=ModelConfig(**raw.get("model", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
        skills=SkillsConfig(**raw.get("skills", {})),
        curator=CuratorConfig(**raw.get("curator", {})),
        context=ContextConfig(**raw.get("context", {})),
        storage=StorageConfig(**raw.get("storage", {})),
    )
