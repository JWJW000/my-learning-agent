"""自我学习 Agent 的程序入口。

本模块负责组装和编排系统中的所有组件：
  Agent ← MemoryManager (记忆管理器) ← MemoryStore (内置文件记忆 Provider)
  Agent ← SkillsManager (技能管理器，渐进式加载)
  Agent ← SessionStore (SQLite 对话持久化 + FTS5 全文检索)
  Agent ← Curator (后台技能生命周期策展器)
  Agent ← ContextCompressor (自动上下文窗口压缩器)

运行方式: python main.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid

# 自动加载 .env 环境变量文件
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    # 如果未安装 python-dotenv，尝试手动解析 .env 文件
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.config import load_config
from agent.curator import Curator
from agent.learn_prompt import build_learn_prompt
from agent.memory_manager import MemoryManager
from agent.run_agent import Agent
from agent.state import SessionStore
from tools.memory_tool import MemoryStore
from tools.session_search import SessionSearch
from tools.skills_tool import SkillsManager

logger = logging.getLogger(__name__)
console = Console()


def setup_logging() -> None:
    """初始化日志配置，向控制台输出 INFO 及以上级别的日志。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    """主控函数，执行系统初始化，建立 REPL 命令行或 TUI 交互。"""
    # 检查是否请求了 TUI 模式 (通过命令行参数 --tui 或 -t)
    use_tui = "--tui" in sys.argv or "-t" in sys.argv

    if use_tui:
        # 启动 Textual TUI
        try:
            from tui_app import MyAgentTUI
            app = MyAgentTUI()
            app.run()
        except ImportError as e:
            print(f"[错误] 无法加载 TUI 组件，可能是未安装 'textual' 依赖: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[错误] TUI 启动异常: {e}", file=sys.stderr)
            sys.exit(1)
        return

    setup_logging()

    # 加载系统配置（config.yaml 或环境变量指定的 YAML 文件）
    config = load_config()

    console.print(
        Panel(
            "[bold cyan]自我学习 Agent (Self-Learning Agent)[/bold cyan]\n"
            "指令集: /learn <主题> | /skills | /search <查询> | /curator | /quit",
            title="My Learning Agent",
            border_style="cyan",
        )
    )

    # -- 1. 初始化持久化存储层 (SQLite + FTS5 全文检索) ------------------------
    store = SessionStore(db_path=config.storage.db_path)
    session_id = str(uuid.uuid4())
    store.create_session(session_id, model=config.model.primary)

    # -- 2. 初始化记忆系统 (包含 MemoryManager 编排层与内置的 MemoryStore) -------
    memory_store = MemoryStore(
        data_dir=config.memory.data_dir,
        agent_char_limit=config.memory.agent_char_limit,
        user_char_limit=config.memory.user_char_limit,
    )
    memory_manager = MemoryManager()
    memory_manager.add_provider(memory_store)
    # initialize 传入当前会话 ID，加载对应记忆快照
    memory_manager.initialize(session_id)

    # -- 3. 初始化技能系统 (管理技能 markdown 的 CRUD) ------------------------
    skills_manager = SkillsManager(skills_dir=config.skills.skills_dir)

    # -- 4. 初始化跨会话搜索组件 (基于 SQLite 连接直接进行 FTS 全文索引检索) -----
    session_search = SessionSearch(store.db)

    # -- 5. 初始化策展生命周期管理器 (自动归档过期技能) ------------------------
    curator = Curator(
        skills_dir=config.skills.skills_dir,
        state_file=str(config.memory.data_dir) + "/.curator_state",
        interval_hours=config.curator.interval_hours,
        stale_after_days=config.curator.stale_after_days,
        archive_after_days=config.curator.archive_after_days,
    )

    # -- 6. 组装 Agent 核心并注册工具 -----------------------------------------
    agent = Agent(config)
    agent.memory_manager = memory_manager
    agent.skill_manager = skills_manager
    agent.session_id = session_id

    # 将技能工具(skills_list, skill_view, skill_create 等)动态注入到 Agent 运行时中
    for schema in skills_manager.get_tool_schemas():
        func = schema["function"]
        agent._tools.append(schema)
        agent._tool_handlers[func["name"]] = lambda _tn=func["name"], **kw: (
            skills_manager.handle_tool_call(_tn, kw)
        )

    # 将跨会话搜索工具(session_search)动态注入到 Agent 运行时中
    for schema in session_search.get_tool_schemas():
        func = schema["function"]
        agent._tools.append(schema)
        agent._tool_handlers[func["name"]] = lambda _tn=func["name"], **kw: (
            session_search.handle_tool_call(_tn, kw)
        )

    # 启动时构建并冻结 System Prompt (快照模式，会话期间写入记忆不改变 prompt，保证 cache 稳定)
    agent.build_system_prompt()

    # -- 7. 启动 REPL 对话循环 ------------------------------------------------
    console.print("[dim]输入您的问题，或使用以 / 开头的系统指令。输入 /quit 退出对话。[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # -- 系统指令分支 (Slash Commands) -----------------------------------

        # 退出系统
        if user_input.lower() == "/quit":
            break

        # 自我学习提炼技能模式
        if user_input.lower().startswith("/learn "):
            topic = user_input[7:].strip()
            if topic:
                # 转换输入为内置标准的学习指令 Prompt，告诉 Agent 基于上下文去写 SKILL.md
                user_input = build_learn_prompt(topic)
                console.print("[dim]系统学习模式已激活...[/dim]")
            else:
                console.print("[yellow]用法: /learn <技能主题名称或数据源>[/yellow]")
                continue

        # 查看已拥有的技能列表 (按 active/stale/archived 状态分类展示)
        if user_input.lower() == "/skills":
            skills = skills_manager.list_skills()
            if skills:
                console.print(Panel(
                    "\n".join(
                        f"[{'green' if s['state'] == 'active' else 'yellow' if s['state'] == 'stale' else 'dim'}]"
                        f"  {s['name']} — {s['description']} (已使用 {s['use_count']} 次, 状态: {s['state']})"
                        f"[/]"
                        for s in skills
                    ),
                    title="技能库 (Skills Library)",
                    border_style="blue",
                ))
            else:
                console.print("[dim]暂未学习任何技能。请使用 /learn 来创建一个技能。[/dim]")
            continue

        # 跨会话搜索指令 (直接展示前 5 个最相关的会话内容片段)
        if user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            if query:
                results = store.search(query, limit=5)
                if results:
                    for r in results:
                        console.print(f"  [{r['role']}] {r['snippet'][:100]}")
                else:
                    console.print("[dim]未找到匹配的内容。[/dim]")
            else:
                console.print("[yellow]用法: /search <查询关键词>[/yellow]")
            continue

        # 手动执行策展维护任务
        if user_input.lower() == "/curator":
            if curator.should_run_now():
                summary = curator.run()
                console.print(f"[cyan]策展器运行完毕: {len(summary['staled'])} 个技能进入过期状态， "
                              f"{len(summary['archived'])} 个技能已归档。[/cyan]")
            else:
                console.print("[dim]策展器：尚未到达下一次计划运行时间。[/dim]")
            continue

        # -- 后台策展机制 (惰性运行) ------------------------------------------
        # 在每次与用户对话轮次前，静默检查是否到了维护周期，如是则静默执行
        if curator.should_run_now():
            summary = curator.run()
            if summary["staled"] or summary["archived"]:
                console.print(
                    f"[dim][策展服务] 自动维护了技能库: "
                    f"{len(summary['staled'])} 个进入过期状态, "
                    f"{len(summary['archived'])} 个归档。[/dim]"
                )

        # -- 执行 Agent 对话轮次 ----------------------------------------------
        try:
            # 运行 Agent 轮次（内部处理前序加载、LLM 调用、工具循环、后序存储、及上下文自动压缩）
            response = agent.run_turn(user_input)

            # 将本轮的交互记录持久化到会话数据库中
            store.save_message(session_id, "user", user_input)
            store.save_message(session_id, "assistant", response)

            # 渲染并输出 Agent 回复内容
            console.print()
            try:
                console.print(Markdown(response))
            except Exception:
                console.print(response)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]对话被用户手动中断。[/yellow]")
        except Exception as exc:
            console.print(f"[red]错误: {exc}[/red]")
            logger.exception("Agent 对话执行失败")

    # -- 清理与会话结束生命周期 ----------------------------------------------
    console.print("\n[dim]正在结束当前会话...[/dim]")
    agent.end_session()
    store.close()
    console.print("[cyan]再见！[/cyan]")


if __name__ == "__main__":
    main()
