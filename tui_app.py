"""My Learning Agent - Textual TUI 客户端。

提供一个比 REPL 命令行更加精美和现代的终端 UI 交互界面，
主要设计要素（CodeWhale 风格）：
  - 侧边栏 (Sidebar)：展示历史会话列表与已学习的技能列表（包括 active/stale/archived 状态标记）
  - 对话显示区 (Chat Log)：使用 RichLog 配合 Markdown 渲染展现流畅优美的对话消息
  - 工具/思考状态栏 (Status Bar)：展现大模型思考中、工具调用中（如：🔧 执行中）的加载提示
  - 底部输入区 (Input Bar)：支持快捷键（如 Esc 切换焦点，Enter 发送，Ctrl+Q 退出，Ctrl+C 清空历史）
  - 异步非阻塞执行：使用 Textual Worker 异步调度同步的 Agent 运行时，防止界面死锁卡顿
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, RichLog, Static
from textual import work

from agent.config import load_config
from agent.curator import Curator
from agent.learn_prompt import build_learn_prompt
from agent.memory_manager import MemoryManager
from agent.run_agent import Agent
from agent.state import SessionStore
from tools.memory_tool import MemoryStore
from tools.session_search import SessionSearch
from tools.skills_tool import SkillsManager

# 自动加载 .env 环境变量文件
import os
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

# 禁用全局 root logger 打印到 stdout，避免污染 Textual 界面
logging.basicConfig(level=logging.WARNING, handlers=[logging.FileHandler("tui.log", encoding="utf-8")])
logger = logging.getLogger("tui_app")
logger.setLevel(logging.INFO)


class SkillListItem(ListItem):
    """技能列表中的行项，带有状态样式"""

    def __init__(self, skill_name: str, desc: str, state: str, use_count: int) -> None:
        super().__init__()
        self.skill_name = skill_name
        self.desc = desc
        self.state = state
        self.use_count = use_count

    def compose(self) -> ComposeResult:
        state_colors = {
            "active": "green",
            "stale": "yellow",
            "archived": "dim",
        }
        color = state_colors.get(self.state, "white")
        label_text = f"[{self.state[:3].upper()}] {self.skill_name}"
        yield Label(Text(label_text, style=f"bold {color}"))
        yield Label(Text(f" {self.desc[:25]}...", style="dim italic"))


class SessionListItem(ListItem):
    """会话列表中的行项"""

    def __init__(self, session_id: str, summary: str | None, created_at: float) -> None:
        super().__init__()
        self.session_id = session_id
        dt = datetime.fromtimestamp(created_at).strftime("%m-%d %H:%M")
        self.display_summary = summary or f"会话 {dt}"
        self.dt_str = dt

    def compose(self) -> ComposeResult:
        yield Label(Text(f"💬 {self.display_summary[:20]}", style="bold cyan"))
        yield Label(Text(f"   {self.dt_str}", style="dim size-9"))


class MyAgentTUI(App):
    """My Learning Agent TUI 主程序"""

    TITLE = "My Learning Agent"
    SUBTITLE = "Self-Learning Agent with Persistent Memory & TUI"

    # 定义键盘快捷键
    BINDINGS = [
        Binding("ctrl+q", "quit", "退出 TUI", show=True),
        Binding("escape", "toggle_focus", "焦点切换", show=True),
        Binding("ctrl+c", "clear_chat", "清空聊天", show=True),
        Binding("ctrl+r", "refresh_sidebar", "刷新侧边栏", show=True),
    ]

    # 定义精美的主题 CSS 样式
    CSS = """
    Screen {
        background: #1e1e2e;
        color: #cdd6f4;
    }

    #sidebar {
        width: 32;
        background: #181825;
        border-right: solid #45475a;
        padding: 1;
    }

    .sidebar-title {
        text-align: center;
        background: #313244;
        color: #89b4fa;
        padding: 1;
        margin-bottom: 1;
        text-style: bold;
    }

    .sidebar-section-header {
        color: #f5c2e7;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
        padding-left: 1;
        border-bottom: solid #45475a;
    }

    #session_list, #skill_list {
        background: transparent;
        border: none;
        height: 1fr;
        margin-bottom: 1;
        scrollbar-size-vertical: 1;
    }

    ListItem {
        padding: 0 1;
        background: transparent;
    }

    ListItem:hover {
        background: #2b2b3c;
    }

    ListItem.--focus {
        background: #313244;
    }

    #chat_area {
        height: 1fr;
        padding: 1;
    }

    #chat_log {
        background: #1e1e2e;
        border: solid #45475a;
        height: 1fr;
        padding: 1;
    }

    #status_bar {
        height: 1;
        background: #11111b;
        color: #a6e3a1;
        padding: 0 2;
        text-style: italic;
    }

    #input_area {
        height: auto;
        margin: 1 1 0 1;
    }

    #user_input {
        background: #181825;
        border: tall #89b4fa;
        color: #cdd6f4;
    }

    #user_input:focus {
        border: tall #a6e3a1;
    }
    """

    # 反应式状态：展现当前的运行状态，会自动刷新底部的状态栏
    status_text = reactive("Idle")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化核心组件，类似 main.py
        self.config = load_config()
        self.store = SessionStore(db_path=self.config.storage.db_path)
        self.session_id = str(uuid.uuid4())
        self.store.create_session(self.session_id, model=self.config.model.primary)

        # 初始化记忆
        self.memory_store = MemoryStore(
            data_dir=self.config.memory.data_dir,
            agent_char_limit=self.config.memory.agent_char_limit,
            user_char_limit=self.config.memory.user_char_limit,
        )
        self.memory_manager = MemoryManager()
        self.memory_manager.add_provider(self.memory_store)
        self.memory_manager.initialize(self.session_id)

        # 初始化技能与搜索
        self.skills_manager = SkillsManager(skills_dir=self.config.skills.skills_dir)
        self.session_search = SessionSearch(self.store.db)

        # 初始化策展器
        self.curator = Curator(
            skills_dir=self.config.skills.skills_dir,
            state_file=str(self.config.memory.data_dir) + "/.curator_state",
            interval_hours=self.config.curator.interval_hours,
            stale_after_days=self.config.curator.stale_after_days,
            archive_after_days=self.config.curator.archive_after_days,
        )

        # 组装 Agent 核心
        self.agent = Agent(self.config)
        self.agent.memory_manager = self.memory_manager
        self.agent.skill_manager = self.skills_manager
        self.agent.session_id = self.session_id

        # 绑定工具集到 Agent 中，并进行包装以便捕获状态
        self._register_tui_tools()

        # 初始化 System Prompt 快照
        self.agent.build_system_prompt()

    def _register_tui_tools(self) -> None:
        """注册并用包装函数代理 Tool 调用以触发状态显示"""
        # 1. 注册技能管理工具
        for schema in self.skills_manager.get_tool_schemas():
            func_name = schema["function"]["name"]
            self.agent._tools.append(schema)
            # 使用闭包包裹，以实现对 UI 状态的实时更新
            self.agent._tool_handlers[func_name] = self._make_tool_proxy(
                func_name, self.skills_manager.handle_tool_call
            )

        # 2. 注册会话搜索工具
        for schema in self.session_search.get_tool_schemas():
            func_name = schema["function"]["name"]
            self.agent._tools.append(schema)
            self.agent._tool_handlers[func_name] = self._make_tool_proxy(
                func_name, self.session_search.handle_tool_call
            )

    def _make_tool_proxy(self, name: str, original_handler: callable):
        """生成 Tool Proxy 闭包，使得在 Worker 线程调用工具时，能安全更新主线程的 UI 状态"""
        def proxy(**kwargs):
            # 将 UI 状态标记更新任务派发到主线程执行
            self.call_from_thread(self.set_status, f"🔧 正在执行工具: {name}...")
            try:
                res = original_handler(name, kwargs)
            except Exception as e:
                res = original_handler(name, **kwargs) if hasattr(original_handler, "__code__") else str(e)
            self.call_from_thread(self.set_status, "🤖 Agent 正在思考...")
            return res
        return proxy

    def set_status(self, text: str) -> None:
        """主线程更新状态的入口"""
        self.status_text = text

    def watch_status_text(self, old_value: str, new_value: str) -> None:
        """状态变化时自动更新 Widget"""
        status_bar = self.query_one("#status_bar", Static)
        if status_bar:
            status_bar.update(f" 状态: {new_value}")

    def compose(self) -> ComposeResult:
        """渲染整体页面布局"""
        yield Header(show_clock=True)
        yield Horizontal(
            # 左侧侧边栏
            Vertical(
                Label("My Learning Agent", classes="sidebar-title"),
                Label("会话历史 (点击载入)", classes="sidebar-section-header"),
                ListView(id="session_list"),
                Label("技能列表 (点击预览)", classes="sidebar-section-header"),
                ListView(id="skill_list"),
                id="sidebar"
            ),
            # 右侧对话区
            Vertical(
                RichLog(id="chat_log", highlight=True, markup=True, wrap=True),
                Static(" 状态: Idle", id="status_bar"),
                Vertical(
                    Input(placeholder="输入您的问题，Esc 切换焦点，Ctrl+Q 退出...", id="user_input"),
                    id="input_area"
                ),
                id="chat_area"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        """界面挂载后的初始化工作"""
        self.refresh_sidebar_data()
        self.query_one("#user_input", Input).focus()

        # 显示欢迎消息
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.write(
            Panel(
                Text.from_markup(
                    "[bold cyan]🤖 自我学习 Agent TUI 客户端已成功启动！[/bold cyan]\n"
                    "在此你可以发起对话，Agent 在回答过程中会分析你的行为并可能保存为 [green]MEMORY[/green] 或提炼为 [magenta]SKILL[/magenta]。\n"
                    "快捷键: [bold yellow]Esc[/bold yellow] 切换输入焦点 | "
                    "[bold yellow]Ctrl+C[/bold yellow] 清空本屏 | "
                    "[bold yellow]Ctrl+R[/bold yellow] 刷新侧边栏 | "
                    "[bold yellow]Ctrl+Q[/bold yellow] 退出程序\n"
                    "左侧支持点击载入历史会话，或点击预览技能细节。"
                ),
                title="系统就绪",
                border_style="cyan"
            )
        )

    def refresh_sidebar_data(self) -> None:
        """刷新侧边栏的历史会话列表和已学技能列表"""
        # 1. 载入会话列表
        session_list = self.query_one("#session_list", ListView)
        session_list.clear()
        try:
            sessions = self.session_search.browse(limit=15)
            # 将当前会话追加进列表顶部（如果未被 browse 返回）
            current_in_list = any(s["session_id"] == self.session_id for s in sessions)
            if not current_in_list:
                session_list.append(SessionListItem(self.session_id, "当前会话 (活动中)", datetime.now().timestamp()))

            for s in sessions:
                if s["session_id"] != self.session_id:
                    session_list.append(SessionListItem(s["session_id"], s["summary"], s["created_at"]))
        except Exception as e:
            logger.error("加载会话历史失败: %s", e)

        # 2. 载入技能列表
        skill_list = self.query_one("#skill_list", ListView)
        skill_list.clear()
        try:
            skills = self.skills_manager.list_skills()
            for s in skills:
                skill_list.append(SkillListItem(s["name"], s.get("description", "无描述"), s["state"], s.get("use_count", 0)))
        except Exception as e:
            logger.error("加载技能列表失败: %s", e)

    # -- 快捷指令与键盘绑定动作实现 ---------------------------------------------

    def action_toggle_focus(self) -> None:
        """在输入框和聊天区域间切换焦点"""
        user_input = self.query_one("#user_input", Input)
        if user_input.has_focus:
            self.query_one("#chat_log", RichLog).focus()
        else:
            user_input.focus()

    def action_clear_chat(self) -> None:
        """清空当前的屏幕对话记录"""
        self.query_one("#chat_log", RichLog).clear()

    def action_refresh_sidebar(self) -> None:
        """手动刷新侧边栏"""
        self.refresh_sidebar_data()
        self.notify("侧边栏已刷新", title="系统提示", severity="information")

    def action_quit(self) -> None:
        """安全退出 TUI 客户端并执行 Agent 销毁钩子"""
        self.set_status("正在保存会话并清理资源...")
        try:
            self.agent.end_session()
            self.store.close()
        except Exception as e:
            logger.error("结束会话清理失败: %s", e)
        self.exit()

    # -- 消息提交与异步处理 Worker ---------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """用户敲击回车提交输入框内容时触发"""
        user_text = event.value.strip()
        if not user_text:
            return

        # 清空输入框
        event.input.value = ""

        # 特殊 slash 指令拦截逻辑 (TUI 原生集成)
        if user_text.lower() == "/quit":
            self.action_quit()
            return
        elif user_text.lower() == "/skills":
            self._render_skills_info()
            return
        elif user_text.lower() == "/curator":
            self._run_curator_manually()
            return
        elif user_text.lower().startswith("/search "):
            query = user_text[8:].strip()
            self._run_fts_search(query)
            return

        # 追加用户消息到屏幕
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.write(Text.from_markup(f"\n[bold green]You:[/bold green] {user_text}"))

        # 判断是否为学习指令
        final_input = user_text
        if user_text.lower().startswith("/learn "):
            topic = user_text[7:].strip()
            if topic:
                final_input = build_learn_prompt(topic)
                chat_log.write(Text.from_markup("[dim]💡 系统已激活学习模式，正在提炼经验，请稍候...[/dim]"))
            else:
                chat_log.write(Text.from_markup("[yellow]⚠️ 用法: /learn <技能主题名称或数据源>[/yellow]"))
                return

        # 唤醒后台 Worker 进行大模型请求，避免死锁 UI
        self.set_status("🤖 Agent 正在思考...")
        self.run_turn_worker(final_input, user_text)

    @work(thread=True)
    def run_turn_worker(self, final_input: str, raw_input: str) -> None:
        """在后台工作线程执行模型调用"""
        try:
            # 1. 惰性触发后台技能生命周期维护
            if self.curator.should_run_now():
                self.call_from_thread(self.set_status, "🧹 正在自动维护技能生命周期...")
                summary = self.curator.run()
                if summary["staled"] or summary["archived"]:
                    info = f"自动维护技能库: {len(summary['staled'])}个过期, {len(summary['archived'])}个归档"
                    self.call_from_thread(self._write_sys_msg, info)

            # 2. 执行核心对话
            self.call_from_thread(self.set_status, "🤖 Agent 正在思考...")
            response = self.agent.run_turn(final_input)

            # 3. 数据入库归档
            self.store.save_message(self.session_id, "user", raw_input)
            self.store.save_message(self.session_id, "assistant", response)

            # 4. 回写回复文本到 UI 并刷新侧边栏
            self.call_from_thread(self._render_agent_response, response)
            self.call_from_thread(self.refresh_sidebar_data)
        except Exception as exc:
            logger.exception("Agent 对话执行异常")
            self.call_from_thread(self._write_error_msg, f"对话执行出错: {exc}")
        finally:
            self.call_from_thread(self.set_status, "Idle")

    # -- 主线程 UI 渲染方法 ---------------------------------------------------

    def _write_sys_msg(self, msg: str) -> None:
        """在屏幕输出一条灰色系统消息"""
        self.query_one("#chat_log", RichLog).write(Text(f"[系统通知] {msg}", style="dim italic"))

    def _write_error_msg(self, msg: str) -> None:
        """在屏幕输出红色异常消息"""
        self.query_one("#chat_log", RichLog).write(Text(f"❌ {msg}", style="bold red"))

    def _render_agent_response(self, response: str) -> None:
        """在屏幕完美排版渲染 Agent 的 Markdown 回复"""
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.write(Text.from_markup("[bold cyan]Agent:[/bold cyan]"))
        chat_log.write(Markdown(response))
        chat_log.write("")  # 换行

    # -- 会话侧边栏交互响应事件 -------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """当侧边栏列表项目被选中时触发"""
        if not event.item:
            return

        # 选中历史会话
        if event.list_view.id == "session_list" and isinstance(event.item, SessionListItem):
            selected_item: SessionListItem = event.item
            self._load_historical_session(selected_item.session_id, selected_item.display_summary)

        # 选中技能条目 (预览技能 Markdown 正文)
        elif event.list_view.id == "skill_list" and isinstance(event.item, SkillListItem):
            selected_item: SkillListItem = event.item
            self._preview_skill_details(selected_item.skill_name)

    def _load_historical_session(self, target_session_id: str, summary: str) -> None:
        """加载并平铺输出历史会话的全部消息"""
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.clear()
        chat_log.write(Text(f"⏳ 正在载入会话 [{summary}] 的历史详情...", style="yellow"))

        try:
            # 载入前 100 条消息
            messages = self.store.get_messages(target_session_id, limit=100)
            chat_log.clear()
            chat_log.write(Panel(Text(f"历史穿越：会话 {target_session_id}\n摘要: {summary}", style="bold green"), border_style="green"))

            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    chat_log.write(Text.from_markup(f"\n[bold green]You:[/bold green] {content}"))
                elif role == "assistant":
                    chat_log.write(Text.from_markup("[bold cyan]Agent:[/bold cyan]"))
                    chat_log.write(Markdown(content))
                    chat_log.write("")

            # 自动滚动到底部
            chat_log.scroll_end()
            self.notify(f"已成功加载历史会话", title="穿越成功", severity="information")
        except Exception as e:
            self._write_error_msg(f"载入历史会话失败: {e}")

    def _preview_skill_details(self, skill_name: str) -> None:
        """加载并在聊天区预览技能的 markdown 操作手册"""
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.write(Text(f"🔍 正在读取技能 '{skill_name}' 手册...", style="magenta"))

        try:
            # 调用技能管理器的 view_skill 会自动递增使用次数
            content = self.skills_manager.view_skill(skill_name)
            if content:
                chat_log.write(
                    Panel(
                        Markdown(content),
                        title=f"技能预览: {skill_name}",
                        border_style="magenta",
                    )
                )
                chat_log.scroll_end()
                self.refresh_sidebar_data()  # 刷新计数
            else:
                self._write_error_msg(f"技能 '{skill_name}' 未找到具体文档文件。")
        except Exception as e:
            self._write_error_msg(f"预览技能详情出错: {e}")

    # -- 辅助工具与 Slash 指令在 TUI 上的适配实现 ---------------------------------

    def _render_skills_info(self) -> None:
        """以 Rich 格式在聊天区展示技能面板"""
        skills = self.skills_manager.list_skills()
        chat_log = self.query_one("#chat_log", RichLog)
        if not skills:
            chat_log.write(Text("💡 暂未学习任何技能。你可以使用 /learn <主题> 创建技能。", style="dim"))
            return

        lines = []
        for s in skills:
            color = "green" if s["state"] == "active" else "yellow" if s["state"] == "stale" else "dim"
            lines.append(
                f"[{color}]• {s['name']}[/] — {s['description']} (使用: {s['use_count']}次, 状态: {s['state']})"
            )
        chat_log.write(
            Panel(
                Text.from_markup("\n".join(lines)),
                title="技能库 (Skills Library)",
                border_style="blue",
            )
        )

    def _run_curator_manually(self) -> None:
        """手动调度策展器并输出结果"""
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.write(Text("🧹 正在手动触发技能策展清理...", style="cyan"))
        try:
            if self.curator.should_run_now():
                summary = self.curator.run()
                info = f"策展运行完毕: {len(summary['staled'])}个技能过期, {len(summary['archived'])}个已物理归档。"
                chat_log.write(Text(info, style="bold green"))
            else:
                chat_log.write(Text("策展期未到 (冷却中)。", style="dim"))
        except Exception as e:
            self._write_error_msg(f"策展运行失败: {e}")

    def _run_fts_search(self, query: str) -> None:
        """执行全文检索并将结果写入聊天区"""
        chat_log = self.query_one("#chat_log", RichLog)
        if not query:
            chat_log.write(Text("⚠️ 用法: /search <查询关键词>", style="yellow"))
            return

        chat_log.write(Text(f"🔎 全局搜索关键词: '{query}'...", style="yellow"))
        try:
            results = self.store.search(query, limit=5)
            if results:
                lines = []
                for r in results:
                    lines.append(
                        f"[cyan][{r['role'].upper()}][/cyan] ({r['created_at']})\n"
                        f"  [dim]{r['snippet']}[/dim]"
                    )
                chat_log.write(
                    Panel(
                        Text.from_markup("\n\n".join(lines)),
                        title=f"搜索结果: {query}",
                        border_style="yellow",
                    )
                )
            else:
                chat_log.write(Text("未找到任何相关的历史消息匹配项。", style="dim"))
        except Exception as e:
            self._write_error_msg(f"检索出错: {e}")


def main() -> None:
    """TUI 主入口"""
    app = MyAgentTUI()
    app.run()


if __name__ == "__main__":
    main()
