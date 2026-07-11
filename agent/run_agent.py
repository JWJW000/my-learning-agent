"""Agent 的主对话循环 — 系统核心会话执行引擎。

负责管理整个单次交互周期（Turn），核心处理链路包含：
  - 系统提示词动态组装（从记忆存储和技能管理器中合并信息）
  - 工具的动态注册与分发机制
  - 每一轮对话之前的记忆预取（Prefetch）与对话结束后的记忆同步写入（Sync）
  - 上下文长度趋近上限时的自动分段压缩
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable

import openai

from agent.config import AppConfig
from agent.context_compressor import ContextCompressor
from agent.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class Agent:
    """具备工具调用循环、持久记忆整合、技能按需调用的自我学习 Agent。"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.session_id = str(uuid.uuid4())

        # 惰性初始化 OpenAI 客户端（避免模块 import 时加载过多 SDK 代码消耗约 240ms 启动开销）
        self._client: openai.OpenAI | None = None

        # 会话历史消息缓冲
        self.messages: list[dict[str, Any]] = []

        # 注册的工具定义和对应的 Python 回调映射表
        self._tools: list[dict[str, Any]] = []
        self._tool_handlers: dict[str, Callable] = {}

        # 外部子系统引用（构建后由 main.py 动态注入）
        self.memory_manager: MemoryManager | None = None
        self.skill_manager = None  # 在 main.py 中实例化后被挂载到此处

        # 上下文窗口压缩处理器
        self.compressor = ContextCompressor(
            auxiliary_model=config.model.auxiliary,
            threshold_percent=config.context.threshold_percent,
            context_length=config.context.context_length,
            protect_first_n=config.context.protect_first_n,
            protect_last_n=config.context.protect_last_n,
        )

        # 冻结的系统提示词 snapshot。在会话开始阶段生成一次后保持只读，
        # 即使对话中途写入了新记忆，当前 snapshot 也不会改变。这样做能够最大化发挥 Prompt Cache 的优势，
        # 并防止大模型由于中途 prompt 漂移发生偏离。
        self._system_prompt_snapshot: str | None = None

    @property
    def client(self) -> openai.OpenAI:
        """获取或惰性创建 OpenAI 客户端。"""
        if self._client is None:
            # 某些中转 API 的 Web 防火墙 (WAF) 拦截了官方 OpenAI Python SDK 默认的 User-Agent 标头 (以 OpenAI/Python 开头)，
            # 导致返回 403 / "Your request was blocked."。这里通过 default_headers 将其覆写为常见的 requests/http 标头。
            self._client = openai.OpenAI(
                default_headers={"User-Agent": "python-requests/2.31.0"}
            )
        return self._client

    # -- 工具注册机制 --------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable,
    ) -> None:
        """向 Agent 注册一个可供 LLM 选择并执行的 Python 外部工具。

        Args:
            name: 工具函数名称（即大模型返回中 tool_calls 的函数名称）。
            description: 详细工具功能描述，这是大模型进行语义路由的核心参考。
            parameters: 基于 JSON Schema 的参数定义规范（类型、必填项等）。
            handler: 工具被调用时实际执行的 Python 可调用对象（如函数或方法）。
        """
        self._tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
        self._tool_handlers[name] = handler
        logger.debug("已注册工具: %s", name)

    # -- 系统提示词组装 ------------------------------------------------------

    def build_system_prompt(self) -> str:
        """从底座配置、静态记忆数据和技能树元数据中组装系统提示词。

        在会话初始启动时或首个 token 请求前调用一次。完成组装后，将缓存在 `_system_prompt_snapshot` 中，
        后续轮次均使用此缓存结果以确保前缀缓存（prefix cache）的稳定与 LLM 长对话稳定性。

        Returns:
            str: 组装好的完整 System Prompt 文本。
        """
        parts = [
            "你是一个能自我学习的智能助手。你可以：\n"
            "1. 使用工具完成任务\n"
            "2. 从工作中学习，将经验提炼为可复用的技能\n"
            "3. 记住重要信息到持久记忆\n"
            "4. 搜索历史对话记录\n"
        ]

        # 注入记忆系统的静态数据区（如 MEMORY.md 与 USER.md 的冻结快照）
        if self.memory_manager:
            memory_block = self.memory_manager.build_system_prompt()
            if memory_block:
                parts.append(memory_block)

        # 注入技能描述目录（采用渐进式加载：只包含技能名称和一行描述摘要，不包含具体技能 body，节省 token）
        if self.skill_manager:
            skills_block = self.skill_manager.get_skills_summary()
            if skills_block:
                parts.append(skills_block)

        self._system_prompt_snapshot = "\n\n".join(parts)
        return self._system_prompt_snapshot

    # -- 核心对话轮次主循环 ----------------------------------------------------

    def run_turn(self, user_input: str) -> str:
        """执行单轮次的用户对话交互。

        执行链路流程：
          1. 预提取记忆（从各个 MemoryProvider 中异步检索与输入相似度高的上下文信息并准备注入）
          2. 将最新的用户消息追加入消息列表中
          3. 进入 Tool-Calling 循环（大模型可能会连续触发多个工具，直到其决定直接用纯文本回复用户）
          4. 交互结束后的同步流程（让 MemoryProvider 在后台提炼此轮对话的有效知识，持久化到本地磁盘）
          5. 计算 Token 消耗，决策是否满足阈值需要对历史消息进行自动化压缩
          6. 将最终得到的助手文本返回给 REPL 层

        Args:
            user_input: 用户的自然语言输入文本。

        Returns:
            str: 助手最终返回给用户的回复文本。
        """
        # 1. 对话前的记忆预拉取
        if self.memory_manager:
            self.memory_manager.prefetch(user_input)

        # 2. 追加用户消息
        self.messages.append({"role": "user", "content": user_input})

        # 3. 执行 Tool-Calling 多级交互循环
        response_text = self._call_llm_loop()

        # 4. 对话后的记忆回写与同步
        if self.memory_manager:
            self.memory_manager.sync_turn(user_input, response_text)

        # 5. 上下文自动压缩决策
        if self.compressor.should_compress():
            logger.info("满足阈值，触发上下文窗口自动压缩")
            self.messages = self.compressor.compress(self.messages, self.client)

        return response_text

    def _call_llm_loop(self) -> str:
        """内部工具交互核心循环。

        LLM 可能会返回一个或多个 `tool_calls`。系统执行这些函数，将结果以 `role="tool"`
        的角色格式化并追加入上下文历史中，然后再次请求 LLM。
        这一循环将持续运行，直到 LLM 不再返回 `tool_calls`（即产生最终的纯文本回复 Choice）为止。
        """
        # 组装当前的工具集：包含 Agent 自身的工具（如技能管理与会话搜索）以及各记忆 Provider 提供的工具
        all_tools = list(self._tools)
        if self.memory_manager:
            all_tools.extend(self.memory_manager.get_all_tool_schemas())

        # 容错：确保系统提示词已被初始化构建
        if self._system_prompt_snapshot is None:
            self.build_system_prompt()

        while True:
            # 必须在每一轮 LLM 调用的头部拼入最新的冻结 System Prompt 保证完整语义
            api_messages = [
                {"role": "system", "content": self._system_prompt_snapshot},
                *self.messages,
            ]

            response = self.client.chat.completions.create(
                model=self.config.model.primary,
                messages=api_messages,
                tools=all_tools if all_tools else openai.NOT_GIVEN,
            )

            choice = response.choices[0]
            msg = choice.message

            # 计算和更新累积使用的 Token 计数（供 ContextCompressor 进行阀值压缩参考）
            if response.usage:
                self.compressor.update_usage(
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                )

            # 情况 A: LLM 决定输出最终的文本回答，退出工具调用循环
            if not msg.tool_calls:
                content = msg.content or ""
                self.messages.append({"role": "assistant", "content": content})
                return content

            # 情况 B: LLM 决定发起工具调用。
            # 首先必须把大模型含有 tool_calls 请求的 Assistant 消息原封不动追加到上下文中（OpenAI 协议强制要求）
            self.messages.append(msg.model_dump())

            # 遍历并依次执行大模型请求的每一个工具调用（支持在一轮中并行执行多个）
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                logger.debug("执行工具回调: %s(%s)", fn_name, fn_args)

                # 工具分发路由器：优先判断是否是记忆组件提供的工具，否则在常规 Agent 注册的工具字典里查找
                if self.memory_manager and self.memory_manager.is_memory_tool(fn_name):
                    result = self.memory_manager.handle_tool_call(fn_name, fn_args)
                elif fn_name in self._tool_handlers:
                    try:
                        result = self._tool_handlers[fn_name](**fn_args)
                    except Exception as exc:
                        logger.exception("工具 '%s' 执行中抛出异常", fn_name)
                        result = f"Error executing {fn_name}: {exc}"
                else:
                    result = f"Unknown tool: {fn_name}"

                # 执行工具后，必须将结果追加，用 tool_call_id 绑定对应的请求
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result) if result is not None else "",
                    }
                )

    # -- 会话结束与释放生命周期 -------------------------------------------------

    def end_session(self) -> None:
        """结束当前交互会话，触发相关组件的收尾钩子。"""
        if self.memory_manager:
            # 告诉记忆管理器进行最终状态固化或执行端侧摘要同步
            self.memory_manager.on_session_end(self.messages)
            # 安全释放线程池或关闭长链接
            self.memory_manager.shutdown()
