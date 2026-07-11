"""技能撰写模板及 /learn 引导提示词服务。

当用户在 REPL 终端输入 `/learn <topic>` 时被触发：
  - 本模块将用户提供的主题或来源，拼装成一整套极具约束力的 Agent 动作指令；
  - 引导 Agent 调动自身的读取文件、网页提取或浏览上下文等工具去收集相关资料；
  - 最终强制命令 Agent 以符合规范的 YAML Frontmatter 和 Markdown 结构撰写 `SKILL.md`，
    并通过调用 `skill_create` 工具持久化到本地技能目录。
"""

from __future__ import annotations

# SKILL.md 技能规格撰写标准（硬线限制说明）
AUTHORING_STANDARDS = """
## 技能规格与格式化要求 (Skill Authoring Standards)

1. **技能命名 (Name)**: 使用 kebab-case（小写短横线连接），禁止空格与下划线，字符数限制在 64 字以内（如 `debug-python-tests`）。
2. **描述信息 (Description)**: 精简为一行话，**绝对不能超过 60 个字符**，以动词开头（如 "Debug failing Python unit tests"）。
   注意：此条规则极其关键。系统提示词的技能摘要会对其进行硬性截断以节约 token，请在生成时务必自检长度。
3. **内容小节规范 (Body sections)**（按以下严格顺序排布）：
   - ## Overview — 技能用途，受众及应用时机
   - ## Steps — 具备可执行性的、清晰的第一步第二步逻辑步骤
   - ## Examples — 真实无杜撰的代码片段或终端命令示例
   - ## Notes — 边缘场景，踩坑记录以及关联知识
4. **质量标准**：
   - 可执行：杜绝空洞的理论宣导，多提供清单、直接复制粘贴即可运行的代码模版。
   - 真实性：绝不允许胡乱杜撰系统 API、文件路径或假想命令。
   - 完备性：覆盖整个任务的起止完整路径。
"""


def build_learn_prompt(user_request: str) -> str:
    """为 Agent 动态组装针对特定主题的“自我技能提炼”引导提示词。

    该提示词将扮演新 turn 的 Input 发送给主模型，主模型据此开始执行工作流分析。

    Args:
        user_request: 用户输入的主题描述或包含原始信息的文档文本。

    Returns:
        str: 拼装好的具体分析并学习命令 Prompt。
    """
    return f"""The user wants you to learn from the following and create a reusable skill:

{user_request}

Follow these steps:
1. Analyze the source material (files, URLs, conversation history, or pasted text)
2. Identify the core reusable pattern or workflow
3. Distill it into a structured skill with clear steps
4. Use the `skill_create` tool to save the skill

{AUTHORING_STANDARDS}

Create the skill now. Choose an appropriate category (e.g. software-development,
data-science, research, devops, creative, general).
"""
