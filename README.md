# My Learning Agent

一个具备**自我学习循环**和**持久化记忆系统**的 AI Agent。

Agent 能够从对话中提炼可复用的技能（Skills），自动管理技能生命周期，并通过文件背书的记忆系统在跨会话间保持上下文连续性。

## 核心特性

| 模块 | 能力 |
|------|------|
| **记忆系统** | 文件背书（MEMORY.md / USER.md），冻结快照，威胁扫描，漂移检测 |
| **技能学习** | 从对话中提炼 SKILL.md，渐进式加载，使用计数追踪 |
| **策展器** | 技能生命周期管理：active → stale(30天) → archived(90天)，pinned 豁免 |
| **上下文压缩** | 头尾保护 + 辅助模型压缩中间消息，维持长对话稳定性 |
| **跨会话搜索** | SQLite FTS5 全文索引，三种搜索模式，零 LLM 开销 |
| **记忆 Provider** | 抽象基类设计，支持可插拔的外部记忆后端 |

## 项目结构

```
my-learning-agent/
├── main.py                         # 入口点 — 组装组件，启动 REPL 循环
├── config.yaml                     # 运行时配置
├── pyproject.toml                  # 项目元数据与依赖
│
├── agent/                          # 核心 Agent 模块
│   ├── config.py                   # 配置加载器（YAML → dataclass）
│   ├── run_agent.py                # Agent 主循环（tool-calling loop）
│   ├── memory_provider.py          # 记忆 Provider 抽象基类
│   ├── memory_manager.py           # 记忆管理器（编排多个 Provider）
│   ├── context_compressor.py       # 上下文压缩器
│   ├── curator.py                  # 策展器（技能生命周期）
│   ├── learn_prompt.py             # /learn 命令提示词构建
│   └── state.py                    # SQLite 持久化层（WAL + FTS5）
│
├── tools/                          # Agent 工具集
│   ├── memory_tool.py              # 内置记忆工具（MemoryStore）
│   ├── skills_tool.py              # 技能管理器（SkillsManager）
│   └── session_search.py           # 跨会话搜索（SessionSearch）
│
├── plugins/                        # 可插拔扩展目录
│   └── memory/                     # 记忆后端插件
│
├── data/                           # 运行时数据（gitignored）
│   ├── state.db                    # SQLite 数据库
│   ├── MEMORY.md                   # Agent 记忆
│   └── USER.md                     # 用户信息
│
├── skills/                         # 技能文件目录（运行时生成）
│
└── tests/                          # 测试套件
    ├── test_memory.py              # 记忆系统测试（9 用例）
    ├── test_skills.py              # 技能系统测试（7 用例）
    ├── test_curator.py             # 策展器测试（6 用例）
    └── test_state.py               # 存储层测试（5 用例）
```

## 快速开始

### 环境要求

- Python 3.11+（< 3.14）
- OpenAI API Key

### 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd my-learning-agent

# 安装依赖（推荐使用虚拟环境）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 配置

```bash
# 设置 OpenAI API Key
export OPENAI_API_KEY="sk-..."

# 可选：自定义配置文件路径
export MYAGENT_CONFIG="/path/to/config.yaml"
```

`config.yaml` 中的关键配置项：

```yaml
model:
  primary: "gpt-4o"           # 主模型（对话 + 推理）
  auxiliary: "gpt-4o-mini"     # 辅助模型（压缩 + 审查）

memory:
  agent_char_limit: 2200       # Agent 记忆字符上限
  user_char_limit: 1375        # 用户记忆字符上限

curator:
  stale_after_days: 30         # 技能闲置多久标记为 stale
  archive_after_days: 90       # stale 多久后归档
```

### 运行

```bash
python main.py
# 或通过 entry point
myagent
```

## 使用方式

### 交互命令

| 命令 | 说明 |
|------|------|
| `/learn <topic>` | 让 Agent 从当前话题中提炼一个可复用的技能 |
| `/skills` | 列出所有已学习的技能及其状态 |
| `/search <query>` | 跨会话全文搜索历史对话 |
| `/curator` | 手动触发策展器，清理过期技能 |
| `/quit` | 退出 Agent |

### 学习循环示例

```
You: 帮我写一个 Python 装饰器来做函数重试
Agent: [生成重试装饰器代码 + 解释]

You: /learn python-retry-decorator
Agent: [分析对话，提炼技能，保存为 skills/python-retry-decorator/SKILL.md]

You: /skills
  python-retry-decorator — Python 函数重试装饰器 (used 0x, active)
```

### 记忆系统

Agent 会自动：
- 将重要信息写入 `MEMORY.md`（Agent 记忆）
- 将 user 偏好写入 `USER.md`（用户画像）
- 会话启动时冻结快照，中途写入不影响当前 system prompt
- 对疑似 prompt injection 的内容进行威胁扫描并拒绝

### 技能生命周期

```
创建 (create)
  ↓
活跃 (active) ──── 被使用 → 重置计时器
  ↓ 30 天未使用
过期 (stale)
  ↓ 再过 90 天
归档 (archived) ── /curator restore 可恢复

* pinned 的技能永不过期
```

## 测试

```bash
# 运行全部测试
pytest

# 带覆盖率
pytest --tb=short -v

# 单个模块
pytest tests/test_memory.py -v
```

## 架构设计

### 冻结快照模式

系统提示中的记忆内容在会话启动时冻结。中途通过工具写入的新记忆只更新磁盘文件，不修改当前 system prompt —— 这确保了 prefix cache 的稳定性，避免 LLM 重执行旧指令。

### 渐进式加载

技能系统采用两层加载策略：
1. **摘要层**（system prompt）：只注入技能名称 + 描述
2. **完整层**（tool call）：Agent 按需通过 `skill_view` 工具加载完整内容

这将 system prompt 大小控制在可预测范围内，不随技能数量膨胀。

### 记忆 Provider 抽象

```python
class MemoryProvider(ABC):
    def system_prompt_block(self) -> str: ...
    def handle_tool_call(self, name, args) -> str: ...
    # ... 完整生命周期钩子
```

通过 `MemoryManager` 编排多个 Provider，支持扩展自定义记忆后端（如 Redis、向量数据库等）。

### 上下文压缩

当对话 token 数超过上下文窗口的 75% 时自动触发：
- 保护头部 3 条消息（system prompt 相关）
- 保护尾部 6 条消息（最新上下文）
- 用辅助模型（gpt-4o-mini）将中间消息压缩为结构化摘要

## License

MIT
