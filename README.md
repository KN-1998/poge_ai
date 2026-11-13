# poge_ai

**一个面向开放环境的「记忆原生、工具原生、异构协同」智能架构研究项目。**

> 当前版本：`0.2.0`，研究原型（Research / Pre-Prototype）
> 本仓库**不是**已训练好的模型，也不宣称任何设计目标已被验证。它包含一份公开的架构提案、一套诚实的研究议程、以及一个可运行、可审计的系统基线。

简体中文 | [English summary](#english-summary)

---

## 这个项目在做什么

主流大模型的路线是「把尽可能多的知识压进参数和上下文，一步给出答案」。本项目探索另一条路：

- **模型核心小而精**：负责理解、判断与调度，不追求记住所有内容；
- **记忆分层**：GPU/NPU 只保留激活特征与索引，内容放在 CPU 内存、本地磁盘和网络上；
- **动作用路由**：在「内部生成 / 记忆检索 / 工具执行 / 委派评审」之间显式选择，而不是硬算；
- **生态化训练**：用异构模型与真实环境反馈做验证，公开蒸馏、公开留痕；
- **规模无关**：端侧小模型是低成本的验证入口，不是架构的最终上限。

这些是**研究假设**，不是结论。仓库把「假设、已有证据、实验结果、尚未解决的问题」分开记录。

## 当前状态（请先读这一段）

| 项目 | 状态 |
|---|---|
| 架构提案文档 | 已完成，`docs/architecture.md` |
| 原始讨论与演进档案 | 已归档，`docs/archive/`（历史观点，不代表当前结论） |
| 相关工作与现成组件 | 已整理，`docs/related_work.md`（**条目未经逐项复核**） |
| 可运行的系统基线 | 已实现，`src/poge_ai/`（SQLite 记忆 + 规则路由 + 受限工具 + JSONL 轨迹） |
| 界面 / 交互 | 无，仅命令行 |
| 模型训练 | **未开始**，也不在本版本范围内 |
| 质量评测 | **无**。默认 `OfflinePlanner` 不做推理，任何问答质量结论都必须接入真实模型后重新测量 |
| 许可 | 代码 AGPL-3.0-or-later；文档 CC BY-SA 4.0 |

## 快速开始

需要 Python 3.10+，无第三方运行时依赖。

```bash
git clone https://github.com/<your-account>/poge_ai.git
cd poge_ai
python -m pip install -e .
```

```console
$ poge-ai ask "3 * (4 + 5) 等于多少"
动作：tool（置信度 0.85）
理由：task contains a computable expression: '3 * (4 + 5)'
工具：calc -> 27

回答：
【离线基线占位输出】
任务：3 * (4 + 5) 等于多少
收到的上下文：
  - 工具 calc 的结果：27
说明：OfflinePlanner 不做推理，接入真实模型后需重新评测。

轨迹：runs/trace.jsonl (run_id=8912e87e1301)
```

```console
$ poge-ai remember "用户偏好中文回答，并要求给出可复现的实验步骤" --kind semantic --source user --confidence 0.9
已写入 009b71e6bf4f4de0（semantic）

$ poge-ai recall "回答偏好"
命中 1 条（查询：回答偏好）：
1. [semantic] score=0.836 id=009b71e6bf4f4de0
   用户偏好中文回答，并要求给出可复现的实验步骤

$ poge-ai stats
数据库：poge_ai.db
总数：1（active=1, archived=0, revoked=0）
分类：working=0, episodic=0, semantic=1
```

以上输出来自本仓库的真实运行结果。注意回答部分是**占位文本**——这正是当前状态。

接入真实模型（任何 OpenAI Chat-Completions 兼容服务）：

```bash
export POGE_AI_BASE_URL="https://api.example.com/v1"
export POGE_AI_API_KEY="..."      # 不会写入轨迹，日志中只留脱敏标记
export POGE_AI_MODEL="some-model"
poge-ai ask "帮我把这段需求拆成三步" --planner openai
```

更多用法见 [`docs/quickstart.md`](docs/quickstart.md)。

## 仓库结构

```text
poge_ai/
├── README.md                  # 本文件
├── LICENSE                    # GNU AGPL-3.0 全文
├── NOTICE                     # 代码/文档/第三方的许可划分
├── CONTRIBUTING.md            # 贡献指南
├── CODE_OF_CONDUCT.md         # 行为准则
├── SECURITY.md                # 安全政策与威胁模型
├── CITATION.cff               # 引用信息
├── CHANGELOG.md               # 版本记录
├── pyproject.toml             # 打包配置（无运行时依赖）
├── docs/
│   ├── architecture.md        # 架构提案与可检验假设
│   ├── research_agenda.md     # MVP 路线、实验问题、发布门槛
│   ├── terminology.md         # 术语表
│   ├── related_work.md        # 相关工作与现成组件
│   ├── governance.md          # 开源治理与许可策略
│   ├── quickstart.md          # 安装、命令、接入模型
│   ├── decisions/             # 架构决策记录（ADR）
│   └── archive/               # 原始对话、演进过程、历次版本、检索报告
├── src/poge_ai/
│   ├── memory.py              # SQLite 记忆：写入/检索/更新/遗忘/巩固
│   ├── router.py              # 规则路由基线
│   ├── tools.py               # 工具注册、参数校验、权限与路径囚笼
│   ├── trace.py               # JSONL 轨迹与脱敏
│   ├── planner.py             # 可插拔生成后端（离线 / OpenAI 兼容）
│   ├── agent.py               # recall -> route -> act -> answer -> remember
│   └── cli.py                 # 命令行
├── examples/demo.py           # 一次完整运行的脚本化演示
├── tests/                     # 单元与端到端测试（仅标准库 unittest）
└── .github/                   # Issue/PR 模板与 CI
```

## 架构总览

```text
                   ┌──────────────────────────┐
                   │       用户 / 环境         │
                   └────────────┬─────────────┘
                                │
                   ┌────────────▼─────────────┐
                   │     决策核心 / Router     │
                   │ 内部生成 | 记忆 | 工具动作 │
                   └───────┬────────┬─────────┘
                           │        │
              ┌────────────▼─┐   ┌──▼────────────────┐
              │ 记忆管理接口  │   │ 工具与执行适配层   │
              │ 写/读/更新/遗忘│   │ 调用/校验/权限/审计 │
              └──────┬───────┘   └──┬────────────────┘
                     │              │
        ┌────────────▼──────────────▼─────────────┐
        │             CPU 调度与状态管理           │
        └────────────┬───────────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │ 本地存储 / 远端资源 / 网络 │
        │ 文档、经验、工具、模型服务 │
        └──────────────────────────┘

        异构评审：独立模型 + 确定性程序 + 环境反馈
        全流程：来源追踪、成本记录、权限控制、实验审计
```

这是一张逻辑示意图。它不表示上面的神经网络级设计已经实现——当前代码只实现了图中系统层的部分。

## 研究路线

| 阶段 | 目标 | 当前状态 |
|---|---|---|
| MVP-0 | 术语、假设、指标与威胁模型 | 部分完成 |
| MVP-1 | 现成模型 + 普通存储 + 工具路由的对照基线 | **已实现最小版本** |
| MVP-2 | 可寻址记忆状态、更新与遗忘策略 | 记录 schema 已实现，评测集未建 |
| MVP-3 | 比较内部回答 / 记忆访问 / 工具调用的决策策略 | 规则基线已实现，其余待做 |
| MVP-4 | 异构评审与信用记录 | 未开始 |
| MVP-5 | 学习式原生记忆（可微门控、主动回顾、联合训练） | 未开始 |

详细拆解与实验问题见 [`docs/research_agenda.md`](docs/research_agenda.md)。

## 科学边界

把架构描述当结论是本项目最想避免的错误。以下表述**不应**被当作既定事实：

- 标准 Transformer 并非简单地「必然遗忘」；长上下文检索、注意力分布与 KV 缓存之间存在多种权衡；
- 固定容量记忆并不自动解决遗忘、冲突、寻址、压缩损失或长期一致性问题；
- 把内容放到 CPU、磁盘或互联网，不等于检索、传输、验证和调度没有成本；
- 工具调用不天然比模型内部计算可靠；工具选择、参数生成、权限与故障恢复都需要评测；
- 多模型投票不等于独立证据，模型之间可能存在训练数据与架构上的相关性；
- 小参数核心能否取得更高成本效益，需要基准测试，不能凭架构推断；
- 「自动接入互联网工具」必须受权限、来源可信度、沙箱、隐私与人工确认约束。

本仓库会把失败、反例和未解决问题一并记录，而不是只保留好看的结论。

## 如何参与

欢迎：相关工作与反例、可复现实验、记忆/路由/评审实现、安全与许可审查。

请先读 [`CONTRIBUTING.md`](CONTRIBUTING.md)；涉及架构方向的改动请提交 ADR，不要直接覆盖历史决策。安全问题请按 [`SECURITY.md`](SECURITY.md) 私下报告。

## 许可与治理

- 软件代码：[GNU AGPL-3.0-or-later](LICENSE)，官方全文，未做修改。
- 文档：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)，详见 [`NOTICE`](NOTICE)。
- 第三方模型、数据集与工具：各自原始许可，不因本仓库而被重新授权。
- 为什么是 AGPL、治理层如何划分、有哪些已知张力：见 [`docs/governance.md`](docs/governance.md)。

## English summary

`poge_ai` is a research project on an environment-native agent architecture: a small decision
core that manages explicit memory tiers, routes between internal generation, memory access and
tool execution, and is evaluated by heterogeneous reviewers rather than a single benchmark.

This repository is **not** a trained model and makes no validated performance claim. It ships a
proposal (`docs/architecture.md`), an honest research agenda with per-MVP status
(`docs/research_agenda.md`), the original discussion archive (`docs/archive/`), and a runnable,
dependency-free Python baseline (`src/poge_ai/`) implementing SQLite memory with provenance and
revocation, a rule-based action router, a permission-scoped tool registry, and JSONL tracing.

Licensed AGPL-3.0-or-later for code, CC BY-SA 4.0 for documentation.
