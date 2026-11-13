# 快速开始

本项目当前是**研究原型**：可以运行，但默认配置下不具备可用的问答质量。它的用途是提供可审计、可复现的工程基线。

## 环境要求

- Python 3.10 及以上
- 无第三方运行时依赖（仅标准库）
- 开发依赖：无（测试用 `unittest`）

## 安装

```bash
git clone https://github.com/<your-account>/poge_ai.git
cd poge_ai
python -m pip install -e .
```

不安装也可以直接用源码树运行：

```bash
PYTHONPATH=src python -m poge_ai --help
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="src"; python -m poge_ai --help
```

## 三条常用命令

```bash
# 1. 问一个问题（离线确定性 Planner，仅验证链路）
poge-ai ask "3 * (4 + 5) 等于多少"

# 2. 写一条记忆再召回
poge-ai remember "用户偏好中文回答，且要求给出可复现的实验步骤" --kind semantic --source user --confidence 0.9
poge-ai recall "回答偏好"

# 3. 查看统计与工具清单
poge-ai stats
poge-ai tools
```

所有命令都支持 `--json` 输出，便于脚本处理。

## 数据放在哪里

- 默认数据库：`./poge_ai.db`（SQLite），可用 `--db` 指定；
- 默认轨迹：`./runs/trace.jsonl`，可用 `--trace` 指定；
- 两者都在 `.gitignore` 中，不会被误提交。

## 接入真实模型

`planner.py` 定义了 `Planner` 协议。仓库内置两种实现：

- `OfflinePlanner`：确定性模板，只用于跑通链路与测试；
- `OpenAICompatPlanner`：兼容 OpenAI Chat Completions 的 HTTP 后端，从环境变量读取配置。

```bash
export POGE_AI_BASE_URL="https://api.example.com/v1"
export POGE_AI_API_KEY="..."        # 不会写入日志，日志中只保留脱敏标记
export POGE_AI_MODEL="some-model"
poge-ai ask "帮我把这段需求拆成三步" --planner openai
```

> 注意：接入真实模型后，任何质量、成本、延迟结论都必须重新测量并记录在 `docs/` 中。默认配置的结果不可用来评价项目。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 阅读顺序

1. `README.md` —— 项目定位与边界
2. `docs/architecture.md` —— 目标架构与假设
3. `docs/research_agenda.md` —— MVP 路线与当前完成度
4. `docs/governance.md` —— 许可与治理
5. `docs/archive/` —— 原始讨论与演进历史（历史观点，不代表当前结论）
