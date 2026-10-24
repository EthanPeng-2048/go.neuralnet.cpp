# 第三方库与技术栈选型

> 状态：v0.2（2026-10-24；M2/M3 实装）
> 原则：**规则、SGF、数据与训练各自只保留一个真源**。项目侧做胶水、特征和验收，
> 不复制成熟库的业务实现。Python 用于数据工程、批处理、可视化和验收；模型与训练
> 执行统一使用 neuralnet.cpp。

## 1. 固定选择

| 范畴 | 选择 | 用途 | 不做什么 |
|---|---|---|---|
| 围棋规则 | KataGo `v1.18.2` `cpp/game` | Board、BoardHistory、Rules；落子、提子、劫/超级劫、禁入、终局、面积/数目计分 | 不自研位棋盘、Zobrist、规则状态机或计分器 |
| 模型与训练 | neuralnet.cpp `v1.12.0` | 张量、层、损失、优化器、checkpoint、训练与 RL 原语 | 不引入 PyTorch/JAX 作为第二训练栈 |
| SGF 解析 | `sgfmill==1.1.1`（MIT） | M2 读取 SGF 主线、让子布置与元数据 | 不把 sgfmill 的棋盘/规则判断当作项目规则真源 |
| HTTP / 重试 | `httpx==0.28.1` + `tenacity==9.2.1` | CWI 下载、流式落盘、校验前重试 | 不做不可审计的静默镜像切换 |
| JSONL I/O | `orjson==3.13.0` | Python → `go_data_cli` 的临时规范化记录 | 不作为最终训练数据格式 |
| 特征张量 | `numpy==2.5.4` | 9 平面、批量列布局、张量检查 | 不承担训练/自动微分 |
| 可视化 | `Pillow==13.0.0` | 特征平面 contact sheet | 不参与推理 |
| C++ 测试 | 项目内轻量断言 + CTest | 规则语义、性能和构建闸门 | 不因测试框架增加运行时依赖 |
| Python 测试 | 标准库 `unittest` | SGF/GnuGo 集成测试，缺少可选工具时明确 skip | 当前不强制引入 pytest |
| 可选对拍 | GNU Go 3.8 | 提子与禁入等外部行为抽查 | 不作为训练、搜索或规则运行时依赖 |

`requirements-dev.txt` 只固定 Python 开发依赖；规则与训练依赖仍由 Git submodule
锁定，构建阶段不联网。

## 2. 分层边界

```text
Python: SGF 解析 / 数据批处理 / 指标与可视化 / 集成验收
                 |
                 v
C++ project adapter: 坐标与生命周期转换、薄 API、CLI
                 |
                 v
KataGo cpp/game: 唯一规则真源

C++ training path: neuralnet.cpp: 唯一模型与训练真源
```

适配层只做三件事：

1. `Stone` / `Point` / `RuleSet` 与 KataGo 类型的机械映射；
2. 一个 `Game` 持有 `Board + BoardHistory`，转发合法性和走子；
3. 输出盘面、合法点、面积归属和即时计分结果。

适配层不得推导劫、气、提子或终局；出现差异时以 KataGo 行为为准并修适配层。

## 3. M2/M3 已实装

SGF 主线、让子/初始布置、贴目、结果、主分变化与显式颜色走法由 Python 规范化；
`.nngames` 最终由 neuralnet.cpp `KeyValueRecord v2` 打包。盘面轨迹由
`go_rules_cli trace` 调用 KataGo 生成，Python 只把轨迹转成 NumPy。

以下库只在对应里程碑出现明确需求时锁定版本，避免现在制造无用依赖：

| 需求 | 首选 | 理由 |
|---|---|---|
| 离线数据审计 | DuckDB | 可直接查询 Parquet/JSONL，无需把全量棋谱载入 Python 对象 |
| Python 单测扩展 | pytest | 仅当参数化/fixture 的收益超过引入依赖的成本 |

100k+ 局原始数据不默认进 pandas。若采用 Parquet 中转格式，再以 DuckDB 做抽样、
分组和质量统计；最终训练格式仍按 D7 的 `.nngames` 定义。

## 4. 明确不选

- 不用 Python 规则库执行训练/搜索时的走子；避免与 KataGo 的劫和计分语义分叉。
- 不把 KataGo 全套搜索、神经网络后端和命令程序链入 M1；只编译 `cpp/game`
  及其最小 `cpp/core` 支撑源。
- 不引入第二套自动微分或张量运行时；ML 已确认统一使用 neuralnet.cpp。
- 不为单元测试直接引入大型测试框架；当前 CTest + unittest 足够覆盖 M1–M3。

## 5. 依赖升级闸门

KataGo 升级必须同时通过：

1. 项目提子、自杀、劫/超级劫、面积/数目计分题库；
2. SGF 回放盘面一致；
3. 200 手回放 `<5 ms`；
4. 有 GnuGo 时执行行为对拍。

SGF/数据依赖升级只要求解析夹具、损坏输入行为与样例集回归，不改变规则真源。
