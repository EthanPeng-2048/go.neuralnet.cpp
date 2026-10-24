# go.neuralnet.cpp 设计与路线图

> 状态：**v0.10**（2026-10-24）· v0.4：范围裁定不做 RL（D9）+ 方案 1/2 文档；
> v0.5：**方案 3（搜索控制 Token 蒸馏）落档**——算法/实践分离
> （[05 算法](05-scheme3-algorithm.md) / [06 实践](06-scheme3-engineering.md)）、
> D3 V100 实测修订、**D10 立项**；
> v0.6：正式锁定上游 **neuralnet.cpp v1.12.0**，其 RL 原语已发布；
> v0.7：**M0 工程与 40HX 基准完成**；M1 裁定为直接消费 KataGo 棋规库，不自行实现规则；
> v0.8：**M0 按公开 4060/KataGo 基准近似闭合**，进入 M1；
> v0.9：**M1 KataGo 棋规适配完成**：题库、SGF 回放、GnuGo 对拍与性能闸门通过；
> v0.10：**M2/M3 完成**：1 万局 `.nngames`、GoTokenizer、9 平面特征与可视化验收
> 上游库：[`neuralnet.cpp` `v1.12.0`](https://github.com/EthanPeng-2048/neuralnet.cpp/tree/v1.12.0)
> （C++26 header-only，CPU/Vulkan 双后端，无 CUDA；完整锁定信息见
> [DEPENDENCIES.md](../../DEPENDENCIES.md)）

## 0. 项目定位

基于 neuralnet.cpp 训练 19×19 围棋模型（策略 + 价值双头 + MCTS），目标是在与同学的
赌约中获胜；对手本人或其 KataGo（跑在 RTX 4060 上）。训练主力 = NVIDIA CMP 40HX
单卡；Tesla V100 依 D3 不投入装机（不进任何里程碑）。

## 1. 决策记录

| # | 决策 | 依据 | 状态 |
|---|------|------|------|
| D1 | **架构只做 B**（双分支编码器）。A（前缀 LM）删除；C（纯 CNN）保留为**基础设施验证器** | A 与 B 在"历史双向可见"前提下等价，做 A 是浪费；C 全部用现成层、零模型代码，最适合验证管线 | ✅ 已定 |
| D2 | **最终模型只用 19×19** 训练与 RL；9×9 / 13×13 仅用于测试与沙盒 | 对战默认尺寸；避免多尺寸干扰 | ✅ 已定 |
| D3 | 硬件：**40HX 单卡** = 唯一训练卡与权威口径。**V100 = 仅供参考/后备，不进任何里程碑**。2026-10-23 实测落档：**Vulkan 可用、无 f16 原生加速** ⇒ 仅 f32 路径、张量核心不可达（库无 CUDA）；装机时间成本未回收。仅当方案 3 教师标注（06 §3）成为全瓶颈且 H1 已过时再评估 | 库无 CUDA → V100 只赚 HBM2 带宽；f16 无加速否定了主要收益论据 | ✅ 已定（v0.5 实测修订） |
| D4 | 管线：**纯监督预训练 + MCTS 推理**。原"阶段 2 轻量自对局微调"随 D9 移出本库（属 RL）；从零 RL 本就不可能（19×19 两周 3–5 万局 ≈ AlphaZero 的 0.1%，见 §3） | D2 算力账 + D9 范围裁定 | ✅ 已定（v0.4 修订） |
| D5 | B 的 19×19 特化：**盘面走卷积平面分支**（非 361 token），**历史截断 K=64 手** | 序列 600+ → 注意力 O(L²) 砍半推理速度，而推理速度 = visits，见 §4 | ⚠️ K 可调参 |
| D6 | 上游补丁政策：**默认零补丁**（ModelSpec 复用 `Transformer` 类型 + `patch_size`），5 行 `ModelType::Go` 补丁为 Plan B | 上游 churn 极高，rebase 成本要预留 | 🔶 待拍板 |
| D7 | 数据格式 `.nngames`（KVRecord v2 容器）：**存走法+结果，不存盘面**；特征训练时按需重放 | 改特征设计不需要重新下载/转换数据 | ✅ 已定 |
| D8 | thinking 路线（B + 推理时思考链）：**评估已出，建议不进主线**，挂 M10（赌约后）研究项；评估见 [01-thinking-eval.md](01-thinking-eval.md)，架构见 [03-scheme2-decoder-thinking.md](03-scheme2-decoder-thinking.md) | 关键路径：thinking 冷启动唯一来源 = MCTS 蒸馏 ⇒ 依赖 M8；训练成本 10×、R1 三前置全缺 | 🔶 待拍板 |
| D9 | **当前范围仍在 M0–M9 不做自对局训练**。通用 RL 原语已由上游 [`v1.12.0`](https://github.com/EthanPeng-2048/neuralnet.cpp/tree/v1.12.0) 提供（具名多输出、PPO/GRPO、rollout pipeline 等），上游依赖阻塞解除；围棋环境、MCTS、自对局与领域 replay 仍在应用侧。**当前交付仍是"监督训练 + MCTS 推理 + 对弈/评估"**；启用项目级 RL 需新的范围决定，不随依赖升级自动进入关键路径 | 上游已消除“重复造 RL 基础设施”的理由，但两周预算仍不支持从零 RL；保持交付最小闭环 | ✅ 已定（v0.4；依赖更新 v0.6） |
| D10 | **方案 3（搜索控制 Token 蒸馏）立项，取代方案 2 成为思考模型主线**：Encoder 直觉 z₀ + Top-K 槽位语法 + 显式可微 PUCT 累加器 + 连续残差头 + 时间条件停止，教师 = KataGo 双预算树蒸馏。算法/实践分离落档（[05](05-scheme3-algorithm.md) / [06](06-scheme3-engineering.md)）。**修复方案 2 两缺陷**（幻觉搜索、分支因子失控），与 D9 兼容（纯蒸馏无 RL）。⚠ 开工阻塞项：软件栈裁决（06 §2，默认 neuralnet.cpp）+ 教师预算 × 赌约条款联定（06 §4） | M0 残差探针为硬闸门（H1，先验不佳：0.040 须先排除竞争解释）；方案 2（03）保留为教师不可得时的回退 | 🔶 立项，阻塞项待裁决 |
| D11 | **M1 棋规全部消费现成库**：锁定 KataGo `v1.18.2` 的 `cpp/game`（Board / BoardHistory / Rules）作为落子、提子、劫/超级劫、禁入、终局与面积计分实现；项目只写薄适配、特征提取与测试，不自行实现位棋盘、Zobrist 或规则状态机 | 棋规已有长期实战验证，重写只会增加规则分叉；同一上游也是 M8 搜索与方案 3 教师管线的一致性来源 | ✅ 已定（2026-10-24） |
| D12 | **数据与特征采用 Python 优先**：SGF 解析用 `sgfmill`，下载/重试用 `httpx + tenacity`，流式记录用 `orjson`，特征张量与可视化用 NumPy + Pillow；规则状态必须经 KataGo，`.nngames`/`.nnvocab` 由 neuralnet.cpp KVRecord v2 编解码。训练执行仍统一 neuralnet.cpp | 批量数据工程用 Python 周期最短；规则与训练各保留单一真源，避免第二套棋规或 ML runtime | ✅ 已定（2026-10-24） |

## 2. 目标校准：诚实评估

**低 visits 的优势归属**（常见直觉错误，此处钉死）：浅搜索下每手质量 ≈ 网络质量，
搜索加成对双方都消失 ⇒ **差距归网络更强的一方**。他背后是 KataGo 在几千倍算力上
训练的网络，所以 19×19 + 低 visits 对我们**更不利**，不是更有利。

| 赌约条件 | 我们的胜率（估算） |
|---|---|
| 19×19，KataGo 任何 visits | **≈ 0%** —— 建议以此为前提规划，不以此为目标 |
| 19×19，双方各限 40–100 visits/手、禁 pondering | 低但仍非零；拼的是网络质量（监督数据量 + 特征质量） |
| 19×19，他本人下（不用 KataGo） | **这是现实的主战场**，监督网 + 250 visits MCTS 对人类 ≈ 碾压 |

**赌约条款建议**（写死数字 visits，不要时间制——时间制下他的 4060 每秒跑的 visits
比我们多，等于先让 2 倍算力）：固定 visits 上限 + 禁 pondering + 固定 komi；
若他坚持满血 KataGo，接受 handicap 起手。

**目标重校准**：本项目的现实交付 = "19×19 强业余～低段位网络 + MCTS 引擎 + 完整
训练基础设施"。对人类高胜率；对 KataGo 承重在赌约条款上，不在算力上。

## 3. 19×19 的算力账（推导口径）

自对局吞吐（可审计推导）：

```
19×19 平均 ~200 手/局 × 250 visits = 5 万次评估/局   （9×9 口径为 1.5 万）
5 万 ÷ 578 次/秒（M0：B-final batch=64，40HX 组件实测，见 §7）
× MCTS 开销 1.5–2× ≈ 130–173 秒/局 ⇒ 约 500–700 局/天（40HX 满负荷）
```

| | 9×9 参考 | **19×19 实际** |
|---|---|---|
| 自对局（40HX） | ~1 万局/天 | **约 500–700 局/天** |
| 两周预算 | 15 万局 | **约 4–6 万局**（visits 爬升后见下） |
| AlphaZero 19×19 实际用量 | — | **~3000 万局** |

⇒ 我们 = AlphaZero 的 **0.13–0.2%**。**从零 RL 数学上不可能**。

**训练期 visits 爬升**（省算力的标准手段，直接改善上表吞吐）：自对局 visits 从
**40 起步**（200 手 × 40 = 8 千次评估/局，含 MCTS 开销 ≈ 21–28 秒/局
≈ 3–4 千局/天），随网络强度分档爬升到 250 ⇒ 两周 RL 预算约 **4–6 万局**。
对弈时（M9）的 visits 是赌约条款参数，与训练期解耦。

管线定为：

1. **监督预训练**（本库唯一训练阶段）：10–30 万局 19×19 人类棋谱 + Leela Zero
   自对局蒸馏；在线随机 8 对称增广（不离线展开）。
2. ~~轻量自对局微调 / 评估晋升门~~ **不属于当前 M0–M9（D9）**。上游
   [`v1.12.0`](https://github.com/EthanPeng-2048/neuralnet.cpp/tree/v1.12.0) 已提供
   RL 原语；未来若重开项目级 RL，只需补围棋环境/自对局/领域 buffer。上方 visits
   爬升与自对局吞吐推导保留为**未来项目的参考口径**。

## 4. 架构 B 定案（19×19 特化）

```
盘面分支:  C 个特征平面 (C, 361, 19×19) → Conv stem（ResBlock ×N，现成 Conv2D/BN）
历史分支:  最近 K=64 手 move ids → gather_rows embedding (+1D 位置 + 段嵌入)
                        ↓
        拼接为单序列 [盘面块 | 历史块 | <DECISION> query]  （Plain 双向掩码，现成）
                        ↓
        TransformerEncoderLayer ×N（现成）
          ├─ query 行 → Linear(d, 362)   策略头（361 点 + pass）
          └─ query 行 → Linear(d, 1)     价值头（tanh）
                        ↓
        打包为单张量 (363, batch) 单进单出 → 自定义 GoLoss（CE + MSE 分行）
```

**完整设计文档**：[02-scheme1-encoder-transformer.md](02-scheme1-encoder-transformer.md)
（盘面→序列候选与 19 质数问题、特征平面、张量布局、消融计划）。

**为什么盘面不做成 361 个 token**：361 + 整局历史 → 序列 600+，注意力 O(L²) 使
推理速度减半；推理速度直接决定 MCTS visits，即决定赌约战力。卷积平面分支保留
二维归纳偏置且几乎不占序列长度。**历史截断 K=64** 同理（K 是调参项，不进 ModelSpec
关键维度之外的地方）。

**为什么"直出坐标"仍然需要 MCTS**（visits 不可省）：策略头给的是**先验**，
价值头给的是**叶子静态评估**——两者都要靠搜索兑现：

- **前馈网络没有推演循环**：死活 / 对杀 / 劫争需要 10–15 手模拟，网络只会模式
  匹配；MCTS 是把价值头转换成阅读深度的唯一部件。
- **不搜索 ⇒ 价值头闲置**（只评一次叶子、无回传）⇒ 双头设计浪费一半参数。
- 对手若开 KataGo（带搜索），纯策略网络对"网络 + 搜索"必败，哪怕网络同强。
  KataGo 的网也"直出"策略+价值，但 raw policy 单下很弱——那张网的意义就是
  给搜索当先验与叶子评估（AlphaGo Zero 的同权重消融：纯策略 vs +搜索一边倒）。
- RL 层面：纯策略自对局（无 MCTS 目标）= 高方差策略梯度路线，样本效率差一个
  量级；§3 管线的前提就是 MCTS 产出 π 目标。

visits **数目**是可谈判条款（§2，40–100 已有实质战力——先验好时搜索只需修错）；
"要不要 MCTS"不是。

**库扩展点**（全部在派生项目侧，不改上游）：

| 扩展 | 方式 | 上游先例 |
|---|---|---|
| `GoModel` 复合层 | `Layer` 派生，`param_children()` 转发 | `GPTModel`（compute_layer_gpt.hpp） |
| 双输出 | 打包为 (363, batch) 单张量 | 无先例，但 Layer 单进单出契约不变 |
| `GoLoss` | `Loss` 派生，forward 分行算 CE/MSE | `Loss` 纯虚基类（compute_loss.hpp） |
| 合法性掩码 | host 侧采样前将非法点置 -inf（自写采样器，不动库内 `sample_next_token_`） | `SamplingConfig` 本就是 host 侧 |
| ModelSpec（D6） | 默认复用 `type=Transformer` + `patch_size`；Plan B = 上游加 `ModelType::Go` 5 行补丁 | `spec_matches` 走 Transformer 分支 |

**C 验证器 DoD**（M4）：用纯现成层（Conv2D/FC/CE）跑通
规则 → 数据 → 特征 → 训练 → checkpoint 往返 → 评估 全链路；4 epoch loss 健康下降、
`.nnckpt`/`.nnmodel` 双角色存取无损、评估器输出可复现。**通过后 B 只替换模型工厂。**

## 5. 基础设施四件套

| 件 | 内容 | 要点 |
|---|---|---|
| 数据下载器 | CWI 90k 档案、HF `go_pgn_string_leela_zero`（蒸馏用）、OGS API；**避开** GoGoD/Go4Go（商业授权） | `.nngames` = KVRecord v2；存走法不存盘面（D7） |
| 分词器 | `GoTokenizer`：点位↔id（361）、PASS、BOS/EOS、`<DECISION>`；**非 BPE** | 盘面格子与落子共用 id 空间，段嵌入区分角色；词表复用 `.nnvocab` 格式 |
| 规则系统 | **KataGo `cpp/game` 的 Board / BoardHistory / Rules**（D11）：落子、提子、劫/超级劫、禁入、终局与面积计分全部复用；项目侧只做小棋盘适配与特征提取 | **不自行实现规则**；零 NN 依赖，M1 最先接；捕提子题库 + TT 测试局面 + 与 GnuGo 交叉核对 |
| 训练器 | `GoTrainer`（照 `TextTrainer` 的钩子/断点/lr 调度形状）+ Arena/GTP；**SelfPlayTrainer 随 D9 移出（属 RL）** | 训练骨架上游现成，只换数据源与 step 回调 |

## 6. 里程碑

| | 里程碑 | 验收（DoD） | 依赖 |
|---|---|---|---|
| **M0** ✅ | 仓库骨架（CMake + `add_subdirectory` + `neuralnet::nn`，开 PCH）+ **基准** | ① 空 main 编译链接；② `layer_bench` 在 40HX 上建立本项目网络尺寸的推理/训练吞吐基线（替换 §7 全部估算）；③ 用公开 RTX 4060 Ti 基准按硬件规模折算 4060，取 **约 1200 visits/s** 的规划值 | — |
| **M1** ✅ | **KataGo 棋规适配层**（D11，不自研规则） | 捕提子题库 + TT 数子用例全绿；单局重放 <5ms；与 GnuGo 交叉核对一致 | KataGo `v1.18.2` |
| **M2** ✅ | 数据下载器 + SGF 解析 + `.nngames` | 1 万局入库，随机回放验证可逆 | M1 |
| **M3** ✅ | `GoTokenizer` + 特征平面（19×19 特化） | 任意局面 → 特征张量，可视化自检 | M1 |
| **M4** | **C 验证器竖切**（见 §4 DoD） | 全链路 + loss 下降 + ckpt 往返 + 评估可复现 | M2,M3 |
| **M5** | `GoTrainer` 正式版：`--resume`、指标落盘、双角色存取 | 断点续训无损；监督配方可 CLI 表达 | M4 |
| **M6** | GTP 适配器 + Arena（SPRT） | Sabaki 亲手可下；vs GnuGo 自动评估 | M1,M5 |
| **M7** | **B 架构**替换 C（复合层 + 双头 + GoLoss + 合法性掩码） | 同数据显著优于 C；存盘/续训无损；推理速度 ≥ 预算（§7） | M5 |
| **M8** | **MCTS 推理引擎**（PUCT / 先验归一化 / 合法性 / 转置表；**只搜索不训练**——自对局训练循环随 D9 移出） | 9×9 沙盒（D2 允许）：vs 方案 1 纯先验显著增益；vs GnuGo 胜率达标；19×19 速度 ≥ §7 预算 | M6,M7 |
| **M9** | 监督 10–30 万局正式训练 + 按赌约条件对 KataGo/对手调参（**无自对局微调**，D9） | 按赌约条件的正式评估达标 | M8 |

开发日历（part-time）：M0–M8 约 6–8 周 + 监督训练 2–3 天 ≈ **2–2.5 个月**到正式对弈（D9 移出 RL 后日历缩短）。

### 6.1 M0 执行记录（2026-10-24）

- ✅ 顶层 CMake、`CMakePresets.json`、`go_neuralnet_m0` 与 `neuralnet::nn` 消费接入完成；
  CPU 与 Windows clang 23.1 + Vulkan（40HX）均构建通过，两边 ctest **2/2**。
- ✅ 新增可重复 `layer_bench` 矩阵：B-small（d=192/8 层）与 B-final
  （d=256/12 层），历史 K=64 + decision query，sequence=65；inference batch=64、
  training batch=256。原始结果见
  [`gpu`](../benchmarks/m0-layer-bench-gpu.md) / [`cpu`](../benchmarks/m0-layer-bench-cpu.md)。
- ✅ 4060/KataGo 对手不可访问，按项目决定采用公开近似：RTX 4060 Ti 的 KataGo 40b
  实测为 OpenCL **1484.78 visits/s** / TensorRT **1956.56 visits/s**；按 core 与带宽
  规模折算到 RTX 4060，得 **约 1040–1860 visits/s**，规划值取 **1200 visits/s**。
  来源、缩放和置信度见 [`公开基准记录`](../benchmarks/m0-opponent-4060.md)。
  这不是对手实测，拿到其 benchmark 输出后必须替换。

### 6.2 M1 执行记录（2026-10-24）

- ✅ 引入并锁定 KataGo `v1.18.2`
  (`fd0723fdbc0e9d82cf269c9630af8c27c57c07c4`)；CMake 只编译
  `cpp/game` 与最小 `cpp/core` 支撑文件。
- ✅ `go_rules` 薄适配层只做坐标/规则配置映射和生命周期管理；提子、禁入、劫/
  位置超级劫、终局、面积与数目计分均直接调用 KataGo。
- ✅ 规则题库、SGF 重放夹具与 CTest 通过；SGF 由 `sgfmill==1.1.1` 解析，
  盘面结果由 KataGo 状态核对。
- ✅ CPU Release 200 手回放中位数 **0.60 ms**、单次公开样本 **0.64 ms**，
  低于 M1 `<5 ms` 门槛；完整记录见
  [`M1 验收报告`](../benchmarks/m1-rules.md)。
- ✅ 使用临时解包的 GNU Go 3.8 对拍提子链与自杀禁入，判定一致；系统未安装
  `gnugo` 时对应 CTest 自动不注册。
- Python/第三方边界已落档：
  [`07-third-party-selection.md`](07-third-party-selection.md)。

### 6.3 M2 执行记录（2026-10-24）

- ✅ CWI 归档固定 SHA-256，下载、解包、SGF 主线解析、失败分类和
  `httpx + tenacity` 重试均已实现。
- ✅ 实际生成 `data/cwi-10k.nngames`：**10,000 局 / 2,076,112 手 /
  2,086,112 局面 / 24,591,541 bytes**。
- ✅ `.nngames` 使用 neuralnet.cpp `KeyValueRecord v2`；全量 10,000 局经 KataGo
  重放成功，容器序列化**逐字节可逆**。
- ✅ 数据集 SHA、来源、许可证、拒绝原因和验收结果写入本地 manifest；完整记录见
  [`M2/M3 验收`](../benchmarks/m2-m3-data.md)。

### 6.4 M3 执行记录（2026-10-24）

- ✅ `GoTokenizer` 固定 367 token：361 点 + PASS/BOS/EOS/DECISION/PAD/UNK。
- ✅ `.nnvocab` 复用 neuralnet.cpp KVRecord v2 容器与 vocab blob 编码。
- ✅ 特征 v1 固定为 9 平面，单局面 `(9,19,19)`，并提供
  `(9*19*19,batch)` 的 neuralnet 列布局。
- ✅ 从真实 `.nngames` 对局经由 KataGo trace 提取第 100 手特征，输出
  [`PNG`](../benchmarks/m3-feature-planes.png) 与
  [`NumPy 张量`](../benchmarks/m3-feature-planes.npy)。
  格式见 [`08-data-and-features.md`](08-data-and-features.md)。

## 7. 算力预算（推导口径，M0 用实测替换）

基准事实（上游实测，40HX）：FP32 理论 9.14 TFLOPS，GEMM 达 45%（4125 GFLOPS @4096³），
小形状有效 **2–4 TFLOPS**（f16 存储）；nn GEMM 为 cuBLAS 的 54–67%。
V100 折算仅作参考、**不投入装机**（D3）：HBM2 900 GB/s vs 448，估 1.7–2.2×，
但无 CUDA → 张量核心不可达，且数据中心驱动的 Vulkan ICD 可用性未验证。

| 阶段 | 工作量 | 40HX（M0 实测，不含数据 I/O） | V100（参考折算，按 D3 不投入） |
|---|---|---|---|
| 监督预训练（10 万局 ≈ 1800 万局面，2–3 遍，在线增广）——**本库唯一训练阶段** | ~7 万 step/遍 @ batch 256 | B-small ≈ **694 pos/s**；B-final ≈ **366 pos/s** | 未实测；按 D3 不投入 |
| ~~自对局微调 / RL 开销~~（**随 D9 移出**，保留为上游 #23 落地后的参考口径） | 见 §3 推导 | （1–2 周） | （0.5–1 周） |

网络规格与推理预算（推导用）：

| 规格 | 参数量 | 单样本 FLOP | 用途 |
|---|---|---|---|
| B-small：d=192 / 8 层 | ~4M | ~1 GFOP | M4 竖切、9×9 沙盒 |
| B-final：d=256 / 12 层 / d_ff=1024 | ~10–15M | ~2.3 GFLOP | 正式训练与对弈 |

**推理预算是硬约束**：网络每大一倍，同等时间下 visits 减半；10–15M 是 40HX 上
"网络质量 × visits 数"的甜点区，超过 15M 先跑 M0 基准再决定。

**M0 组件基线（40HX，batch=64，B-small / B-final）**：单层 Transformer
forward **7.818 / 9.128 ms**；按 1×Conv2D + N×Transformer + 1×Linear 求和，
估计 **915 / 578 positions/s**。B=256 训练估计 **694 / 366 positions/s**。
该数字不含 MCTS 树操作、数据加载与 optimizer update（`layer_bench` 的 train 口径 =
fwd+bwd）。M3 已将输入冻结为 9 个特征平面；M0 的 17 平面是当时用于层基准的保守
占位，不代表最终拓扑。Conv stem 层数与 8 个 attention head 仍需在 M7 定案，
并用真实 `GoModel` 端到端基准替换组件求和。

## 8. 开放问题

| 问题 | 阻塞 | 备注 |
|---|---|---|
| 赌约条款（visits 上限 / handicap / 是否允许 KataGo） | 不阻塞开发；阻塞 §2 目标定档 | 建议：固定 visits + 禁 pondering |
| D6 补丁政策（零补丁 vs `ModelType::Go` 5 行补丁） | 不阻塞（M7 前拍板即可） | 零补丁语义略别扭但可用 |
| 特征平面 v2 消融（气 / 超级劫历史 / 征子 / 提子分布） | 不阻塞 M4 | M3 v1 已固定 9 平面；先跑通 C 验证器，再以消融决定增量 |
| 项目级 RL 应用层（围棋环境 / 自对局 / 领域 buffer） | 不阻塞 M0–M9；若启用则阻塞方案 2 Step 2 | 上游 `v1.12.0` 已提供 PPO/GRPO 与 rollout 原语；按 D9，本轮不实现，重开范围时再排期 |
