# 方案 1（主线）：Encoder Transformer 围棋模型

> 状态：**v1.1：M3 数据/特征首版实装**（2026-10-24）· 对应路线图 D1 / D5 / D12 ·
> 方案 2（Decoder
> 思考模型）见 [03-scheme2-decoder-thinking.md](03-scheme2-decoder-thinking.md)，
> 两者关系见其 §7
> 上游库：[`neuralnet.cpp` `v1.12.0`](https://github.com/EthanPeng-2048/neuralnet.cpp/tree/v1.12.0)
> （CPU/Vulkan，无 CUDA；版本锁定见 [DEPENDENCIES.md](../../DEPENDENCIES.md)）

## 1. 定位

B 方案的正式设计文档：**双分支编码器 + decision query + 策略/价值双头**，
一次前向直出坐标，MCTS 叶子评估网络。19×19 特化（D2/D5）。

## 2. 为什么是 Encoder（而不是 Decoder LM / 纯 CNN）

| 论据 | 说明 |
|---|---|
| 历史全可见 ⇒ 双向合法 | 推理时整段落子历史都是**过去**，第 5 手可以看第 30 手的应手——因果掩码是人为约束。Encoder 用 `PlainScoreMask`（双向），**上游现成**（`compute_layer_attention.hpp`），零库手术 |
| 输出 = 分类头，不是 token 生成 | 要的就是"一个坐标"，query token → `Linear(d, 362)` 天然吻合；decoder 的自回归解码机制在此是死重 |
| 相对 Decoder LM（方案 2） | 方案 2 是它的**进阶变体**（同一盘面分支 + 思考链），不是替代：encoder 先交付，decoder 思考层后加（依赖 MCTS 蒸馏，见 03 §5） |
| 相对纯 CNN（C 验证器） | CNN 丢掉全部落子历史（定式次序、劫争来往、征子走向）；且 C 是棋盘尺寸锁死的，B 的 token+位置编码结构天然支持 9/13/19 变尺寸（测试期沙盒可用） |

## 3. 架构总览（19×19 特化）

```
盘面分支:  C 个特征平面 (C, 19×19)
           → Conv stem（Conv2D+BN ×N，残差在复合层内部实现，先例 = GPTBlock）
           → 下采样为空间 token 序列（候选见 §3.1，默认 5×5 = 25 个区域 token）
历史分支:  最近 K=64 手 move ids → gather_rows embedding
           + 1D 位置编码（Learned/Sinusoidal，现成 PositionEncoder）
           + 段嵌入（可学习，加法，复合层内 add_inplace）
决策:      <DECISION> query token（可学习）
           ↓ 拼接单序列 [25 区域 | 64 历史 | 1 query] ≈ 90 token
           Plain 双向掩码 TransformerEncoderLayer ×N（现成）
           ↓
           query 行（slice_rows 取出）→ Linear(d, 362) 策略头（361 点 + pass）
                          └→ Linear(d, 1)    价值头（tanh）
           ↓ 打包为单张量 (363, batch) 单进单出（insert_rows）
           GoLoss（自定义 Loss 派生）：前 362 行 CE + 末行 MSE
```

### 3.1 盘面 → 序列：三个候选（含 19 质数问题）

19 是质数，**任何等分 patch 划分都不存在**，k×k 网格只能靠下采样 + padding：

| 候选 | 序列代价（总 T） | 单样本 FLOP* | 信息瓶颈 | 增广兼容性 |
|---|---|---|---|---|
| (a) 全局池化 → **1 board token** | 66 | 2.0 GFLOP | **有**：全盘细节压缩进 1 个 d=256 向量 | 完全对称，安全 |
| (b) **5×5 = 25 区域 token**（默认） | 90 | 2.7 GFLOP | 轻 | pad 20 再 4×4 均分（补 1 格非对称）⇒ 8 对称增广在**输入侧**做、每次重新提取特征，镜像对应有半格噪声（已知缺陷，可接受） |
| (c) 361 全格 token | 426 | 12.8 GFLOP | 无 | 安全，但推理慢 **4.8×** = 同等时间 visits 减半，违反 D5 |

\* B-final（~15M 参数）口径，`2 × params × T`。

**默认 (b)**，消融计划覆盖 (a)/(b)/(c)（§9）。兜底设计（若 (b) 消融不佳）：
**盘面细节走 CNN 直通头**——1×1 conv 直接在特征图上出 361 logits（AlphaGo 式，
无瓶颈、零序列代价），与 query logits 相加融合；价值头仍走 transformer 侧。
该兜底不需要新层，只是策略头多一条加法路径。

### 3.2 关键张量布局（上游约定）

| 张量 | 形状 | 说明 |
|---|---|---|
| 特征平面 | `(C·19·19, batch)` | 卷积列布局，先例 = `domain_cnn.hpp` |
| 序列 | `(d, batch·T)` | batch-major 列序，先例 = `GPTModel` |
| query 行 | `(d, batch)` | `slice_rows` |
| 打包输出 | `(363, batch)` | `insert_rows`；策略 362 行 + 价值 1 行 |
| GoLoss 目标 | `(363, batch)` | 策略目标（软/硬）+ 价值目标 z∈{−1,0,+1}（**从当前手方视角**） |

## 4. 特征平面清单（v1 已实装）

| 平面 | 说明 | 备注 |
|---|---|---|
| 己方子 / 对方子 | 2 | 按 to-play 翻转（颜色无关） |
| to-play 奇偶 | 1 | 常量平面 |
| 简单劫禁着点 | 1 | 规则系统直接给出 |
| 最近 3 手标记 | 3 | 落点 one-hot，捕捉近因 |
| 空位 | 1 | |
| 边界距离编码 | 1 | 非空平面，辅助气/眼直觉 |

当前实现为 `(9,19,19)` 的 `float32` 张量；详细定义、scalar 输入与 neuralnet 列布局
见 [`08-data-and-features.md`](08-data-and-features.md)。

标量输入（komi、手数、提子差）**不进平面**：以 embedding 加到 board token
（复合层内加法）。v2 消融再考虑超劫相关历史平面。合法性由规则系统在
host 侧给出，**不进网络**（见 §7）。

## 5. 训练配方（监督，D4 管线阶段 1）

| 项 | 配方 |
|---|---|
| 数据 | 10–30 万局 19×19 人类棋谱（`.nngames`，D7），在线随机 8 对称增广 |
| 策略损失 | CE（362 类）。**验收项**：确认上游 `CrossEntropyLoss` 接受软目标（公式 `Σ target·log p` 天然接受，实现侧 M7 验证；不行则 GoLoss 内自写 softmax-CE，DSL 表达有先例） |
| 价值损失 | MSE(z, v)，z ∈ {−1,0,+1} 从当前手方视角（每局每手生成两个视角样本或单视角 + 签号翻转） |
| 优化器 | AdamW + warmup+cosine（`lr_scheduler.hpp` 现成）；f16 存储按 `PrecisionProfile` 注入 |
| 增广 | 8 个二面体变换，**在输入侧做**（平面 + 坐标同步变换，重新提取特征）；§3.1(b) 的半格噪声记录在案 |

## 6. 规格与成本（M0 基准校准前的估算）

| 规格 | 参数量 | T | FLOP/次 | 有效吞吐（40HX） |
|---|---|---|---|---|
| B-small：d=192 / 8 层 / d_ff=768 | ~4M | 90 | ~0.7 GFLOP | ~2–4k 次/秒 |
| B-final：d=256 / 12 层 / d_ff=1024 | ~10–15M | 90 | ~2.7 GFLOP | ~1–1.5k 次/秒 |

口径：有效 2–4 TFLOPS（f32 存储，上游实测 GEMM 达理论 45%）；**估算，M0 用
`layer_bench` 在 B-small/B-final 实测替换**。下表仍是推导值；f16 存储的
“40HX −3.2% 端到端”属于 v1.8.2 历史口径，**不得直接当作 v1.12.0 实测**，
M0 必须在锁定版本上重新建立基线。

## 7. 合法性与推理集成

- **训练期**：CE 目标在非法点恒为 0，梯度 `p_i − 0` 自然把非法 logits 压低——
  合法性**统计涌现**，但不保证。
- **推理期**：host 侧采样/先验归一化前将非法点置 −inf（自写采样器，不动库内
  `sample_next_token_`）。MCTS 先验只在合法集上归一化——**规则系统是唯一真源**。
- **MCTS 叶子评估**：batch × 90 token 单次前向；无自回归 ⇒ 无 KV cache 需求
  （方案 2 才需要，先例 = `GPTBlock::forward_step`）。

## 8. 库接入点（零上游改动，D6 默认零补丁）

| 扩展 | 方式 | 先例 |
|---|---|---|
| `GoModel` 复合层 | `Layer` 派生；残差/段嵌入/打包全部在层内，`param_children()` 转发 | `GPTModel` |
| 双头单进单出 | `(363, batch)` 打包 + `GoLoss` 分行（`Loss` 派生） | `Loss` 纯虚基类 |
| embedding | `gather_rows` 查表 | `GPTModel` token_emb |
| ModelSpec | 复用 `type=Transformer` + `patch_size` 字段记盘面网格（D6）；Plan B = `ModelType::Go` 5 行补丁 | `spec_matches` Transformer 分支 |

## 9. 实验计划

**主对照**：同数据、同训练器下 B vs C（验证器）——策略 top-1/top-3 准确率、
价值 AUC、vs GnuGo 胜率（Arena SPRT）。

**消融矩阵**（按预期收益排序）：

1. 盘面 token 数：1 / 25 / 361（§3.1）；
2. 历史长度 K：0 / 32 / 64 / 128（K=0 即退化为无历史变体，对照"历史序列是否真有增益"——这是本项目创新点的核心验证）；
3. 策略头融合：query-only vs +CNN 直通 logits（§3.1 兜底）；
4. 特征平面逐项移除（§4）；
5. 段嵌入有无、位置编码类型。

**里程碑映射**：M4（C 竖切）→ M7（B 替换 + 消融 1/2）→ M9（正式训练）。

## 10. 风险

| 风险 | 缓解 |
|---|---|
| 单 query 瓶颈（策略信息不足） | §3.1 兜底（CNN 直通 logits）；消融 3 前置 |
| 区域网格与增广的半格噪声 | §3.1 已知缺陷；消融对比"无增广/仅旋转 180°/全 8 对称" |
| 价值监督噪声（人类棋局胜负含运气成分） | buffer 混入 MCTS 自评估目标（上游 v1.12.0 已有 RL 基础设施，但当前按 D9 不启用自对局）；先用纯胜负基线 |
| 监督天花板（19×19 无自对局改进，D9） | 已知且接受：交付 = 监督 + MCTS 推理；对人胜率不受影响 |
