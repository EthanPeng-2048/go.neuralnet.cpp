# 问题总账、研究过程与 MCTS 开发史（M8–M10）

> 状态快照：**2026-10-28 20:37（Asia/Shanghai）**
> 性质：工作日志式档案——把「打不赢 GnuGo L1」这一主线问题的全部已知问题、
> 排查研究过程、以及 MCTS（M8）到 M10 价值头实验的开发历史钉死在一个文件里。
> 证据来源：docs/benchmarks/m8-mcts.md、docs/design/00-roadmap.md §6.8/§6.10、
> .cache/m10/*.jsonl、.cache/m8-*.json、git log、以及本文所记录的本地/远程排查。
> 本文只做归档与索引；每个问题的**修复**必须回到对应源文件与验收文档。

---

## 0. 一句话现状

MCTS 搜索树本身已验收通过（同一网络下搜索 6:0 碾压纯先验）；
**绑定瓶颈是价值头**：所有 B 模型对 GnuGo L1 累计 **0 胜 24 负**，
价值 AUC 始终在 0.50–0.51 死区。M10b 探针式空间价值头（--value-spatial 1）
是当前候选，epoch 2 双门（top1 > 20% **且** value_auc > 0.53）未过，
已按预设规则停训，等待下一步决策（见 §6）。

**2026-11-01 更新**：P2 结案（测试数据无符号下溢，非工具链）；M10b 诊断代码
（value_from_spatial 具名 spec 键 / debug_grad_norms / --grad-debug / 分项损失）
按 v1.14.1 开放规格落地进 main；训练器重写为 RunDriver + IndexedBatchSource +
session.save_checkpoint/resume（库口径断点编排）。

---

## 1. 目标与判据背景

- **最终目标**：赌约——19×19、低 visits 条件下赢下对手（人或其 KataGo）。
  GnuGo level 1 是当前代理里程碑（对手 KataGo 不可及，GnuGo L1 是最弱可自动化的
  人类以下基准）。
- **架构**：B 模型（双分支编码器：盘面卷积区域 token + K=64 历史 embedding
  → 90 token 单序列 → TransformerEncoderLayer ×N → decision query 行
  → 策略头 362 类 + 价值头 tanh，打包 (363, batch)，GoLoss = 稀疏 CE + 价值 MSE）。
  C（纯 CNN）只当基础设施验证器；A 已删。
- **MCTS 验收（M8 DoD）已达成的部分**：同一网络下 mcts:128 vs 纯先验 argmax
  **6:0**；40HX batch-1 126 visits/s。
- **未达成的部分**：vs GnuGo L1 胜率。归因为网络质量（价值头 + 数据量），不是树。
- **M10 双门判据（预设，用户确认过）**：每个 epoch 后检查
  top1 > 20%（策略健康）**且** value_auc > 0.53（价值头活着）；
  AUC ≤ 0.51 → 停训并做价值梯度诊断；0.51–0.53 → 再等一个 epoch。

---

## 2. 问题总账（按严重度排序）

### P1【主线·未解】价值头学不到东西（value AUC 死区 0.50–0.51）

- **现象**：所有 B 模型在共享评估集（~104k 价值样本）上 AUC ≈ 随机：
  - v1 死头谱系：m8-b-big 0.5006、m10-b（5 epoch）0.5005
    （轨迹 0.511 → 0.516 → 0.500007，后期塌向常数）；
  - M10b 探针谱系：m10-b3 init 0.4951 → epoch1 0.5080 → epoch2 0.5109
    （**不塌常数**，疑似梯度在流但信号弱）。
- **代价**：MCTS 叶子评估≈无效 ⇒ 搜索退化为先验搜索 ⇒ 对 GnuGo 0–24
  （详见 §4 汇总表）。
- **已排除的解释**：
  - 价值**目标**健康：两个缓存抽样核对——train 8512 样本
    −1:4206 / 0:46 / +1:4260；eval 8639 样本 −1:4243 / 0:127 / +1:4269
    （左右基本均衡，零样本 <2%）；
  - 策略头健康（m10-b3 epoch2 top1 30.5%），排除整体训练崩溃；
  - trunk 洪泛（m10-b2 事故）已通过「价值头不回传 trunk」规避——但这本身
    是一把双刃剑：**探针读的是 LayerNorm(r2)，其梯度只训练价值头自己的
    权重，盘面/历史特征的质量只能间接受益**。
- **未做的关键实验**（被 P6 的 stash 状态卡了一半）：价值头/价值归一化的
  **梯度量级实测**——区分「梯度死了」vs「梯度活着但监督信号弱」。
  诊断代码已写完但躺在 git stash（见 P6）。
- **候选方向**（待拍板，见 §6）：① --grad-debug 实测梯度后再对症；
  ② 探针改读 transformer 之后的特征（当前 r2 是 conv 输出区域 token，
  未经过注意力融合）；③ trunk 归一化后开联合塑形（回到 v2 但修好洪泛根因）。

### P2【已结案·2026-11-01】稀疏 CE "返回垃圾" = 测试数据无符号下溢（不是上游/工具链 bug）

- **现象（旧记录）**：CPU 构建中，forward_sparse（logsumexp fold 路径）
  输出 −0 或 1e18 量级垃圾：batch=1,2 → ce=-0；batch≥3 → 未初始化内存值；
  logsumexp 中间量为 inf。发现路径：给 GoLoss 测试加「分项损失为正」断言
  → 失败 → debug 打出 combined=0.729167 policy=-0.000000 value=0.729167
  （combined 纯粹是价值 MSE）。
- **结案结论（2026-11-01，WP1 落地诊断埋点时顺手破案）**：
  - 此前所有"复现"探针（含纯头文件的 /tmp/ce_sweep.cpp）都用
    `(i % 7) - 3` 构造测试 logits——`i` 是 `std::size_t`，余数 < 3 时
    `0-3` 下溢成 `2^64-3` ≈ 1.8e19，转 float 得 1.8e18 级垃圾。
    **exp(1.8e18) = inf ⇒ CE 输出 NaN/−0/1e18，全是构造出来的烂数据**；
  - 决定性证据：同一引擎、同一路径，把 `(i % 7) - 3` 换成
    `static_cast<int>(i % 7) - 3` 后，classes ∈ {8,64,362} ×
    batch ∈ {1,3} 全部返回正确 CE（1.86–6.01）；
  - 位模式验证：write 后 read 回来的 back[0] 位模式 = `0x5dcccccd`
    （= 0.1f 的高位被下溢整数污染），与源缓冲逐位相同——引擎读写无误，
    错在宿主侧数据构造；
  - **上游 v1.14.1 的 fold logsumexp / 稀疏 CE 实现无 bug**；工具链嫌疑
    （clang-24 / libstdc++ 16 与症状同期升级）为**误判**，时间线巧合。
    此前的"与优化级别/编译器无关"排查之所以没抓到，是因为所有复现体
    都复制了同一个错误构造。
- **修复**：tests/go_model_tests.cpp 三处 `(size_t 表达式) - 正数` 改为
  先转 `int` 再做减法（test_forward_backward 平面数据 / grad_data /
  test_go_loss logits）。修复后 go_model_tests 全绿。
- **解除的限制**：「禁止本机 CPU 训练结论入库」自 2026-11-01 起作废。
- **教训**：大批量"与工具链无关、纯头文件复现"的排查里，先检查
  **测试数据构造本身**（位模式验证 > 推断时间线）。

### P3【已修复·教训】m10-b2 trunk 洪泛事故

- v2 接法让价值头梯度回传 trunk；clip_grad_norm 是**全局**范数，
  价值分支的大梯度淹没策略分支 → epoch1 top1 从 27.8% 崩到 **0.8%**
  （m10-b2.jsonl：loss 7.37、top1 0.76%、AUC 0.491）。
- 修复：探针式价值头（M10b）在 value_norm_ 处**截断回传**，
  策略恢复（m10-b3 top1 29→30.5%）。但代价见 P1——特征侧对价值头近乎盲。
- 教训：多头损失 + 全局梯度裁剪时，头间梯度量级差一个数量级就会互相投毒；
  要么分组裁剪，要么把价值监督做成对归一化特征的读出（现状）。

### P4【数据】训练规模仍是已知短板（与 P1 独立）

- 当前主力数据：CWI 10k 局（m10-train-10k.ngf，1,986,753 样本）
  + 500 局评估集。M7 消融已证明数据规模是 B 的第一杠杆
  （30 局 B 过拟合 / 910 局 B 欠拟合、C 随数据线性涨）。
- 路线图 M9 计划 10–30 万局。数据本身健康（P1 已排除目标问题）。

### P5【运维·已绕过】Windows 远程环境杂项（已知 workaround）

- WSL 代理横幅「已设置 http…」污染 ssh stdout → 取流时 grep -v '^已设置'；
- 传文件：ssh … 'wsl -e bash -c "cat > path"' < file；
- 本机（Linux dev box，GTX 850M）只做开发与 CPU 测试，训练权威口径 =
  Windows 主机 192.168.1.102（ethan）的 CMP 40HX 原生 Vulkan；
- WSL 内 2060S GPU 已废弃（Dozen descriptor pool 耗尽）。

### P6【已清偿·2026-11-01】诊断代码在 stash，工作树有临时 printf

- ✅ **已清偿（2026-11-01）**：stash 内容按 v1.14.1 开放规格重做后落进
  main（value_from_spatial 从 layer_dims[7] 魔数改为 GoTransformerSpec
  具名键 `value_from_spatial`，旧 spec 缺键回退旧接线；[dbg] printf 噪声
  剥除；新增价值头路由回归门 test_value_head_routing + 分项损失恒等式）。
  stash 已 drop。
- ⚠ 仍未在 GPU 管线上验证（Windows 构建未同步）——M10b 下一轮训练前先
  touch 后重建 build\gpu。

### P7【历史事故账】（均已修复，留档防再犯）

| 事故 | 症状 | 根因 | 修复 |
|---|---|---|---|
| 价值目标符号 | AUC 恒等 -1 | black vs b 字段恒等 -1 | M7 期修复 |
| AUC 并列名次公式 | 数值错 | 并列名次未平均 | M7 期修复 |
| 续训 K 取错源 | 特征错位 | 从 --history-len 而非断点 spec 取 K | M7 期修复 |
| 数据边界棋谱 | 整锅缓存硬失败 | cwi:CJShuko/1988/40 第 288 手颜色不交替 | 跳过整局 + 记 manifest |
| Windows GTP 管道 | 对局挂死 | 管道未配对 + 孤儿子进程 | arena 修复（9ed3f9f） |
| GnuGo PASS 大写 | genmove 解析失败 | GnuGo 3.8 返回大写 PASS | GUI 期修复 |
| untrained baseline 慢 | 评估卡死 | 基线评估未限样本 | 32k 上限（e3c80df） |

---

## 3. MCTS 开发史（M8，2026-10-25 落地）

### 3.1 实现（commit db66494，+1475 行）

- go::MctsSearch + go::LeafEvaluator（go_mcts 库，
  include/go/mcts.hpp、src/go/mcts.cpp 449 行）：
  - **PUCT 选择**（含 FPU 折扣）；
  - **先验重归一**：只在 KataGo legal_moves() 集合上归一，
    全零退化为均匀；
  - **negamax 备份**：边值按父方行棋方视角记账；
  - **终局备份**：pass-pass 走 KataGo 计分（中国规则）；
  - **转置表 = 评估缓存**：key 覆盖评估器所见的一切
    （盘面 + 行棋方 + 简单劫 + 近 64 手历史窗口——同时覆盖 C 的近期着法平面
    与 B 的 K=64 历史分支）；节点**不做** DAG 共享，命中只省一次评估，
    合法性永远按真实对局历史判定。
- **双线接入**（D1 的 C=验证器语义延续）：B 用价值头当叶子评估；
  C 无价值头走 ValueFallback::area_score（tanh(目差/15)）回退。
- **引擎链路**：arena / 竞技台 / GUI 共用引擎 spec
  mcts:VISITS:model:FILE[=DEVICE]；go_gtp_cli --visits N
  （--visits 0 = 纯先验，缺省行为不变）。

### 3.2 契约测试（go_mcts_tests，mock 评估器，无模型）

1. PUCT：0.9 先验点吃掉几乎全部根访问；
2. 根子先验在合法集上和为 1；
3. 占点上的非法先验永远进不了树；
4. negamax 符号：「白好 0.9」评估器对黑根出负值；
5. pass-pass 中国规则终局给黑备份 −1（空盘 + 7.5 贴目）；
6. 转置表：重复局面命中缓存、评估调用数下降。

test_m8_mcts.py 覆盖产品接线：GTP --visits（Sabaki 路径、合法应手）、
arena 引擎 spec 解析与畸形 spec 拒绝。

### 3.3 验收结果（docs/benchmarks/m8-mcts.md）

| 配对（同网络，唯一差异=选点） | 结果 | 证据 |
|---|---|---|
| mcts:128 vs 纯先验（40HX GPU） | **6–0** | .cache/m8-b-gpu-vs-prior.json（M8 DoD 过） |
| mcts:128 vs GnuGo L1 | 0–4 | .cache/m8-b-gpu-vs-gnugo.json |
| C 线 prior / mcts-256 / mcts-512 vs GnuGo | 0–4 / 0–4 / 0–4 | .cache/m8-c-*.json |

速度（batch-1，B 编码器 22MB）：开发机 CPU 13 visits/s；40HX Vulkan
**126 visits/s**（mcts-128 ≈ 1.0 s/手，纯先验 10.6 ms/手）。
树循环开销 ≈ 0（C 模型 1.1 ms/visit ≡ 一次 batch-1 前向）——
**MCTS 吞吐就是模型 batch-1 延迟**。

口径注记：原计划 9×9 沙盒，但 M3 v1 特征契约 19×19 专用
（PolicyPlayer 拒绝其它尺寸）⇒ 树验证用 5×5 mock 单测，
增益/速度测量在 19×19 做。

### 3.4 M8 立场结论（钉死，防止后续误判）

> **同一网络、同一搜索，6:0 打赢无搜索版本** ⇒ 搜索环没有问题；
> vs GnuGo 全负是**网络质量问题**。此后所有战力工作都应打在
> 价值头（P1）与数据规模（P4）上，不在树上。

---

## 4. 价值头实验史（M7 → M10，三代接法）

| 代 | 接法 | 代表 | 策略 top-1 | value AUC | 结局 |
|---|---|---|---|---|---|
| v1 | query 行 → Linear(d,1)（M7 原设计） | m7-b-smoke / m8-b-big / m10-b | 24.9% / 32.4% | 0.5006 / 0.5005 | 死头：后期塌向常数（0.511→0.516→0.500007） |
| v2 | 读 LayerNorm 特征、梯度回传 trunk | m10-b2 | **0.8%**（洪泛崩溃） | 0.491 | 事故：全局 clip 被价值梯度投毒（P3） |
| v3 = M10b | 探针：value_norm_->forward(*r2)，**回传截断** | m10-b3 | 30.5% ✓ | 0.5109 ✗ | 双门未过，epoch 2 停训；AUC 不塌常数（疑信号弱非梯度死） |

### M10 系列训练账

全部 40HX、batch 64、AdamW lr 1e-3 cosine、--value-weight 1.0、
CWI 10k 数据、共享 500 局评估集 ~104k 价值样本。

| run | 配方 | epoch 轨迹（top1 / AUC） | 结局 |
|---|---|---|---|
| m8-b-big | 30 局数据过拟合（前史） | 24.9% / 0.5006 | 0–4 GnuGo；6–0 纯先验 |
| m10-b | --value-spatial 0（v1 头），6 epoch | e5: 32.4% / 0.5005 | AUC 死区；vs GnuGo（纯先验）0–4（.cache/m10/m10-b-prior-vs-gnugo.json，SPRT LLR −0.42） |
| m10-b2 | v2 回传接法 | e1: 0.76% / 0.491 | 洪泛事故，即刻废弃 |
| m10-b3 | --value-spatial 1（M10b 探针），resume 续到 e2 | e1: 29.1% / 0.5080；e2: 30.5% / 0.5109 | 双门未过，按规则停训；checkpoint 停在 e2 |

关键 epoch 记录（m10-b3，来自 m10-b3.jsonl）：

~~~json
{"epoch":1,"eval":{"loss":4.3999,"top1":0.2911,"top3":0.4796,"value_auc":0.5080,"value_samples":104155},"train_loss":4.7986,"global_step":31044}
{"epoch":2,"eval":{"loss":4.2121,"top1":0.3047,"top3":0.4952,"value_auc":0.5109,"value_samples":104155},"train_loss":4.1151,"global_step":62088}
~~~

### GnuGo 战绩总账

| 时间线 | 模型 | vs GnuGo L1 | 备注 |
|---|---|---|---|
| M8（10-25） | C + fallback（visits 0/256/512） | 0–12 | C 无价值头，叶子=静态计分 |
| M8（10-25） | m8-b-bench + mcts-128 | 0–4 | 30 局过拟合网 |
| 10-26 | m8-b-big + mcts-128 | 0–4 | top1 24.9% |
| 10-27 | m10-b 纯先验 | 0–4 | 无搜索，确认先验本身不够 |
| 累计 | — | **0–24** | |

---

## 5. 研究过程实录（可复查的推理链）

### 5.1 价值头诊断链（P1）

1. **假设「目标坏了」→ 排除**：抽样两个 .ngf 缓存核对价值标签分布
   （train −1:4206/0:46/+1:4260；eval −1:4243/0:127/+1:4269）——健康。
2. **假设「整体训练崩了」→ 排除**：策略 top1 稳步 27→30%，loss 单调降。
3. **假设「v2 洪泛重演」→ 排除**：m10-b3 top1 正常。
4. **剩余假设**：梯度死（接线/尺度问题）vs 信号弱（r2 特征没价值信息）。
   ⇒ 需要**梯度量级实测**：debug_grad_norms()（embeddings/conv1/conv2/
   blocks/ln_f/policy_head/value_head/value_norm/cnn_policy 分组 L2）
   + --grad-debug N 每 N 步落 *.grads.jsonl。
5. **卡点**：加「分项损失为正」断言时引爆 P2（CPU CE=0），
   诊断代码转入 stash，梯度实验未跑。

### 5.2 CPU CE=0 排查链（P2，方法论留档）

~~~
断言失败(policy split ≈ -0)
→ debug: combined=0.729 = 纯 value MSE；lse=[inf inf inf]，sum=0/num_valid=3
→ git stash 全部改动 → HEAD 原码同症状 ⇒ 非本次改动引入
→ /tmp/ce_sweep.cpp 纯头文件探针（零 go 代码）复现
   batch 1,2 → -0；batch ≥3 → 未初始化垃圾值 ⇒ fold 路径输出从未被写入
→ 矩阵排除：-O0/-O1/-O2/-O3 × clang-24(-std=c++26 -fexperimental-library)
   × g++-15.2 × CpuEngine::create / cli 工厂 → 全部复现、症状一致
→ PCH 排除（探针不经 CMake）
→ 时间线取证：10-24 m7-c CPU 训练正常；10-25 起 clang-24 首装+多轮升级、
   libstdc++/gcc-16-base 同期更新 ⇒ 工具链/头文件升级嫌疑最大
→ 未定案。GPU 训练不受影响（真实 loss 构成分正确），本机 CPU 路径不可信。
~~~

### 5.3 已实施但未跑的实验清单

- [ ] Windows GPU 构建同步诊断源码（已进 main，touch 后重建 build\gpu）；
- [ ] 用 m10-b3 epoch-2 checkpoint + --grad-debug 200 短跑几百步：
  看 value_head/value_norm 的 grad_l2 与 policy_head 对比，
  以及 train_value_loss 是否有下降趋势；
- [x] P2 根因 —— **2026-11-01 结案**：测试数据无符号下溢（见 §2 P2），
  上游无 bug，无需 bisect 工具链。

---

## 6. 下一步决策点（2026-10-28 时点，待拍板）

| 选项 | 内容 | 前提/风险 |
|---|---|---|
| A | 恢复 stash，Windows GPU 跑 --grad-debug，用数据决定价值头生死 | 最小代价，回答 P1 的核心二分 |
| B | 修/绕 P2（钉死工具链版本或最小 bisect），恢复本机 CPU 测试可信度 | 与 A 不互斥；不修则本机测试长期带伤 |
| C | 放弃探针接法，改「trunk 归一化 + 联合塑形」（v2 修根因版：分组梯度裁剪） | 风险=P3 事故重演，需先落分组裁剪 |
| D | 探针改读 transformer 后特征（r2 未经注意力，信息量存疑） | 中等改动量 |

推荐顺序：**A →（并行）B → 依 A 结果选 C/D**。

### 附：m10-b3 续训命令（曾成功执行，当前进程已杀）

~~~
.\build\gpu\go_train_cli.exe train --dataset D:\Codes\m10\m10-train-10k.ngf --eval-dataset D:\Codes\m10\m10-eval-500.ngf --gpu=40HX --batch-size 64 --epochs 6 --warmup-epochs 1 --schedule cosine --lr 0.001 --min-lr 0.00001 --weight-decay 0.0001 --value-weight 1.0 --seed 20261030 --arch b --cnn-policy 1 --value-spatial 1 --history-len 64 --resume D:\Codes\m10\m10-b3.nnckpt --checkpoint D:\Codes\m10\m10-b3.nnckpt --model D:\Codes\m10\m10-b3.nnmodel --metrics-out D:\Codes\m10\m10-b3.json --metrics-log D:\Codes\m10\m10-b3.jsonl
~~~

### 附：工件清单（截至 2026-10-28）

| 工件 | 位置 | 说明 |
|---|---|---|
| m10-b.nnmodel | .cache/m10/（本地备份，md5 f61723c4f04d20b0592fa8ab68d63f15）+ D:\Codes\m10\ | v1 头，AUC 0.5005 |
| m10-b3.nnckpt | D:\Codes\m10\（68,081,792 B，10-28 0:38） | M10b epoch-2，当前最佳候选 |
| m10-b2.* | D:\Codes\m10\ | 洪泛事故留证 |
| m10-train-10k.ngf | D:\Codes\m10\（26.1 GB） | CWI 10k 局缓存 |
| m10-eval-500.ngf | D:\Codes\m10\ | 500 局评估集 |
| m8 原始战报 | .cache/m8-*.json | §3.3 表格来源 |
| 诊断探针 | /tmp/ce_sweep.cpp 等 | P2 复现用，易失 |
| stash diag-go-loss | 本仓库 git | §2 P6 |

---

## 7. 模型总谱（口径统一，供写文档/汇报引用）

| 模型 | 大小 | top-1 | value AUC | 对局证据 | 状态 |
|---|---|---|---|---|---|
| m4-c-validator | 1.5MB | 10.2% | —（无价值头） | vs random 3胜 | 退役（基础设施验证器） |
| m5 straight/split | — | — | — | — | 等价性证据 |
| m7-c-base | 1.5MB | — | — | — | C 基线 |
| m7-b-smoke | 15.7MB | — | — | — | B 冒烟 |
| m8-b-bench | 22.6MB | 2% | — | vs prior 6–0；vs GnuGo 0–4 | M8 搜索验收用网 |
| m8-b-big | 22.6MB | 24.9% | 0.5006 | vs GnuGo 0–4；vs prior 6–0 | v1 头 |
| m10-b | 22.6MB | 32.4% | 0.5005 | vs GnuGo（纯先验）0–4 | v1 头，5 epoch |
| m10-b2 | — | 0.8% | 0.491 | — | **失败**（洪泛事故） |
| m10-b3 | 22.6MB | 30.5% | 0.5109 | — | M10b 探针，停在 e2，当前候选 |

GnuGo L1 累计对局：**0–24**。
