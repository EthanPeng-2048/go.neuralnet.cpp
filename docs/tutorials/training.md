# 训练教程：从数据集到可用模型

> 面向第一次跑通完整训练链路的读者。所有命令在仓库根目录执行，产物全部落在
> 不进 Git 的路径（`.cache/`、`data/`、`*.nnckpt`/`*.nnmodel`）。
> 每节给出最小可跑命令与产物；出错处理集中在 [§2.2](#22-sha-256-对不上重新下载)
> 与 [§8 常见错误速查](#8-常见错误速查)。

## 0. 前置条件

按 [`CONTRIBUTING.md`](../../CONTRIBUTING.md) 完成一次 CPU 构建与虚拟环境：

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -e .

cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu --output-on-failure
```

教程命令依赖以下已构建产物（`--preset gpu` 时把 `build/cpu` 换成 `build/gpu`）：

| 产物 | 作用 |
|---|---|
| `.venv/bin/go-pipeline` | 数据管线（下载/入库/验收/词表/特征/训练缓存） |
| `build/cpu/go_data_cli` | `.nngames` 打包、校验、导出 |
| `build/cpu/go_rules_cli` | KataGo 规则回放与轨迹 |
| `build/cpu/go_train_cli` | 训练、续训、评估（neuralnet.cpp） |

## 1. 链路总览

```text
CWI 棋谱归档 (games.tgz, SHA-256 钉死)
  → go-pipeline download        校验 + 解压           .cache/cwi/games/*.sgf
  → go-pipeline ingest          过滤 19×19 + 编码      data/cwi-10k.nngames
  → go-pipeline verify          逐字节可逆 + KataGo 重放验收
  → go-pipeline training-cache  KataGo 轨迹 → 特征缓存 data/m4-c-train.ngf / m4-c-eval.ngf
  → go_train_cli train          监督训练               run.nnckpt / run.nnmodel / run.jsonl
  → go_train_cli evaluate       评估                   JSON（loss/top1/top3/value_auc）
  → go_gtp_cli / go_arena_cli / go_gui_cli             对局与评估
```

## 2. 数据集：下载与 SHA-256 校验

### 2.1 下载

```bash
.venv/bin/go-pipeline download
# {"archive": ".../.cache/downloads/cwi-go-games.tgz",
#  "games_root": ".../.cache/cwi/games"}
```

下载源是项目 Releases 上的字节快照（tag `data-cwi-games-v1`）——2026-10-25
上游 CWI 重新发布归档后固化，任何时间下载都得到同一字节，上游漂移不会
再让校验失败；原始上游地址只作为 provenance 记录。下载器把归档写到
`.cache/downloads/cwi-go-games.tgz`：先落
`.tgz.part` 临时文件，**完整下载后才原子改名**；网络错误自动重试 4 次
（指数退避）。文件就位后计算 SHA-256，与代码中钉死的
`CWI_ARCHIVE_SHA256`（`python/go_pipeline/cwi.py`，同时记录在
[`DEPENDENCIES.md`](../../DEPENDENCIES.md)）比对，不一致即报错：

```text
RuntimeError: archive checksum mismatch:
  expected c57ad2ba3c5f3a3ee145bb77bcc9c0167c0d7b5478cf691a8939f750511071e2,
  got <实际值>
```

### 2.2 SHA-256 对不上：重新下载

先理解机制，再对症处理：

- **校验失败不会自动重试下载**（防止对确定性坏内容反复拉取），需要手动触发；
- **只要 `.cache/downloads/cwi-go-games.tgz` 已存在，`download` 和
  `ingest` 都不会重新下载**，只做校验 —— 一个坏的缓存文件会永远报同一个错，
  直到你强制刷新或删掉它。

处理步骤：

```bash
# 1) 看实际哈希，确认报错来自当前缓存文件
sha256sum .cache/downloads/cwi-go-games.tgz

# 2) 强制重新下载（推荐做法）
.venv/bin/go-pipeline download --refresh

# 等价做法：删掉缓存再下（.part 是中断残留，一并删掉）
rm -f .cache/downloads/cwi-go-games.tgz .cache/downloads/cwi-go-games.tgz.part
.venv/bin/go-pipeline download

# 3) 归档没问题、但怀疑解压产物（extract_archive 见到已有 games/*.sgf
#    会直接复用，不会重解压）→ 连解压目录一起清掉再来一次
rm -rf .cache/cwi
.venv/bin/go-pipeline download

# 4) 网络不稳时手动下载，交给管线做校验（文件已存在时管线只校验、不覆盖）
curl -L --retry 5 -o .cache/downloads/cwi-go-games.tgz \
  https://github.com/EthanPeng-2048/go.neuralnet.cpp/releases/download/data-cwi-games-v1/cwi-go-games.tgz
.venv/bin/go-pipeline download
```

判断重新下载是否"修好了"：

| 现象 | 结论 | 动作 |
|---|---|---|
| `sha256sum` 等于 `c57ad2b…` | 归档完好 | 继续 §3 |
| `got` 每次都不同 | 传输持续被破坏（代理/运营商劫持、磁盘坏道） | 换网络或用上面第 4 步手动下载 |
| `got` 每次相同但 ≠ 期望值 | 下载到的不是钉死资产（镜像/代理劫持/资产被误换） | 核对 `gh release view data-cwi-games-v1` 的资产哈希，换网络重下 |

钉死的期望值（与
[`DEPENDENCIES.md`](../../DEPENDENCIES.md) 一致）：

```text
c57ad2ba3c5f3a3ee145bb77bcc9c0167c0d7b5478cf691a8939f750511071e2
```

**上游变了怎么办**：`download` 现在钉死在项目 Releases 快照上，上游漂移
**不再影响**新用户的下载。这个策略正是被真实事件固化的：2026-10-25
上游重新发布 `games.tgz`（新哈希 `9a7e4069…`，仅在 `games/people/
Cho_Hunhyun/` 下新增 54 局、无删除无修改；新增文件排序位置在入库扫描
窗口 10,066 之外，1 万局数据集身份不受影响），项目随即把旧版字节快照到
tag `data-cwi-games-v1` 并把下载点切过去 —— 归档/数据集 SHA-256 均未变，
零重验收。仍然撞到 checksum 错就按上面步骤重下。想跟进上游新增棋谱
属于数据集变更：需要新 tag、新 SHA-256，并按
[`M2/M3 报告`](../benchmarks/m2-m3-data.md) 口径重新验收；归档变化会连带
下游身份变化（`.nngames`、`.ngf` 的 SHA-256 都变），**旧 checkpoint 不能
再 `--resume`**（身份门禁硬错，见 §6），需从推理模型重开一轮训练。
Release 资产视为不可变：改动它会破坏所有已记录的身份。

> 附加参数：`--archive PATH` 指定归档落点，`--destination PATH` 指定解压
> 目录；不给时用上面的 `.cache/` 默认路径。

## 3. 入库为 `.nngames` 并验收

```bash
.venv/bin/go-pipeline ingest --limit 10000
.venv/bin/go-pipeline verify --sample 1000
```

`ingest` 过滤出 19×19 对局，经 KataGo 规则编码为
`data/cwi-10k.nngames`，并在 `data/cwi-10k.nngames.manifest.json` 里记录
对局数、手数、数据集 SHA-256 与归档 SHA-256。`verify` 做逐字节可逆 +
抽样 KataGo 重放验收。当前钉死口径（M2/M3 实测）：

| 项 | 值 |
|---|---:|
| 对局 | 10,000 |
| 手数 | 2,076,112 |
| 局面 | 2,086,112 |
| 数据集 SHA-256 | `bc4ea0c7…e49d2313` |

重新下载归档后，对比 manifest 里的 `dataset_sha256` 与上表，即可确认
"同一归档 ⇒ 同一数据集"的可复现性没有被破坏。

## 4. 生成训练缓存 `.ngf`

```bash
.venv/bin/go-pipeline training-cache \
  --max-games 1000 --eval-games 100 --seed 20261024 \
  --history-len 64 --manifest .cache/cache-manifest.json
```

管线按固定 seed 抽对局、用 KataGo 重放生成 9 平面特征、策略标签、价值
标签与最近 K=64 手历史，写成只读 `.ngf` 缓存（格式见
[`09-training-cache.md`](../design/09-training-cache.md)）：

- 产物：`data/m4-c-train.ngf`（训练）与 `data/m4-c-eval.ngf`（验证）；
- 当前缓存是 **v2**（带历史段），B 模型必须用 v2；`--history-len` 要与
  训练时的 `--history-len` 一致（默认都是 64）；
- 磁盘要留足：v2 每条记录 13,148 字节（≈13 KB/局面），
  1,000 局 ≈ 2.7 GB，M8 用的 3,000 局实测 8.2 GB，10,000 局约 27 GB；
- 用训练器查看缓存元信息：

```bash
./build/cpu/go_train_cli inspect --dataset data/m4-c-train.ngf
# {"format_version": 2, "records": ..., "history_len": 64, ...}
```

只想先跑通链路时，用 40 局做冒烟（M5 验收同款）：
`--max-games 40 --eval-games 10`，产物只有几 MB。

## 5. 训练 B 模型（主线）

```bash
./build/cpu/go_train_cli train \
  --dataset data/m4-c-train.ngf --eval-dataset data/m4-c-eval.ngf \
  --arch b --epochs 4 --batch-size 64 --lr 0.001 --seed 20261024 \
  --checkpoint run.nnckpt --model run.nnmodel \
  --metrics-out run.json --metrics-log run.jsonl
```

要点：

- `--arch b` 是默认值（B 双分支编码器 + 策略/价值双头）；`--arch c` 是
  M4 的 CNN 验证器，只用于 B/C 对照与回归，不带价值头；
- 默认配方：AdamW、cosine 退火（`--schedule cosine|constant|step_cosine`）、
  lr 1e-3 → min-lr 1e-5、weight-decay 1e-4、clip-grad-norm 1.0、
  warmup 0；架构默认 d_model=192 / 8 层 / 8 头 / d_ff=768 / K=64，
  可用 `--d-model --num-layers --num-heads --d-ff --conv-channels
  --history-len --value-weight` 覆盖；
- CPU 构建加 `--cpu`；GPU 用 `--gpu` 或 `--gpu=设备名片段`（如
  `--gpu=40HX`）。小显存 GPU 记得降 batch：M7 实测 GTX 850M 上
  batch 32 直接 `vkAllocateMemory failed`，batch 16 可行；
- 产物：`.nnckpt`（训练态 + 优化器状态，续训用）、`.nnmodel`（纯推理
  模型，喂给 GTP/Arena/GUI）、`.json`（最终指标）与 `.jsonl`（逐 epoch
  追加）；
- 每个 epoch 结束时 stderr 打一行
  `epoch i/N train_loss=… eval_loss=… top1=… top3=… value_auc=… seconds=…`。

## 6. 中断续训与指标

断点永远写在 epoch 边界，崩溃最多丢一个 epoch 的进度：

```bash
# 训到第 2 个 epoch 后存档退出
./build/cpu/go_train_cli train ... --stop-after-epoch 2

# 补完剩余 epoch（指标续写同一个 JSONL）
./build/cpu/go_train_cli train ... --resume run.nnckpt
```

断点续训与一口气训练**逐位一致**（M5 在真实 CWI 数据上验收，见
[`m5-trainer.md`](../benchmarks/m5-trainer.md)）。`.jsonl` 以
`run` / `epoch` / `final` 三种记录跨进程追加，`global_step` 跨续训累计。

续训有三道硬门禁，违反即报错（这是设计，不是故障）：

1. **数据集身份**：`.ngf` 的 SHA-256 与样本数必须与断点记录一致 ——
   换数据集 = 新任务，请从 `.nnmodel` 重开一轮；
2. **`--epochs` 必须与断点配方一致**（lr 曲线形状跨续训不能变）；
3. **`.nnmodel` 不能当断点**（推理模型没有训练状态段）。

## 7. 评估与对局

```bash
# 单独评估一个推理模型（JSON 输出：loss/top1/top3/value_auc）
./build/cpu/go_train_cli evaluate \
  --dataset data/m4-c-eval.ngf --model run.nnmodel

# 之后拿模型去下棋（详见 README 对应章节）
./build/cpu/go_gtp_cli --model run.nnmodel          # Sabaki 可接入的 GTP
./build/cpu/go_arena_cli --player model:run.nnmodel \
  --opponent "gtp:gnugo --mode gtp --quiet" --games 100
./build/cpu/go_gui_cli --model run.nnmodel          # 浏览器对弈/竞技台
```

B 模型的评估比 C 多两个口径：`top3`（策略前三命中）与 `value_auc`
（价值头对局势的排序质量）；C 没有价值头，这两项恒为 0，且双头/单头
loss 绝对值不可比 —— B/C 对照请用同一份 `.ngf` 缓存比 eval top-1
（见 [`m7-b-vs-c.md`](../benchmarks/m7-b-vs-c.md)）。

## 8. 常见错误速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `archive checksum mismatch: expected …, got …` | 归档损坏 / 传输被劫持 / 上游变更 | [§2.2](#22-sha-256-对不上重新下载)：`download --refresh` 或删缓存重下 |
| `download` 秒回且不显示下载进度 | 缓存文件已存在，只做了校验 | 需要重下时用 `--refresh` |
| `ingest` 报同样的 checksum 错 | `ingest` 也会校验已存在的归档（不重下） | 先按 §2.2 修好归档，再 `ingest` |
| `B model needs move history but cache '…' has history_len=0` | 喂了 v1 缓存（无历史段）给 B 模型 | 用 `training-cache --history-len 64` 重新生成 v2 缓存 |
| `feature cache history span has wrong size … K=…` | 训练 `--history-len` 与缓存 K 不一致 | 两边统一为 64，或按报错提示重新生成缓存 |
| `--resume 的数据集与断点不一致` | `.ngf` 内容身份变了（重生成/换数据） | 换数据集属新任务：从 `.nnmodel` 重开，不要 `--resume` |
| `--epochs 与断点配方不一致` | 续训时改了总 epoch 数 | 用与断点相同的 `--epochs`；真要改配方就重开一轮 |
| `vkAllocateMemory failed`（GPU） | batch 超出显存 | 降 `--batch-size`（850M 实测 16 可行、32 不行） |
| `only found N usable 19x19 games` | 数据集不足 `--limit` 局 | 降低 `--limit` 或换数据源（换源按 §2.2 上游变更流程） |

## 相关文档

- 数据与特征格式：[`docs/design/08-data-and-features.md`](../design/08-data-and-features.md)
- `.ngf` 训练缓存：[`docs/design/09-training-cache.md`](../design/09-training-cache.md)
- 数据源版本与 SHA-256 唯一真源：[`DEPENDENCIES.md`](../../DEPENDENCIES.md)
- 验收证据：[`m2-m3-data.md`](../benchmarks/m2-m3-data.md)、
  [`m5-trainer.md`](../benchmarks/m5-trainer.md)、
  [`m7-b-vs-c.md`](../benchmarks/m7-b-vs-c.md)
