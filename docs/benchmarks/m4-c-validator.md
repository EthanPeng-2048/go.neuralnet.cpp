# M4 C 验证器验收

> 日期：2026-10-24
> 结论：**M4 通过**。C 验证器在 NVIDIA CMP 40HX 上完成 4 epoch policy
> 训练，loss 健康下降，checkpoint 往返无损，独立评估指标可复现。

C 只验证“规则 → `.nngames` → 特征缓存 → neuralnet.cpp 训练 → 模型存取 →
评估”这条公共管线。它不是最终 B 架构，也不用于宣称 C 比 B 更好。

## 1. 实测环境

| 项目 | 值 |
|---|---|
| 主机 | Windows 11 |
| GPU | NVIDIA CMP 40HX，8192 MiB |
| 驱动 | 616.92 |
| Vulkan | 1.4.351（40HX） |
| 编译器 | Clang 23.1.0 |
| CMake / Ninja | 4.3.3 / Ninja |
| 后端 | neuralnet.cpp Vulkan，f32 |
| 上游库 | neuralnet.cpp `v1.12.0`，KataGo `v1.18.2` |

训练只在用户指定的远端训练机执行；本机没有运行正式训练。

## 2. 数据

从 CWI 10,000 局数据中用 seed `20261024` 固定抽取 1,000 局，按对局切分：

| Split | 对局 | 局面 | 文件大小 | SHA-256 |
|---|---:|---:|---:|---|
| train | 900 | 187,432 | 2,440,364,696 B | `811f98c6c2a7f4e2899f5ca385d9f97d27aa71605ccdbb281329ca3320557d94` |
| eval | 100 | 20,367 | 265,178,396 B | `19041a11af43d44d312972a3824fcd6d88721c75114d637c3728b4abd6fe0d1e` |

每个非终局 frame 生成一条样本。policy target 为下一手点位或 PASS；value
目标从当前行棋方视角记录，但 M4 C 只训练 policy。缓存格式见
[`09-training-cache.md`](../design/09-training-cache.md)。

## 3. 模型与配方

C 验证器全部使用 neuralnet.cpp 现成层：

```text
(9,19,19)
→ Conv2D(9→32, 3x3, pad=1) → MaxPool2D(2)
→ Conv2D(32→64, 3x3, pad=1) → MaxPool2D(2)
→ Linear(1024→256) → ReLU → Linear(256→362)
```

- 参数量：376,554；
- 输出：361 点位 + PASS；
- 归一化：关闭（C 是管线探针，不做架构 A/B）；
- optimizer：AdamW；
- lr：`1e-3` cosine 到 `1e-5`；
- weight decay：`1e-4`；
- batch：64；
- epochs：4；
- seed：`20261024`。

## 4. 结果

| 阶段 | Train loss | Validation loss | Validation top-1 |
|---|---:|---:|---:|
| 初始随机模型 | — | 6.030586 | 0.2995% |
| epoch 1 | 4.787182 | 4.479491 | 6.4320% |
| epoch 2 | 4.370605 | 4.378817 | 8.2634% |
| epoch 3 | 4.150168 | 4.351417 | 9.5596% |
| epoch 4 | 3.941637 | 4.353983 | **10.2175%** |

362 类均匀随机 top-1 为 0.2762%。validation loss 在第 4 轮比第 3 轮高
0.002566，但 final validation loss 仍显著低于初始值，top-1 继续上升。
正式训练加四次评估总耗时 **328.43 s**；每 epoch 约 81–83 s。

健康门槛：

- 所有 train/eval loss 均为有限值，无 NaN/Inf；
- final train loss 低于 first epoch train loss；
- final eval loss 低于初始随机模型；
- final top-1 明显高于 `1/362` 随机基线。

## 5. 存取与可复现性

| 产物 | 大小 | SHA-256 | 重载 logits 最大绝对差 |
|---|---:|---|---:|
| `.nnmodel` | 1,507,243 B | `1790215962d59d3344ca1b59af0aef3219f87be62bbbbac49ddbae6441d936d5` | 0.0 |
| `.nnckpt` | 1,507,243 B | `ba60b790f70d9a72f4f6cc16c9a7f5f1d775c70151d517498c83eea6899aa054` | 0.0 |

独立加载 `.nnmodel` 连续评估两次，得到完全相同的：

```text
loss       = 4.353982925415039
samples    = 20367
top1       = 0.10217508715078313
```

两次仅实际耗时不同。重跑同一 seed 的完整 4 epoch 训练也得到逐轮相同的
train/validation 指标。机器可读结果见
[`m4-c-validator.json`](m4-c-validator.json)。

本阶段 `.nnckpt` 只验证模型参数和 non-learnable state 的往返；optimizer state
与真正 `--resume` 属于 M5，不计入本次通过条件。

## 6. 验收命令

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

cmake --preset gpu
cmake --build --preset gpu

.\.venv\Scripts\go-pipeline.exe training-cache `
  --dataset data\cwi-10k.nngames `
  --data-cli build\gpu\go_data_cli.exe `
  --rules-cli build\gpu\go_rules_cli.exe `
  --train-output data\m4-c-train.ngf `
  --eval-output data\m4-c-eval.ngf `
  --manifest data\m4-c-cache-manifest.json `
  --max-games 1000 --eval-games 100 --seed 20261024

build\gpu\go_train_cli.exe train `
  --dataset data\m4-c-train.ngf `
  --eval-dataset data\m4-c-eval.ngf `
  --epochs 4 --batch-size 64 --lr 0.001 --min-lr 0.00001 `
  --weight-decay 0.0001 --seed 20261024 --gpu=40HX `
  --checkpoint artifacts\m4-c-validator.nnckpt `
  --model artifacts\m4-c-validator.nnmodel `
  --metrics-out artifacts\m4-c-validator.json
```

同一提交的 CPU CTest 为 **6/6**，远端 GPU 构建 CTest 为 **6/6**。

## 7. 实测发现的兼容问题

- KataGo `sha2.cpp` 在 Windows 上需要显式 `BYTE_ORDER/LITTLE_ENDIAN`；
- KataGo `rand.cpp` 链接需要 `ws2_32`；
- neuralnet.cpp 的流式 SHA-256 使用 1 MiB 栈缓冲，而 Windows 默认线程栈约
  1 MiB；`go_train_cli` 在 Windows 链接 8 MiB 栈，不修改 submodule；
- 多 Conv 模型在评估时需显式包 Vulkan batch 作用域，否则第二次 Conv 分配
  会报 `VK_ERROR_OUT_OF_DEVICE_MEMORY`。

## 8. 解释边界

- 100 局验证集只用于管线验收，不能作为正式棋力估计；
- M4 没有 value loss、历史分支、合法点 mask、8 对称增广或 MCTS；
- 不根据本次 C 结果对 B 架构下结论；
- 模型文件和 `.ngf` 缓存均为本地产物，不提交 Git；报告只记录大小和哈希。
