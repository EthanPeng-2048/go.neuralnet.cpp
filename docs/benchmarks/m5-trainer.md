# M5 GoTrainer 验收

> 日期：2026-10-24
> 结论：**M5 通过**。断点续训在真实 CWI 数据上逐位无损；指标按 epoch
> 落盘且跨进程续写；`.nnckpt`（Checkpoint 角色，含优化器状态）与
> `.nnmodel`（Model 角色，纯推理）双角色存取成立；监督配方全部可经
> CLI 表达并随断点落盘。

## 1. 实测环境

| 项目 | 值 |
|---|---|
| 主机 | Linux（开发机） |
| 后端 | neuralnet.cpp CPU，f32 |
| 编译器 | clang 24.0.0，Ninja，Release |
| 上游库 | neuralnet.cpp `v1.12.0`，KataGo `v1.18.2` |
| 提交 | 本次工作树（见 CHANGELOG） |

正式 40HX 训练仍按 M4 口径在远端训练机执行；本报告的数值证据来自
本机 CPU，用真实数据验证语义正确性（续训等价性与硬件无关）。

## 2. 数据

从 `data/cwi-10k.nngames`（M2 产物，10,000 局）以 seed `20261024`
抽取 40 局，按对局切分为 30 局训练 / 10 局验证：

| Split | 对局 | 局面 | SHA-256（.ngf） |
|---|---:|---:|---|
| train | 30 | 2,129 | `f6da253c763c05fd69d8a656053189b4d943f6800d89660437854734a5587391` |
| eval | 10 | 2,129（同格式独立切分） | 见 `.cache/m5-cache-manifest.json` |

缓存由 M4 的 `training-cache` 管线生成，格式不变
（[`09-training-cache.md`](../design/09-training-cache.md)）。

## 3. 配方（全部 CLI 可表达）

```text
--optimizer adamw --schedule cosine --lr 0.001 --min-lr 0.00001
--weight-decay 0.0001 --beta1 0.9 --beta2 0.999 --clip-grad-norm 1.0
--warmup-epochs 0 --epochs 4 --batch-size 64 --seed 20261024
```

新增 CLI 面（M4 没有的）：`--resume`、`--stop-after-epoch`、
`--metrics-log`、`--optimizer`、`--schedule`、`--beta1`、`--beta2`、
`--clip-grad-norm`、`--warmup-epochs`，以及 `make-model` 子命令
（M6 测试用的确定性未训练模型）。

## 4. 核心验收：断点续训无损

**方法**：同一份数据、同一配方，比较
(a) 一口气 4 epoch；(b) 2 epoch → 存档 → `--resume` 再 2 epoch。
两者的最终指标必须逐位一致。

**结果（真实 CWI 数据，CPU）**：

| Epoch | 一口气 train/eval loss | 断点续训 train/eval loss | 逐位相等 |
|---|---|---|---|
| 1 | 5.86442 / 5.78620 | 5.86442 / 5.78620 | ✅ |
| 2 | 5.69511 / 5.75350 | 5.69511 / 5.75350 | ✅ |
| 3 | 5.55740 / 5.76431 | 5.55740 / 5.76431 | ✅ |
| 4 | 5.43041 / 5.77227 | 5.43041 / 5.77227 | ✅ |

epoch 3/4 的 train_loss、eval loss、top-1 **完全相等**
（`5.557397365570068 == 5.557397365570068` 等，非四舍五入后相等）。
续训进程恢复了 16 个优化器状态张量（AdamW 的 m/v × 参数组）与步计数
`optimizer_step=200`。

**逐位无损的三个前提**（实现里各自成立）：

1. 优化器状态（m/v/t）随断点落盘并原样恢复；
2. 每 epoch 的洗牌序是 `(seed, epoch)` 的纯函数——每 epoch 先还原自然
   序再以派生 RNG 洗牌（`std::shuffle` 的输出依赖输入序，沿用上一
   epoch 的排列会让续训进程与一口气训练拿到不同数据序）；
3. lr 曲线形状（`--epochs`、schedule）随断点落盘，续训时与 CLI 交叉
   校验，不一致即硬错。

## 5. 指标落盘（JSONL 追加）

`--metrics-log`（默认从 `--metrics-out` 派生 `.jsonl`）逐条**追加**：

```jsonl
{"record":"run","mode":"fresh",...}
{"record":"epoch","epoch":1,"lr":0.001,"train_loss":5.86442,...}
{"record":"epoch","epoch":2,...}
{"record":"final","mode":"fresh",...}
{"record":"run","mode":"resume","resume_from":"...","start_epoch":2,...}
{"record":"epoch","epoch":3,...}
{"record":"epoch","epoch":4,...}
{"record":"final","mode":"resume",...}
```

一个文件跨进程完整记录两段训练；崩溃最多丢一个 epoch 的记录
（checkpoint 同样写在 epoch 边界）。epoch 记录含 `global_step`，跨
续训累计（200 → 400）。

## 6. 双角色存取

| 产物 | 大小 | 角色 | 内容 |
|---|---:|---|---|
| `.nnckpt` | 4,521,355 B | Checkpoint | 模型参数 + TrainState（计数器/配方/数据集身份）+ 16 个优化器状态张量 |
| `.nnmodel` | 1,507,243 B | Model | 仅推理参数 |

`.nnckpt` 是 `.nnmodel` 的 3.0 倍，差值即优化器状态。独立加载
`.nnmodel` 评估得 `loss=5.772269248962402, top1=1.2682%`，与训练收尾
一致。用 `.nnmodel` 做 `--resume` 会被明确拒绝（推理模型没有训练状态
段），这正是角色门禁的语义。

## 7. 数据集身份门禁

`.ngf` 文件的 SHA-256 与样本数随断点落盘；续训时不一致即硬错并给出
两条身份的对照。这堵死了"换数据集续训"这一不存在正确语义的路径
（step 号、lr 曲线、shuffle 网格全部随数据集变化）。`--epochs` 与断点
配方不一致同样硬错。

## 8. 自动化测试

CTest `go_trainer_m5_resume`（4 例，`tests/test_m5_resume.py`）：

- `test_resume_equivalence_and_artifacts`：合成数据上 2+2 == 4 的逐位
  等价 + JSONL 记录数与顺序 + checkpoint 大于 model；
- `test_resume_rejects_changed_dataset`：换数据集必须被拒；
- `test_resume_rejects_epoch_mismatch`：改 `--epochs` 必须被拒；
- `test_warm_start_from_inference_model`：`.nnmodel` 不能当断点用。

同一提交 CPU CTest **9/9**（M0–M6 全量）。

## 9. 验收命令

```bash
cmake --preset cpu && cmake --build --preset cpu
ctest --test-dir build/cpu --output-on-failure
```

真实数据复现（本报告 §4 的数字）：

```bash
python -m go_pipeline.cli training-cache --dataset data/cwi-10k.nngames \
  --data-cli build/cpu/go_data_cli --rules-cli build/cpu/go_rules_cli \
  --train-output .cache/m5-train.ngf --eval-output .cache/m5-eval.ngf \
  --manifest .cache/m5-cache-manifest.json \
  --max-games 40 --eval-games 10 --seed 20261024

build/cpu/go_train_cli train --dataset .cache/m5-train.ngf \
  --eval-dataset .cache/m5-eval.ngf --cpu --batch-size 64 --epochs 4 \
  --checkpoint s.nnckpt --model s.nnmodel \
  --metrics-out s.json --metrics-log s.jsonl --seed 20261024
# 与 --stop-after-epoch 2 + --resume s.nnckpt 的尾部逐位对比
```

## 10. 解释边界

- 本报告的 loss 数值是**管线验收口径**（40 局、4 epoch、CPU），不是
  棋力估计；
- 续训等价性在 CPU 上逐位验证；GPU 上的等价性依赖 Vulkan kernel 的
  确定性，属于 M7（B 架构）在 40HX 上的验收范围；
- 断点粒度是 epoch 边界：崩溃恢复最多损失一个 epoch 的进度，不做
  batch 级续训（上游 `TrainState.step_in_epoch` 字段已预留）。
