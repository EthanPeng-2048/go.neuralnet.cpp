# M7 acceptance: B model vs C validator on identical data

- Run (UTC): 2026-10-24
- Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43
- Device: CPU (build/cpu) and NVIDIA GTX 850M (build/gpu, Vulkan)
- Data: `.cache/m7-train.ngf` (6353 samples / 30 games) and
  `.cache/m7-eval.ngf` (2129 samples / 10 games) — the same v2 cache, so C and
  B see identical boards, identical policy targets, identical value targets.
  Only the model differs.
- Seed: 20261024; AdamW lr 1e-3, cosine to 1e-5; batch 16 on the 850M
  (batch 32 exceeds its VRAM: `vkAllocateMemory failed`).
- Reproduce: `ctest -R go_model_m7` for the contracts, and the two
  `go_train_cli train … --arch c|b` commands below for the numbers.

## What changed for B

C consumes the 9 feature planes and predicts a policy over 362 classes. B adds
two things C cannot express:

1. a **history branch** — the last K=64 moves as GoTokenizer point ids
   (`0..360` point, `361` pass, `365` pad), left-padded and right-aligned so
   the slot at `K-1` is always the move immediately before the one being
   predicted;
2. a **value head** — row 362 of the packed `(363, batch)` output, trained with
   MSE against the game outcome from the side-to-move's perspective.

Implemented as a single `nn::Layer` (`GoModel`) + `nn::Loss` (`GoLoss`),
following the `GPTModel` composition pattern:

```
盘面 (9·19·19, batch) → Conv(9→64, 3×3) → ReLU → Conv(64→192, 4×4, s4) → ReLU
                     → (192·25, batch) → 区域重排 → (192, batch·25)
历史 (K, batch) ids  → transpose → gather_rows((367,192) 表) → (192, batch·64)
query                → gather_rows((1,192) 表) → (192, batch)
序列 = [25 区域 | 64 历史 | 1 query]  (d, batch·90)：gather_rows + transpose
     + 位置表(T,192) + 段表(3,192)   一次融合加法（dsl::compute）
     → TransformerEncoderLayer × 8（Pre-Norm，Plain 双向掩码）→ LayerNorm
     → query 行 → Linear(192,362) 策略头 / Linear(192,1) 价值头 → tanh
     → 打包 (363, batch)
GoLoss = 稀疏 CE（行 0..361）+ MSE（行 362）
```

Layout notes (all batch-major, `i = b·T + t`): Conv2D emits `c·25+p` rows, so
one `gather_rows` + `rearrange_3d` converts to `(d, batch·25)`; the three
segments are assembled in a "token bag" and reordered to batch-major with a
single precomputed permutation gather. Element-wise math (position+segment
addition, tanh, the tanh Jacobian `1-t²`) is written with `dsl::compute`.

GPU builds register the fused shader structures through neuralnet.cpp's
standard mechanism: each executable compiles the model TU into its own
collector and calls `nn_enable_gpu_fusion(<target> MAIN <main源>)`
(`CMakeLists.txt`), with no upstream patch.

Because C has no value head, the losses are not comparable in absolute terms.
The acceptance metric is **policy top-1 accuracy on the shared eval set**, plus
B's own value-head metrics.

## Measured results

Both runs below are the same recipe on the same cache (30 train games / 6353
samples, 10 eval games / 2129 samples), differing only in `--arch`; numbers
come from the run JSONs.

| Metric | C validator | B encoder | Notes |
|---|---:|---:|---|
| Eval policy top-1 (final) | 1.74% | 1.27% | higher is better |
| Eval policy top-3 (final) | 2.91% | 3.19% | |
| Eval loss (final) | 5.735 | 8.10 | B's loss includes the value MSE — not comparable |
| Value AUC (B only) | — | 0.37 | 0.5 = chance |
| Checkpoint round-trip diff | 0.0 | 0.0 | bit-exact reload (CPU and GPU) |
| Seconds / epoch | 5.4 | 216 | 850M, batch 16 |

**At this data scale B is not yet better than C on eval top-1, and the reason is
visible in the train-set numbers:** evaluated on the *training* cache, C reaches
4.2% top-1 / 8.1% top-3 while B reaches 13.4% / 24.6% (value AUC 0.999). B has
the capacity and learns the task, but 30 games (6.4k samples) cannot support
~4M parameters: it memorizes. The M7 DoD ("同数据下 B 显著优于 C") therefore
needs the dataset scale that M9 uses (10k+ games); the 40HX run reported in
this file is the acceptance vehicle, not the 850M smoke.

Also note the value head *hurts* at this scale: its eval AUC (0.37) is below
chance while its train AUC is 0.999, i.e. it memorizes outcomes, and its
gradient shares the trunk with the policy head.

## Input-contract verification

The history input is verified frame by frame against the Python tokenizer in
`tests/test_feature_cache_history.py`, which is the guard that keeps the two
branches' notion of "the last K moves" identical between the C++ trainer and
the Python cache writer. A mismatch there would train the history branch on
shuffled moves and show up only as "B is barely better than C", so it is
asserted directly rather than inferred from accuracy.

A v1 cache (no history block) is refused rather than silently substituted with
zeros; that rejection is also covered by the same test file.

The value targets themselves are checked end to end: the cache writer maps the
game outcome ("black"/"white") to the side-to-move's perspective, and
`tests/test_m7_b_model.py` asserts that `value_auc` is computed over a
non-degenerate target set (an all-one-sign cache leaves it at 0).

## Speed

End-to-end throughput on the 850M (batch 16, d=192/8 layers, T=90):
`go_train_cli evaluate` on the 2129-sample eval cache takes 12.5 s
(≈170 samples/s, ≈5.9 ms/sample). Layer-level baselines live in
`m0-layer-bench-{cpu,gpu}.md`.
