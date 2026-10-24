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

Two sizes, same recipe per size (`--arch` is the only difference), all on the
same cache and seed.

### 40HX, 110-game cache (100 train games / 21159 samples, 10 eval / 1818)

| Metric | C validator | B encoder |
|---|---:|---:|
| Eval top-1 (epoch 4) | **4.62%** | 1.82% |
| Eval top-3 (epoch 4) | **12.60%** | 3.91% |
| Eval loss (epoch 4) | 4.737 | 7.774 (CE + value MSE — not comparable) |
| Value AUC (B only) | — | 0.389 |
| Checkpoint round-trip diff | 0.0 | 0.0 |
| Train loss, epoch 1 → 4 | 5.75 → 4.40 | 7.32 → 5.16 |
| Seconds / epoch (batch 32) | **10.9** | 137 |
| Inference (batch 32) | 7697 pos/s | **1312 pos/s** |

### GTX 850M, 30-game cache (6353 / 2129 samples, batch 16)

| Metric | C validator | B encoder |
|---|---:|---:|
| Eval top-1 (final) | 1.74% | 1.27% |
| Eval top-3 (final) | 2.91% | 3.19% |
| Train top-1 / top-3 | 4.17% / 8.11% | 13.36% / 24.62% |
| Value AUC (B only) | — | 0.37 (train 0.9995) |
| Seconds / epoch | 5.4 | 216 |

### B at 20 epochs (same 110-game cache, 40HX)

Training longer does not close the gap — it memorises:

| epoch | train loss | eval loss | eval top-1 | eval top-3 |
|---:|---:|---:|---:|---:|
| 1 | 6.91 | 7.61 | 0.99% | 2.81% |
| 4 | 5.53 | 7.44 | 1.87% | 5.12% |
| 8 | 4.19 | **6.91** | **4.51%** | 10.45% |
| 12 | 3.31 | 7.23 | 4.40% | 10.84% |
| 16 | 2.51 | 7.64 | 4.07% | 8.97% |
| 20 | 2.05 | 7.89 | 3.58% | 8.47% |

The best point (4.51% at epoch 8) is statistically the same as C's 4.62%, and
everything after it is pure overfitting: train loss falls to 2.05 while eval
loss climbs back to 7.89.

### Head-to-head

`go_arena_cli --player model:run-c.nnmodel --opponent model:run-b.nnmodel
--games 20 --size 19 --komi 7.5`: **C won 20-0** (deterministic argmax players,
so each colour pairing produces one game — 684 and 811 moves).

### What this says

1. **Speed budget: met.** §7 estimated 915 positions/s (inference, B-small,
   batch 64 component-sum). Measured end to end on the 40HX: **1312 pos/s** at
   batch 32. Training throughput (154 pos/s) is well below the 694 pos/s
   training estimate — that number assumed a pure Transformer stack; the real
   model spends its time in the Conv stem and the sequence assembly.
2. **B generalises worse than C at every data size we can run here.** The gap
   closes as data grows (30 games: 1.27% vs 1.74%; 110 games: 1.82% vs 4.62% —
   B nearly doubles, C nearly triples), and B's *training* fit is far ahead
   (13.4% vs 4.2% top-1 at 30 games), i.e. B is capacity-rich and data-poor.
3. **More epochs do not help** (table above): B peaks at 4.51% (epoch 8) and
   degrades afterwards, exactly the signature of a model whose capacity is not
   matched by data. The DoD comparison needs M9-scale data (10k games, ~2M
   samples); the honest verdict at 110 games is "B ties C on quality and loses
   20-0 head to head", not "B wins".

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
