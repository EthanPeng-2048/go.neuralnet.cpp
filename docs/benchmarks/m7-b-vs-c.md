# M7 acceptance: B model vs C validator on identical data

- Run (UTC): 2026-10-24
- Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43
- Device: CPU (build/cpu) and CMP 40HX (build/gpu)
- Data: `.cache/m7-train.ngf` / `.cache/m7-eval.ngf` — the same v2 cache, so
  C and B see identical boards, identical policy targets, identical value
  targets. Only the model differs.
- Seed: 20261024; batch 32; AdamW lr 1e-3, cosine to 1e-5.

## What changed for B

C consumes the 9 feature planes and predicts a policy over 362 classes. B adds
two things C cannot express:

1. a **history branch** — the last K moves as GoTokenizer point ids
   (`0..360` point, `361` pass, `365` pad), left-padded and right-aligned so
   the slot at `K-1` is always the move immediately before the one being
   predicted;
2. a **value head** — row 362 of the packed `(363, batch)` output, trained with
   MSE against the game outcome from the side-to-move's perspective.

The two branches plus a decision query token are concatenated into one
sequence of `25 + K + 1` tokens and consumed by a Transformer encoder; the
query row feeds both heads.

Because C has no value head, the losses are not comparable in absolute terms.
The acceptance metric is therefore **policy top-1 accuracy on the shared eval
set**, plus B's own value-head metrics.

## Measured results

Pending: the B implementation is being rewritten on top of `dsl::compute` with
AOT fusion (`nn_enable_gpu_fusion`). No B-vs-C run has been recorded yet, so
this table is intentionally empty — it is filled only from real runs.

| Metric | C validator | B encoder | Notes |
|---|---:|---:|---|
| Eval policy top-1 (final) | | | higher is better |
| Eval loss (final) | | | not comparable across architectures |
| Value AUC (B only) | — | | 0.5 = chance |
| Checkpoint round-trip diff | 0.0 | | must be 0 (bit-exact reload) |

## Input-contract verification

The history input is verified frame by frame against the Python tokenizer in
`tests/test_feature_cache_history.py`, which is the guard that keeps the two
branches' notion of "the last K moves" identical between the C++ trainer and
the Python cache writer. A mismatch there would train the history branch on
shuffled moves and show up only as "B is barely better than C", so it is
asserted directly rather than inferred from accuracy.

A v1 cache (no history block) is refused rather than silently substituted with
zeros; that rejection is also covered by the same test file.

## Speed

Inference and training throughput are the §7 budget check. Layer-level
baselines live in `m0-layer-bench-{cpu,gpu}.md`; end-to-end B throughput is
measured with `go_train_cli` on the shared cache once the rewrite lands.
