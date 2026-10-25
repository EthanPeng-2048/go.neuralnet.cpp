# Changelog

All notable project changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning once public model or data formats are declared stable.

## [0.3.0] - 2026-10-24

### Added

- M7 `go::GoModel`: the B architecture as a single `nn::Layer`, written the way
  neuralnet.cpp writes its own models (GPTModel composition pattern): Conv stem
  (9→64→d, 5×5 region tokens) + per-move history embedding + learnable
  decision query → `25+K+1`-token sequence → `TransformerEncoderLayer` ×N
  (Pre-Norm, Plain bidirectional) → LayerNorm → policy (362) / value (tanh)
  heads, packed into one `(363, batch)` output. Position + segment embeddings,
  tanh and the tanh Jacobian are `dsl::compute` expressions.
- M7 `go::GoLoss`: sparse CE over rows 0..361 + MSE over row 362, with both
  gradients re-packed into a single `(363, batch)` tensor.
- GPU AOT fusion via neuralnet.cpp's standard mechanism: each executable
  compiles the model TU into its own collector and calls
  `nn_enable_gpu_fusion(<target> MAIN <main source>)`; the fused structures are
  registered by `FusedAnchor` at static-init time. Verified on the GTX 850M:
  no closed-world miss, checkpoint round-trip diff 0.0.
- ModelSpec (D6 zero patch): `type=Transformer` + `patch_size=5` +
  `layer_dims[0]=0x476F4D31`; trainer, evaluator and policy player rebuild the
  exact model from the spec (resume / `evaluate` / GTP / arena).
- Tests: `go_model_tests` (contract constants, spec round-trip, tiny CPU
  forward/backward, GoLoss), `test_m7_b_model` (dual-head metrics with a
  non-degenerate value target set, B resume equivalence, GTP genmove legality).

### Fixed

- Value targets were wrong for every sample: the cache writer compared the game
  outcome ("black"/"white") with the M2 colour tag ("b"/"w"), so `z` was always
  -1 and the value head trained on a constant. Now mapped through the
  side-to-move; caches must be regenerated.
- Value AUC rank statistic used `(rank+len)/2` for tie groups instead of
  `rank+(len-1)/2`, which biased the metric (it could even go negative).
- `--resume` of a B checkpoint derived K from `--history-len` before reading
  the checkpoint spec; it now takes K from the checkpoint and gates the caches
  against that.

### Known limitations

- The M7 DoD "same data, B significantly better than C" is **not achieved**.
  Measured on three data scales (30 / 110 / 910 games): C 1.74% / 4.62% /
  **10.00%** eval top-1, B 1.27% / 4.57% / 5.80%. At 30 games B overfits, at
  910 it underfits; the design's §3.1 fallback (CNN-direct policy logits,
  `--cnn-policy 1`) lifts B to 8.23% but C still wins the head-to-head arena
  20-0. Remaining levers are M9-scale data and a wider model, not tuning.
  Full evidence: `docs/benchmarks/m7-b-vs-c.md`, roadmap §6.8.

## [0.2.0] - 2026-10-24

### Added

- M5 `GoTrainer`: `--resume` restores model weights, optimizer state
  (m/v/t), and the lr-curve position from a `.nnckpt` training-state
  section; resumed tails are bitwise identical to an uninterrupted run,
  verified on real CWI data (2+2 epochs == 4). Dataset identity
  (`.ngf` SHA-256 + sample count) and `--epochs` are hard-gated against
  the checkpoint recipe.
- M5 metrics logging: per-epoch JSONL records appended to `--metrics-log`
  across processes; checkpoints are written at epoch boundaries so a crash
  loses at most one epoch.
- M5 dual-role artifacts: `.nnckpt` (Checkpoint role: weights + TrainState
  + optimizer tensors) and `.nnmodel` (Model role: inference only);
  a `.nnmodel` is rejected as a resume source by the role gate.
- M5 recipe flags: `--optimizer`, `--schedule`, `--beta1`, `--beta2`,
  `--clip-grad-norm`, `--warmup-epochs`, `--stop-after-epoch`, plus the
  `make-model` subcommand for deterministic untrained models.
- M6 `go::extract_features`: C++ live extraction of the M3 v1 nine planes,
  bitwise identical to the Python trainer (verified frame-by-frame).
- M6 `go_gtp_cli`: GTP v2 adapter over stdin/stdout (Sabaki-compatible);
  move generation masks illegal points via KataGo and answers on the
  model's argmax; rules, ko, captures, and scoring stay in KataGo.
- M6 `go_arena_cli`: batch matches between `random`, `model:`, and
  external `gtp:` engines with color alternation and an SPRT win-rate
  test (binomial LLR with alpha/beta bounds, early stop on decision).
- `.ngf` cache v2: per-sample move history as K little-endian `u16`
  GoTokenizer point ids (left `<pad>`-padded, right-aligned, excluding the
  predicted move), written by `--history-len` (default 64). The header keeps
  `history_len`/`history_vocab` in the reserved bytes; v1 files stay readable
  and a v1 cache is refused when history rows are requested.
- Trainer metrics: `top3` (exact top-3 policy accuracy) and `value_auc`
  (Mann-Whitney rank AUC over decisive games) added to the per-epoch and
  final JSON/JSONL records.

### Known limitations

- M7 onward (B architecture rewrite, MCTS, formal training) is not implemented.
  Model construction is stubbed while the B architecture is rebuilt on
  `dsl::compute`; `train`/`make-model`/`evaluate` / GTP move generation report
  it explicitly instead of falling back to a hand-assembled stand-in.
- The RTX 4060 visits rate is a public-data estimate, not a local measurement.
- Resume equivalence is verified on CPU; GPU equivalence depends on Vulkan
  kernel determinism and belongs to M7 acceptance on the 40HX.
- GnuGo arena runs auto-activate only where `gnugo` is installed.

## [0.1.0] - 2026-10-24

### Added

- CMake CPU/Vulkan presets and the `neuralnet::nn` integration.
- Reproducible layer benchmarks and an RTX 4060 planning estimate.
- KataGo `v1.18.2` rule adapter, rule tests, SGF replay, and optional GnuGo
  cross-check.
- CWI data pipeline and a 10,000-game 19×19 `.nngames` dataset with full KataGo
  replay verification.
- `GoTokenizer`, `.nnvocab` export, nine feature planes, scalar features, and
  feature visualization.
- Python-built versioned `.ngf` training caches with single-process bulk export
  and KataGo trace.
- The M4 C validator CLI, four-epoch policy training, reproducible evaluation,
  and exact `.nnckpt`/`.nnmodel` round-trip validation on a CMP 40HX.

### Known limitations

- M5 onward is not implemented.
- The RTX 4060 visits rate is a public-data estimate, not a local measurement.
- Generated datasets, tokenizers, and model artifacts are intentionally
  excluded from Git.
