# Changelog

All notable project changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning once public model or data formats are declared stable.

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
