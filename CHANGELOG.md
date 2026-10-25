# Changelog

All notable project changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning once public model or data formats are declared stable.

## [0.4.0] - 2026-10-25

### Added

- `go_gui_cli`: a local play/arena GUI served to the browser (single-file
  `gui/index.html` embedded at build time). Play mode drives human vs engine
  games (click to move, undo, pass, resign, SGF export); arena mode streams
  engine vs engine matches with pacing, colour alternation, a scoreboard, and
  auto-started follow-up games. Rules are fixed to Chinese rules
  (`go::RuleSet::chinese`: positional superko, area/stone scoring, no suicide,
  7.5 komi) and every legality/score decision stays in KataGo.
- `go::PolicyPlayer::evaluate`: masked-softmax policy distribution over legal
  moves plus the value head from the same forward pass that `select_move` uses
  (backing the GUI's candidate heatmap and position judgement; the M8 MCTS
  prior will reuse it). Engines without a policy (`random`, external GTP) do
  not error out: they synthesize a "next move at 100%" preview
  (`synthetic = true`) — `random` caches the announced move and plays it,
  external GTP engines are asked with `reg_genmove`.
- The GUI's default opponent is now GNU Go (`gtp:gnugo --mode gtp --quiet`)
  whenever `gnugo` is on the PATH, falling back to the auto-discovered model
  and then `random`; the arena defaults to project model (black) vs that
  opponent (white). Arena scoreboards track wins per engine: the totals follow
  the engines across the per-game colour swap.
- `go::arena_engine`: the arena engine abstraction (`random` /
  `model:FILE[=DEVICE]` / `gtp:COMMAND`) extracted from `go_arena_cli` so the
  GUI arena and the SPRT evaluator share one implementation.
- Tests: `go_gui_tests` (session state machine: play flow, undo, Chinese-rule
  scoring, resign, arena stepping/colour swap/scoreboard, SGF, state JSON) and
  `tests/test_gui_http.py` (drives a real `go_gui_cli` over HTTP, including
  model-backed `/api/evaluate`).
- The GnuGo arena variant now runs the opponent as
  `gtp:gnugo --mode gtp --quiet --level 1`: the test covers the external GTP
  subprocess path, not GnuGo's strength, and full-strength games no longer fit
  the ctest budget now that GNU Go 3.8 is actually installed on this machine.

### Fixed

- External GTP opponents whose `genmove` answers with an upper-case `PASS`
  (GNU Go 3.8 does) aborted the match with "invalid vertex". Pass/resign tokens
  are now matched case-insensitively in `go::arena_engine`.

## [0.3.1] - 2026-10-25

### Fixed

- `go_arena_cli` external GTP opponents were unusable on Windows: `spawn()`
  wired the two `CreatePipe` calls crossed — the child's stdin and stdout ended
  up on the same pipe — so the parent read EOF before the first response while
  the child leaked for good. Holding the host's stderr handle, that orphan kept
  `subprocess.run(capture_output=True)` blocked past its timeout and turned
  `go_gtp_arena_m6` into a 600 s ctest timeout. Pipe ends are now paired
  correctly and `reap()` terminates a child that outlives its 5 s grace period
  (same semantics as the existing POSIX `SIGKILL` path).

### Changed

- M6 acceptance scope corrected in `docs/benchmarks/m6-gtp-arena.md`: the
  external GTP subprocess path had only been validated through the POSIX fork
  path. The Windows path is now fixed and re-verified on the CMP 40HX
  (ctest 12/12, 2026-10-25).

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
