# M8 acceptance: MCTS search vs raw-prior play

- Run (UTC): 2026-10-25
- Host: Linux x86_64, NVIDIA GTX 850M (Vulkan). This is the same dev host as
  the M7 acceptance; the authoritative 40HX re-verification is still pending
  (same caveat as M6/M7 records).
- Model: `.cache/m8-b-bench.nnmodel` — B encoder (default d=192 / 8 layers /
  K=64, `--cnn-policy 1`, `--value-weight 0.25`), trained 8 epochs on
  `.cache/m7-train.ngf` (6,353 samples / 30 games) with the M5 trainer.
- Reproduce: the commands quoted in each section below.

## What the comparison isolates

Both sides use the **same network**. The only difference is the move
selection in front of it:

| engine spec | selection |
|---|---|
| `model:FILE` | raw prior argmax over legal moves ("方案 1 纯先验") |
| `mcts:128:model:FILE` | 128-visit PUCT search: priors + value-head leaves |

So any win-rate gap is the search loop's doing, not the model's.

## Tree contracts (ctest)

`go_mcts_m8_tests` (mock evaluator, no model) locks the four M8 DoD pieces
plus the sign convention:

1. **PUCT selection** — a 0.9-prior point takes essentially all root visits;
2. **prior renormalization** — root child priors sum to 1 on the legal set;
3. **legality** — priors stacked on occupied points never enter the tree;
4. **negamax sign** — a "white is better by 0.9" evaluator comes back as a
   negative root value for black on every root edge;
5. **terminal scoring** — pass-pass under Chinese rules backs up -1 for black
   (7.5 komi on an empty board);
6. **transposition table** — re-searching the same position hits the cache
   and cuts evaluator calls.

`go_mcts_m8_python` covers the product wiring: `go_gtp_cli --visits N`
(Sabaki path, legal replies), `--visits 0` = pure prior, and the arena engine
spec `mcts:VISITS:model:FILE[=DEVICE]` including rejection of malformed
specs.

## C line: search cannot rescue a value-head-less model

The C validator drives the same engine through the `ValueFallback::area_score`
leaf heuristic (it has no value head). Four 19×19 games per pairing, Chinese
rules, komi 7.5, fixed seed 20261025, opponent
`gtp:gnugo --mode gtp --quiet --level 1`:

| player | result | game lengths (moves) | wall time |
|---|---|---|---|
| `model:…m4-c-validator.nnmodel` (pure prior) | **0 – 4** | 341 / 498 / 433 / 486 | 127 s |
| `mcts:256:model:…` | **0 – 4** | 455 / 414 / 439 / 458 | 434 s |
| `mcts:512:model:…` | **0 – 4** | 433 / 500 / 426 / 443 | 806 s |

Raw reports: `.cache/m8-c-prior-vs-gnugo.json`,
`.cache/m8-c-mcts-vs-gnugo.json`, `.cache/m8-c-mcts512-vs-gnugo.json`.

Why: in the opening and early middle game the area margin is essentially the
komi constant, so the fallback leaf value carries no positional information —
the tree is steered by C's priors alone (10.2% top-1 on the shared eval set),
and those are weaker than GnuGo's pattern library even at level 1. This is a
model limitation, not a tree limitation: the four tree contracts pass, and the
B line below has a real value head.

## Speed (19×19, CPU, batch 1)

Per-move latency measured through `go_gtp_cli --visits N` (timed genmove,
model load excluded) on the dev host's CPU (a GTX 850M-era laptop):

| model | pure prior (1 eval/move) | mcts-32 | mcts-128 | visits/s |
|---|---|---|---|---|
| C validator (1.5 MB CNN) | 1.1 ms | — | 141 ms | **909** |
| B bench encoder (22 MB, d=192/8 + CNN-direct head) | 84 ms | 2465 ms | 9675 ms | **13** |

Two things fall out of this table:

1. **The tree loop is essentially free.** C's per-visit cost is ~1.1 ms —
   the same as its single batch-1 forward — so selection, replay, backup and
   the transposition lookup together cost far less than one evaluation.
   MCTS throughput *is* the model's batch-1 latency.
2. **The model, not the engine, sets visits/s.** The 22 MB B encoder is ~75x
   slower per evaluation than the C CNN on this CPU, so 13 visits/s vs 909.
   This is exactly why the acceptance matchups moved to the 32-core box and
   why the 40HX GPU build matters for the visits budget: §7's planning
   numbers (915/578 pos/s) were component-sum estimates at batch 64 for the
   40HX, and batch-1 search traffic behaves differently. 40HX batch-1 numbers
   land here once the Vulkan build completes.

40HX GPU (Vulkan, `go_gtp_cli.exe --gpu=40HX`, warmup moves discarded,
median of 18 timed genmoves, batch 1):

| selection | ms/move | visits/s |
|---|---|---|
| pure prior | 10.6 | — (1 eval) |
| mcts-32 | 300 | 107 |
| mcts-128 | 1019 | **126** |

The 40HX runs the B encoder at ~8 ms per batch-1 evaluation vs ~77 ms on the
dev CPU: **126 visits/s vs 13 visits/s**. At the bet-terms range of 40–100
visits that is 0.3–0.8 s per move, comfortably inside a playable clock. Note
the §7 budget (915/578 pos/s) was estimated at batch 64; search traffic is
batch 1, so the two are not directly comparable — the M7 inference gate
(1312 pos/s at batch 32 on the 40HX) stays the throughput acceptance, and
this table is the search-side latency record.

## Gain (arena, 19×19, Chinese rules)

<!-- FILLED: arena result -->

## Scope notes (honest)

- **Transposition table = evaluation cache.** The key covers everything the
  evaluator sees (board + side to move + simple ko + last-64-move window,
  which includes both C's recent-move planes and B's K=64 history branch);
  nodes are *not* shared as a DAG. Legality is always adjudicated by KataGo
  against the actual game history of each line, so a cache hit can only save
  an evaluation — it can never legalize an illegal move.
- **C models work through the same engine** but have no value head
  (`PolicyEvaluation::value == nullopt`); `make_policy_evaluator` falls back
  to `ValueFallback::area_score` (tanh of the area margin / 15) for leaves.
  The tree mechanics are identical; leaf quality is static, so the gain over
  raw priors is expected to be smaller than with B.
- **Board size:** the engine itself is size-agnostic (the unit tests run on
  5×5 boards). Real models are 19×19-only per the M3 v1 feature contract, so
  the "search vs prior" measurement here is on 19×19, not a 9×9 sandbox.
- **Root noise / temperature** are implemented but off by default
  (D9: no self-play in this milestone).
