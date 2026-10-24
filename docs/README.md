# Documentation

This directory contains the project design, milestone records, and reproducible
acceptance evidence. Start with the roadmap when deciding what to work on, and
with the dependency baseline before changing a third-party component.

## Entry points

| Document | Purpose |
|---|---|
| [Roadmap and decisions](design/00-roadmap.md) | Scope, decisions, milestones, budgets, and open questions |
| [Third-party selection](design/07-third-party-selection.md) | Responsibilities and boundaries of KataGo, neuralnet.cpp, and Python tools |
| [Data and feature format](design/08-data-and-features.md) | `.nngames`, GoTokenizer, `.nnvocab`, and feature-plane contracts |
| [M4 training cache](design/09-training-cache.md) | Versioned `.ngf` cache and C validator data flow |
| [Dependencies](../DEPENDENCIES.md) | Exact versions, commits, licenses, and upgrade gates |
| [Contributing](../CONTRIBUTING.md) | Build, test, dependency, and data-artifact rules |

## Design documents

| Document | Status | Topic |
|---|---|---|
| [00-roadmap.md](design/00-roadmap.md) | Current | Decisions and M0–M9 roadmap |
| [01-thinking-eval.md](design/01-thinking-eval.md) | Research | Whether a thinking model belongs in the main path |
| [02-scheme1-encoder-transformer.md](design/02-scheme1-encoder-transformer.md) | Main candidate | Board encoder plus history Transformer |
| [03-scheme2-decoder-thinking.md](design/03-scheme2-decoder-thinking.md) | Deferred | Decoder/thinking model |
| [05-scheme3-algorithm.md](design/05-scheme3-algorithm.md) | Proposed | Search-controlled token distillation |
| [06-scheme3-engineering.md](design/06-scheme3-engineering.md) | Proposed | Engineering and teacher budget for scheme 3 |
| [07-third-party-selection.md](design/07-third-party-selection.md) | Current | Library boundaries |
| [08-data-and-features.md](design/08-data-and-features.md) | Current | Data and feature contracts |
| [09-training-cache.md](design/09-training-cache.md) | Current | M4 `.ngf` cache format |

Document 04 is intentionally absent. It was never created; numbering is kept
stable so existing links do not change.

## Acceptance evidence

| Report | Evidence |
|---|---|
| [M0 opponent estimate](benchmarks/m0-opponent-4060.md) | Public benchmark scaling for a non-accessible RTX 4060 |
| [M0 CPU layers](benchmarks/m0-layer-bench-cpu.md) | CPU component benchmark |
| [M0 Vulkan layers](benchmarks/m0-layer-bench-gpu.md) | CMP 40HX component benchmark |
| [M1 rules](benchmarks/m1-rules.md) | Rule tests, replay, latency, and GnuGo cross-check |
| [M2/M3 data and features](benchmarks/m2-m3-data.md) | Dataset counts, replay, round-trip, tokenizer, and feature evidence |
| [M4 C validator](benchmarks/m4-c-validator.md) | Measured 40HX training, reproducibility, and model round-trip |

## Evidence labels

Use these labels consistently in new reports:

- **Measured**: produced by a named commit and command on named hardware.
- **Public estimate**: calculated from externally published measurements; not a
  substitute for running the benchmark locally.
- **Plan**: a target or budget that has not been validated.

Every performance report should include the hardware, compiler, build type,
dataset, command, and commit. A component benchmark must not be presented as
end-to-end model or match throughput.
