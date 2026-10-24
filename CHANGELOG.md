# Changelog

All notable project changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning once public model or data formats are declared stable.

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
