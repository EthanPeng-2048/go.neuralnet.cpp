# Contributing

## Scope

This project builds a 19×19 Go model and training/inference pipeline. The
current implementation covers M0–M3. M4 and later milestones are tracked in
[`docs/design/00-roadmap.md`](docs/design/00-roadmap.md).

The project intentionally favors a small C++ boundary and Python for data
engineering:

- reuse mature libraries instead of reimplementing rules or model primitives;
- keep KataGo as the only source of truth for Go rules;
- keep neuralnet.cpp as the only model and training runtime;
- use Python for SGF handling, feature generation, orchestration, reporting,
  and integration tests.

Large features that duplicate an existing dependency should be discussed
before implementation.

## Development setup

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu --output-on-failure
```

The optional GnuGo cross-check runs only when `gnugo` is installed. GPU tests
require a Vulkan-capable environment and should be reported separately from
CPU results.

## Before opening a pull request

1. Keep changes focused and update the affected design or acceptance document.
2. Run the CPU build and CTest suite.
3. Run the Python data/feature tests directly if CTest cannot discover
   Python:

   ```bash
   PYTHONPATH=python .venv/bin/python -m unittest discover -s tests -p 'test_m2_m3.py' -v
   PYTHONPATH=python .venv/bin/python tests/test_sgf_replay.py \
     --cli build/cpu/go_rules_cli
   ```

4. For dependency, model-format, feature, or licensing changes, update
   [`DEPENDENCIES.md`](DEPENDENCIES.md) and
   [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
5. State the exact hardware, compiler, dataset, and commit for performance
   claims. Do not mix measured results, public estimates, and plans.

Do not commit build outputs, virtual environments, downloaded data, generated
`.nngames`/`.nnvocab` files, checkpoints, or model artifacts. The ignore rules
are in [`.gitignore`](.gitignore).

## Dependency and format rules

- Pin third-party versions to immutable releases or commits.
- Prefer a submodule gitlink over vendoring third-party source.
- Do not change `.nngames`, `.nnvocab`, checkpoint, or feature formats without
  a versioned migration and round-trip tests.
- Do not add a second Go rules engine or a second ML/autograd runtime.
- Preserve upstream license and attribution files when changing a dependency.

## License of contributions

By submitting a contribution, you agree that it may be distributed under the
repository's LGPL-3.0-only license. You retain copyright in your contribution.
