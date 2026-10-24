# Third-Party Notices

This file summarizes third-party components used by go.neuralnet.cpp. It does
not replace the license text shipped with each component.

## Project license

Original source code and documentation in this repository are licensed under
LGPL-3.0-only. See [LICENSE](LICENSE).

The submodules, Python packages, data, and externally referenced material are
separate works and remain under their own terms.

## neuralnet.cpp

- Project: <https://github.com/EthanPeng-2048/neuralnet.cpp>
- Version: `v1.12.0`
- Commit: `05c78eaf6eb994981204dca9434febc0c447c6a5`
- License: LGPL-3.0
- Local license: `3rd_party/neuralnet.cpp/LICENSE`

The library is consumed through the public `neuralnet::nn` CMake target.
Distributed combined works must satisfy the LGPL relinking and notice
requirements for the library.

## KataGo

- Project: <https://github.com/lightvector/KataGo>
- Version: `v1.18.2`
- Commit: `fd0723fdbc0e9d82cf269c9630af8c27c57c07c4`
- License: MIT for KataGo-authored code, except material with embedded or
  vendored notices
- Local license: `3rd_party/KataGo/LICENSE`

This project compiles only the rule and minimal core sources needed by
`cpp/game`. Other KataGo components and vendored dependencies are not part of
the default build. Their notices remain in the KataGo submodule.

## Python packages

The pinned development dependencies in `pyproject.toml` are separate packages:

| Package | Version | License family |
|---|---:|---|
| httpx | 0.28.1 | BSD-3-Clause |
| numpy | 2.5.4 | BSD-3-Clause and other permissive notices |
| orjson | 3.13.0 | MPL-2.0 AND (Apache-2.0 OR MIT) |
| Pillow | 13.0.0 | MIT-CMU |
| sgfmill | 1.1.1 | MIT |
| tenacity | 9.2.1 | Apache-2.0 |

Installed package metadata and license files are authoritative. The Python
code in this repository calls these packages but does not relicense them.

## CWI Go games

The CWI Go game archive is downloaded at runtime and is not committed to this
repository. The CWI source page states that the games are in the public domain.
The archive and generated `.nngames` datasets remain separate from the project
source license.

- Source: <https://homepages.cwi.nl/~aeb/go/games/games.tgz>
- Archive SHA-256:
  `c57ad2ba3c5f3a3ee145bb77bcc9c0167c0d7b5478cf691a8939f750511071e2`

## Benchmark references

Public benchmark pages and standards referenced by project reports are cited
at their point of use. Project-authored benchmark tables are licensed with the
rest of the documentation.
