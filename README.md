# go.neuralnet.cpp

基于 [neuralnet.cpp](https://github.com/EthanPeng-2048/neuralnet.cpp) 的
19×19 围棋训练与推理项目。模型、训练与 checkpoint 统一使用 neuralnet.cpp；
棋规、SGF 和数据工程分别复用 KataGo、sgfmill 与 Python 生态。

**当前状态：M0–M3 已完成，M4 尚未开始。**

| 里程碑 | 状态 | 产物 |
|---|---|---|
| M0 工程与基准 | ✅ | CPU/Vulkan 构建、neuralnet.cpp 接入、层基准 |
| M1 棋规适配 | ✅ | KataGo v1.18.2 薄适配、规则题库、SGF 回放 |
| M2 数据管线 | ✅ | 10,000 局 19×19 `.nngames`、KataGo 全量重放 |
| M3 Token 与特征 | ✅ | GoTokenizer、`.nnvocab`、9 平面特征与可视化 |
| M4 验证器竖切 | 未开始 | 规则 → 特征 → C 模型训练 → 存取 → 评估 |

路线图与设计决策见 [`docs/design/00-roadmap.md`](docs/design/00-roadmap.md)，
全部文档入口见 [`docs/README.md`](docs/README.md)，第三方版本唯一真源见
[`DEPENDENCIES.md`](DEPENDENCIES.md)。

## 快速开始

构建要求：

- CMake 3.30+、Ninja；
- 支持 C++26 的编译器（CPU 构建不要求 GPU）；
- Python 3.12+；
- GPU 构建另需 Vulkan SDK 和可用驱动。

```bash
git clone --recurse-submodules https://github.com/EthanPeng-2048/go.neuralnet.cpp.git
cd go.neuralnet.cpp

python3 -m venv .venv
.venv/bin/python -m pip install -e .

cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu --output-on-failure
```

已有检出目录时，首次构建前执行：

```bash
git submodule update --init --recursive
```

有 Vulkan SDK 且需要 GPU 时，把上述 `cpu` preset 换成 `gpu`。

## 数据管线

原始数据和生成数据不进入 Git；下载源、SHA-256、格式和验收结果均固定记录。

```bash
.venv/bin/go-pipeline download
.venv/bin/go-pipeline ingest --limit 10000
.venv/bin/go-pipeline verify --sample 1000
.venv/bin/go-pipeline tokenizer
.venv/bin/go-pipeline features --index 0 --frame 100
```

CWI 数据集的当前验收结果是 10,000 局、2,076,112 手、2,086,112 个局面，
全部经 KataGo 重放，`.nngames` 容器逐字节可逆。格式和边界见
[`数据与特征格式`](docs/design/08-data-and-features.md)。

## 棋规

规则唯一真源是 KataGo `v1.18.2` 的 `Board + BoardHistory + Rules`。项目 API
见 [`include/go/game.hpp`](include/go/game.hpp)，覆盖合法点、落子/弃着、提子、
劫、超级劫、面积归属与即时计分。

```bash
./build/cpu/go_rules_cli replay \
  --size 5 --rules chinese --komi 7.5 \
  b:1,1 w:2,1 b:1,2 w:0,0 b:3,1 w:4,4 b:2,2 w:0,4 b:2,0
```

## 基准与证据

M0 的 CPU/Vulkan 层基准可通过以下命令重跑：

```bash
python3 bench/m0_layer_bench.py \
  --layer-bench build/cpu/neuralnet_cpp/layer_bench

python3 bench/m0_layer_bench.py \
  --layer-bench build/gpu/neuralnet_cpp/layer_bench \
  --gpu
```

这些数字是项目相关层（Conv2D、Transformer、策略输出 Linear）的组件基准，
不是完整 GoModel 或对局吞吐。RTX 4060 无法由项目作者实测；当前只采用公开数据
折算的约 **1200 visits/s** 规划值，详见
[`M0 对手基准说明`](docs/benchmarks/m0-opponent-4060.md)。文档中始终区分实测结果、
公开估算和计划值。

## 开发边界

- 不自行复制提子、劫、禁入、终局或计分逻辑；规则一律交给 KataGo。
- 不引入 PyTorch/JAX 作为第二训练栈；ML 统一使用 neuralnet.cpp。
- Python 用于 SGF、批处理、特征、可视化和验收；训练执行不属于 Python。
- 第三方依赖只引用固定版本；升级前更新 [`DEPENDENCIES.md`](DEPENDENCIES.md)。

参与开发前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。安全问题按
[`SECURITY.md`](SECURITY.md) 私下报告。

## 许可证

项目原创代码采用 [GNU Lesser General Public License v3.0 only](LICENSE)
（LGPL-3.0-only）。Submodule、Python 依赖和数据集继续遵循各自许可证；
详情见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
