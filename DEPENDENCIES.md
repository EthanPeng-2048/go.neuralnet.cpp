# 第三方依赖基线

> 状态：**已锁定**（2026-10-24）
> 本文件是 go.neuralnet.cpp 第三方依赖版本的唯一真源。修改依赖前先更新本文件，
> 再同步路线图与设计文档。

## neuralnet.cpp

| 项目 | 值 |
|---|---|
| 上游仓库 | <https://github.com/EthanPeng-2048/neuralnet.cpp> |
| 正式引用 | [`v1.12.0`](https://github.com/EthanPeng-2048/neuralnet.cpp/tree/v1.12.0) |
| annotated tag object | `e8cae9b4031f9427e36f8557f7382b0594bd5200` |
| tag 指向提交 | `05c78eaf6eb994981204dca9434febc0c447c6a5` |
| 发布提交时间 | `2026-10-23T23:55:23+08:00` |
| 许可证 | **LGPL-3.0**（v1.12.0 起；v1.11.0 及以前为 MIT） |
| 消费入口 | `neuralnet::nn` |
| 聚合头 | `<neuralnet.cpp/nn.hpp>`；RL-only 入口为 `<neuralnet.cpp/rl.hpp>` |
| 引用方式 | Git submodule（名称 `neuralnet.cpp`） |
| 检出路径 | `3rd_party/neuralnet.cpp` |
| 构建要求 | CMake 3.30+、C++26；仅 CPU / Vulkan 后端，无 CUDA |

上游已经作为 submodule 纳入本仓库，`.gitmodules` 和父仓库 gitlink 均指向固定提交。
首次克隆父仓库后执行：

```bash
git submodule update --init --recursive
```

项目已建立顶层 CMake 目标 `go_neuralnet_m0`，直接消费已检出的 submodule：

```cmake
add_subdirectory(3rd_party/neuralnet.cpp)
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

## 为什么锁定 v1.12.0

- v1.12.0 正式发布 RL 与可组合训练基础设施；此前“等上游 RL 原语”的阻塞已解除。
- 可用的公共能力包括具名多输出、schema 化 experience、GAE/return/group
  normalization、REINFORCE/PPO/GRPO objectives、`TrainingGroupSession` 与
  `RLTrainingPipeline`。
- 上游仍只提供训练基础设施；围棋规则、MCTS、自对局 worker、replay policy 和领域
  buffer 仍由本项目实现。
- 项目当前范围仍按路线图 D9 执行：M0–M9 不做自对局训练。启用项目级 RL 需要新的
  范围决定，不得仅因依赖已升级就把 RL 自动并入关键路径。

## 升级规则

1. 只引用不可变的正式 tag；不使用 `main`、`dev/rl_training` 或浮动分支作为构建输入。
2. 升级时同时更新 tag、annotated tag object、peeled commit 与许可证影响。
3. 先阅读上游 release notes，再复验“作为库消费”的构建、RL 头文件与 Vulkan AOT
   闭合世界；历史性能锚不得跨版本直接复用。
4. `3rd_party/neuralnet.cpp` 的检出提交必须等于本表的 peeled commit；父仓库只记录
   gitlink，不把上游源码重复提交进本仓库。

依据：[v1.12.0 release notes](https://github.com/EthanPeng-2048/neuralnet.cpp/blob/v1.12.0/docs/release-notes/v1.12.0.md)、
[RL 使用指南](https://github.com/EthanPeng-2048/neuralnet.cpp/blob/v1.12.0/docs/usage/06-reinforcement-learning.md)。

## KataGo（M1 棋规库）

> 状态：**已接入并锁定**（2026-10-24）。M1 直接消费 `cpp/game`，项目不复制规则。

| 项目 | 值 |
|---|---|
| 上游仓库 | <https://github.com/lightvector/KataGo> |
| 正式引用 | `v1.18.2` |
| tag 指向提交 | `fd0723fdbc0e9d82cf269c9630af8c27c57c07c4` |
| 发布提交时间 | `2026-08-29T21:17:14-04:00` |
| 许可证 | MIT（KataGo 自有代码）；vendored `nlohmann_json` 为 MIT |
| submodule 名称 | `3rd_party/KataGo` |
| 检出路径 | `3rd_party/KataGo` |
| 消费范围 | `cpp/game/` 的 Board / BoardHistory / Rules + 其必需的 `cpp/core/` 头 |
| 项目侧职责 | 薄适配、19×19 特征提取、回放与题库测试 |
| 项目侧不实现 | 位棋盘、Zobrist、提子、简单劫/超级劫、禁入、终局与面积计分 |

选择理由：

- 规则实现经过 KataGo 训练、自对局、GTP 与异构规则配置长期验证；
- M8 搜索和方案 3 的教师树必须与学生使用同一棋规，直接复用可消除规则分叉；
- 只消费库代码，不把 KataGo 的对局循环、搜索或模型实现带进项目主线。

CMake 目标 `go_katago_game` 只编译 `cpp/game` 的四个源文件，并补入它实际依赖的
最小 `cpp/core` 源。`go_rules` 是对外薄适配层，业务代码不应直接 include KataGo
内部头。升级规则库前先跑项目题库、SGF 回放和 GnuGo 交叉对拍，再更新本表。

## Python 开发依赖

| 项目 | 值 |
|---|---|
| 文件 | `requirements-dev.txt` |
| SGF 解析 | `sgfmill==1.1.1` |
| HTTP 下载与重试 | `httpx==0.28.1` + `tenacity==9.2.1` |
| JSONL 流式 I/O | `orjson==3.13.0` |
| 特征张量 | `numpy==2.5.4` |
| 可视化 | `Pillow==13.0.0` |
| 许可证 | sgfmill MIT；httpx BSD-3-Clause；tenacity Apache-2.0；orjson MPL-2.0 AND (Apache-2.0 OR MIT)；NumPy BSD-3-Clause/0BSD/MIT/Zlib/CC0；Pillow MIT-CMU |
| 使用边界 | Python 负责 SGF/批处理/特征；合法性与计分仍由 KataGo 决定 |

首次开发环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

完整选型边界见
[`docs/design/07-third-party-selection.md`](docs/design/07-third-party-selection.md)。

## CWI 数据源

| 项目 | 值 |
|---|---|
| 地址 | <https://homepages.cwi.nl/~aeb/go/games/games.tgz> |
| 归档 SHA-256 | `c57ad2ba3c5f3a3ee145bb77bcc9c0167c0d7b5478cf691a8939f750511071e2` |
| 许可声明 | CWI 页面声明棋谱属于 public domain |
| 项目消费 | 1 万局 19×19，生成 `.nngames`；原始归档与派生数据不提交 Git |
| 数据集 SHA-256 | `bc4ea0c7929350395a0d632f53b2a59f270b6bb7510b3dcd5095c670e49d2313` |

数据的下载、过滤、编码与回放验收见
[`M2/M3 报告`](docs/benchmarks/m2-m3-data.md)。任何来源许可变化都必须重新审查。
