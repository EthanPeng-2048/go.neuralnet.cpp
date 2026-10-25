# M6 GTP 适配器与 Arena 验收

> 日期：2026-10-24
> 结论：**M6 通过**。GTP v2 适配器可被 Sabaki 类 GUI 驱动（协议命令
> 全覆盖、genmove 恒为合法着、错误走 `? ` 前缀）；Arena 支持
> random / model / 外部 GTP 三种引擎并用 SPRT 给出统计判定；C++ 实时
> 特征提取与 Python 训练管线**逐位一致**。

## 1. 实测环境

与 [`m5-trainer.md`](m5-trainer.md) §1 相同（Linux 开发机，CPU，f32，
clang 24 + Ninja Release，neuralnet.cpp `v1.12.0`，KataGo `v1.18.2`）。

## 2. 组件

| 组件 | 内容 |
|---|---|
| `go::extract_features`（`go/features.hpp`） | M3 v1 九平面的 C++ 实时提取器；GTP 与 Arena 共用 |
| `go::PolicyPlayer`（`go/policy.hpp`） | 特征 → 前向 → **合法点** argmax（含 pass）；两条路径同一份选点语义 |
| `go_gtp_cli` | GTP v2 over stdin/stdout；规则全部走 KataGo |
| `go_arena_cli` | 批量对局 + SPRT；引擎 = `random` / `model:FILE[=DEVICE]` / `gtp:COMMAND` |

## 3. 特征一致性（最重要的正确性前提）

GTP 对局时的输入必须与训练时**逐位相同**，否则网络在对局中看到的是
它没见过的分布。`go_rules_cli features` 输出每一帧九平面的 float32
小端 hex，`tests/test_m6_features.py` 与 Python `extract_features`
逐位比对。

4 组对局（含 corner 提子、ko 形、pass 占位、36 手长局）× 全部帧
× 9×19×19 个 float：**全部 `np.array_equal` 通过**。

覆盖的语义点：

- own/opponent 相对行棋方视角；white_to_play 全盘填充；
- 简单劫禁入点（KataGo `ko_loc`）；
- 最近三手窗口：plane 4 = 倒数第三手、plane 6 = 最近一手；
  **pass 占用窗口槽位但不置位**（与 Python 的 `if point is None:
  continue` 一致）；
- 边界距离归一化 `min(x,y,W-1-x,H-1-y) / ((size-1)/2)`。

## 4. GTP 适配器

命令集：`protocol_version` `name` `version` `known_command`
`list_commands` `quit` `boardsize` `clear_board` `komi` `play`
`genmove` `showboard` `final_score` `time_settings` `time_left`
`reg_genmove`。

行为要点：

- `play` 走 KataGo tolerant 回放路径（GUI 摆子/悔棋权威指令），
  pass 同路径；
- `genmove` 在**合法着**（`game.is_legal`，含禁入点/劫）上取模型
  argmax；模型选不出合法点时回退 pass，协议上永远有应答；
- 非 19×19 的 genmove 返回 `? `（M3 v1 特征契约限制，写明原因）；
  行棋方不符时同样 `? `；
- `komi` 拒绝在有落子后修改（避免静默重建丢历史）；
- `final_score` 用 KataGo 中国规则计分。

测试（`tests/test_m6_gtp_arena.py::GtpAdapterTests`，6 例）覆盖
身份命令、完整对局序列（genmove 点位经盘面占用校验）、非 19×19 拒绝、
错误前缀、pass/resign 与终局计分。

**Sabaki 手工验证**：GTP v2 命令集与响应格式是 Sabaki 的连接契约；
本次验收以协议级自动化测试为准（开发机无 GUI）。人工连接步骤：

```text
Sabaki → Engine → Install Engine → Command:
  /path/to/go_gtp_cli --model /path/to/model.nnmodel --gpu=40HX
```

## 5. Arena 与 SPRT

- 每局由 KataGo 面积计分（中国规则 + komi）定胜负；和局不进似然；
- `--player` 颜色逐局轮换（消除先手偏差）；
- SPRT：`LLR = w·ln(p₁/p₀) + l·ln((1-p₁)/(1-p₀))`，边界
  `±ln((1-β)/α)`，越界即停（未达 `--games` 也停）；
- 外部 GTP 引擎经子进程管道驱动：每步把对手着用 `play` 转发给它，
  否则它的内部盘面脱节（实测踩过：不转发时下一步 genmove 报
  "color to move is b"）。

**实测**（M5 训出的 C 验证器 vs random，19×19，seed 20261024）：

| 局 | player 颜色 | 结果 | 累计 W/L/J | LLR |
|---|---|---|---|---|
| 1 | b | 胜 | 1/0/0 | 0.182 |
| 2 | w | 负 | 1/1/0 | −0.041 |
| 3 | b | 胜 | 2/1/0 | 0.142 |
| 4 | w | 胜 | 3/1/0 | 0.324 |

胜率 75%（4 局样本，仅证明管线，无统计意义；SPRT 判定 `continue`
正确——`p₀=0.5, p₁=0.6` 下 4 局远不足以过界）。

## 6. 自动化测试

CTest 两个（`tests/test_m6_features.py` 4 例 +
`tests/test_m6_gtp_arena.py` 14 例，其中 GnuGo 组在装有 `gnugo` 时
才跑，与 M1 crosscheck 同款"未装即跳"语义）：

- `go_features_m6_parity`：特征逐位一致性；
- `go_gtp_arena_m6`：GTP 协议 6 例 + Arena/SPRT 8 例（含
  LLR 对拍手算参考值、早停、报告落盘、非法引擎拒绝、GTP 子进程对手）。

同一提交 CPU CTest **9/9**（M0–M6 全量）。

## 7. vs GnuGo

`--opponent "gtp:gnugo --mode gtp --quiet"` 即为 DoD 的自动评估路径。
本机未安装 gnugo（与 M1 验收时相同），故：

- 测试里用**我们自己的 `go_gtp_cli` 作为外部 GTP 子进程**，走与
  gnugo 完全相同的管道路径（boardsize/clear_board/komi/play 转发 +
  genmove 解析 + 退出回收），证明子进程机制可用；
- `GnuGoArenaTests` 在装有 gnugo 的机器上自动激活
  （`shutil.which("gnugo")`）。

## 8. 验收命令

```bash
cmake --preset cpu && cmake --build --preset cpu
ctest --test-dir build/cpu --output-on-failure

# 手工对局（Sabaki 连接或管道驱动）：
build/cpu/go_gtp_cli --model artifacts/model.nnmodel --gpu=40HX

# 自动评估（装了 gnugo 的机器）：
build/cpu/go_arena_cli --player model:artifacts/model.nnmodel \
  --opponent "gtp:gnugo --mode gtp --quiet" \
  --games 100 --size 19 --sprt-p0 0.5 --sprt-p1 0.55 \
  --report arena-gnugo.json
```

## 9. 解释边界

- Arena 的每局时长随随机对手漂移（random 不主动 pass，会打满防御性
  上限 4×19² 手）；正式评估的对手是会正常终局的引擎；
- 本报告的对局结果只证明协议与管线，模型本身仍是 M4 的 C 验证器
  （无搜索、无合法性训练），棋力评估属于 M8/M9；
- GTP 适配器当前不做 time control（`time_settings` 应答但不生效），
  M8 之前无搜索，无时钟需求。
