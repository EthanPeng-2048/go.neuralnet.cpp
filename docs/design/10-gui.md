# 对弈室 GUI（go_gui_cli）

> 状态：**已落地（2026-10-25）** · 规则口径：**中国规则**（与挑战对手约定）
> · 依赖：M1 棋规（KataGo）、M6 引擎抽象（random / model / GTP 子进程）、
> M7 策略模型（go::PolicyPlayer）。

## 1. 目标

一个本地起、浏览器开的对弈室，两个模式：

| 模式 | 给谁用 | 做什么 |
|---|---|---|
| **对弈** | 人 | 人类 vs 策略模型（或 random / 外部 GTP 引擎）：点按落子、悔棋、停一手、认输、SGF 导出 |
| **竞技台** | 观众 | 引擎自动对局观战：后台线程节奏落子、换先开新局、累计比分、速度可调 |

观赏性来自两件 M8 前就能做的小事：**候选着热力图**（模型在合法着上的
softmax 分布）与**价值头形势判断**（B 编码器的 tanh 价值输出）。没有
策略分布的引擎（random / 外部 GTP）不报错，而是给出**"将下点 100%"的
合成预告**（`synthetic = true`）：random 会把预告的一步记住、实战就下它，
外部 GTP 引擎走 `reg_genmove` 问出来。

竞技台比分按**引擎**统计：每局换先后，胜局随引擎换到新席位，界面上
"引擎名 + 胜局"永远对应同一引擎（黑/白只是当前席位）。

**不做**（边界）：MCTS 搜索（属 M8）、多客户端协同编辑、鉴权/TLS（只听
127.0.0.1）、多盘并行观战。

## 2. 规则口径

固定 `go::RuleSet::chinese(7.5F)`：

- 位置超级劫（positional superko）；
- **面积/数子计分**（中国规则），终局由连续两次停一手触发 KataGo
  `endAndScoreGameNow`；
- 禁止多子自杀；贴 **7.5 目**（无和棋）；
- SGF 标注 `RU[Chinese]`、`KM[7.5]`。

棋规全部走 KataGo（`go::Game`），与 M1/M2/M6/M7 同源；GUI 不实现任何
规则逻辑。komi 可在新对局时改（`--komi` / 界面），计分仍是中国规则。

## 3. 架构

```
gui/index.html ──HTTP──> go_gui_cli（路由）
                            │
                       go::gui::GuiSession  ←─ 单互斥锁保护全部状态
                        │        │
              go::Game（KataGo）  go::ArenaEngine（random / model / gtp:）
                                        │
                                go::PolicyPlayer（M3 特征 → 模型 → 合法着掩码）
```

| 文件 | 职责 |
|---|---|
| `gui/index.html` | 单文件前端（内联 CSS/JS，无外部资源），构建期嵌入二进制 |
| `src/go/go_gui_cli.cpp` | CLI、HTTP 路由、页面/JSON/SGF 出口 |
| `include/go/gui_session.hpp` `src/go/gui_session.cpp` | 会话状态机：对弈流程、竞技台后台线程、状态 JSON、SGF |
| `include/go/http_server.hpp` `src/go/http_server.cpp` | 极简 HTTP/1.1（每请求一连接，GET/POST，无 keep-alive） |
| `include/go/arena_engine.hpp` `src/go/arena_engine.cpp` | 引擎抽象，从 `go_arena_cli` 抽出，竞技台与 SPRT 评估共用 |
| `cmake/embed_text.cmake` | 文本 → raw string 头的构建期嵌入 |

`PolicyPlayer::evaluate` 是本里程碑给推理侧的唯一扩展：与 `select_move`
同一次前向，输出合法着 softmax 分布 + 价值头（C 验证器无价值头 → null），
M8 的 MCTS 先验会复用同一接口。

### 线程模型

一把互斥锁保护 `GuiSession` 全部状态；HTTP 每连接一个工作线程，竞技台一个
后台线程按 `speed_ms` 节奏落子。引擎前向在锁内执行（模型推理毫秒级，
简单正确优先）；stop() 只置原子标志，accept 循环 200 ms 内退出。

## 4. HTTP 契约

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 内嵌单页前端 |
| GET | `/api/state` | 局面快照（棋盘 361 字符、历史、提子、结果、模式字段） |
| GET | `/api/evaluate` | 候选着 top10 + 价值头 + `synthetic` 标记（无分布引擎 = 将下点 100% 预告） |
| GET | `/api/sgf` | 当前对局 SGF（`RU[Chinese]`） |
| POST | `/api/play` `/api/pass` `/api/engine-move` `/api/undo` `/api/resign` | 对弈动作 |
| POST | `/api/new` `/api/mode` `/api/config` | 新对局（执黑/执白/自由摆棋 + komi）、模式切换、速度/自动开新局/komi |
| POST | `/api/arena/start|pause|step|next|reset` | 竞技台控制 |

失败一律 `{"ok": false, "error": ...}` + 非 2xx（400 参数错 / 409 状态不允许 /
500 引擎故障）。完整 JSON 形状以 `tests/test_gui_http.py` 为准。

## 5. 使用

```bash
# 默认对手 = GnuGo（PATH 里有 gnugo 时；否则退回自动发现的模型 / random）
# 竞技台默认 = 项目模型（黑） vs 默认对手（白）
./build/cpu/go_gui_cli

# 指定模型与 GPU、竞技台双方、节奏
./build/cpu/go_gui_cli --model run.nnmodel --device 40HX \
  --black-engine model:run.nnmodel=40HX --white-engine "gtp:gnugo --mode gtp --quiet" \
  --speed 800
```

浏览器打开打印的 `http://127.0.0.1:8765` 即可。引擎 spec 与 `go_arena_cli`
一致：`random | model:FILE[=DEVICE] | gtp:COMMAND`（竞技台可以摆
`gtp:gnugo --mode gtp --quiet` 看模型 vs GnuGo）。

## 6. 测试与验收

| 测试 | 覆盖 |
|---|---|
| `go_gui_session_tests` | 对弈流程/悔棋/中国规则数子（空盘 7.5 目白胜）/认输、竞技台步进/换先/比分、SGF、状态 JSON、评估拒绝 |
| `go_gui_http`（tests/test_gui_http.py） | 起真实 `go_gui_cli`：页面服务、对弈+引擎应手+悔棋、两次停一手数子、认输、SGF、竞技台单步/换先/重置/自动对局、random 引擎评估 409、模型引擎评估分布 |

## 7. 已知边界

- 无搜索：`model:` 引擎是纯策略 argmax（M8 前的口径），棋力弱于带搜索的
  KataGo 属预期。
- `/api/mode` 切换会开新局（不留半盘棋）；悔棋只在对弈模式可用。
- GTP 子进程引擎的思考时间会短暂阻塞局面轮询（单锁模型的取舍）。
- 外部 GTP 引擎的 `genmove` 返回 `pass`/`resign` 按大小写不敏感解析
  （GnuGo 3.8 返回大写 `PASS`）。
