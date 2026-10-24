# `.nngames`、Token 与特征平面

> 状态：v0.1（2026-10-24）· 对应 M2 / M3
> 约束：SGF 用 `sgfmill` 解析；规则与局面用 KataGo 重放；`.nngames` 的
> KVRecord v2 编解码直接使用 neuralnet.cpp；训练仍只使用 neuralnet.cpp。

## 1. 数据流

```text
CWI tar.gz
  → sgfmill 主线解析
  → Python 规范化 JSONL（临时）
  → go_data_cli + nn::KeyValueRecord v2
  → .nngames（存初始布置 + 走法 + 结果，不存逐手盘面）
  → KataGo 重放校验 / Python 特征提取
```

原始 SGF、下载归档和生成的 `.nngames` 均属于可再生成数据，不纳入 Git。
项目只跟踪下载 URL、归档 SHA-256、容器格式和验收脚本。

## 2. `.nngames` v1

文件：

```text
[magic "NNGM0001"（8 bytes）]
[根 KeyValueRecord v2，base_offset = 8]
```

根记录字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `format` | Str | 固定 `go.nngames` |
| `format_version` | UInt | 当前为 1 |
| `created_at` | UInt | Unix 秒 |
| `source` | Record | 名称、URL、归档 SHA-256、许可证、获取时间 |
| `game_count` | UInt | 对局数 |
| `games` | Record | 键 `g00000000`、`g00000001`… |

单局字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | Str | 稳定来源 ID |
| `source_path` | Str | 归档内相对路径 |
| `source_sha256` | Str | 原始 SGF 字节摘要 |
| `board_size` | UInt | 训练集只收 19 |
| `komi_twice` | UInt | 2×贴目，避免浮点歧义；负数按二补码写入 |
| `handicap` | UInt | SGF `HA` |
| `to_move` | UInt | 1 黑 / 2 白 |
| `outcome` | Str | `unknown/black/white/draw` |
| `result` | Str | 原始 SGF `RE` |
| `rules` | Str | `chinese/japanese` |
| `black_player`, `white_player`, `event`, `date` | Str | 元数据 |
| `setup_black`, `setup_white`, `setup_empty` | UIntArray | 初始布置点 |
| `moves` | UIntArray | 颜色感知走法 token |

走法 token：

```text
point = y * 19 + x                 # 0..360
pass_point = 361
token = point * 2 + is_white       # 黑 0..720，白 1..721
```

因此黑弃着 = 722，白弃着 = 723。初始布置点也使用 19 列固定坐标，
小棋盘取左上前缀；9×9 / 13×13 只保留为测试与沙盒，不进入正式训练集。

回放顺序固定为：建立初始盘面 → 按显式颜色逐手调用 KataGo tolerant replay →
比较 `Game.history()` 解码后的 token。v1 不支持第一手之后的 SGF setup，此类记录
在导入时拒绝；当前 CWI 1 万局中未触发该限制。

## 3. `GoTokenizer`

词表复用 `.nnvocab` 容器：`[label "NNVC"][词表 KVRecord v2]`。

| token | ID |
|---|---:|
| 19×19 点位 `y*19+x` | 0–360 |
| `<pass>` | 361 |
| `<bos>` | 362 |
| `<eos>` | 363 |
| `<decision>` | 364 |
| `<pad>` | 365 |
| `<unk>` | 366 |

`.nnvocab` 中 `type=go`、`version=1`、`max_board_size=19`；`vocab` 字段使用
neuralnet.cpp 自带 `encode_vocab_blob`，不走 BPE。

历史分支使用最近 K 手点位 token，左端补 `<pad>`；颜色由段/颜色嵌入表达。
完整对局编码为 `[BOS, ..., EOS]`。

## 4. 特征平面 v1

特征以当前行棋方视角生成，形状 `(9, H, W)`、`float32`：

| # | 名称 | 含义 |
|---:|---|---|
| 0 | `own_stones` | 当前行棋方棋子 |
| 1 | `opponent_stones` | 对手棋子 |
| 2 | `white_to_play` | 白方行棋时全 1，否则全 0 |
| 3 | `simple_ko_ban` | KataGo 给出的当前简单劫禁着点 |
| 4–6 | `move_minus_{1,2,3}` | 最近三手落点 one-hot；弃着为空平面 |
| 7 | `empty` | 空点 |
| 8 | `border_distance` | 到最近边界的曼哈顿距离，按半盘归一化 |

Scalar 特征：

| 名称 | 含义 |
|---|---|
| `komi` | 贴目 |
| `move_number` | 当前手数 |
| `black_stones_captured` | 被提黑子数 |
| `white_stones_captured` | 被提白子数 |
| `captured_balance_for_black` | 被提白子 − 被提黑子 |

单样本 `planes` 为 `(9,19,19)`；喂给 neuralnet.cpp 的卷积样本列为
`flatten_nn()` 的 `(9*19*19,)`，批量列布局为 `(9*19*19, batch)`。
合法点集合不塞进特征平面，M7/M8 在 host 侧从 KataGo 单独取得。

## 5. 局面轨迹边界

Python 不实现提子或劫。`go_rules_cli trace --input JSON` 输入初始 ASCII 盘面与
显式走法，逐手输出：

- 盘面
- 下一手方
- 当前简单劫禁着点
- 双方被提子数
- 终局状态

Python feature extractor 只把该轨迹转换为 NumPy 张量。SGF/`.nngames` 中的非法或
历史异例走法用 KataGo tolerant replay 处理，原始记录不被静默改写。
