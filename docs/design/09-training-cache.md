# `.ngf` M4 训练缓存

> 状态：v1（2026-10-24）· 对应 M4
> 职责：Python 将 `.nngames` + KataGo 轨迹转换成只读训练样本；C++ 只读取缓存并
> 驱动 neuralnet.cpp。缓存是可再生成的本地数据，不进入 Git。

## 1. 数据流

```text
.nngames
  → go_data_cli export-range（单进程批量导出 JSONL）
  → Python 选择固定 seed 的对局并构造 trace-batch 请求
  → go_rules_cli trace-batch（每批 32 局）
  → Python 提取 9 平面 + policy/value 标签
  → .ngf（train / eval）
  → go_train_cli（neuralnet.cpp）
```

规则、提子、简单劫和行棋方全部来自 KataGo；`.ngf` 不复制规则状态机。

## 2. 文件格式 v1

所有整数为 little-endian，浮点为 IEEE-754 binary32。

文件头 56 字节：

| 偏移 | 类型 | 内容 |
|---:|---|---|
| 0 | char[8] | magic `NGFEAT01` |
| 8 | u32 | version = 1 |
| 12 | u32 | header size = 56 |
| 16 | u64 | record count |
| 24 | u32 | feature count = 3249 |
| 28 | u32 | board size = 19 |
| 32 | u8[24] | reserved，写零 |

后续每条记录固定 13,020 字节：

| 偏移 | 类型 | 内容 |
|---:|---|---|
| 0 | u64 | sample id |
| 8 | u32 | 原 `.nngames` 对局索引 |
| 12 | u32 | 0-based move number |
| 16 | u16 | policy target：`y*19+x` 或 PASS=`361` |
| 18 | i8 | outcome value：当前行棋方胜 `+1`、负 `-1`、未知/和棋 `0` |
| 19 | u8 | flags，保留为 0 |
| 20 | u8[4] | reserved，写零 |
| 24 | f32[3249] | 9×19×19，C-order；喂给 Conv2D 时为 3249 行 |

记录按文件顺序存储，因此第 `i` 个平面块偏移为
`56 + i*13020 + 24`。读取器在打开时只验证头部和文件大小，元数据按需读取，
避免扫描整个多 GB 缓存。

## 3. 切分与标签

`go-pipeline training-cache` 从数据集中按固定 seed 无放回抽样，再按对局切分，
防止同一局相邻局面跨 train/eval 泄漏。M4 正式配置为：

- seed `20261024`
- 900 局训练、100 局验证
- 每个非终局 frame 一条样本
- policy target = 该 frame 的下一手
- value target = 最终胜负从该 frame 行棋方视角的符号

对局结果未知时 value 为 0；M4 C 验证器只训练 policy，value 字段留给 B 的双头。

## 4. 命令

```bash
.venv/bin/go-pipeline training-cache \
  --dataset data/cwi-10k.nngames \
  --data-cli build/cpu/go_data_cli \
  --rules-cli build/cpu/go_rules_cli \
  --train-output data/m4-c-train.ngf \
  --eval-output data/m4-c-eval.ngf \
  --manifest data/m4-c-cache-manifest.json \
  --max-games 1000 --eval-games 100 --seed 20261024

./build/cpu/go_train_cli inspect --dataset data/m4-c-train.ngf
```

`*_cli` 可用显式路径指向 GPU 构建产物。正式训练使用 GPU 构建，但缓存格式与设备无关。
