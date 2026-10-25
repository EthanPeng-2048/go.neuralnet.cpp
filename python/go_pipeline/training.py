from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .features import sample_from_trace, trace_records
from .model import GameRecord, PASS_POINT, decode_move
from .tokenizer import PAD, VOCAB_SIZE

CACHE_MAGIC = b"NGFEAT01"
# M7 v2: per-sample move history (u16 point ids, left <pad>-padded). v1 files
# remain readable in C++ (version-dispatched stride); the writer only emits v2.
CACHE_VERSION = 2
# Header layout v2: same 56 bytes, reserved bytes 32..40 carry u32 history_len
# + u32 history_vocab (all-zero in v1 = no history).
CACHE_HEADER = struct.Struct("<8sIIQIIII16x")
CACHE_RECORD = struct.Struct("<QIIHbB4x")
CACHE_HEADER_SIZE = CACHE_HEADER.size
CACHE_RECORD_SIZE = CACHE_RECORD.size
FEATURE_PLANES = 9
BOARD_SIZE = 19
FEATURE_COUNT = FEATURE_PLANES * BOARD_SIZE * BOARD_SIZE
POLICY_CLASSES = PASS_POINT + 1
DEFAULT_HISTORY_LEN = 64


@dataclass(frozen=True, slots=True)
class SkippedGame:
    """一局被跳过的棋谱 + 原因（数据边界，如重放颜色不一致）。"""

    game: str
    reason: str

    def to_json(self) -> dict[str, str]:
        return {"game": self.game, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class CacheBuildResult:
    path: Path
    split: str
    games: int
    records: int
    bytes: int
    sha256: str
    skipped: tuple[SkippedGame, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "split": self.split,
            "games": self.games,
            "records": self.records,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "skipped": [entry.to_json() for entry in self.skipped],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _export_records(
    data_cli: Path,
    dataset: Path,
    available: int,
    indices: list[int],
) -> list[tuple[int, GameRecord]]:
    wanted = set(indices)
    records: dict[int, GameRecord] = {}
    process = subprocess.Popen(
        [
            str(data_cli),
            "export-range",
            "--input",
            str(dataset),
            "--start",
            "0",
            "--limit",
            str(available),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    for row, line in enumerate(process.stdout):
        if row in wanted:
            records[row] = GameRecord.from_json(json.loads(line))
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"export-range failed: {stderr.strip()}")
    missing = wanted.difference(records)
    if missing:
        raise RuntimeError(f"export-range omitted indices: {sorted(missing)[:8]}")
    return [(index, records[index]) for index in indices]


def _dataset_game_count(data_cli: Path, dataset: Path) -> int:
    result = subprocess.run(
        [str(data_cli), "stats", "--input", str(dataset)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(json.loads(result.stdout)["games"])


def _value_for_result(outcome: str, to_move: str) -> int:
    """Outcome from the side-to-move's perspective: +1 win, -1 loss, 0 unknown.

    `outcome` is the game result ("black" / "white" / "draw"), `to_move` is the
    M2 colour tag ("b" / "w"). Comparing them directly would make every label
    -1 — the value head would then train on a constant target and only surface
    as "value AUC = 0" much later.
    """
    if outcome == "draw" or outcome == "unknown":
        return 0
    side = "black" if to_move == "b" else "white"
    return 1 if outcome == side else -1


def _point_token(token: int) -> int:
    """M2 color-tagged token → M3 point-id token (0..360 point, 361 pass)."""
    _, point = decode_move(token)
    return PASS_POINT if point is None else point[1] * BOARD_SIZE + point[0]


def _samples_for_game(
    record: GameRecord,
    trace: dict[str, object],
    *,
    first_sample_id: int,
    game_index: int,
    history_len: int,
) -> list[tuple[int, int, int, int, int, np.ndarray, np.ndarray]]:
    if record.board_size != BOARD_SIZE:
        raise ValueError(f"{record.id}: M4 cache requires 19x19 games")
    frames = trace["frames"]
    assert isinstance(frames, list)
    if len(frames) != len(record.moves) + 1:
        raise ValueError(
            f"{record.id}: trace frame count {len(frames)} != moves + 1"
        )

    samples: list[tuple[int, int, int, int, int, np.ndarray, np.ndarray]] = []
    for move_index, token in enumerate(record.moves):
        color, point = decode_move(token)
        frame = frames[move_index]
        assert isinstance(frame, dict)
        if str(frame["to_move"]) != color:
            raise ValueError(
                f"{record.id}: move {move_index} color disagrees with replay"
            )
        target = PASS_POINT if point is None else point[1] * BOARD_SIZE + point[0]
        sample = sample_from_trace(
            trace,
            frame_index=move_index,
            record=record,
        )
        value = _value_for_result(record.outcome, color)
        # History for the sample predicting move i = the i moves before it,
        # left <pad>-padded and right-aligned (row 0 = oldest slot), point ids.
        recent = [
            _point_token(move_token)
            for move_token in record.moves[max(0, move_index - history_len) : move_index]
        ]
        history = [PAD] * (history_len - len(recent)) + recent
        samples.append(
            (
                first_sample_id + move_index,
                game_index,
                move_index,
                target,
                value,
                sample.flatten_nn().astype(np.float32, copy=False),
                np.asarray(history, dtype="<u2"),
            )
        )
    return samples


def _trace_chunks(
    chunks: list[list[tuple[int, GameRecord]]],
    rules_cli: Path,
    jobs: int,
) -> Iterator[list[dict[str, object]]]:
    """按 chunk 原序产出 trace（每元素 = 该 32 局 chunk 的 trace 列表）。

    jobs=1：串行起 trace-batch 子进程（原行为）。jobs>1：线程池并发跑
    子进程（subprocess 不占 GIL），最多 jobs 个 chunk 在飞；消费端按
    chunk 序取结果 ⇒ 写盘顺序、sample_id 序列与串行逐位一致。
    """
    if jobs <= 1:
        for chunk in chunks:
            yield trace_records([record for _, record in chunk], rules_cli)
        return
    iterator = iter(chunks)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending: deque[Future[list[dict[str, object]]]] = deque()

        def submit_next() -> bool:
            chunk = next(iterator, None)
            if chunk is None:
                return False
            pending.append(
                pool.submit(trace_records, [record for _, record in chunk], rules_cli)
            )
            return True

        for _ in range(jobs):
            if not submit_next():
                break
        while pending:
            yield pending.popleft().result()
            submit_next()


def write_feature_cache(
    output: Path,
    *,
    split: str,
    records: list[tuple[int, GameRecord]],
    rules_cli: Path,
    history_len: int = DEFAULT_HISTORY_LEN,
    jobs: int = 1,
) -> CacheBuildResult:
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    record_count = sum(len(record.moves) for _, record in records)
    if record_count == 0:
        raise ValueError(f"{split}: no samples generated")

    header = CACHE_HEADER.pack(
        CACHE_MAGIC,
        CACHE_VERSION,
        CACHE_HEADER_SIZE,
        record_count,
        FEATURE_COUNT,
        BOARD_SIZE,
        history_len,
        VOCAB_SIZE,
    )
    skipped: list[SkippedGame] = []
    with temporary.open("wb") as stream:
        stream.write(header)
        sample_id = 0
        chunks = [records[start : start + 32] for start in range(0, len(records), 32)]
        for chunk, traces in zip(
            chunks, _trace_chunks(chunks, rules_cli, jobs), strict=True
        ):
            for (game_index, record), trace in zip(chunk, traces, strict=True):
                try:
                    game_samples = _samples_for_game(
                        record,
                        trace,
                        first_sample_id=sample_id,
                        game_index=game_index,
                        history_len=history_len,
                    )
                except ValueError as error:
                    # 数据边界（如重放颜色不一致的棋谱）：跳过整局并记入
                    # manifest，不为一局脏数据废掉整个缓存构建。
                    skipped.append(SkippedGame(game=record.id, reason=str(error)))
                    continue
                for (
                    current_id,
                    current_game,
                    move_number,
                    target,
                    value,
                    planes,
                    history,
                ) in game_samples:
                    if planes.shape != (FEATURE_COUNT,):
                        raise ValueError(f"{record.id}: invalid cached feature shape")
                    if history.shape != (history_len,):
                        raise ValueError(f"{record.id}: invalid cached history shape")
                    stream.write(
                        CACHE_RECORD.pack(
                            current_id,
                            current_game,
                            move_number,
                            target,
                            value,
                            0,
                        )
                    )
                    stream.write(history.tobytes(order="C"))
                    stream.write(planes.astype("<f4", copy=False).tobytes(order="C"))
                    sample_id += 1
        if sample_id == 0:
            raise ValueError(f"{split}: no samples generated")
        if skipped:
            # 跳过整局改变了样本总数：回头修正文件头部的记录数。
            stream.seek(0)
            stream.write(
                CACHE_HEADER.pack(
                    CACHE_MAGIC,
                    CACHE_VERSION,
                    CACHE_HEADER_SIZE,
                    sample_id,
                    FEATURE_COUNT,
                    BOARD_SIZE,
                    history_len,
                    VOCAB_SIZE,
                )
            )
    temporary.replace(output)
    return CacheBuildResult(
        path=output,
        split=split,
        games=len(records) - len(skipped),
        records=sample_id,
        bytes=output.stat().st_size,
        sha256=_sha256_file(output),
        skipped=tuple(skipped),
    )


def build_training_caches(
    *,
    dataset: Path,
    data_cli: Path,
    rules_cli: Path,
    train_output: Path,
    eval_output: Path,
    max_games: int,
    eval_games: int,
    seed: int,
    history_len: int = DEFAULT_HISTORY_LEN,
    jobs: int = 1,
) -> dict[str, object]:
    if max_games <= 0:
        raise ValueError("max_games must be positive")
    if not 0 < eval_games < max_games:
        raise ValueError("eval_games must be in [1, max_games)")
    if history_len <= 0:
        raise ValueError("history_len must be positive")

    available = _dataset_game_count(data_cli, dataset)
    if max_games > available:
        raise ValueError(f"requested {max_games} games but dataset has {available}")
    selected = random.Random(seed).sample(range(available), max_games)
    selected.sort()
    eval_indices = set(selected[-eval_games:])
    train_indices = [index for index in selected if index not in eval_indices]
    eval_indices_ordered = [index for index in selected if index in eval_indices]
    train_records = _export_records(
        data_cli, dataset, available, train_indices
    )
    eval_records = _export_records(
        data_cli, dataset, available, eval_indices_ordered
    )
    train = write_feature_cache(
        train_output,
        split="train",
        records=train_records,
        rules_cli=rules_cli,
        history_len=history_len,
        jobs=jobs,
    )
    evaluation = write_feature_cache(
        eval_output,
        split="eval",
        records=eval_records,
        rules_cli=rules_cli,
        history_len=history_len,
        jobs=jobs,
    )
    return {
        "format": "go.feature-cache",
        "format_version": CACHE_VERSION,
        "history_len": history_len,
        "dataset": str(dataset),
        "seed": seed,
        "selected_games": selected,
        "train": train.to_json(),
        "eval": evaluation.to_json(),
    }
