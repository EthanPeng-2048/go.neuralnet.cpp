from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import sample_from_trace, trace_records
from .model import GameRecord, PASS_POINT, decode_move

CACHE_MAGIC = b"NGFEAT01"
CACHE_VERSION = 1
CACHE_HEADER = struct.Struct("<8sIIQII24x")
CACHE_RECORD = struct.Struct("<QIIHbB4x")
CACHE_HEADER_SIZE = CACHE_HEADER.size
CACHE_RECORD_SIZE = CACHE_RECORD.size
FEATURE_PLANES = 9
BOARD_SIZE = 19
FEATURE_COUNT = FEATURE_PLANES * BOARD_SIZE * BOARD_SIZE
POLICY_CLASSES = PASS_POINT + 1


@dataclass(frozen=True, slots=True)
class CacheBuildResult:
    path: Path
    split: str
    games: int
    records: int
    bytes: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "split": self.split,
            "games": self.games,
            "records": self.records,
            "bytes": self.bytes,
            "sha256": self.sha256,
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
    if outcome == "draw" or outcome == "unknown":
        return 0
    return 1 if outcome == to_move else -1


def _samples_for_game(
    record: GameRecord,
    trace: dict[str, object],
    *,
    first_sample_id: int,
    game_index: int,
) -> list[tuple[int, int, int, int, int, np.ndarray]]:
    if record.board_size != BOARD_SIZE:
        raise ValueError(f"{record.id}: M4 cache requires 19x19 games")
    frames = trace["frames"]
    assert isinstance(frames, list)
    if len(frames) != len(record.moves) + 1:
        raise ValueError(
            f"{record.id}: trace frame count {len(frames)} != moves + 1"
        )

    samples: list[tuple[int, int, int, int, int, np.ndarray]] = []
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
        samples.append(
            (
                first_sample_id + move_index,
                game_index,
                move_index,
                target,
                value,
                sample.flatten_nn().astype(np.float32, copy=False),
            )
        )
    return samples


def write_feature_cache(
    output: Path,
    *,
    split: str,
    records: list[tuple[int, GameRecord]],
    rules_cli: Path,
) -> CacheBuildResult:
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
    )
    with temporary.open("wb") as stream:
        stream.write(header)
        sample_id = 0
        for chunk_start in range(0, len(records), 32):
            chunk = records[chunk_start : chunk_start + 32]
            traces = trace_records([record for _, record in chunk], rules_cli)
            for (game_index, record), trace in zip(chunk, traces, strict=True):
                for (
                    current_id,
                    current_game,
                    move_number,
                    target,
                    value,
                    planes,
                ) in _samples_for_game(
                    record,
                    trace,
                    first_sample_id=sample_id,
                    game_index=game_index,
                ):
                    if planes.shape != (FEATURE_COUNT,):
                        raise ValueError(f"{record.id}: invalid cached feature shape")
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
                    stream.write(planes.astype("<f4", copy=False).tobytes(order="C"))
                    sample_id += 1
    temporary.replace(output)
    return CacheBuildResult(
        path=output,
        split=split,
        games=len(records),
        records=record_count,
        bytes=output.stat().st_size,
        sha256=_sha256_file(output),
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
) -> dict[str, object]:
    if max_games <= 0:
        raise ValueError("max_games must be positive")
    if not 0 < eval_games < max_games:
        raise ValueError("eval_games must be in [1, max_games)")

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
    )
    evaluation = write_feature_cache(
        eval_output,
        split="eval",
        records=eval_records,
        rules_cli=rules_cli,
    )
    return {
        "format": "go.feature-cache",
        "format_version": CACHE_VERSION,
        "dataset": str(dataset),
        "seed": seed,
        "selected_games": selected,
        "train": train.to_json(),
        "eval": evaluation.to_json(),
    }
