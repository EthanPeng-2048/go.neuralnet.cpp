#!/usr/bin/env python3
"""`.ngf` v2 move-history contract.

The v2 cache carries a K-slot move history per sample (GoTokenizer point ids,
left `<pad>`-padded, right-aligned, excluding the move being predicted). It is
verified frame by frame against the Python tokenizer: a mismatch here would
train a history branch on shuffled moves and only ever show up as "the history
branch is barely better than nothing", so it is asserted directly instead of
being inferred from accuracy.

The v1 rejection case is kept here because it is the same contract seen from
the other side: a v1 cache (no history block) must be refused loudly rather
than silently substituted with zeros.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from go_pipeline.model import GameRecord, decode_move, encode_move
from go_pipeline.tokenizer import PAD, PASS, VOCAB_SIZE
from go_pipeline.training import write_feature_cache

ROOT = Path(__file__).resolve().parents[1]


def binary(name: str) -> Path:
    directories = []
    if value := os.environ.get("GO_TEST_BINARY_DIR"):
        directories.append(Path(value))
    directories.extend((ROOT / "build/cpu", ROOT / "build/gpu"))
    for directory in directories:
        candidate = directory / name
        if candidate.is_file():
            return candidate
        executable = candidate.with_suffix(".exe")
        if executable.is_file():
            return executable
    searched = ", ".join(str(path) for path in directories)
    raise RuntimeError(f"cannot find {name}; searched: {searched}")


RULES_CLI = binary("go_rules_cli")
TRAIN_CLI = binary("go_train_cli")

SEED = 20261024
BATCH = 16

# The contract is K-agnostic; a small K keeps the test fast.
TINY_K = 8


def make_records(count: int, moves_per_game: int) -> list[tuple[int, GameRecord]]:
    """Synthetic 19x19 games with a shared, peaked point vocabulary."""
    points = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10),
              (12, 12), (14, 14), (16, 16), (18, 18), (1, 3), (3, 5)]
    assert moves_per_game <= len(points)
    records: list[tuple[int, GameRecord]] = []
    for index in range(count):
        moves = []
        for step in range(moves_per_game):
            color = "b" if step % 2 == 0 else "w"
            x, y = points[(step + index) % len(points)]
            moves.append(encode_move(color, (x, y)))
        records.append((
            index,
            GameRecord(
                id=f"cache-history-{index}",
                source_path=f"cache-history-{index}.sgf",
                source_sha256="0" * 64,
                board_size=19,
                komi_twice=15,
                handicap=0,
                to_move="b",
                outcome="black" if index % 2 == 0 else "white",
                result="B+R" if index % 2 == 0 else "W+R",
                rules="chinese",
                black_player="black",
                white_player="white",
                event="cache-history-test",
                date="2026-10-24",
                setup_black=(),
                setup_white=(),
                setup_empty=(),
                moves=tuple(moves),
            ),
        ))
    return records


def point_token(token: int) -> int:
    """M2 color-tagged token → M3 point-id token, mirroring the C++ reader."""
    _, point = decode_move(token)
    return PASS if point is None else point[1] * 19 + point[0]


def read_cache_history(path: Path, history_len: int) -> list[list[int]]:
    """Decode the v2 history block of every record straight from the file."""
    data = path.read_bytes()
    magic, version, _header_size, records, features, board = struct.unpack_from(
        "<8sIIQII", data, 0
    )
    assert magic == b"NGFEAT01", magic
    assert version == 2, version
    assert board == 19
    stored_len, stored_vocab = struct.unpack_from("<II", data, 32)
    assert stored_len == history_len, (stored_len, history_len)
    assert stored_vocab == VOCAB_SIZE, stored_vocab
    plane_bytes = features * 4
    stride = 24 + 2 * history_len + plane_bytes
    assert len(data) == 56 + records * stride, (len(data), records, stride)

    out: list[list[int]] = []
    for index in range(records):
        offset = 56 + index * stride + 24
        out.append(list(struct.unpack_from(f"<{history_len}H", data, offset)))
    return out


class FeatureCacheSkipTests(unittest.TestCase):
    """数据边界：颜色不交替的棋谱跳过并记账，其余照常缓存。"""

    def test_non_alternating_game_is_skipped(self) -> None:
        good = make_records(count=1, moves_per_game=6)[0][1]
        bad_moves = list(good.moves)
        # 第 3 手改成与第 2 手同色（重放 to_move 必然对不上）。
        bad_moves[3] = encode_move("b", (6, 6))
        bad = replace(good, id="cache-history-bad", moves=tuple(bad_moves))

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "skipped.ngf"
            result = write_feature_cache(
                cache, split="train",
                records=[(0, good), (1, bad)],
                rules_cli=RULES_CLI, history_len=TINY_K,
            )
            self.assertEqual(result.games, 1)
            self.assertEqual(result.records, len(good.moves))
            self.assertEqual(
                [entry.game for entry in result.skipped], ["cache-history-bad"]
            )
            self.assertIn("color disagrees", result.skipped[0].reason)
            decoded = read_cache_history(cache, TINY_K)
            self.assertEqual(len(decoded), len(good.moves))
            self.assertEqual(result.to_json()["skipped"][0]["game"], "cache-history-bad")


class FeatureCacheHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        work = Path(cls._temporary.name)
        cls.work = work
        cls.train_records = make_records(count=16, moves_per_game=12)
        cls.eval_records = make_records(count=4, moves_per_game=12)
        cls.train_cache = work / "train.ngf"
        cls.eval_cache = work / "eval.ngf"
        write_feature_cache(
            cls.train_cache, split="train", records=cls.train_records,
            rules_cli=RULES_CLI, history_len=TINY_K,
        )
        write_feature_cache(
            cls.eval_cache, split="eval", records=cls.eval_records,
            rules_cli=RULES_CLI, history_len=TINY_K,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_cache_history_matches_tokenizer(self) -> None:
        """Every stored history must equal the tokenizer's view of the same
        game: the K moves before the predicted one, left <pad>-padded."""
        decoded = read_cache_history(self.train_cache, TINY_K)
        expected: list[list[int]] = []
        for _, record in self.train_records:
            tokens = [point_token(token) for token in record.moves]
            for move_index in range(len(tokens)):
                recent = tokens[max(0, move_index - TINY_K):move_index]
                expected.append([PAD] * (TINY_K - len(recent)) + recent)
        self.assertEqual(len(decoded), len(expected))
        for index, (actual, want) in enumerate(zip(decoded, expected, strict=True)):
            self.assertEqual(actual, want, f"record {index}: history mismatch")

    def test_cache_history_excludes_predicted_move(self) -> None:
        """The last history slot must never be the move being predicted —
        that would leak the answer into the input."""
        record = self.train_records[0][1]
        tokens = [point_token(token) for token in record.moves]
        decoded = read_cache_history(self.train_cache, TINY_K)
        offset = 0
        for move_index in range(len(tokens)):
            slot = decoded[offset + move_index]
            if move_index == 0:
                self.assertEqual(slot, [PAD] * TINY_K)
            else:
                self.assertEqual(slot[-1], tokens[move_index - 1])
                self.assertNotIn(tokens[move_index], slot[1:])
            # A pad may only appear as a left-aligned prefix.
            pads = sum(1 for value in slot if value == PAD)
            self.assertEqual(slot[:pads], [PAD] * pads)

    def test_v1_cache_rejected(self) -> None:
        """A v1 cache (no history) must be refused loudly. Silently feeding
        zeros would hide a data-pipeline regression behind "no gain"."""
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            v1 = work / "v1.ngf"
            written_k = 1
            write_feature_cache(
                v1, split="train", records=self.train_records,
                rules_cli=RULES_CLI, history_len=written_k,
            )
            header = bytearray(v1.read_bytes()[:56])
            struct.pack_into("<I", header, 8, 1)  # version = 1
            struct.pack_into("<II", header, 32, 0, 0)  # no history in v1
            body = v1.read_bytes()[56:]
            # A v1 record is the v2 record minus the history block: the 24-byte
            # meta and the planes are byte-identical across versions, so v1 is
            # exactly what the reader reconstructs with stride 13020.
            v1_records = sum(len(record.moves) for _, record in self.train_records)
            v2_stride = 24 + 2 * written_k + 9 * 19 * 19 * 4
            v1_stride = 24 + 9 * 19 * 19 * 4
            self.assertEqual(v1_stride, 13020)
            stripped = bytearray()
            for index in range(v1_records):
                start = index * v2_stride
                stripped += body[start:start + 24]
                stripped += body[start + 24 + 2 * written_k:start + v2_stride]
            self.assertEqual(len(stripped), v1_records * v1_stride)
            (work / "v1-stripped.ngf").write_bytes(bytes(header) + bytes(stripped))

            result = subprocess.run(
                [
                    str(TRAIN_CLI), "train",
                    "--dataset", str(work / "v1-stripped.ngf"),
                    "--eval-dataset", str(self.eval_cache),
                    "--cpu", "--batch-size", str(BATCH), "--epochs", "1",
                    "--arch", "b", "--seed", str(SEED),
                    "--checkpoint", str(work / "c.nnckpt"),
                    "--model", str(work / "m.nnmodel"),
                    "--metrics-out", str(work / "metrics.json"),
                ],
                capture_output=True, text=True,
            )
            # Either the cache reader or the trainer must refuse; both are
            # acceptable, silently succeeding is not.
            self.assertNotEqual(result.returncode, 0)
            combined = result.stdout + result.stderr
            self.assertIn("history", combined.lower())


if __name__ == "__main__":
    unittest.main()
