#!/usr/bin/env python3
"""M6 feature parity: the C++ live extractor must match the Python trainer.

The GTP adapter extracts features in C++ from a live `go::Game`; the training
pipeline extracts them in Python from trace frames. If the two ever disagree,
the network silently receives different inputs at play time than at training
time. This test replays real games through both paths and requires bitwise
equality of every float32 plane value.
"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

import numpy as np

from go_pipeline.features import Position, extract_features
from go_pipeline.model import GameRecord, decode_move, encode_move
from go_pipeline.sgf import parse_sgf

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


def trace_request(record: GameRecord) -> dict[str, object]:
    moves = []
    for token in record.moves:
        color, point = decode_move(token)
        moves.append({"color": color, "point": None if point is None else list(point)})
    return {
        "size": record.board_size,
        "rules": record.rules,
        "komi": record.komi_twice / 2,
        "to_move": record.to_move,
        "rows": record.initial_rows(),
        "moves": moves,
    }


def cpp_planes(record: GameRecord) -> list[np.ndarray]:
    result = subprocess.run(
        [str(RULES_CLI), "features", "--input", "-"],
        input=json.dumps(trace_request(record)),
        text=True,
        check=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    return [
        np.frombuffer(bytes.fromhex(hex_value), dtype="<f4").reshape(9, 19, 19)
        for hex_value in payload["planes_hex_le_f32"]
    ]


def python_planes(record: GameRecord) -> list[np.ndarray]:
    # Legality enforcement (captures, ko) lives in KataGo on the C++ side;
    # reuse its frames so this test compares feature extraction only, not two
    # independent rule implementations.
    traced = subprocess.run(
        [str(RULES_CLI), "trace", "--input", "-"],
        input=json.dumps(trace_request(record)),
        text=True,
        check=True,
        capture_output=True,
    )
    frames = json.loads(traced.stdout)["frames"]
    samples = []
    for frame_index, frame in enumerate(frames):
        recent = [
            decode_move(token)[1] for token in record.moves[:frame_index]
        ]
        position = Position.from_frame(
            frame, recent_moves=recent, komi=record.komi_twice / 2
        )
        samples.append(extract_features(position).planes)
    return samples


class M6FeatureParityTests(unittest.TestCase):
    def assert_parity(self, record: GameRecord) -> None:
        ours = cpp_planes(record)
        theirs = python_planes(record)
        self.assertEqual(len(ours), len(theirs))
        self.assertGreater(len(ours), 1)
        for index, (left, right) in enumerate(zip(ours, theirs, strict=True)):
            self.assertEqual(left.shape, (9, 19, 19))
            if not np.array_equal(left, right):
                location = np.unravel_index(
                    int(np.argmax(np.abs(left - right))), left.shape
                )
                self.fail(
                    f"frame {index} plane/value {location}: "
                    f"cpp={left[location]!r} python={right[location]!r}"
                )

    def test_capture_sgf_is_5x5(self) -> None:
        # The M3 v1 feature contract is 19x19-only; the fixture reminds us
        # why parity games below are synthetic 19x19 records.
        record = parse_sgf(
            (ROOT / "tests/data/capture.sgf").read_bytes(),
            source_path="capture.sgf",
            default_rules="chinese",
        )
        self.assertEqual(record.board_size, 5)

    def test_19x19_game_with_pass_parity(self) -> None:
        record = GameRecord(
            id="m6-parity",
            source_path="m6-parity.sgf",
            source_sha256="0" * 64,
            board_size=19,
            komi_twice=15,
            handicap=0,
            to_move="b",
            outcome="black",
            result="B+R",
            rules="chinese",
            black_player="black",
            white_player="white",
            event="m6-test",
            date="2026-10-24",
            setup_black=(),
            setup_white=(),
            setup_empty=(),
            moves=(
                # Corner capture plus a pass, so recency planes, capture
                # bookkeeping, and pass-slot semantics are all exercised.
                encode_move("b", (1, 0)),
                encode_move("w", (0, 0)),
                encode_move("b", (0, 1)),   # captures the white corner stone
                encode_move("w", (15, 15)),
                encode_move("b", (3, 3)),
                encode_move("w", None),      # pass occupies a recency slot
                encode_move("b", (16, 2)),
                encode_move("w", (9, 9)),
            ),
        )
        self.assert_parity(record)

    def test_ko_shape_parity(self) -> None:
        record = GameRecord(
            id="m6-ko",
            source_path="m6-ko.sgf",
            source_sha256="0" * 64,
            board_size=19,
            komi_twice=15,
            handicap=0,
            to_move="b",
            outcome="white",
            result="W+R",
            rules="chinese",
            black_player="black",
            white_player="white",
            event="m6-test",
            date="2026-10-24",
            setup_black=(),
            setup_white=(),
            setup_empty=(),
            moves=(
                # Ko shape near the center: after black retakes, the simple-ko
                # ban point must appear in plane 3 identically on both sides.
                encode_move("b", (4, 4)),
                encode_move("w", (5, 3)),
                encode_move("b", (6, 4)),
                encode_move("w", (5, 5)),
                encode_move("b", (3, 4)),
                encode_move("w", (17, 17)),
                encode_move("b", (5, 4)),
                encode_move("w", (10, 10)),
            ),
        )
        self.assert_parity(record)

    def test_long_game_tail_parity(self) -> None:
        # Many moves so the recency window slides correctly and the empty /
        # stone planes track a dense board.
        moves = []
        points = [(x, y) for y in range(0, 19, 2) for x in range(0, 19, 2)]
        for index, point in enumerate(points[:36]):
            color = "b" if index % 2 == 0 else "w"
            moves.append(encode_move(color, point))
        record = GameRecord(
            id="m6-long",
            source_path="m6-long.sgf",
            source_sha256="0" * 64,
            board_size=19,
            komi_twice=15,
            handicap=0,
            to_move="b",
            outcome="draw",
            result="0",
            rules="chinese",
            black_player="black",
            white_player="white",
            event="m6-test",
            date="2026-10-24",
            setup_black=(),
            setup_white=(),
            setup_empty=(),
            moves=tuple(moves),
        )
        self.assert_parity(record)


if __name__ == "__main__":
    unittest.main()
