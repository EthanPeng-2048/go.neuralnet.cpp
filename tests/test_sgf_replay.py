#!/usr/bin/env python3
"""Parse an SGF with sgfmill, replay it through KataGo, and verify the board."""

from __future__ import annotations

import argparse
import json
import subprocess
import unittest
from pathlib import Path

try:
    from sgfmill import sgf
except ImportError:  # pragma: no cover - exercised when dev extras are absent
    sgf = None


DATA = Path(__file__).parent / "data"


def cli_move(color: str, point: tuple[int, int] | None, size: int) -> str:
    if point is None:
        return f"{color}:pass"
    row, column = point
    # sgfmill uses row 0 at the bottom; the C++ adapter uses row 0 at the top.
    top_row = size - 1 - row
    return f"{color}:{column},{top_row}"


class SgfReplayTest(unittest.TestCase):
    cli: Path

    @classmethod
    def setUpClass(cls) -> None:
        if sgf is None:
            raise unittest.SkipTest("install requirements-dev.txt to run the SGF test")

    def replay(self, path: Path) -> dict:
        game = sgf.Sgf_game.from_bytes(path.read_bytes())
        root = game.get_root()
        size = root.get_size()
        komi = root.get("KM")
        rules = (root.get("RU") or "Chinese").lower()
        moves: list[str] = []
        for node in game.get_main_sequence():
            color, point = node.get_move()
            if color is not None:
                moves.append(cli_move(color, point, size))

        command = [
            str(self.cli),
            "replay",
            "--size",
            str(size),
            "--rules",
            "japanese" if "japanese" in rules else "chinese",
        ]
        if komi is not None:
            command.extend(["--komi", str(float(komi))])
        command.extend(moves)
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    def test_capture_sgf_replay(self) -> None:
        replay = self.replay(DATA / "capture.sgf")
        self.assertEqual(replay["width"], 5)
        self.assertEqual(replay["height"], 5)
        self.assertEqual(replay["moves"], 9)
        self.assertEqual(
            replay["board"],
            [
                "o.x..",
                ".x.x.",
                ".xx..",
                ".....",
                "o...o",
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cli", type=Path, required=True)
    known, unittest_args = parser.parse_known_args()
    SgfReplayTest.cli = known.cli.resolve()
    unittest.main(argv=[__file__, *unittest_args])


if __name__ == "__main__":
    main()
