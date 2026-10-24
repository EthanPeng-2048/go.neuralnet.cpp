#!/usr/bin/env python3
"""Compare selected legality decisions with GnuGo when it is installed."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import unittest
from pathlib import Path


def gtp_point(point: tuple[int, int], size: int) -> str:
    x, y = point
    return f"{chr(ord('A') + x)}{size - y}"


def gtp_moves(moves: list[tuple[str, tuple[int, int] | None]], size: int) -> list[str]:
    result = ["boardsize " + str(size), "clear_board"]
    for color, point in moves:
        target = "pass" if point is None else gtp_point(point, size)
        result.append(f"play {color.upper()} {target}")
    return result


class GnuGoCrossCheck(unittest.TestCase):
    cli: Path
    gnugo: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.gnugo = shutil.which("gnugo") or ""
        if not cls.gnugo:
            raise unittest.SkipTest("gnugo is not installed")

    def gnugo_results(self, commands: list[str]) -> list[bool]:
        result = subprocess.run(
            [self.gnugo, "--mode", "gtp", "--quiet"],
            input="\n".join(commands) + "\n",
            check=True,
            text=True,
            capture_output=True,
        )
        lines = [
            line for line in result.stdout.splitlines()
            if line.startswith(("= ", "? "))
        ]
        self.assertEqual(len(lines), len(commands), "unexpected GnuGo response count")
        return [line.startswith("= ") for line in lines]

    def ours(self, moves: list[tuple[str, tuple[int, int] | None]], size: int) -> bool:
        command = [str(self.cli), "replay", "--size", str(size), "--rules", "japanese"]
        command.extend(
            f"{color}:" + ("pass" if point is None else f"{point[0]},{point[1]}")
            for color, point in moves
        )
        return subprocess.run(command, text=True, capture_output=True).returncode == 0

    def test_capture_and_suicide(self) -> None:
        capture = [
            ("b", (1, 1)),
            ("w", (2, 1)),
            ("b", (1, 2)),
            ("w", (0, 0)),
            ("b", (3, 1)),
            ("w", (4, 4)),
            ("b", (2, 2)),
            ("w", (0, 4)),
            ("b", (2, 0)),
        ]
        self.assertTrue(all(self.gnugo_results(gtp_moves(capture, 5))))
        self.assertTrue(self.ours(capture, 5))

        ring = [
            (1, 1), (2, 1), (3, 1), (1, 2),
            (3, 2), (1, 3), (2, 3), (3, 3),
        ]
        fillers = [(0, 0), (4, 4), (0, 4), (4, 0), (0, 2), (4, 2), (2, 0), (2, 4)]
        alternating: list[tuple[str, tuple[int, int] | None]] = []
        for index, point in enumerate(ring):
            alternating.append(("b", point))
            alternating.append(("w", fillers[index]))
        suicide = [*alternating, ("w", (2, 2))]

        gnugo = self.gnugo_results(gtp_moves(suicide, 5))
        self.assertTrue(all(gnugo[:-1]))
        self.assertFalse(gnugo[-1])
        self.assertFalse(self.ours(suicide, 5))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cli", type=Path, required=True)
    known, unittest_args = parser.parse_known_args()
    GnuGoCrossCheck.cli = known.cli.resolve()
    unittest.main(argv=[__file__, *unittest_args])


if __name__ == "__main__":
    main()
