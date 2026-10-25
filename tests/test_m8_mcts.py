#!/usr/bin/env python3
"""M8 integration: the MCTS engine behind the GTP and arena front ends.

The tree engine itself is covered by tests/go_mcts_tests.cpp (mock evaluator,
no model). This file exercises the two product paths that wrap it:
  * go_gtp_cli --visits N — the Sabaki-facing GTP adapter selects moves with
    MCTS (default stays raw policy argmax, so M6 semantics are unchanged);
  * go_arena_cli engine spec mcts:VISITS:model:FILE[=DEVICE] — batch games
    through the shared engine abstraction, plus spec validation.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

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


TRAIN_CLI = binary("go_train_cli")
GTP_CLI = binary("go_gtp_cli")
ARENA_CLI = binary("go_arena_cli")

GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"


def parse_vertex(vertex: str, size: int) -> tuple[int, int] | None:
    if vertex in {"pass", "resign"}:
        return None
    x = GTP_COLUMNS.index(vertex[0])
    y = size - int(vertex[1:])
    return (x, y)


def make_untrained_model(directory: Path) -> Path:
    model = directory / "m8-smoke.nnmodel"
    result = subprocess.run(
        [str(TRAIN_CLI), "make-model", "--output", str(model), "--cpu"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"make-model failed: {result.stderr}")
    return model


class GtpSession:
    """Minimal GTP client driving a go_gtp_cli subprocess."""

    def __init__(self, model: Path, visits: int) -> None:
        self.process = subprocess.Popen(
            [str(GTP_CLI), "--model", str(model), "--cpu", "--visits", str(visits)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def command(self, command: str) -> str:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        lines = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError("gtp engine closed")
            line = line.rstrip("\n")
            if line == "" and lines:
                break
            if line != "":
                lines.append(line)
        first = lines[0]
        if not first.startswith(("=", "?")):
            raise RuntimeError(f"bad gtp response: {first}")
        if first.startswith("?"):
            raise RuntimeError(f"gtp rejected '{command}': {first}")
        payload = first[1:].strip()
        return "\n".join([payload] + lines[1:]).strip()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.command("quit")
            except RuntimeError:
                pass
            self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


class GtpMctsTests(unittest.TestCase):
    model: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.model = make_untrained_model(Path(cls._temporary.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_mcts_genmove_is_legal(self) -> None:
        session = GtpSession(self.model, visits=4)
        try:
            self.assertEqual(session.command("protocol_version"), "2")
            self.assertEqual(session.command("boardsize 19"), "")
            self.assertEqual(session.command("clear_board"), "")
            self.assertEqual(session.command("komi 7.5"), "")

            first = parse_vertex(session.command("genmove b"), 19)
            # 未训练模型：点位或 pass 都合法，但必须是盘内着。
            if first is not None:
                x, y = first
                self.assertTrue(0 <= x < 19 and 0 <= y < 19)
                human = (0, 0) if first != (0, 0) else (18, 18)
            else:
                human = (0, 0)
            self.assertEqual(
                session.command(f"play w {GTP_COLUMNS[human[0]]}{19 - human[1]}"), ""
            )
            second_point = parse_vertex(session.command("genmove b"), 19)
            if second_point is not None:
                x, y = second_point
                self.assertTrue(0 <= x < 19 and 0 <= y < 19)
        finally:
            session.close()

    def test_visits_zero_is_pure_prior(self) -> None:
        # --visits 0 = 纯先验 argmax（与缺省一致）；非数字才拒绝。
        session = GtpSession(self.model, visits=0)
        try:
            session.command("boardsize 19")
            session.command("clear_board")
            point = parse_vertex(session.command("genmove b"), 19)
            if point is not None:
                x, y = point
                self.assertTrue(0 <= x < 19 and 0 <= y < 19)
        finally:
            session.close()

        result = subprocess.run(
            [str(GTP_CLI), "--model", str(self.model), "--cpu", "--visits", "abc"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--visits", result.stderr)


class ArenaMctsTests(unittest.TestCase):
    model: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.model = make_untrained_model(Path(cls._temporary.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def run_arena(self, *arguments: str) -> dict[str, object]:
        result = subprocess.run(
            [str(ARENA_CLI), *arguments],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_mcts_vs_model_wellformed(self) -> None:
        report = self.run_arena(
            "--player", f"mcts:4:model:{self.model}=cpu",
            "--opponent", f"model:{self.model}=cpu",
            "--games", "1", "--size", "19", "--seed", "5",
            "--sprt-p0", "0.45", "--sprt-p1", "0.6",
        )
        self.assertEqual(report["games_played"], 1)
        self.assertTrue(str(report["player"]).startswith("mcts:"))
        self.assertGreater(report["games"][0]["moves"], 0)

    def test_bad_mcts_specs_rejected(self) -> None:
        for spec in (
            "mcts:0:model:x.nnmodel",
            "mcts:abc:model:x.nnmodel",
            "mcts:4:gtp:gnugo --mode gtp",
            "mcts:4",
        ):
            result = subprocess.run(
                [str(ARENA_CLI), "--player", spec, "--games", "1"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0, spec)
            self.assertIn("mcts", result.stderr, spec)


if __name__ == "__main__":
    unittest.main()
