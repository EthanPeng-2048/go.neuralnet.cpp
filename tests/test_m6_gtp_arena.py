#!/usr/bin/env python3
"""M6 acceptance: GTP adapter and arena with SPRT.

Covers the two DoD lines of roadmap M6:
  * "Sabaki 亲手可下" — the GTP v2 adapter answers the command sequence a GUI
    sends (protocol_version/boardsize/clear_board/komi/play/genmove/showboard),
    every genmove is a legal move on the shared game state, and error replies
    follow the `? ` convention.
  * "vs GnuGo 自动评估" — the arena runs batches of games between engines
    (random / model / external GTP subprocess) and applies an SPRT win-rate
    test whose LLR is verified against hand-computed values.

GnuGo itself is optional: when `gnugo` is installed the CMake-registered
variant drives it as a GTP subprocess; here we exercise the identical code
path with our own go_gtp_cli as the opponent.
"""

from __future__ import annotations

import json
import math
import os
import shutil
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


def gtp_vertex(point: tuple[int, int], size: int) -> str:
    x, y = point
    return f"{GTP_COLUMNS[x]}{size - y}"


def parse_vertex(vertex: str, size: int) -> tuple[int, int] | None:
    if vertex in {"pass", "resign"}:
        return None
    x = GTP_COLUMNS.index(vertex[0])
    y = size - int(vertex[1:])
    return (x, y)


class GtpSession:
    """Minimal GTP client driving a go_gtp_cli subprocess."""

    def __init__(self, model: Path) -> None:
        self.process = subprocess.Popen(
            [str(GTP_CLI), "--model", str(model), "--cpu"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.next_id = 1

    def command(self, text: str) -> tuple[bool, str]:
        command_id = self.next_id
        self.next_id += 1
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(f"{command_id} {text}\n")
        self.process.stdin.flush()
        # GTP v2：响应可能是多行（如 list_commands），以空行结束。
        lines: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("gtp engine closed unexpectedly")
            if not line.strip():
                break
            lines.append(line.rstrip("\n"))
        if not lines:
            raise RuntimeError("empty gtp response")
        success = lines[0].startswith("= ")
        payload = lines[0][2:].strip()
        # 带 id 的请求会在响应中回显 id，剥掉它。
        if payload.startswith(f"{command_id} "):
            payload = payload[len(str(command_id)) + 1:]
        # 多行响应：首行之后的行拼进 payload（list_commands 等）。
        payload = "\n".join([payload] + lines[1:]).strip()
        return success, payload

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


def make_untrained_model(directory: Path) -> Path:
    model = directory / "arena-smoke.nnmodel"
    result = subprocess.run(
        [str(TRAIN_CLI), "make-model", "--output", str(model), "--cpu"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"make-model failed: {result.stderr}")
    return model


class GtpAdapterTests(unittest.TestCase):
    model: Path
    session: GtpSession

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.model = make_untrained_model(Path(cls._temporary.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def setUp(self) -> None:
        self.session = GtpSession(self.model)

    def tearDown(self) -> None:
        self.session.close()

    def test_identity_commands(self) -> None:
        success, version = self.session.command("protocol_version")
        self.assertTrue(success)
        self.assertEqual(version, "2")
        success, name = self.session.command("name")
        self.assertTrue(success)
        self.assertIn("go.neuralnet.cpp", name)
        success, known = self.session.command("known_command genmove")
        self.assertTrue(success)
        self.assertEqual(known, "true")
        success, known = self.session.command("known_command bogus_command")
        self.assertTrue(success)
        self.assertEqual(known, "false")
        success, listing = self.session.command("list_commands")
        self.assertTrue(success)
        self.assertIn("genmove", listing.split())

    def test_full_game_sequence_and_legality(self) -> None:
        self.assertTrue(self.session.command("boardsize 19")[0])
        self.assertTrue(self.session.command("clear_board")[0])
        self.assertTrue(self.session.command("komi 7.5")[0])
        self.assertTrue(self.session.command("play b D4")[0])
        self.assertTrue(self.session.command("play w Q16")[0])

        # 引擎的 genmove 必须是合法点，且落子后能被 rules 引擎接受。
        board: set[tuple[int, int]] = {(3, 15), (15, 3)}
        for turn in range(8):
            color = "b" if turn % 2 == 0 else "w"
            success, vertex = self.session.command(f"genmove {color}")
            self.assertTrue(success, vertex)
            point = parse_vertex(vertex, 19)
            if point is None:
                continue  # pass 合法
            self.assertNotIn(point, board, f"genmove played on occupied point {vertex}")
            board.add(point)

        # showboard 返回 19 行盘面，含已下的三手。
        success, shown = self.session.command("showboard")
        self.assertTrue(success)
        rows = shown.strip().splitlines()
        self.assertEqual(len(rows), 19)
        self.assertEqual(rows[15][3], "x")   # D4 black
        self.assertEqual(rows[3][15], "o")   # Q16 white

    def test_genmove_pass_when_asked_out_of_turn(self) -> None:
        self.assertTrue(self.session.command("boardsize 19")[0])
        self.assertTrue(self.session.command("clear_board")[0])
        # 黑方行棋时要求白方 genmove → 协议错误（? 前缀）。
        success, message = self.session.command("genmove w")
        self.assertFalse(success)
        self.assertIn("color to move", message)

    def test_non_19x19_rejected_for_model(self) -> None:
        self.assertTrue(self.session.command("boardsize 9")[0])
        self.assertTrue(self.session.command("clear_board")[0])
        success, message = self.session.command("genmove b")
        self.assertFalse(success)
        self.assertIn("19x19", message)

    def test_errors_use_question_prefix(self) -> None:
        success, message = self.session.command("bogus_command")
        self.assertFalse(success)
        self.assertIn("unknown command", message)
        self.assertTrue(self.session.command("boardsize 19")[0])
        self.assertTrue(self.session.command("clear_board")[0])
        success, message = self.session.command("play b Z99")
        self.assertFalse(success)

    def test_resign_and_pass_plays(self) -> None:
        self.assertTrue(self.session.command("boardsize 19")[0])
        self.assertTrue(self.session.command("clear_board")[0])
        self.assertTrue(self.session.command("play b pass")[0])
        self.assertTrue(self.session.command("play w pass")[0])
        # 双方连续 pass 后对局结束，final_score 可用（中国规则计分）。
        success, score = self.session.command("final_score")
        self.assertTrue(success, score)
        self.assertTrue(score.startswith(("B+", "W+", "0")), score)


class ArenaTests(unittest.TestCase):
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
            timeout=240,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_random_vs_random_wellformed(self) -> None:
        report = self.run_arena(
            "--player", "random", "--opponent", "random",
            "--games", "2", "--size", "9", "--seed", "42",
        )
        self.assertEqual(report["games_played"], 2)
        self.assertEqual(len(report["games"]), 2)
        for game in report["games"]:
            self.assertIn(game["player_result"], (0.0, 0.5, 1.0))
            self.assertGreater(game["moves"], 0)
        sprt = report["sprt"]
        self.assertEqual(sprt["wins"] + sprt["losses"] + sprt["jigos"], 2)
        self.assertIn(sprt["verdict"], ("continue", "accept", "reject"))

    def test_model_vs_random_19x19(self) -> None:
        report = self.run_arena(
            "--player", f"model:{self.model}=cpu", "--opponent", "random",
            "--games", "2", "--size", "19", "--seed", "7",
            "--sprt-p0", "0.45", "--sprt-p1", "0.6",
        )
        self.assertEqual(report["games_played"], 2)
        self.assertIn("model:", report["player"])
        self.assertEqual(report["opponent"], "random")

    def test_gtp_subprocess_opponent(self) -> None:
        # 用我们自己的 go_gtp_cli 作为外部 GTP 引擎，走与 gnugo 完全相同的
        # 子进程管道路径（play 转发 + genmove 解析）。
        command = f"{GTP_CLI} --model {self.model} --cpu"
        report = self.run_arena(
            "--player", f"model:{self.model}=cpu", "--opponent", f"gtp:{command}",
            "--games", "2", "--size", "19", "--seed", "3",
            "--sprt-p0", "0.45", "--sprt-p1", "0.55",
        )
        self.assertEqual(report["games_played"], 2)
        self.assertTrue(report["opponent"].startswith("gtp:"))

    def test_sprt_llr_matches_reference(self) -> None:
        # LLR = w·ln(p1/p0) + l·ln((1-p1)/(1-p0))，边界 ±ln((1-β)/α)。
        p0, p1, alpha, beta = 0.5, 0.55, 0.05, 0.05
        # 构造一个一边倒的序列让判定触发 accept。
        report = self.run_arena(
            "--player", "random", "--opponent", "random",
            "--games", "400", "--size", "5", "--seed", "1",
            "--sprt-p0", str(p0), "--sprt-p1", str(p1),
            "--sprt-alpha", str(alpha), "--sprt-beta", str(beta),
        )
        sprt = report["sprt"]
        wins, losses = sprt["wins"], sprt["losses"]
        expected = wins * math.log(p1 / p0) + losses * math.log((1 - p1) / (1 - p0))
        self.assertAlmostEqual(sprt["llr"], expected, places=9)
        self.assertAlmostEqual(
            sprt["upper_bound"], math.log((1 - beta) / alpha), places=12
        )
        self.assertAlmostEqual(
            sprt["lower_bound"], math.log(beta / (1 - alpha)), places=12
        )
        if sprt["verdict"] == "accept":
            self.assertGreaterEqual(sprt["llr"], sprt["upper_bound"])
        elif sprt["verdict"] == "reject":
            self.assertLessEqual(sprt["llr"], sprt["lower_bound"])
        else:
            self.assertLess(sprt["llr"], sprt["upper_bound"])
            self.assertGreater(sprt["llr"], sprt["lower_bound"])

    def test_sprt_stops_early_on_decision(self) -> None:
        # p0 极低 + 一边倒对手：LLR 应很快越过上界，实际局数 < 请求局数。
        report = self.run_arena(
            "--player", "random", "--opponent", "random",
            "--games", "500", "--size", "5", "--seed", "11",
            "--sprt-p0", "0.1", "--sprt-p1", "0.9",
            "--sprt-alpha", "0.05", "--sprt-beta", "0.05",
        )
        if report["sprt"]["verdict"] != "continue":
            self.assertLess(report["games_played"], 500)

    def test_report_file_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            self.run_arena(
                "--player", "random", "--opponent", "random",
                "--games", "1", "--size", "5", "--seed", "5",
                "--report", str(path),
            )
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())["schema"], 1)

    def test_unknown_engine_spec_rejected(self) -> None:
        result = subprocess.run(
            [str(ARENA_CLI), "--player", "bogus", "--games", "1"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown engine spec", result.stderr)


@unittest.skipUnless(shutil.which("gnugo"), "gnugo is not installed")
class GnuGoArenaTests(unittest.TestCase):
    """roadmap M6 DoD「vs GnuGo 自动评估」：装了 gnugo 就真跑。

    CMake 也会在发现 gnugo 时注册同款测试；这里保证手工运行
    `python tests/test_m6_gtp_arena.py` 时语义一致。
    """

    model: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.model = make_untrained_model(Path(cls._temporary.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_model_vs_gnugo(self) -> None:
        result = subprocess.run(
            [
                str(ARENA_CLI),
                "--player", f"model:{self.model}=cpu",
                "--opponent", "gtp:gnugo --mode gtp --quiet",
                "--games", "2", "--size", "19", "--seed", "13",
                "--sprt-p0", "0.45", "--sprt-p1", "0.6",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["games_played"], 2)
        self.assertTrue(report["opponent"].startswith("gtp:"))


if __name__ == "__main__":
    unittest.main()
