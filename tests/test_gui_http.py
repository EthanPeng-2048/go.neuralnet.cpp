#!/usr/bin/env python3
"""GUI acceptance: go_gui_cli served over HTTP.

Starts a real go_gui_cli and drives the same JSON API the browser page uses:

  * the embedded single-page frontend is served at /;
  * play mode: human move -> engine reply -> undo, illegal moves are refused;
  * Chinese rules end the game by two passes and score it (empty board with
    7.5 komi is a white win by 7.5);
  * the arena tab: step/auto-play/next/reset and colour alternation;
  * SGF export carries SZ/KM/RU[Chinese];
  * /api/evaluate returns a real distribution for a model engine and a
    synthetic "next move at 100%" preview for engines without one.

The model engine variant runs only when a model artifact is available.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import unittest
import urllib.error
import urllib.request
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


GUI_CLI = binary("go_gui_cli")

MODEL_CANDIDATES = [
    ROOT / "artifacts/m4/m4-c-validator.nnmodel",
    ROOT / ".cache/m7-b-smoke.nnmodel",
    ROOT / ".cache/m7-c-base.nnmodel",
]


def find_model() -> Path | None:
    if value := os.environ.get("GO_GUI_MODEL"):
        return Path(value)
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


class GuiServer:
    """A go_gui_cli child process with a tiny HTTP client."""

    def __init__(self, *extra: str):
        self.process = subprocess.Popen(
            [str(GUI_CLI), "--port", "0", *extra],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert self.process.stdout is not None
        banner = self.process.stdout.readline().strip()
        prefix = "go.gui listening on "
        if not banner.startswith(prefix):
            self.stop()
            raise RuntimeError(f"unexpected go_gui_cli banner: {banner!r}")
        self.base = banner[len(prefix):]

    def request(self, method: str, path: str, body=None):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8")

    def get_json(self, path: str):
        status, text = self.request("GET", path)
        return status, json.loads(text)

    def post_json(self, path: str, body=None):
        status, text = self.request("POST", path, body if body is not None else {})
        return status, json.loads(text)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.process.stdout is not None:
            self.process.stdout.close()


class GuiHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = GuiServer(
            "--engine", "random",
            "--black-engine", "random",
            "--white-engine", "random",
            "--speed", "50",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def enter_play(self, human_color: str = "black") -> None:
        """每个用例先切回对弈模式并开新局（unittest 按字母序共享一台服务器）。"""
        self.server.post_json("/api/mode", {"mode": "play"})
        self.server.post_json("/api/new", {"human_color": human_color, "komi": 7.5})

    def enter_arena(self) -> None:
        self.server.post_json("/api/mode", {"mode": "arena"})

    # -- 页面 ------------------------------------------------------------

    def test_index_page_served(self) -> None:
        status, text = self.server.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("对弈", text)
        self.assertIn("竞技台", text)

    def test_unknown_route_is_404(self) -> None:
        status, _ = self.server.request("GET", "/nope")
        self.assertEqual(status, 404)

    # -- 对弈 ------------------------------------------------------------

    def test_initial_state(self) -> None:
        self.enter_play()
        status, state = self.server.get_json("/api/state")
        self.assertEqual(status, 200)
        self.assertTrue(state["ok"])
        self.assertEqual(state["mode"], "play")
        self.assertEqual(state["size"], 19)
        self.assertEqual(state["komi"], 7.5)  # 中国规则贴 7.5 目
        self.assertEqual(state["to_move"], "black")
        self.assertEqual(len(state["board"]), 361)
        self.assertTrue(state["clickable"])
        self.assertFalse(state["play"]["engine_turn"])

    def test_play_engine_undo(self) -> None:
        self.enter_play()
        status, state = self.server.post_json("/api/play", {"x": 3, "y": 3})
        self.assertEqual(status, 200)
        self.assertEqual(state["move_number"], 1)
        self.assertEqual(state["board"][3 * 19 + 3], "B")
        self.assertTrue(state["play"]["engine_turn"])

        status, state = self.server.post_json("/api/engine-move")
        self.assertEqual(status, 200)
        self.assertEqual(state["move_number"], 2)
        self.assertFalse(state["play"]["engine_turn"])
        self.assertEqual(len(state["history"]), 2)

        # 引擎手不能由人类重复触发
        status, _ = self.server.post_json("/api/engine-move")
        self.assertEqual(status, 409)
        # 占用点与越界点都拒绝
        status, _ = self.server.post_json("/api/play", {"x": 3, "y": 3})
        self.assertEqual(status, 409)
        status, _ = self.server.post_json("/api/play", {"x": 19, "y": 0})
        self.assertEqual(status, 400)

        status, state = self.server.post_json("/api/undo")
        self.assertEqual(status, 200)
        self.assertEqual(state["move_number"], 0)
        self.assertEqual(state["to_move"], "black")

    def test_two_passes_score_chinese_rules(self) -> None:
        self.enter_play("none")
        self.server.post_json("/api/pass")
        status, state = self.server.post_json("/api/pass")
        self.assertEqual(status, 200)
        self.assertTrue(state["game_over"])
        self.assertEqual(state["result"]["reason"], "two-passes")
        self.assertEqual(state["result"]["winner"], "white")
        self.assertAlmostEqual(state["result"]["margin"], 7.5)

    def test_resign(self) -> None:
        self.enter_play()
        status, state = self.server.post_json("/api/resign")
        self.assertEqual(status, 200)
        self.assertTrue(state["game_over"])
        self.assertEqual(state["result"]["winner"], "white")
        self.assertEqual(state["result"]["reason"], "resign")

    def test_sgf_export(self) -> None:
        self.enter_play()
        self.server.post_json("/api/play", {"x": 3, "y": 3})
        status, text = self.server.request("GET", "/api/sgf")
        self.assertEqual(status, 200)
        self.assertIn("SZ[19]", text)
        self.assertIn("KM[7.5]", text)
        self.assertIn("RU[Chinese]", text)
        self.assertIn(";B[dd]", text)

    def test_evaluate_synthetic_for_random(self) -> None:
        # 不支持评估的引擎不再报错：给"将下点 100%"的合成预告（synthetic）。
        self.enter_play()
        status, body = self.server.get_json("/api/evaluate")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertTrue(body["synthetic"])
        self.assertEqual(len(body["top"]), 1)
        self.assertAlmostEqual(body["top"][0]["p"], 1.0)
        self.assertIsNone(body["value"])

    # -- 竞技台 ----------------------------------------------------------

    def test_arena_step_next_and_reset(self) -> None:
        self.enter_arena()
        status, state = self.server.get_json("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(state["mode"], "arena")
        self.assertEqual(state["arena"]["game_index"], 1)
        self.assertFalse(state["clickable"])

        status, state = self.server.post_json("/api/arena/step")
        self.assertEqual(status, 200)
        self.assertEqual(state["move_number"], 1)

        first_black = state["arena"]["black_name"]
        first_white = state["arena"]["white_name"]

        status, state = self.server.post_json("/api/arena/next")
        self.assertEqual(status, 200)
        self.assertEqual(state["arena"]["game_index"], 2)
        self.assertEqual(state["move_number"], 0)
        self.assertEqual(state["arena"]["black_name"], first_white)
        self.assertEqual(state["arena"]["white_name"], first_black)

        status, state = self.server.post_json("/api/arena/reset")
        self.assertEqual(status, 200)
        self.assertEqual(state["arena"]["game_index"], 1)
        self.assertEqual(state["arena"]["games_played"], 0)

    def test_arena_autoplay(self) -> None:
        self.enter_arena()
        self.server.post_json("/api/arena/reset")
        self.server.post_json("/api/config", {"speed_ms": 50, "auto_next": True})
        status, state = self.server.post_json("/api/arena/start")
        self.assertEqual(status, 200)
        self.assertTrue(state["arena"]["running"])
        time.sleep(1.5)
        self.server.post_json("/api/arena/pause")
        _, state = self.server.get_json("/api/state")
        self.assertFalse(state["arena"]["running"])
        self.assertGreater(state["move_number"], 2, "自动对局应持续落子")


class GuiDefaultOpponentTest(unittest.TestCase):
    """默认对手：装了 gnugo 就用 GnuGo，不再拿随机引擎当对手。"""

    def test_default_opponent_is_gnugo(self) -> None:
        if shutil.which("gnugo") is None:
            self.skipTest("gnugo is not installed")
        server = GuiServer("--speed", "50")
        try:
            _, state = server.get_json("/api/state")
            self.assertTrue(
                state["play"]["engine"].startswith("gtp:gnugo"),
                f"对弈默认对手应是 GnuGo，实际 {state['play']['engine']!r}",
            )
            server.post_json("/api/mode", {"mode": "arena"})
            _, state = server.get_json("/api/state")
            self.assertTrue(state["arena"]["white_engine"].startswith("gtp:gnugo"))
        finally:
            server.stop()


class GuiModelEngineTest(unittest.TestCase):
    """模型引擎：/api/evaluate 的候选着分布 + 价值头。"""

    def test_evaluate_with_model(self) -> None:
        model = find_model()
        if model is None:
            self.skipTest("no model artifact available")
        server = GuiServer("--model", str(model), "--speed", "50")
        try:
            status, body = server.get_json("/api/evaluate")
            self.assertEqual(status, 200)
            self.assertTrue(body["ok"])
            self.assertEqual(body["to_move"], "black")
            self.assertFalse(body["synthetic"], "模型引擎给真实分布")
            self.assertTrue(body["top"], "应给出候选着")
            total = sum(entry["p"] for entry in body["top"])
            self.assertLessEqual(total, 1.0 + 1e-3)
            self.assertGreater(total, 0.0)
            for entry in body["top"]:
                self.assertIn("p", entry)
                if "pass" not in entry:
                    self.assertTrue(0 <= entry["x"] < 19)
                    self.assertTrue(0 <= entry["y"] < 19)

            # 引擎先走（人类执白）也应成功
            server.post_json("/api/new", {"human_color": "white", "komi": 7.5})
            status, state = server.post_json("/api/engine-move")
            self.assertEqual(status, 200)
            self.assertEqual(state["move_number"], 1)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
