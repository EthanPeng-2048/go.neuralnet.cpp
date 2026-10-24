#!/usr/bin/env python3
"""M7 acceptance: the B model (dual-branch encoder + dual head) on real caches.

The roadmap M7 DoD is "同数据下 B 显著优于 C". This test does not attempt the
"significantly better" claim — that needs a full CWI training run and lives in
docs/benchmarks/m7-b-vs-c.md. What it pins down here is every *contract* the B
path depends on, because each of these fails silently rather than loudly:

  * B reports the value head's metrics (value_auc), which C cannot produce,
    and the value targets are not degenerate (a cache written from the wrong
    side-to-move perspective trains on a constant label and only shows up as
    "value AUC = 0");
  * B resume reproduces the uninterrupted tail exactly;
  * an untrained B model still answers GTP genmove with a legal move, proving
    the legality mask is applied on the host side regardless of model output.

The v2 move-history contract itself lives in tests/test_feature_cache_history.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from go_pipeline.model import GameRecord, encode_move
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
GTP_CLI = binary("go_gtp_cli")

SEED = 20261024
BATCH = 16

# A small B config so the CPU tests finish quickly. The architecture is the
# real one (25 region tokens + K history + query → 363 rows); only the widths
# shrink, which the spec round-trip must preserve exactly.
TINY_B = {
    "--d-model": "32",
    "--num-layers": "2",
    "--num-heads": "4",
    "--d-ff": "64",
    "--conv-channels": "16",
    "--history-len": "8",
}
TINY_K = 8


def make_records(count: int, moves_per_game: int) -> list[tuple[int, GameRecord]]:
    """Synthetic 19x19 games with a shared, peaked point vocabulary.

    Outcomes alternate black/white so the value targets contain both +1 and -1
    (a single-sign cache would make value_auc meaningless).
    """
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
                id=f"m7-{index}",
                source_path=f"m7-{index}.sgf",
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
                event="m7-test",
                date="2026-10-24",
                setup_black=(),
                setup_white=(),
                setup_empty=(),
                moves=tuple(moves),
            ),
        ))
    return records


class M7BModelTests(unittest.TestCase):
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

    def common_args(self, work: Path) -> list[str]:
        args = [
            "--dataset", str(self.train_cache),
            "--eval-dataset", str(self.eval_cache),
            "--cpu",
            "--batch-size", str(BATCH),
            "--epochs", "2",
            "--lr", "0.001",
            "--min-lr", "0.00001",
            "--weight-decay", "0.0001",
            "--seed", str(SEED),
            "--arch", "b",
            "--checkpoint", str(work / "c.nnckpt"),
            "--model", str(work / "m.nnmodel"),
            "--metrics-out", str(work / "metrics.json"),
            "--metrics-log", str(work / "metrics.jsonl"),
        ]
        args += [arg for pair in TINY_B.items() for arg in pair]
        return args

    def train(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(TRAIN_CLI), "train", *extra], capture_output=True, text=True
        )

    def test_b_train_reports_value_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            result = self.train(*self.common_args(work))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((work / "metrics.json").read_text())
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["arch"], "b")
            self.assertEqual(report["history_len"], TINY_K)
            final = report["history"][-1]["eval"]
            # C cannot produce these two: the value head is B-only.
            self.assertIn("value_auc", final)
            self.assertIn("top3", final)
            # Every eval sample is decisive here (win/loss), so the AUC must be
            # computed over real pairs. A degenerate target set (all one sign)
            # would leave value_auc at 0 and hide the bug.
            self.assertGreaterEqual(final["value_samples"], 1)
            self.assertGreater(final["value_auc"], 0.0)
            self.assertLessEqual(final["value_auc"], 1.0)
            self.assertGreaterEqual(final["top3"], final["top1"])
            self.assertTrue(report["health"]["checkpoint_roundtrip"])

    def test_b_resume_equivalence(self) -> None:
        """Resuming a B run must reproduce the uninterrupted tail, exactly as
        M5 proved for C — the dual-head loss goes through the same optimizer
        state path, so the property must hold for B too."""
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            straight_dir = work / "straight"
            split_dir = work / "split"
            straight_dir.mkdir()
            split_dir.mkdir()

            result = self.train(*self.common_args(straight_dir))
            self.assertEqual(result.returncode, 0, result.stderr)
            straight = json.loads((straight_dir / "metrics.json").read_text())
            self.assertEqual(straight["status"], "pass")

            phase1 = self.train(
                *self.common_args(split_dir), "--stop-after-epoch", "1"
            )
            self.assertEqual(phase1.returncode, 0, phase1.stderr)
            phase2 = self.train(
                *self.common_args(split_dir), "--resume",
                str(split_dir / "c.nnckpt")
            )
            self.assertEqual(phase2.returncode, 0, phase2.stderr)
            resumed = json.loads((split_dir / "metrics.json").read_text())
            self.assertEqual(resumed["mode"], "resume")
            self.assertEqual(resumed["start_epoch"], 1)

            whole = straight["history"][1]
            tail = resumed["history"][-1]
            self.assertAlmostEqual(
                tail["train_loss"], whole["train_loss"], delta=1e-6
            )
            self.assertAlmostEqual(
                tail["eval"]["loss"], whole["eval"]["loss"], delta=1e-6
            )
            self.assertEqual(tail["eval"]["top1"], whole["eval"]["top1"])
            self.assertEqual(resumed["global_step"], straight["global_step"])

    def test_b_model_answers_gtp_genmove(self) -> None:
        """An *untrained* B model must still produce legal genmove replies:
        legality comes from KataGo on the host, never from the network."""
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            model = work / "b.nnmodel"
            built = subprocess.run(
                [str(TRAIN_CLI), "make-model", "--output", str(model),
                 "--arch", "b", "--cpu",
                 *[arg for pair in TINY_B.items() for arg in pair]],
                capture_output=True, text=True,
            )
            self.assertEqual(built.returncode, 0, built.stderr)

            process = subprocess.Popen(
                [str(GTP_CLI), "--model", str(model), "--cpu"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
            )
            assert process.stdin is not None
            assert process.stdout is not None

            def respond(command: str) -> list[str]:
                """Send one GTP command and read its response up to the blank
                line that terminates every answer (GTP v2 framing)."""
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(f"1 {command}\n")
                process.stdin.flush()
                lines: list[str] = []
                while True:
                    line = process.stdout.readline()
                    if not line.strip():
                        break
                    lines.append(line.strip())
                return lines

            try:
                for command in ("protocol_version", "boardsize 19", "clear_board",
                                "play b D4", "play w Q16"):
                    answer = respond(command)
                    self.assertTrue(answer, command)
                    self.assertTrue(answer[0].startswith("= "), answer)

                lines = respond("genmove b")
                self.assertTrue(lines, "genmove produced no response")
                # GTP echoes the command id: "= 1 M6" for our id-1 commands.
                fields = lines[0].split()
                self.assertEqual(fields[0], "=", lines)
                payload = fields[2:] if len(fields) > 2 else []
                reply = payload[0] if payload else ""
                self.assertTrue(reply, lines)
                self.assertNotIn("?", reply)
                # Legal vertex: pass, or an empty point not occupied by D4/Q16.
                if reply.lower() not in {"pass", "resign"}:
                    columns = "ABCDEFGHJKLMNOPQRST"
                    x = columns.index(reply[0])
                    y = 19 - int(reply[1:])
                    self.assertTrue(0 <= x < 19 and 0 <= y < 19, reply)
                    self.assertNotIn((x, y), {(3, 15), (15, 2)}, reply)
            finally:
                process.stdin.close()
                process.terminate()
                process.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
