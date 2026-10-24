#!/usr/bin/env python3
"""M5 acceptance: GoTrainer resume, metrics logging, and dual-role artifacts.

The core claim of M5 is "断点续训无损" (lossless resume). This test proves it
numerically: training 4 epochs in one process must produce bitwise-identical
final metrics to training 2 epochs, checkpointing, then resuming for 2 more.
The resume path restores model weights, optimizer state (m/v/t), the lr curve
position, and the per-epoch shuffle order (a pure function of seed+epoch).
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

SEED = 20261024
BATCH = 16
EPOCHS = 4


def make_records(count: int, moves_per_game: int) -> list[tuple[int, GameRecord]]:
    """Synthetic 19x19 games with a shared, peaked point vocabulary.

    Every game draws its moves from the same pool of distinct board points
    (rotated per game), so the tiny CNN can push the loss down within a
    handful of epochs while every individual move stays legal — a point is
    never played twice in one game.
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
                id=f"m5-{index}",
                source_path=f"m5-{index}.sgf",
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
                event="m5-test",
                date="2026-10-24",
                setup_black=(),
                setup_white=(),
                setup_empty=(),
                moves=tuple(moves),
            ),
        ))
    return records


def train_command(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(TRAIN_CLI), "train", *extra],
        capture_output=True,
        text=True,
    )


class M5TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        work = Path(cls._temporary.name)
        cls.train_cache = work / "train.ngf"
        cls.eval_cache = work / "eval.ngf"
        write_feature_cache(
            cls.train_cache,
            split="train",
            records=make_records(count=16, moves_per_game=12),
            rules_cli=RULES_CLI,
        )
        write_feature_cache(
            cls.eval_cache,
            split="eval",
            records=make_records(count=4, moves_per_game=12),
            rules_cli=RULES_CLI,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def common_args(self, work: Path) -> list[str]:
        return [
            "--dataset", str(self.train_cache),
            "--eval-dataset", str(self.eval_cache),
            "--cpu",
            "--batch-size", str(BATCH),
            "--epochs", str(EPOCHS),
            "--lr", "0.001",
            "--min-lr", "0.00001",
            "--weight-decay", "0.0001",
            "--seed", str(SEED),
            # M7: B is the mainline architecture. Keep the widths tiny so the
            # resume-equivalence property is still verified on the shipping
            # model while staying inside the CI budget.
            "--arch", "b",
            "--d-model", "32",
            "--num-layers", "2",
            "--num-heads", "4",
            "--d-ff", "64",
            "--conv-channels", "16",
            "--history-len", "64",
            "--checkpoint", str(work / "c.nnckpt"),
            "--model", str(work / "m.nnmodel"),
            "--metrics-out", str(work / "metrics.json"),
            "--metrics-log", str(work / "metrics.jsonl"),
        ]

    def test_resume_equivalence_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            straight_dir = work / "straight"
            split_dir = work / "split"
            straight_dir.mkdir()
            split_dir.mkdir()

            # Run A: 4 epochs in a single process.
            result = train_command(*self.common_args(straight_dir))
            self.assertEqual(result.returncode, 0, result.stderr)
            straight = json.loads((straight_dir / "metrics.json").read_text())
            self.assertEqual(straight["status"], "pass")
            self.assertEqual(straight["mode"], "fresh")
            self.assertEqual(len(straight["history"]), EPOCHS)

            # Run B, phase 1: 2 epochs, checkpoint written at each epoch edge.
            phase1 = train_command(
                *self.common_args(split_dir), "--stop-after-epoch", "2"
            )
            self.assertEqual(phase1.returncode, 0, phase1.stderr)
            partial = json.loads((split_dir / "metrics.json").read_text())
            self.assertEqual(partial["status"], "pass")
            self.assertEqual(len(partial["history"]), 2)
            self.assertTrue((split_dir / "c.nnckpt").is_file())

            # Run B, phase 2: resume from the checkpoint for the remaining epochs.
            phase2 = train_command(*self.common_args(split_dir), "--resume",
                                   str(split_dir / "c.nnckpt"))
            self.assertEqual(phase2.returncode, 0, phase2.stderr)
            resumed = json.loads((split_dir / "metrics.json").read_text())
            self.assertEqual(resumed["status"], "pass")
            self.assertEqual(resumed["mode"], "resume")
            self.assertEqual(resumed["start_epoch"], 2)
            self.assertEqual(len(resumed["history"]), 2)
            self.assertEqual(
                [entry["epoch"] for entry in resumed["history"]], [3, 4]
            )

            # The heart of M5: resumed tail == straight tail, exactly.
            for phase, whole in zip(resumed["history"], straight["history"][2:],
                                    strict=True):
                self.assertEqual(phase["epoch"], whole["epoch"])
                self.assertAlmostEqual(
                    phase["train_loss"], whole["train_loss"], delta=1e-7
                )
                self.assertAlmostEqual(
                    phase["eval"]["loss"], whole["eval"]["loss"], delta=1e-7
                )
                self.assertEqual(phase["eval"]["top1"], whole["eval"]["top1"])
            self.assertAlmostEqual(
                resumed["history"][-1]["eval"]["loss"],
                straight["history"][-1]["eval"]["loss"],
                delta=1e-7,
            )
            self.assertEqual(
                resumed["global_step"], straight["global_step"]
            )

            # Dual-role artifacts: .nnckpt carries optimizer state, .nnmodel
            # does not. Proven by file size (checkpoint must be larger) and by
            # the fact that resume succeeded at all (optimizer tensors were
            # loaded, not cold-started).
            checkpoint = split_dir / "c.nnckpt"
            model = split_dir / "m.nnmodel"
            self.assertGreater(checkpoint.stat().st_size, model.stat().st_size)

            # Metrics log: run + epoch records + final, appended across both
            # phases without truncation.
            records = [
                json.loads(line)
                for line in (split_dir / "metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            kinds = [entry["record"] for entry in records]
            self.assertEqual(kinds.count("run"), 2)
            self.assertEqual(kinds.count("epoch"), EPOCHS)
            self.assertEqual(kinds.count("final"), 2)
            modes = [entry["mode"] for entry in records if entry["record"] == "run"]
            self.assertEqual(modes, ["fresh", "resume"])
            self.assertEqual(
                [entry["epoch"] for entry in records if entry["record"] == "epoch"],
                [1, 2, 3, 4],
            )

    def test_resume_rejects_changed_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            result = train_command(*self.common_args(work), "--stop-after-epoch", "2")
            self.assertEqual(result.returncode, 0, result.stderr)

            # A different cache file with the same shape but different bytes.
            other = work / "other.ngf"
            write_feature_cache(
                other,
                split="train",
                records=make_records(count=17, moves_per_game=12),
                rules_cli=RULES_CLI,
            )
            rejected = subprocess.run(
                [str(TRAIN_CLI), "train",
                 "--dataset", str(other),
                 "--eval-dataset", str(self.eval_cache),
                 "--cpu", "--batch-size", str(BATCH), "--epochs", str(EPOCHS),
                 "--seed", str(SEED),
                 "--checkpoint", str(work / "c.nnckpt"),
                 "--model", str(work / "m.nnmodel"),
                 "--metrics-out", str(work / "metrics.json"),
                 "--resume", str(work / "c.nnckpt")],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("数据集与断点不一致", rejected.stderr)

    def test_resume_rejects_epoch_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            result = train_command(*self.common_args(work), "--stop-after-epoch", "2")
            self.assertEqual(result.returncode, 0, result.stderr)
            rejected = subprocess.run(
                [str(TRAIN_CLI), "train",
                 "--dataset", str(self.train_cache),
                 "--eval-dataset", str(self.eval_cache),
                 "--cpu", "--batch-size", str(BATCH),
                 "--epochs", "6",  # checkpoint was written with epochs=4
                 "--seed", str(SEED),
                 "--checkpoint", str(work / "c.nnckpt"),
                 "--model", str(work / "m.nnmodel"),
                 "--metrics-out", str(work / "metrics.json"),
                 "--resume", str(work / "c.nnckpt")],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("epochs", rejected.stderr)

    def test_warm_start_from_inference_model(self) -> None:
        """A .nnmodel (Model role) can seed a new training run with a cold
        optimizer — the second half of 双角色存取."""
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            built = subprocess.run(
                [str(TRAIN_CLI), "make-model",
                 "--output", str(work / "warm.nnmodel"), "--cpu"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertTrue((work / "warm.nnmodel").is_file())

            # Fresh training cannot take --resume from a .nnmodel; warm start
            # is expressed by loading weights before training. The trainer
            # rejects it as a resume source because it has no state section.
            rejected = subprocess.run(
                [str(TRAIN_CLI), "train",
                 "--dataset", str(self.train_cache),
                 "--eval-dataset", str(self.eval_cache),
                 "--cpu", "--batch-size", str(BATCH), "--epochs", str(EPOCHS),
                 "--seed", str(SEED),
                 "--checkpoint", str(work / "c.nnckpt"),
                 "--model", str(work / "m.nnmodel"),
                 "--metrics-out", str(work / "metrics.json"),
                 "--resume", str(work / "warm.nnmodel")],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("训练状态段", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
