#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from go_pipeline.features import (
    BLACK,
    EMPTY,
    WHITE,
    Position,
    batch_columns,
    extract_features,
    sample_from_trace,
    save_feature_contact_sheet,
    trace_record,
)
from go_pipeline.model import decode_move, encode_move
from go_pipeline.sgf import parse_sgf
from go_pipeline.tokenizer import BOS, EOS, PAD, PASS, GoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_CLI = ROOT / "build/cpu/go_data_cli"
RULES_CLI = ROOT / "build/cpu/go_rules_cli"


class M2DataTests(unittest.TestCase):
    def test_sgf_record_and_move_codec(self) -> None:
        record = parse_sgf(
            (ROOT / "tests/data/capture.sgf").read_bytes(),
            source_path="capture.sgf",
            default_rules="chinese",
        )
        self.assertEqual(record.board_size, 5)
        self.assertEqual(record.komi_twice, 15)
        self.assertEqual(len(record.moves), 9)
        self.assertEqual(decode_move(record.moves[0]), ("b", (1, 1)))
        self.assertEqual(encode_move("b", (1, 1)), record.moves[0])
        self.assertEqual(encode_move("w", None), 723)

    def test_nngames_pack_stats_verify_export(self) -> None:
        record = parse_sgf(
            (ROOT / "tests/data/capture.sgf").read_bytes(),
            source_path="capture.sgf",
            default_rules="chinese",
        )
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            jsonl = work / "records.jsonl"
            dataset = work / "games.nngames"
            jsonl.write_text(json.dumps(record.to_json()) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    str(DATA_CLI),
                    "pack",
                    "--input",
                    str(jsonl),
                    "--output",
                    str(dataset),
                    "--source-name",
                    "unit-test",
                    "--created-at",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stats = json.loads(
                subprocess.run(
                    [str(DATA_CLI), "stats", "--input", str(dataset)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            self.assertEqual(stats["games"], 1)
            self.assertEqual(stats["moves"], 9)
            verified = json.loads(
                subprocess.run(
                    [
                        str(DATA_CLI),
                        "verify",
                        "--input",
                        str(dataset),
                        "--sample",
                        "1",
                        "--seed",
                        "7",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            self.assertTrue(verified["byte_identical"])
            self.assertEqual(verified["sampled_games"], 1)
            self.assertEqual(verified["replayed_moves"], 9)

            exported = json.loads(
                subprocess.run(
                    [
                        str(DATA_CLI),
                        "export",
                        "--input",
                        str(dataset),
                        "--index",
                        "0",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            self.assertEqual(exported["id"], record.id)
            self.assertEqual(exported["moves"], list(record.moves))


class M3FeatureTests(unittest.TestCase):
    def test_tokenizer_roundtrip_and_nnvocab(self) -> None:
        tokenizer = GoTokenizer()
        moves = [("b", (3, 3)), ("w", None), ("b", (15, 15))]
        tokens = tokenizer.encode_game(moves)
        self.assertEqual(tokens[0], BOS)
        self.assertEqual(tokens[-1], EOS)
        self.assertEqual(tokenizer.decode_game(tokens), moves)
        self.assertEqual(tokenizer.history(moves, 5), [PAD, PAD, tokenizer.point_id(3, 3), PASS, tokenizer.point_id(15, 15)])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "go.nnvocab"
            tokenizer.save(output, DATA_CLI)
            self.assertEqual(output.read_bytes()[:4], b"NNVC")

    def test_feature_planes_for_arbitrary_position(self) -> None:
        board = np.zeros((5, 5), dtype=np.uint8)
        board[1, 1] = BLACK
        board[2, 2] = WHITE
        board[0, 0] = EMPTY
        position = Position(
            board=board,
            to_move="w",
            ko=(4, 4),
            recent_moves=((1, 1), None, (2, 2)),
            move_number=12,
            komi=6.5,
            black_stones_captured=2,
            white_stones_captured=4,
        )
        sample = extract_features(position)
        self.assertEqual(sample.planes.shape, (9, 5, 5))
        self.assertEqual(sample.flatten_nn().shape, (225,))
        self.assertEqual(sample.planes[0, 2, 2], 1.0)
        self.assertEqual(sample.planes[1, 1, 1], 1.0)
        self.assertTrue(np.all(sample.planes[2] == 1.0))
        self.assertEqual(sample.planes[3, 4, 4], 1.0)
        self.assertEqual(sample.planes[4, 1, 1], 1.0)
        self.assertEqual(sample.planes[5].sum(), 0.0)
        self.assertEqual(sample.planes[6, 2, 2], 1.0)
        self.assertEqual(sample.planes[7, 0, 0], 1.0)
        self.assertEqual(sample.planes[8, 0, 0], 0.0)
        self.assertEqual(sample.planes[8, 2, 2], 1.0)
        self.assertEqual(batch_columns([sample, sample]).shape, (225, 2))

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "planes.png"
            save_feature_contact_sheet(sample, image)
            self.assertGreater(image.stat().st_size, 1000)

    def test_trace_and_feature_self_check(self) -> None:
        record = parse_sgf(
            (ROOT / "tests/data/capture.sgf").read_bytes(),
            source_path="capture.sgf",
            default_rules="chinese",
        )
        trace = trace_record(record, RULES_CLI)
        frames = trace["frames"]
        self.assertEqual(len(frames), len(record.moves) + 1)
        sample = sample_from_trace(trace, frame_index=len(frames) - 1, record=record)
        self.assertEqual(sample.planes.shape, (9, 5, 5))
        self.assertEqual(sample.planes[7, 1, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
