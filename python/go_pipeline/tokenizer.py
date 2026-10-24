from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .model import BOARD_SIZE, PASS_POINT, decode_move, encode_move

POINT_COUNT = BOARD_SIZE * BOARD_SIZE
PASS = 361
BOS = 362
EOS = 363
DECISION = 364
PAD = 365
UNK = 366
VOCAB_SIZE = 367


@dataclass(frozen=True, slots=True)
class GoTokenizer:
    max_board_size: int = BOARD_SIZE

    def point_id(self, x: int, y: int, size: int = BOARD_SIZE) -> int:
        if not (0 <= x < size and 0 <= y < size and size <= self.max_board_size):
            raise ValueError(f"point ({x},{y}) outside {size}x{size} board")
        return y * self.max_board_size + x

    def point_from_id(self, token: int, size: int = BOARD_SIZE) -> tuple[int, int]:
        if not 0 <= token < POINT_COUNT:
            raise ValueError(f"token is not a point: {token}")
        x = token % self.max_board_size
        y = token // self.max_board_size
        if x >= size or y >= size:
            raise ValueError(f"point token outside {size}x{size} board: {token}")
        return x, y

    def encode_move(self, color: str, point: tuple[int, int] | None) -> int:
        if point is None:
            return PASS
        return self.point_id(point[0], point[1])

    def encode_game(self, moves: list[tuple[str, tuple[int, int] | None]]) -> list[int]:
        return [BOS, *(self.encode_move(color, point) for color, point in moves), EOS]

    def decode_game(self, tokens: list[int]) -> list[tuple[str, tuple[int, int] | None]]:
        if not tokens or tokens[0] != BOS or tokens[-1] != EOS:
            raise ValueError("game token sequence must start with BOS and end with EOS")
        return [
            ("b" if index % 2 == 0 else "w", None if token == PASS else self.point_from_id(token))
            for index, token in enumerate(tokens[1:-1])
        ]

    def history(self, moves: list[tuple[str, tuple[int, int] | None]], length: int = 64) -> list[int]:
        if length <= 0:
            raise ValueError("history length must be positive")
        recent = [self.encode_move(color, point) for color, point in moves[-length:]]
        return [PAD] * (length - len(recent)) + recent

    def vocab(self) -> list[str]:
        names = [
            f"P_{x:02d}_{y:02d}"
            for y in range(self.max_board_size)
            for x in range(self.max_board_size)
        ]
        return [*names, "<pass>", "<bos>", "<eos>", "<decision>", "<pad>", "<unk>"]

    def save(self, output: Path, data_cli: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(data_cli), "write-vocab", "--output", str(output)],
            check=True,
            text=True,
            capture_output=True,
        )
        inspect = json.loads(
            subprocess.run(
                [str(data_cli), "inspect-vocab", "--input", str(output)],
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        )
        if inspect["type"] != "go" or inspect["vocab_size"] != VOCAB_SIZE:
            raise RuntimeError(f"unexpected vocabulary contents: {inspect}")


def game_move_records(record: object) -> list[tuple[str, tuple[int, int] | None]]:
    tokens = getattr(record, "moves")
    return [decode_move(token) for token in tokens]


def legacy_token_id(record_token: int) -> int:
    """Convert the M2 color-tagged token to the M3 point-only token."""
    _, point = decode_move(record_token)
    return PASS if point is None else point[1] * BOARD_SIZE + point[0]


def record_token_from_point(color: str, point: tuple[int, int] | None) -> int:
    return encode_move(color, point)
