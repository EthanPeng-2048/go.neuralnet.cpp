from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .model import BOARD_SIZE, GameRecord, decode_move

EMPTY = 0
BLACK = 1
WHITE = 2

PLANE_NAMES = (
    "own_stones",
    "opponent_stones",
    "white_to_play",
    "simple_ko_ban",
    "move_minus_1",
    "move_minus_2",
    "move_minus_3",
    "empty",
    "border_distance",
)
FEATURE_PLANES = len(PLANE_NAMES)


@dataclass(frozen=True, slots=True)
class Position:
    board: np.ndarray
    to_move: str
    ko: tuple[int, int] | None
    recent_moves: tuple[tuple[int, int] | None, ...]
    move_number: int
    komi: float
    black_stones_captured: int
    white_stones_captured: int

    @classmethod
    def from_frame(
        cls,
        frame: dict[str, object],
        *,
        recent_moves: list[tuple[int, int] | None],
        komi: float,
    ) -> "Position":
        rows = frame["board"]
        assert isinstance(rows, list)
        board = np.zeros((len(rows), len(rows)), dtype=np.uint8)
        for y, row in enumerate(rows):
            assert isinstance(row, str)
            for x, value in enumerate(row):
                board[y, x] = {".": EMPTY, "x": BLACK, "o": WHITE}[value]
        ko_value = frame["ko"]
        ko = None if ko_value is None else (int(ko_value[0]), int(ko_value[1]))
        return cls(
            board=board,
            to_move=str(frame["to_move"]),
            ko=ko,
            recent_moves=tuple(recent_moves[-3:]),
            move_number=int(frame["move_number"]),
            komi=komi,
            black_stones_captured=int(frame["black_stones_captured"]),
            white_stones_captured=int(frame["white_stones_captured"]),
        )


@dataclass(frozen=True, slots=True)
class FeatureSample:
    planes: np.ndarray
    scalar_names: tuple[str, ...]
    scalars: np.ndarray

    def flatten_nn(self) -> np.ndarray:
        return self.planes.reshape(-1)


def extract_features(position: Position) -> FeatureSample:
    board = position.board
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise ValueError(f"expected a square HxW board, got {board.shape}")
    size = board.shape[0]
    if position.to_move not in {"b", "w"}:
        raise ValueError(f"invalid to_move: {position.to_move}")

    planes = np.zeros((FEATURE_PLANES, size, size), dtype=np.float32)
    own = BLACK if position.to_move == "b" else WHITE
    opponent = WHITE if own == BLACK else BLACK
    planes[0] = board == own
    planes[1] = board == opponent
    planes[2].fill(1.0 if position.to_move == "w" else 0.0)
    planes[7] = board == EMPTY

    if position.ko is not None:
        x, y = position.ko
        if 0 <= x < size and 0 <= y < size:
            planes[3, y, x] = 1.0

    for offset, point in enumerate(position.recent_moves):
        if point is None:
            continue
        x, y = point
        if 0 <= x < size and 0 <= y < size:
            planes[4 + offset, y, x] = 1.0

    y_grid, x_grid = np.indices((size, size))
    border_distance = np.minimum.reduce(
        [x_grid, y_grid, size - 1 - x_grid, size - 1 - y_grid]
    ).astype(np.float32)
    planes[8] = border_distance / max(1.0, (size - 1) / 2.0)

    scalar_names = (
        "komi",
        "move_number",
        "black_stones_captured",
        "white_stones_captured",
        "captured_balance_for_black",
    )
    scalars = np.asarray(
        [
            position.komi,
            position.move_number,
            position.black_stones_captured,
            position.white_stones_captured,
            position.white_stones_captured - position.black_stones_captured,
        ],
        dtype=np.float32,
    )
    return FeatureSample(planes=planes, scalar_names=scalar_names, scalars=scalars)


def batch_columns(samples: list[FeatureSample]) -> np.ndarray:
    if not samples:
        raise ValueError("at least one feature sample is required")
    return np.stack([sample.flatten_nn() for sample in samples], axis=1)


def trace_record(record: GameRecord, rules_cli: Path) -> dict[str, object]:
    moves = []
    for token in record.moves:
        color, point = decode_move(token)
        moves.append({"color": color, "point": None if point is None else list(point)})
    request = {
        "size": record.board_size,
        "rules": record.rules,
        "komi": record.komi_twice / 2,
        "to_move": record.to_move,
        "rows": record.initial_rows(),
        "moves": moves,
    }
    result = subprocess.run(
        [str(rules_cli), "trace", "--input", "-"],
        input=json.dumps(request),
        text=True,
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def sample_from_trace(
    trace: dict[str, object],
    *,
    frame_index: int,
    record: GameRecord,
) -> FeatureSample:
    frames = trace["frames"]
    assert isinstance(frames, list)
    if not 0 <= frame_index < len(frames):
        raise IndexError(f"frame {frame_index} outside [0,{len(frames)})")
    recent = []
    for token in record.moves[:frame_index]:
        _, point = decode_move(token)
        recent.append(point)
    position = Position.from_frame(
        frames[frame_index],
        recent_moves=recent,
        komi=record.komi_twice / 2,
    )
    return extract_features(position)


def save_feature_contact_sheet(
    sample: FeatureSample,
    output: Path,
    *,
    title: str = "M3 Go feature planes",
    scale: int = 18,
) -> None:
    planes = sample.planes
    count, height, width = planes.shape
    columns = 3
    rows = (count + columns - 1) // columns
    margin = 18
    label_height = 24
    cell_width = width * scale
    cell_height = height * scale
    image = Image.new(
        "RGB",
        (
            columns * (cell_width + margin) + margin,
            label_height + rows * (cell_height + margin) + margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(image)
    draw.text((margin, 4), title, fill="black")
    for index, name in enumerate(PLANE_NAMES):
        row, column = divmod(index, columns)
        values = np.clip(planes[index], 0.0, 1.0)
        pixels = Image.fromarray((values * 255).astype(np.uint8), mode="L").resize(
            (cell_width, cell_height), Image.Resampling.NEAREST
        )
        x = margin + column * (cell_width + margin)
        y = label_height + margin + row * (cell_height + margin)
        image.paste(pixels.convert("RGB"), (x, y))
        draw.rectangle((x - 1, y - 1, x + cell_width, y + cell_height), outline="black")
        draw.text((x, y + cell_height + 3), f"{index}: {name}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def save_feature_tensor(sample: FeatureSample, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, sample.planes, allow_pickle=False)
