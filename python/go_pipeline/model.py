from __future__ import annotations

from dataclasses import dataclass

BOARD_SIZE = 19
PASS_POINT = BOARD_SIZE * BOARD_SIZE


class SgfUnsupported(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GameRecord:
    id: str
    source_path: str
    source_sha256: str
    board_size: int
    komi_twice: int
    handicap: int
    to_move: str
    outcome: str
    result: str
    rules: str
    black_player: str
    white_player: str
    event: str
    date: str
    setup_black: tuple[int, ...]
    setup_white: tuple[int, ...]
    setup_empty: tuple[int, ...]
    moves: tuple[int, ...]

    @classmethod
    def from_json(cls, value: dict[str, object]) -> "GameRecord":
        return cls(
            id=str(value["id"]),
            source_path=str(value["source_path"]),
            source_sha256=str(value["source_sha256"]),
            board_size=int(value["board_size"]),
            komi_twice=int(value["komi_twice"]),
            handicap=int(value["handicap"]),
            to_move=str(value["to_move"]),
            outcome=str(value["outcome"]),
            result=str(value.get("result", "")),
            rules=str(value.get("rules", "japanese")),
            black_player=str(value.get("black_player", "")),
            white_player=str(value.get("white_player", "")),
            event=str(value.get("event", "")),
            date=str(value.get("date", "")),
            setup_black=tuple(int(item) for item in value["setup_black"]),
            setup_white=tuple(int(item) for item in value["setup_white"]),
            setup_empty=tuple(int(item) for item in value["setup_empty"]),
            moves=tuple(int(item) for item in value["moves"]),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "board_size": self.board_size,
            "komi_twice": self.komi_twice,
            "handicap": self.handicap,
            "to_move": self.to_move,
            "outcome": self.outcome,
            "result": self.result,
            "rules": self.rules,
            "black_player": self.black_player,
            "white_player": self.white_player,
            "event": self.event,
            "date": self.date,
            "setup_black": list(self.setup_black),
            "setup_white": list(self.setup_white),
            "setup_empty": list(self.setup_empty),
            "moves": list(self.moves),
        }

    def initial_rows(self) -> list[str]:
        board = ["."] * (self.board_size * self.board_size)
        for point in self.setup_black:
            x, y = point % BOARD_SIZE, point // BOARD_SIZE
            board[y * self.board_size + x] = "x"
        for point in self.setup_white:
            x, y = point % BOARD_SIZE, point // BOARD_SIZE
            board[y * self.board_size + x] = "o"
        for point in self.setup_empty:
            x, y = point % BOARD_SIZE, point // BOARD_SIZE
            board[y * self.board_size + x] = "."
        return [
            "".join(board[y * self.board_size : (y + 1) * self.board_size])
            for y in range(self.board_size)
        ]


def encode_move(color: str, point: tuple[int, int] | None) -> int:
    if color not in {"b", "w"}:
        raise ValueError(f"invalid move color: {color}")
    if point is None:
        point_index = PASS_POINT
    else:
        x, y = point
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError(f"point outside 19x19 board: {point}")
        point_index = y * BOARD_SIZE + x
    return point_index * 2 + (1 if color == "w" else 0)


def decode_move(token: int) -> tuple[str, tuple[int, int] | None]:
    if not 0 <= token < (PASS_POINT + 1) * 2:
        raise ValueError(f"move token outside 19x19 vocabulary: {token}")
    color = "w" if token & 1 else "b"
    point_index = token >> 1
    if point_index == PASS_POINT:
        return color, None
    return color, (point_index % BOARD_SIZE, point_index // BOARD_SIZE)
