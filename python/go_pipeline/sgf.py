from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sgfmill import sgf

from .model import GameRecord, SgfUnsupported, encode_move


def _property(node: Any, name: str, default: Any = None) -> Any:
    return node.get(name) if node.has_property(name) else default


def _text(node: Any, name: str, default: str = "") -> str:
    value = _property(node, name, default)
    return "" if value is None else str(value)


def _number(node: Any, name: str, default: float) -> float:
    value = _property(node, name, default)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise SgfUnsupported(f"invalid {name}: {value!r}") from error


def _point_index(point: tuple[int, int], size: int) -> int:
    row, column = point
    if not (0 <= row < size and 0 <= column < size):
        raise SgfUnsupported(f"setup point outside {size}x{size}: {point}")
    # M2 stores coordinates in the fixed 19x19 token order. Small boards use
    # the same top-left prefix so their records remain in one vocabulary.
    return (size - 1 - row) * 19 + column


def _outcome(result: str) -> str:
    normalized = result.strip().upper()
    if normalized.startswith("B+"):
        return "black"
    if normalized.startswith("W+"):
        return "white"
    if normalized in {"0", "DRAW", "JIGO"}:
        return "draw"
    return "unknown"


def _rules(node: Any, default_rules: str) -> str:
    raw = _text(node, "RU").lower()
    if "chinese" in raw:
        return "chinese"
    if "japanese" in raw:
        return "japanese"
    return default_rules


def parse_sgf(
    data: bytes,
    *,
    source_path: str,
    source_sha256: str | None = None,
    default_rules: str = "japanese",
) -> GameRecord:
    try:
        game = sgf.Sgf_game.from_bytes(data)
    except Exception as error:
        raise SgfUnsupported(f"sgfmill parse failed: {error}") from error

    root = game.get_root()
    try:
        size = root.get_size()
    except Exception as error:
        raise SgfUnsupported(f"invalid SZ: {error}") from error
    if not 1 <= size <= 19:
        raise SgfUnsupported(f"board size {size} exceeds supported range")

    setup_black: set[int] = set()
    setup_white: set[int] = set()
    setup_empty: set[int] = set()
    moves: list[int] = []
    first_move_color: str | None = None
    declared_to_move: str | None = None

    for node in game.get_main_sequence():
        if declared_to_move is None:
            player = _property(node, "PL")
            if player in {"b", "w"}:
                declared_to_move = player

        black, white, empty = node.get_setup_stones()
        if black or white or empty:
            if moves:
                raise SgfUnsupported("setup stones after the first move are not supported")
            setup_black.update(_point_index(point, size) for point in black)
            setup_white.update(_point_index(point, size) for point in white)
            setup_empty.update(_point_index(point, size) for point in empty)

        color, point = node.get_move()
        if color is None:
            continue
        if color not in {"b", "w"}:
            raise SgfUnsupported(f"invalid move color: {color}")
        first_move_color = first_move_color or color
        coordinate = None
        if point is not None:
            row, column = point
            if not (0 <= row < size and 0 <= column < size):
                raise SgfUnsupported(f"move outside {size}x{size}: {point}")
            coordinate = (column, size - 1 - row)
        moves.append(encode_move(color, coordinate))

    if not moves:
        raise SgfUnsupported("game contains no moves")

    handicap_raw = _property(root, "HA", 0)
    try:
        handicap = int(handicap_raw or 0)
    except (TypeError, ValueError) as error:
        raise SgfUnsupported(f"invalid HA: {handicap_raw!r}") from error

    result = _text(root, "RE")
    to_move = declared_to_move or first_move_color
    if to_move is None:
        to_move = "w" if handicap >= 2 else "b"

    return GameRecord(
        id=f"cwi:{Path(source_path).with_suffix('').as_posix()}",
        source_path=source_path,
        source_sha256=source_sha256 or hashlib.sha256(data).hexdigest(),
        board_size=size,
        komi_twice=round(_number(root, "KM", 5.5) * 2),
        handicap=handicap,
        to_move=to_move,
        outcome=_outcome(result),
        result=result,
        rules=_rules(root, default_rules),
        black_player=_text(root, "PB"),
        white_player=_text(root, "PW"),
        event=_text(root, "EV"),
        date=_text(root, "DT"),
        setup_black=tuple(sorted(setup_black)),
        setup_white=tuple(sorted(setup_white)),
        setup_empty=tuple(sorted(setup_empty)),
        moves=tuple(moves),
    )
