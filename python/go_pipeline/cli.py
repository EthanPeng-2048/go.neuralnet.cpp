from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import orjson

from .cwi import (
    build_cwi_dataset,
    ensure_archive,
    extract_archive,
)
from .features import (
    sample_from_trace,
    save_feature_contact_sheet,
    save_feature_tensor,
    trace_record,
)
from .model import GameRecord
from .tokenizer import GoTokenizer
from .training import DEFAULT_HISTORY_LEN, build_training_caches


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_cli_path(root: Path, value: str | None) -> Path:
    return Path(value).resolve() if value else root / "build/cpu/go_data_cli"


def rules_cli_path(root: Path, value: str | None) -> Path:
    return Path(value).resolve() if value else root / "build/cpu/go_rules_cli"


def cmd_download(args: argparse.Namespace) -> None:
    root = default_root()
    archive = Path(args.archive or root / ".cache/downloads/cwi-go-games.tgz").resolve()
    destination = Path(args.destination or root / ".cache/cwi").resolve()
    ensure_archive(archive, refresh=args.refresh)
    games = extract_archive(archive, destination)
    print(json.dumps({"archive": str(archive), "games_root": str(games)}, indent=2))


def cmd_ingest(args: argparse.Namespace) -> None:
    root = default_root()
    archive = Path(args.archive or root / ".cache/downloads/cwi-go-games.tgz").resolve()
    games_root = Path(args.games_root or root / ".cache/cwi/games").resolve()
    output = Path(args.output or root / "data/cwi-10k.nngames").resolve()
    ensure_archive(archive)
    if not games_root.is_dir():
        games_root = extract_archive(archive, games_root.parent)
    manifest = build_cwi_dataset(
        games_root=games_root,
        archive=archive,
        output=output,
        data_cli=data_cli_path(root, args.data_cli),
        work_dir=root / ".cache",
        limit=args.limit,
        sample_size=args.sample,
        seed=args.seed,
    )
    print(orjson.dumps(manifest, option=orjson.OPT_INDENT_2).decode())


def cmd_verify(args: argparse.Namespace) -> None:
    root = default_root()
    dataset = Path(args.dataset or root / "data/cwi-10k.nngames").resolve()
    command = [
        str(data_cli_path(root, args.data_cli)),
        "verify",
        "--input",
        str(dataset),
        "--seed",
        str(args.seed),
    ]
    if args.sample:
        command.extend(["--sample", str(args.sample)])
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout, end="")


def cmd_tokenizer(args: argparse.Namespace) -> None:
    root = default_root()
    output = Path(args.output or root / "data/go.nnvocab").resolve()
    GoTokenizer().save(output, data_cli_path(root, args.data_cli))
    print(json.dumps({"vocabulary": str(output), "vocab_size": 367}, indent=2))


def cmd_features(args: argparse.Namespace) -> None:
    root = default_root()
    dataset = Path(args.dataset or root / "data/cwi-10k.nngames").resolve()
    exported = subprocess.run(
        [
            str(data_cli_path(root, args.data_cli)),
            "export",
            "--input",
            str(dataset),
            "--index",
            str(args.index),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    value = json.loads(exported.stdout)
    record = GameRecord(
        id=value["id"],
        source_path=value["source_path"],
        source_sha256=value["source_sha256"],
        board_size=value["board_size"],
        komi_twice=value["komi_twice"],
        handicap=value["handicap"],
        to_move=value["to_move"],
        outcome=value["outcome"],
        result=value["result"],
        rules=value["rules"],
        black_player=value["black_player"],
        white_player=value["white_player"],
        event=value["event"],
        date=value["date"],
        setup_black=tuple(value["setup_black"]),
        setup_white=tuple(value["setup_white"]),
        setup_empty=tuple(value["setup_empty"]),
        moves=tuple(value["moves"]),
    )
    trace = trace_record(record, rules_cli_path(root, args.rules_cli))
    frames = trace["frames"]
    frame_index = args.frame if args.frame is not None else max(0, len(frames) // 2)
    sample = sample_from_trace(trace, frame_index=frame_index, record=record)
    png = Path(args.output or root / "docs/benchmarks/m3-feature-planes.png").resolve()
    save_feature_contact_sheet(
        sample,
        png,
        title=f"{record.id} frame {frame_index}/{len(frames) - 1}",
    )
    npy = Path(args.npy_output or png.with_suffix(".npy")).resolve()
    save_feature_tensor(sample, npy)
    print(
        json.dumps(
            {
                "game": record.id,
                "frame": frame_index,
                "total_frames": len(frames),
                "planes": list(sample.planes.shape),
                "scalar_names": list(sample.scalar_names),
                "scalars": sample.scalars.tolist(),
                "image": str(png),
                "tensor": str(npy),
            },
            indent=2,
        )
    )


def cmd_training_cache(args: argparse.Namespace) -> None:
    root = default_root()
    dataset = Path(args.dataset or root / "data/cwi-10k.nngames").resolve()
    train_output = Path(
        args.train_output or root / "data/m4-c-train.ngf"
    ).resolve()
    eval_output = Path(
        args.eval_output or root / "data/m4-c-eval.ngf"
    ).resolve()
    manifest = build_training_caches(
        dataset=dataset,
        data_cli=data_cli_path(root, args.data_cli),
        rules_cli=rules_cli_path(root, args.rules_cli),
        train_output=train_output,
        eval_output=eval_output,
        max_games=args.max_games,
        eval_games=args.eval_games,
        seed=args.seed,
        history_len=args.history_len,
    )
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(manifest, option=orjson.OPT_INDENT_2).decode())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="go-pipeline")
    sub = result.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download and extract CWI public-domain games")
    download.add_argument("--archive")
    download.add_argument("--destination")
    download.add_argument("--refresh", action="store_true")
    download.set_defaults(func=cmd_download)

    ingest = sub.add_parser("ingest", help="convert 10k 19x19 SGFs to .nngames")
    ingest.add_argument("--archive")
    ingest.add_argument("--games-root")
    ingest.add_argument("--output")
    ingest.add_argument("--data-cli")
    ingest.add_argument("--limit", type=int, default=10_000)
    ingest.add_argument("--sample", type=int, default=500)
    ingest.add_argument("--seed", type=int, default=20261024)
    ingest.set_defaults(func=cmd_ingest)

    verify = sub.add_parser("verify", help="verify KVRecord round-trip and replay")
    verify.add_argument("--dataset")
    verify.add_argument("--data-cli")
    verify.add_argument("--sample", type=int, default=500)
    verify.add_argument("--seed", type=int, default=20261024)
    verify.set_defaults(func=cmd_verify)

    tokenizer = sub.add_parser("tokenizer", help="write GoTokenizer .nnvocab")
    tokenizer.add_argument("--output")
    tokenizer.add_argument("--data-cli")
    tokenizer.set_defaults(func=cmd_tokenizer)

    features = sub.add_parser("features", help="trace a game and render feature planes")
    features.add_argument("--dataset")
    features.add_argument("--data-cli")
    features.add_argument("--rules-cli")
    features.add_argument("--index", type=int, default=0)
    features.add_argument("--frame", type=int)
    features.add_argument("--output")
    features.add_argument("--npy-output")
    features.set_defaults(func=cmd_features)

    cache = sub.add_parser(
        "training-cache",
        help="build versioned .ngf feature caches for the C validator",
    )
    cache.add_argument("--dataset")
    cache.add_argument("--data-cli")
    cache.add_argument("--rules-cli")
    cache.add_argument("--train-output")
    cache.add_argument("--eval-output")
    cache.add_argument("--manifest")
    cache.add_argument("--max-games", type=int, default=1000)
    cache.add_argument("--eval-games", type=int, default=100)
    cache.add_argument("--seed", type=int, default=20261024)
    # M7: per-sample move history for the B model's history branch (K slots,
    # left <pad>-padded). Must match the model's --history-len at train time.
    cache.add_argument("--history-len", type=int, default=DEFAULT_HISTORY_LEN)
    cache.set_defaults(func=cmd_training_cache)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
