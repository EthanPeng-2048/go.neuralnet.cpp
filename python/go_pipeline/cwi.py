from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterator

import httpx
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential

from .model import SgfUnsupported
from .sgf import parse_sgf

CWI_URL = "https://homepages.cwi.nl/~aeb/go/games/games.tgz"
CWI_ARCHIVE_SHA256 = "c57ad2ba3c5f3a3ee145bb77bcc9c0167c0d7b5478cf691a8939f750511071e2"
CWI_LICENSE = "CWI states that the games in this collection are in the public domain."
CWI_METADATA_EPOCH = 1_792_800_000  # 2026-10-24T00:00:00Z


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _download(url: str, output: Path) -> None:
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0)) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes(1024 * 1024):
                    stream.write(chunk)
    os.replace(partial, output)


def ensure_archive(
    archive: Path,
    *,
    url: str = CWI_URL,
    expected_sha256: str = CWI_ARCHIVE_SHA256,
    refresh: bool = False,
) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not archive.exists():
        _download(url, archive)
    actual = sha256_file(archive)
    if actual != expected_sha256:
        raise RuntimeError(
            f"archive checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    return archive


def extract_archive(archive: Path, destination: Path) -> Path:
    games = destination / "games"
    if games.is_dir() and any(games.rglob("*.sgf")):
        return games
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")
    if not games.is_dir():
        raise RuntimeError(f"archive did not contain games/: {archive}")
    return games


def iter_sgf_files(games_root: Path) -> Iterator[Path]:
    yield from sorted(games_root.rglob("*.sgf"), key=lambda path: path.as_posix())


def build_cwi_dataset(
    *,
    games_root: Path,
    archive: Path,
    output: Path,
    data_cli: Path,
    work_dir: Path | None = None,
    limit: int = 10_000,
    sample_size: int = 500,
    seed: int = 20261024,
) -> dict[str, object]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or output.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl = work_dir / f"{output.stem}.jsonl"
    rejected: Counter[str] = Counter()
    sizes: Counter[int] = Counter()
    accepted = 0
    scanned = 0
    imported_moves = 0

    with jsonl.open("wb") as stream:
        for path in iter_sgf_files(games_root):
            scanned += 1
            relative = path.relative_to(games_root).as_posix()
            try:
                record = parse_sgf(
                    path.read_bytes(),
                    source_path=relative,
                    default_rules="japanese",
                )
                sizes[record.board_size] += 1
                if record.board_size != 19:
                    rejected[f"board_size_{record.board_size}"] += 1
                    continue
                stream.write(orjson.dumps(record.to_json()))
                stream.write(b"\n")
                imported_moves += len(record.moves)
                accepted += 1
                if accepted % 1000 == 0:
                    print(f"parsed {accepted} games ({scanned} files scanned)", flush=True)
                if accepted >= limit:
                    break
            except (OSError, SgfUnsupported, ValueError) as error:
                message = str(error).replace("\n", " ")[:180]
                rejected[f"{type(error).__name__}: {message}"] += 1
            if scanned % 5000 == 0:
                print(f"scan {scanned}: accepted {accepted}, rejected {sum(rejected.values())}", flush=True)

    if accepted < limit:
        raise RuntimeError(f"only found {accepted} usable 19x19 games; requested {limit}")

    archive_sha = sha256_file(archive)
    metadata_epoch = int(os.environ.get("SOURCE_DATE_EPOCH", CWI_METADATA_EPOCH))
    retrieved_at = metadata_epoch
    created_at = metadata_epoch
    subprocess.run(
        [
            str(data_cli),
            "pack",
            "--input",
            str(jsonl),
            "--output",
            str(output),
            "--source-name",
            "cwi-go-games",
            "--source-url",
            CWI_URL,
            "--archive-sha256",
            archive_sha,
            "--license",
            CWI_LICENSE,
            "--retrieved-at",
            str(retrieved_at),
            "--created-at",
            str(created_at),
        ],
        check=True,
    )
    verified = json.loads(
        subprocess.run(
            [
                str(data_cli),
                "verify",
                "--input",
                str(output),
                "--sample",
                str(sample_size),
                "--seed",
                str(seed),
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )
    if not verified["byte_identical"]:
        raise RuntimeError("nngames container did not round-trip byte-identically")
    fully_verified = json.loads(
        subprocess.run(
            [str(data_cli), "verify", "--input", str(output)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )
    if not fully_verified["byte_identical"]:
        raise RuntimeError("full nngames verification failed")
    jsonl.unlink(missing_ok=True)

    manifest: dict[str, object] = {
        "format": "go.nngames",
        "dataset": str(output),
        "games": accepted,
        "moves": imported_moves,
        "positions": imported_moves + accepted,
        "bytes": output.stat().st_size,
        "dataset_sha256": sha256_file(output),
        "files_scanned": scanned,
        "rejected": dict(rejected),
        "sizes_seen": {str(key): value for key, value in sorted(sizes.items())},
        "source": {
            "name": "cwi-go-games",
            "url": CWI_URL,
            "archive": str(archive),
            "archive_sha256": archive_sha,
            "license": CWI_LICENSE,
        },
        "verification": fully_verified,
        "sampled_verification": verified,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    return manifest
