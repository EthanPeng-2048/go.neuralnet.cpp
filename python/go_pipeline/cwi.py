from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Iterator

import httpx
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential

from .model import SgfUnsupported
from .sgf import parse_sgf

# 下载地址钉死在项目 Releases 的字节快照（data-cwi-games-v1）。上游
# homepages.cwi.nl 于 2026-10-25 重新发布过归档（新 SHA-256 9a7e4069…，
# 仅新增 54 局 Cho Hunhyun 棋谱，在入库扫描窗口之外、不影响 1 万局数据集），
# 为免上游漂移破坏既有验收与断点的数据身份，下载点固定为快照。
# 快照是未改动的上游副本，作为数据资产分发：不属于源码树，不受本仓库
# LGPL-3.0-only 覆盖，沿用源许可（CWI 声明的 public domain）。
CWI_URL = (
    "https://github.com/EthanPeng-2048/go.neuralnet.cpp/releases/download/"
    "data-cwi-games-v1/cwi-go-games.tgz"
)
# 来源（provenance）：快照的原始上游地址。写入 pack 与 manifest 的身份字段
# （改它会改变 .nngames 容器字节 ⇒ 数据集 SHA-256 变化），不用于下载。
CWI_SOURCE_URL = "https://homepages.cwi.nl/~aeb/go/games/games.tgz"
CWI_ARCHIVE_SHA256 = "c57ad2ba3c5f3a3ee145bb77bcc9c0167c0d7b5478cf691a8939f750511071e2"
CWI_LICENSE = "CWI states that the games in this collection are in the public domain."
CWI_METADATA_EPOCH = 1_792_800_000  # 2026-10-24T00:00:00Z


def _run_checked(command: list[str]) -> str:
    """执行子进程并返回 stdout；失败时把 stderr 带进错误信息。

    CalledProcessError 的默认消息只有 "returned non-zero exit status 1"，
    看不到 go_data_cli 打在 stderr 上的真正原因。
    """
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"{Path(command[0]).name} exited with status "
            f"{result.returncode}: {detail[-2000:]}"
        )
    return result.stdout


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


# 解析结果五元组：(JSONL payload, board_size, moves, game_id, error)。
ParseResult = tuple[bytes | None, int | None, int | None, str | None, str | None]


# 进程池 worker（ingest 并行解析）。顶层函数才能被 pickle；返回的 payload
# 是 orjson 序列化 + 换行的成品，父进程按文件序消费 ⇒ 输出字节与串行一致。
def _parse_sgf_file(arguments: tuple[str, str]) -> ParseResult:
    path, relative = arguments
    try:
        record = parse_sgf(
            Path(path).read_bytes(),
            source_path=relative,
            default_rules="japanese",
        )
    except (OSError, SgfUnsupported, ValueError) as error:
        message = str(error).replace("\n", " ")[:180]
        return None, None, None, None, f"{type(error).__name__}: {message}"
    payload = orjson.dumps(record.to_json()) + b"\n"
    return payload, record.board_size, len(record.moves), record.id, None


def build_cwi_dataset(
    *,
    games_root: Path,
    archive: Path,
    output: Path,
    data_cli: Path,
    work_dir: Path | None = None,
    limit: int = 10_000,  # 0 = 不设上限，转换全部可用 19×19 对局
    sample_size: int = 500,
    seed: int = 20261024,
    jobs: int = 1,
) -> dict[str, object]:
    if limit < 0:
        raise ValueError("limit must be non-negative (0 = convert everything)")
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or output.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl = work_dir / f"{output.stem}.jsonl"
    rejected: Counter[str] = Counter()
    sizes: Counter[int] = Counter()
    seen_ids: set[str] = set()
    accepted = 0
    scanned = 0
    imported_moves = 0

    def consume(result: ParseResult) -> bool:
        """按文件序消费一个解析结果；返回 True = 已达 limit，应停止。

        坏棋局一律抛弃并计数（含重复 id —— pack 会对此硬失败），
        绝不让单局问题废掉整个 ingest。
        """
        nonlocal accepted, scanned, imported_moves
        payload, board_size, moves, game_id, error = result
        scanned += 1
        if error is not None:
            rejected[error] += 1
        else:
            assert (
                payload is not None
                and board_size is not None
                and moves is not None
                and game_id is not None
            )
            sizes[board_size] += 1
            if board_size != 19:
                rejected[f"board_size_{board_size}"] += 1
                return False
            if game_id in seen_ids:
                rejected[f"duplicate game id: {game_id}"] += 1
                return False
            seen_ids.add(game_id)
            stream.write(payload)
            imported_moves += moves
            accepted += 1
            if accepted % 1000 == 0:
                print(f"parsed {accepted} games ({scanned} files scanned)", flush=True)
            if limit > 0 and accepted >= limit:
                return True
        if scanned % 5000 == 0:
            print(
                f"scan {scanned}: accepted {accepted}, rejected {sum(rejected.values())}",
                flush=True,
            )
        return False

    with jsonl.open("wb") as stream:
        files = iter_sgf_files(games_root)
        if jobs <= 1:
            for path in files:
                relative = path.relative_to(games_root).as_posix()
                try:
                    record = parse_sgf(
                        path.read_bytes(),
                        source_path=relative,
                        default_rules="japanese",
                    )
                    result: ParseResult = (
                        orjson.dumps(record.to_json()) + b"\n",
                        record.board_size,
                        len(record.moves),
                        record.id,
                        None,
                    )
                except (OSError, SgfUnsupported, ValueError) as error:
                    message = str(error).replace("\n", " ")[:180]
                    result = (
                        None,
                        None,
                        None,
                        None,
                        f"{type(error).__name__}: {message}",
                    )
                if consume(result):
                    break
        else:
            # 并行解析：窗口化提交 + 按文件序消费（map保序）。窗口尾部
            # 超出 limit 的解析属于浪费但有界；计数/写盘/进度输出与串行
            # 逐条一致 ⇒ JSONL 与 manifest 字节不变。
            window = max(jobs * 8, 32)
            pool = ProcessPoolExecutor(max_workers=jobs)
            try:
                while True:
                    batch = list(islice(files, window))
                    if not batch:
                        break
                    work = [
                        (str(path), path.relative_to(games_root).as_posix())
                        for path in batch
                    ]
                    done = False
                    for parsed in pool.map(_parse_sgf_file, work, chunksize=4):
                        if consume(parsed):
                            done = True
                            break
                    if done:
                        break
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

    if limit > 0 and accepted < limit:
        raise RuntimeError(f"only found {accepted} usable 19x19 games; requested {limit}")

    archive_sha = sha256_file(archive)
    metadata_epoch = int(os.environ.get("SOURCE_DATE_EPOCH", CWI_METADATA_EPOCH))
    retrieved_at = metadata_epoch
    created_at = metadata_epoch

    def run_pack() -> None:
        _run_checked(
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
                CWI_SOURCE_URL,
                "--archive-sha256",
                archive_sha,
                "--license",
                CWI_LICENSE,
                "--retrieved-at",
                str(retrieved_at),
                "--created-at",
                str(created_at),
            ]
        )

    def run_verify(*, sampled: bool, tolerant: bool) -> dict[str, object]:
        command = [str(data_cli), "verify", "--input", str(output)]
        if sampled:
            command += ["--sample", str(sample_size), "--seed", str(seed)]
        if tolerant:
            command.append("--skip-unreplayable")
        command += ["--threads", str(max(jobs, 1))]
        return json.loads(_run_checked(command))

    run_pack()
    verified = run_verify(sampled=True, tolerant=True)
    if not verified["byte_identical"]:
        raise RuntimeError("nngames container did not round-trip byte-identically")
    fully_verified = run_verify(sampled=False, tolerant=True)
    if not fully_verified["byte_identical"]:
        raise RuntimeError("full nngames verification failed")

    # 坏棋局直接抛弃：容错重放列出的不可重放局从 JSONL 剔除后重新
    # pack，再做一次严格复验（此时必须全部可重放）。
    bad: dict[str, str] = {}
    for report in (verified, fully_verified):
        for entry in report.get("skipped_games", []):
            bad.setdefault(entry["id"], entry["reason"])
    dropped: list[dict[str, str]] = []
    if bad:
        filtered = jsonl.with_suffix(".filtered.jsonl")
        dropped_moves = 0
        with jsonl.open("rb") as source, filtered.open("wb") as target:
            for line in source:
                record = json.loads(line)
                if record["id"] in bad:
                    dropped.append({"id": record["id"], "reason": bad[record["id"]]})
                    dropped_moves += len(record["moves"])
                    continue
                target.write(line)
        filtered.replace(jsonl)
        accepted -= len(dropped)
        imported_moves -= dropped_moves
        print(f"dropped {len(dropped)} unreplayable games, re-packing", flush=True)
        run_pack()
        verified = run_verify(sampled=True, tolerant=False)
        fully_verified = run_verify(sampled=False, tolerant=False)
        if not verified["byte_identical"] or not fully_verified["byte_identical"]:
            raise RuntimeError(
                "nngames container did not round-trip after dropping games"
            )
    # manifest 不随机器变化：--threads 只影响速度，记录前剥掉。
    verified.pop("threads", None)
    fully_verified.pop("threads", None)
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
        "dropped": dropped,
        "sizes_seen": {str(key): value for key, value in sorted(sizes.items())},
        "source": {
            "name": "cwi-go-games",
            "url": CWI_SOURCE_URL,
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
