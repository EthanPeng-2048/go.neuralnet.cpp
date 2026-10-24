#!/usr/bin/env python3
"""Run the M0 project-shaped layer_bench matrix and emit JSON + Markdown."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


PROFILES = {
    "b-small": {
        "dmodel": 192,
        "heads": 8,
        "dff": 768,
        "transformer_layers": 8,
    },
    "b-final": {
        "dmodel": 256,
        "heads": 8,
        "dff": 1024,
        "transformer_layers": 12,
    },
}

PHASES = {
    "inference": 64,
    "training": 256,
}

LAYER_LINE = re.compile(
    r"^\s+(?P<layer>\w+)\s*:\s*"
    r"fwd\s+(?P<fwd_ms>[0-9.]+)\s+ms\s+"
    r"(?P<fwd_gflops>[0-9.]+)\s+GFLOPS"
    r"(?:\s+\|\s+train\s+(?P<train_ms>[0-9.]+)\s+ms\s+"
    r"(?P<train_tok_s>[0-9]+)\s+tok/s)?"
)


@dataclass
class Measurement:
    profile: str
    phase: str
    layer: str
    batch: int
    sequence: int
    dmodel: int
    heads: int
    dff: int
    forward_ms: float
    forward_gflops: float
    train_ms: float | None
    train_tokens_per_s: float | None


def run_command(command: list[str]) -> str:
    proc = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n{proc.stdout}"
        )
    return proc.stdout


def benchmark(
    layer_bench: Path,
    gpu: bool,
    iterations: int,
    warmup: int,
    history: int,
    feature_planes: int,
) -> tuple[list[Measurement], list[str]]:
    measurements: list[Measurement] = []
    transcripts: list[str] = []
    sequence = history + 1

    for profile_name, profile in PROFILES.items():
        for phase_name, batch in PHASES.items():
            command = [
                str(layer_bench),
                "--layer",
                "conv2d,transformer,linear",
                "--batch",
                str(batch),
                "--seq",
                str(sequence),
                "--dmodel",
                str(profile["dmodel"]),
                "--heads",
                str(profile["heads"]),
                "--dff",
                str(profile["dff"]),
                "--in",
                str(profile["dmodel"]),
                "--out",
                "363",
                "--cin",
                str(feature_planes),
                "--cout",
                "64",
                "--kernel",
                "3",
                "--hw",
                "19",
                "--iter",
                str(iterations),
                "--warmup",
                str(warmup),
            ]
            if gpu:
                command.append("--gpu")
            if phase_name == "inference":
                command.append("--forward-only")

            output = run_command(command)
            transcripts.append(
                f"$ {' '.join(command)}\n{output.rstrip()}\n"
            )

            found: set[str] = set()
            for line in output.splitlines():
                match = LAYER_LINE.match(line)
                if not match:
                    continue
                layer = match.group("layer")
                found.add(layer)
                measurements.append(
                    Measurement(
                        profile=profile_name,
                        phase=phase_name,
                        layer=layer,
                        batch=batch,
                        sequence=sequence,
                        dmodel=profile["dmodel"],
                        heads=profile["heads"],
                        dff=profile["dff"],
                        forward_ms=float(match.group("fwd_ms")),
                        forward_gflops=float(match.group("fwd_gflops")),
                        train_ms=(
                            float(match.group("train_ms"))
                            if match.group("train_ms")
                            else None
                        ),
                        train_tokens_per_s=(
                            float(match.group("train_tok_s"))
                            if match.group("train_tok_s")
                            else None
                        ),
                    )
                )

            missing = {"conv2d", "transformer", "linear"} - found
            if missing:
                raise RuntimeError(
                    f"{profile_name}/{phase_name}: layer_bench omitted "
                    f"{sorted(missing)}\n{output}"
                )

    return measurements, transcripts


def estimates(measurements: list[Measurement]) -> list[dict[str, float | str]]:
    by_key = {
        (m.profile, m.phase, m.layer): m
        for m in measurements
    }
    rows: list[dict[str, float | str]] = []

    for profile_name, profile in PROFILES.items():
        layers = int(profile["transformer_layers"])
        for phase_name, batch in PHASES.items():
            conv = by_key[(profile_name, phase_name, "conv2d")]
            transformer = by_key[(profile_name, phase_name, "transformer")]
            head = by_key[(profile_name, phase_name, "linear")]

            forward_ms = (
                conv.forward_ms
                + layers * transformer.forward_ms
                + head.forward_ms
            )
            row: dict[str, float | str] = {
                "profile": profile_name,
                "phase": phase_name,
                "batch": batch,
                "estimated_forward_ms": forward_ms,
                "estimated_positions_per_s": batch * 1000.0 / forward_ms,
            }

            if transformer.train_ms is not None and conv.train_ms is not None and head.train_ms is not None:
                train_ms = conv.train_ms + layers * transformer.train_ms + head.train_ms
                row["estimated_train_ms"] = train_ms
                row["estimated_train_positions_per_s"] = batch * 1000.0 / train_ms
            rows.append(row)

    return rows


def render_markdown(
    measurements: list[Measurement],
    estimate_rows: list[dict[str, float | str]],
    transcripts: list[str],
    command_line: str,
    gpu: bool,
) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    device = "Vulkan GPU" if gpu else "CPU"
    lines = [
        "# M0 layer benchmark raw results",
        "",
        f"- Run (UTC): {now}",
        f"- Host: {platform.platform()}",
        f"- Device: {device}",
        f"- Command: `{command_line}`",
        "- Board: 19x19",
        "- Sequence: `K=64` history moves + 1 decision query = 65",
        "- Conv input: 17 feature planes (provisional until M3 freezes features)",
        "- Head count: 8 (provisional until M7 freezes the model)",
        "- Profiled layers: conv2d, transformer, linear head",
        "- Batch: inference 64 positions; training 256 positions",
        "- Train column = forward + backward; optimizer update is excluded",
        "",
        "## Raw layer results",
        "",
        "| Profile | Phase | Layer | Batch | Fwd ms | Fwd GFLOPS | Train ms | Train tok/s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for m in measurements:
        train_ms = "" if m.train_ms is None else f"{m.train_ms:.3f}"
        train_tok = (
            ""
            if m.train_tokens_per_s is None
            else f"{m.train_tokens_per_s:.0f}"
        )
        lines.append(
            f"| {m.profile} | {m.phase} | {m.layer} | {m.batch} | "
            f"{m.forward_ms:.3f} | {m.forward_gflops:.1f} | "
            f"{train_ms} | {train_tok} |"
        )

    lines.extend(
        [
            "",
            "## Component-sum estimate",
            "",
            "> This is a component-sum baseline, not a measured end-to-end GoModel:",
            "> `conv2d + N * transformer + linear_head`. Conv-stem/ResBlock count and",
            "> feature planes remain M3/M7 decisions. Replace this estimate with the",
            "> real model benchmark when those decisions land.",
            "",
            "| Profile | Phase | Batch | Est. total ms | Est. positions/s |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in estimate_rows:
        ms_key = (
            "estimated_forward_ms"
            if row["phase"] == "inference"
            else "estimated_train_ms"
        )
        pos_key = (
            "estimated_positions_per_s"
            if row["phase"] == "inference"
            else "estimated_train_positions_per_s"
        )
        lines.append(
            f"| {row['profile']} | {row['phase']} | {row['batch']} | "
            f"{float(row[ms_key]):.3f} | {float(row[pos_key]):.1f} |"
        )

    lines.extend(["", "## Raw transcripts", "", "```text"])
    lines.extend(transcript.rstrip() for transcript in transcripts)
    lines.extend(["```", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer-bench",
        type=Path,
        required=True,
        help="Path to the neuralnet.cpp layer_bench executable",
    )
    parser.add_argument("--gpu", action="store_true", help="Run on Vulkan GPU")
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--history", type=int, default=64)
    parser.add_argument("--feature-planes", type=int, default=17)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("docs/benchmarks/m0-layer-bench.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("docs/benchmarks/m0-layer-bench.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layer_bench = args.layer_bench.resolve()
    if not layer_bench.is_file():
        print(f"layer_bench not found: {layer_bench}", file=sys.stderr)
        return 2
    if args.iter < 1 or args.warmup < 0:
        print("--iter must be >= 1 and --warmup must be >= 0", file=sys.stderr)
        return 2

    try:
        measurements, transcripts = benchmark(
            layer_bench=layer_bench,
            gpu=args.gpu,
            iterations=args.iter,
            warmup=args.warmup,
            history=args.history,
            feature_planes=args.feature_planes,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    estimate_rows = estimates(measurements)
    command_line = " ".join(sys.argv)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.platform(),
        "device": "vulkan" if args.gpu else "cpu",
        "command": command_line,
        "assumptions": {
            "board_size": 19,
            "history_moves": args.history,
            "sequence_length": args.history + 1,
            "feature_planes": args.feature_planes,
            "inference_batch": PHASES["inference"],
            "training_batch": PHASES["training"],
        },
        "measurements": [asdict(m) for m in measurements],
        "component_sum_estimates": estimate_rows,
    }
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(
        render_markdown(
            measurements,
            estimate_rows,
            transcripts,
            command_line,
            args.gpu,
        ),
        encoding="utf-8",
    )

    print(f"wrote {args.json_out}")
    print(f"wrote {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
