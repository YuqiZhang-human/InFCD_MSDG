#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Command helpers for the local visualization app."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "data" / "runs" / "optimal_solutions_compare_module"
DEFAULT_INPUT_DIR = DEFAULT_RUN_DIR / "input"
DEFAULT_RESULT_DIR = DEFAULT_RUN_DIR / "result"
DEFAULT_TTFT_RESULT_DIR = DEFAULT_RUN_DIR / "results_ttft"
DEFAULT_SUMMARY_DIR = DEFAULT_RUN_DIR / "summary"
DEFAULT_VALIDATION_DIR = DEFAULT_RUN_DIR / "validation"

STRATEGY_LABELS = {
    "MSDG": "multi_max_profit",
    "LMP": "lmp",
    "LMU": "multi_max_user",
    "SPC": "shortest_path",
}

DEFAULT_REFERENCE_RESULT_DIR = (
    Path("E:/PycharmProjects/InFCD_optimal_solution")
    / "data"
    / "runs"
    / "comnet"
    / "final_main_experiment_3"
    / "result"
)
DEFAULT_REFERENCE_LMP_RESULT_DIR = (
    Path("E:/PycharmProjects/InFCD_optimal_solution")
    / "data"
    / "runs"
    / "comnet"
    / "final_main_experiment_3"
    / "LMP_result"
)


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def as_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_csv_files(path: str | Path) -> list[Path]:
    resolved = as_path(path)
    if resolved.is_file():
        return [resolved]
    if not resolved.exists():
        return []
    return sorted(
        [
            item
            for item in resolved.iterdir()
            if item.is_file() and item.suffix.lower() == ".csv"
        ],
        key=lambda item: item.name,
    )


def run_command(command: list[str]) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_placement_experiment(
    run_dir: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    strategies: Iterable[str],
    processes: int,
    limit: Optional[int] = None,
    input_file: Optional[str] = None,
) -> CommandResult:
    strategy_arg = ",".join(strategies)
    command = [
        sys.executable,
        "-m",
        "src.experiment.run_sparse_experiment",
        "--run_dir",
        str(as_path(run_dir)),
        "--input_dir",
        str(as_path(input_dir)),
        "--output_dir",
        str(as_path(output_dir)),
        "--strategies",
        strategy_arg,
        "--processes",
        str(max(1, int(processes))),
    ]
    if input_file:
        command = [
            item
            for item in command
            if item not in {"--input_dir", str(as_path(input_dir))}
        ]
        command.extend(["--input", input_file])
    if limit is not None and int(limit) > 0:
        command.extend(["--limit", str(int(limit))])
    return run_command(command)


def add_ttft_metrics(
    input_dir: str | Path,
    result_dir: str | Path,
    output_dir: str | Path,
    load_factors: str = "1.0",
    ttft_slo: str = "",
    tpot_slo: str = "",
) -> CommandResult:
    command = [
        sys.executable,
        "-m",
        "data.src.add_ttft_to_results_sparse",
        "--input_dir",
        str(as_path(input_dir)),
        "--result_dir",
        str(as_path(result_dir)),
        "--output_dir",
        str(as_path(output_dir)),
        "--load_factors",
        load_factors,
    ]
    if ttft_slo.strip():
        command.extend(["--TTFT_SLO", ttft_slo.strip()])
    if tpot_slo.strip():
        command.extend(["--TPOT_SLO", tpot_slo.strip()])
    return run_command(command)


def add_ttft_metrics_for_file(
    input_file: str | Path,
    result_file: str | Path,
    output_file: str | Path,
    load_factors: str = "1.0",
    ttft_slo: str = "",
    tpot_slo: str = "",
) -> CommandResult:
    command = [
        sys.executable,
        "-m",
        "data.src.add_ttft_to_results_sparse",
        "--input",
        str(as_path(input_file)),
        "--result",
        str(as_path(result_file)),
        "--output",
        str(as_path(output_file)),
        "--load_factors",
        load_factors,
    ]
    if ttft_slo.strip():
        command.extend(["--TTFT_SLO", ttft_slo.strip()])
    if tpot_slo.strip():
        command.extend(["--TPOT_SLO", tpot_slo.strip()])
    return run_command(command)


def generate_summary(
    run_dir: str | Path,
    result_dir: str | Path,
    output_dir: str | Path,
) -> CommandResult:
    command = [
        sys.executable,
        "-m",
        "data.src.analysis.summarize_results_sparse",
        "--run_dir",
        str(as_path(run_dir)),
        "--result_dir",
        str(as_path(result_dir)),
        "--output",
        str(as_path(output_dir)),
    ]
    return run_command(command)


def validate_against_reference(
    new_result_dir: str | Path,
    reference_result_dir: str | Path,
    reference_lmp_result_dir: str | Path,
    output_dir: str | Path,
    atol: float = 1e-6,
    rtol: float = 1e-9,
    row_limit: Optional[int] = None,
    only_new_files: bool = False,
) -> CommandResult:
    command = [
        sys.executable,
        "-m",
        "visualization.compare_results",
        "--new_result_dir",
        str(as_path(new_result_dir)),
        "--reference_result_dir",
        str(as_path(reference_result_dir)),
        "--reference_lmp_result_dir",
        str(as_path(reference_lmp_result_dir)),
        "--output_dir",
        str(as_path(output_dir)),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
    ]
    if row_limit is not None and int(row_limit) > 0:
        command.extend(["--row_limit", str(int(row_limit))])
    if only_new_files:
        command.append("--only_new_files")
    return run_command(command)
