#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run the bundled sample workflow end to end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data" / "runs" / "optimal_solutions_compare_module"
SMOKE_RESULT_DIR = RUN_DIR / "smoke_result"
SMOKE_TTFT_DIR = RUN_DIR / "smoke_results_ttft"
SMOKE_SUMMARY_DIR = RUN_DIR / "smoke_summary"


def run(cmd: list[str]) -> None:
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run([
        python,
        "-m",
        "src.experiment.run_sparse_experiment",
        "--run_dir",
        str(RUN_DIR),
        "--output_dir",
        str(SMOKE_RESULT_DIR),
        "--strategies",
        "multi_max_profit,lmp,multi_max_user,shortest_path",
        "--processes",
        "1",
        "--limit",
        "1",
    ])
    run([
        python,
        "-m",
        "data.src.add_ttft_to_results_sparse",
        "--input_dir",
        str(RUN_DIR / "input"),
        "--result_dir",
        str(SMOKE_RESULT_DIR),
        "--output_dir",
        str(SMOKE_TTFT_DIR),
        "--load_factors",
        "1.0",
    ])
    run([
        python,
        "-m",
        "data.src.analysis.summarize_results_sparse",
        "--run_dir",
        str(RUN_DIR),
        "--result_dir",
        str(SMOKE_TTFT_DIR),
        "--output",
        str(SMOKE_SUMMARY_DIR),
    ])
    print("[OK] Sample workflow completed.")


if __name__ == "__main__":
    main()
