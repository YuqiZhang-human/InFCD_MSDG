#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mix-chain sparse experiment entry point for InFCD_MSDG.

This entry uses the same sparse optimizers as the main runner and writes
strategy names with a `mix_` prefix.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from src.experiment import run_sparse_experiment as main_runner
from src.experiment import sparse_experiment_common as common

MIX_STRATEGY_MAP = {
    "mix_multi_max_profit": "multi_max_profit",
    "mix_lmp": "lmp",
    "mix_multi_max_user": "multi_max_user",
    "mix_brute": "brute",
    "mix_shortest_path": "shortest_path",
}


def parse_mix_strategies(raw: Optional[str]) -> set[str]:
    if not raw:
        return set(MIX_STRATEGY_MAP)
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    invalid = requested - set(MIX_STRATEGY_MAP)
    if invalid:
        raise SystemExit(
            "Unknown mix strategies: "
            + ",".join(sorted(invalid))
            + "\nValid mix strategies: "
            + ",".join(sorted(MIX_STRATEGY_MAP))
        )
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sparse mix-chain strategies.")
    parser.add_argument("--run_name", default="optimal_solutions_compare_module")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output", default="result_{input_name}.csv")
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    mix_enabled = parse_mix_strategies(args.strategies)
    public_enabled = {MIX_STRATEGY_MAP[name] for name in mix_enabled}
    run_dir = main_runner.resolve_run_dir(args.run_name, args.run_dir)
    output_dir = main_runner.resolve_abs_path(args.output_dir, common.project_root) if args.output_dir else os.path.join(run_dir, "mix_result")
    input_files = main_runner.resolve_input_files(args, run_dir)

    for input_file in input_files:
        stem = os.path.splitext(os.path.basename(input_file))[0]
        temp_output = os.path.join(output_dir, args.output.replace("{input_name}", stem))
        main_runner.run_file(input_file, temp_output, public_enabled, args.limit, max(1, int(args.processes)))
        # Rename public prefixes to the mix-chain namespace after the equivalent public run.
        import pandas as pd
        df = pd.read_csv(temp_output, encoding="utf-8-sig", low_memory=False)
        rename = {}
        for mix_name, public_name in MIX_STRATEGY_MAP.items():
            if mix_name not in mix_enabled:
                continue
            for col in df.columns:
                if col.startswith(public_name + "_"):
                    rename[col] = mix_name + col[len(public_name):]
        df = df.rename(columns=rename)
        df.to_csv(temp_output, index=False, encoding="utf-8-sig")
        print(f"[OK] mix-chain result -> {temp_output}")


if __name__ == "__main__":
    main()
