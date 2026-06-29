#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Summarize sparse experiment results into compact CSV tables."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_NAME = "optimal_solutions_compare_module"
DEFAULT_RESULT_SUBDIR = "results_ttft"
DEFAULT_SUMMARY_SUBDIR = "summary"
GROUP_COLUMNS = [
    "target_total_users",
    "module_count",
    "profit_per_user",
    "topology_node_count",
    "seq_len",
    "topology_params",
]
BASE_METRICS = [
    "profit",
    "users",
    "cost",
    "deploy_cost",
    "comm_cost",
    "nodes",
    "avg_modules",
    "chain_count",
    "avg_cost_per_user",
    "avg_profit_per_user",
    "ttft_ms",
    "tpot_ms",
    "queue_ms",
    "prefill_pipeline_ms",
    "decode_pipeline_interval_ms",
    "avg_chain_time",
    "total_deploy_time",
]


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def natural_key(path: Path) -> List[Any]:
    out: List[Any] = []
    for part in re.split(r"(\d+)", path.name):
        out.append(int(part) if part.isdigit() else part.lower())
    return out


def list_csv_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted([p for p in path.iterdir() if p.suffix.lower() == ".csv"], key=natural_key)


def detect_strategy_prefixes(columns: Iterable[str]) -> List[str]:
    prefixes = set()
    for col in columns:
        if col.endswith("_deployment_list"):
            prefixes.add(col[: -len("_deployment_list")])
    return sorted(prefixes)


def parse_list(value: Any) -> list:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    for parser in (ast.literal_eval,):
        try:
            parsed = parser(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass
    return []


def mean_numeric(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def mean_list_length(series: pd.Series) -> float:
    values = []
    for value in series:
        parsed = parse_list(value)
        if parsed:
            values.append(float(np.mean([float(x) for x in parsed if str(x).strip()])))
    return float(np.mean(values)) if values else float("nan")


def build_summary_for_group(df: pd.DataFrame, group_col: str, strategies: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(group_col, dropna=True)
    for strategy in strategies:
        for metric in BASE_METRICS:
            col = f"{strategy}_{metric}"
            if col not in df.columns:
                continue
            row: Dict[str, Any] = {"strategy": strategy, "metric": metric}
            for value, part in grouped:
                row[str(value)] = mean_numeric(part[col])
            rows.append(row)
        chain_len_col = f"{strategy}_chain_len_list"
        if chain_len_col in df.columns:
            row = {"strategy": strategy, "metric": "mean_chain_length"}
            for value, part in grouped:
                row[str(value)] = mean_list_length(part[chain_len_col])
            rows.append(row)
    return pd.DataFrame(rows)


def build_analysis_view(df: pd.DataFrame, group_col: str, strategies: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for value, part in df.groupby(group_col, dropna=True):
        for strategy in strategies:
            row: Dict[str, Any] = {group_col: value, "strategy": strategy, "row_count": int(len(part))}
            for metric in BASE_METRICS:
                col = f"{strategy}_{metric}"
                if col in part.columns:
                    row[metric] = mean_numeric(part[col])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sparse placement results.")
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir) if args.run_dir else PROJECT_ROOT / "data" / "runs" / args.run_name
    result_dir = resolve_path(args.result_dir) if args.result_dir else run_dir / DEFAULT_RESULT_SUBDIR
    output_dir = resolve_path(args.output) if args.output else run_dir / DEFAULT_SUMMARY_SUBDIR
    files = list_csv_files(result_dir)
    if not files:
        raise SystemExit(f"No result CSV files found in {result_dir}")

    frames = [pd.read_csv(path, encoding="utf-8-sig", low_memory=False) for path in files]
    df = pd.concat(frames, ignore_index=True)
    strategies = detect_strategy_prefixes(df.columns)
    if not strategies:
        raise SystemExit("No strategy deployment columns found.")
    output_dir.mkdir(parents=True, exist_ok=True)

    group_columns = [col for col in GROUP_COLUMNS if col in df.columns]
    if not group_columns:
        raise SystemExit("No supported grouping columns found.")
    for group_col in group_columns:
        summary = build_summary_for_group(df, group_col, strategies)
        if not summary.empty:
            summary.to_csv(output_dir / f"summary_{group_col}.csv", index=False, encoding="utf-8-sig")
        view = build_analysis_view(df, group_col, strategies)
        if not view.empty:
            view.to_csv(output_dir / f"analysis_view_{group_col}.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote summaries for {len(group_columns)} group columns to {output_dir}")


if __name__ == "__main__":
    main()
