#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare reproduced InFCD_MSDG results against reference experiment CSVs."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_ROOT = (
    Path("E:/PycharmProjects/InFCD_optimal_solution")
    / "data"
    / "runs"
    / "comnet"
    / "final_main_experiment_3"
)

STRATEGY_MAPPINGS = {
    "multi_max_profit": ("multi_func_max_profit", "main"),
    "lmp": ("current_module_greedy_max_profit", "lmp"),
    "multi_max_user": ("multi_func_max_users", "main"),
    "shortest_path": ("sp", "main"),
}

NUMERIC_METRICS = [
    "cost",
    "deploy_cost",
    "comm_cost",
    "profit",
    "users",
    "nodes",
    "avg_modules",
    "chain_count",
    "avg_cost_per_user",
    "avg_profit_per_user",
]

STRUCTURED_METRICS = [
    "chain_len_list",
    "chain_used_nodes_list",
    "chain_capacity_users_list",
    "chain_served_users_list",
    "deployment_list",
]

NAT_SPLIT = re.compile(r"(\d+)")


def resolve_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def natural_key(path: Path) -> List[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in NAT_SPLIT.split(path.name)
    ]


def list_result_files(path: Path) -> Dict[str, Path]:
    if not path.exists():
        return {}
    return {
        item.name: item
        for item in sorted(path.glob("result_input_*.csv"), key=natural_key)
        if item.is_file()
    }


def read_csv(path: Path, row_limit: Optional[int]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        nrows=row_limit if row_limit and row_limit > 0 else None,
    )


def report_row(
    rows: List[Dict[str, Any]],
    status: str,
    file_name: str,
    strategy: str,
    metric: str,
    detail: str,
    max_abs_diff: float = float("nan"),
    failing_rows: int = 0,
) -> None:
    rows.append({
        "status": status,
        "file": file_name,
        "strategy": strategy,
        "metric": metric,
        "detail": detail,
        "max_abs_diff": max_abs_diff,
        "failing_rows": int(failing_rows),
    })


def parse_structured(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (list, dict, tuple)):
        return value
    text = str(value).strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return text


def normalize_structured(value: Any) -> str:
    parsed = parse_structured(value)
    return json.dumps(parsed, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def compare_numeric_columns(
    new_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    new_col: str,
    ref_col: str,
    atol: float,
    rtol: float,
) -> Tuple[bool, float, int]:
    left = pd.to_numeric(new_df[new_col], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(ref_df[ref_col], errors="coerce").to_numpy(dtype=float)
    both_nan = np.isnan(left) & np.isnan(right)
    ok = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=True) | both_nan
    diffs = np.abs(left - right)
    finite = diffs[np.isfinite(diffs)]
    max_abs_diff = float(np.max(finite)) if finite.size else 0.0
    return bool(np.all(ok)), max_abs_diff, int((~ok).sum())


def compare_structured_columns(
    new_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    new_col: str,
    ref_col: str,
) -> Tuple[bool, int]:
    left = new_df[new_col].map(normalize_structured)
    right = ref_df[ref_col].map(normalize_structured)
    ok = left == right
    return bool(ok.all()), int((~ok).sum())


def compare_strategy(
    rows: List[Dict[str, Any]],
    file_name: str,
    strategy: str,
    new_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    ref_prefix: str,
    atol: float,
    rtol: float,
) -> None:
    if len(new_df) != len(ref_df):
        report_row(
            rows,
            "fail",
            file_name,
            strategy,
            "__row_count__",
            f"row count mismatch: new={len(new_df)}, reference={len(ref_df)}",
            failing_rows=abs(len(new_df) - len(ref_df)),
        )
        return

    for metric in NUMERIC_METRICS:
        new_col = f"{strategy}_{metric}"
        ref_col = f"{ref_prefix}_{metric}"
        if new_col not in new_df.columns:
            report_row(rows, "fail", file_name, strategy, metric, f"missing new column {new_col}")
            continue
        if ref_col not in ref_df.columns:
            report_row(rows, "fail", file_name, strategy, metric, f"missing reference column {ref_col}")
            continue
        ok, max_abs_diff, failing_rows = compare_numeric_columns(
            new_df,
            ref_df,
            new_col,
            ref_col,
            atol,
            rtol,
        )
        report_row(
            rows,
            "pass" if ok else "fail",
            file_name,
            strategy,
            metric,
            "ok" if ok else "numeric mismatch",
            max_abs_diff=max_abs_diff,
            failing_rows=failing_rows,
        )

    for metric in STRUCTURED_METRICS:
        new_col = f"{strategy}_{metric}"
        ref_col = f"{ref_prefix}_{metric}"
        if new_col not in new_df.columns:
            report_row(rows, "fail", file_name, strategy, metric, f"missing new column {new_col}")
            continue
        if ref_col not in ref_df.columns:
            report_row(rows, "fail", file_name, strategy, metric, f"missing reference column {ref_col}")
            continue
        ok, failing_rows = compare_structured_columns(new_df, ref_df, new_col, ref_col)
        report_row(
            rows,
            "pass" if ok else "fail",
            file_name,
            strategy,
            metric,
            "ok" if ok else "structured value mismatch",
            failing_rows=failing_rows,
        )


def compare_result_dirs(
    new_result_dir: Path,
    reference_result_dir: Path,
    reference_lmp_result_dir: Path,
    output_dir: Path,
    atol: float,
    rtol: float,
    row_limit: Optional[int] = None,
    only_new_files: bool = False,
) -> Dict[str, Any]:
    new_files = list_result_files(new_result_dir)
    reference_files = list_result_files(reference_result_dir)
    reference_lmp_files = list_result_files(reference_lmp_result_dir)
    file_names = set(new_files)
    if not only_new_files:
        file_names |= set(reference_files) | set(reference_lmp_files)
    all_names = sorted(file_names, key=lambda name: natural_key(Path(name)))

    rows: List[Dict[str, Any]] = []
    for file_name in all_names:
        new_path = new_files.get(file_name)
        if new_path is None:
            report_row(rows, "fail", file_name, "__all__", "__file__", "missing reproduced result file")
            continue

        new_df = read_csv(new_path, row_limit)
        for strategy, (ref_prefix, ref_kind) in STRATEGY_MAPPINGS.items():
            ref_map = reference_lmp_files if ref_kind == "lmp" else reference_files
            ref_path = ref_map.get(file_name)
            if ref_path is None:
                report_row(
                    rows,
                    "warning",
                    file_name,
                    strategy,
                    "__file__",
                    "reference file is missing for this strategy",
                )
                continue
            ref_df = read_csv(ref_path, row_limit)
            compare_strategy(
                rows,
                file_name,
                strategy,
                new_df,
                ref_df,
                ref_prefix,
                atol,
                rtol,
            )

    report_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_report.csv"
    summary_path = output_dir / "validation_summary.json"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    status_counts = report_df["status"].value_counts().to_dict() if not report_df.empty else {}
    summary = {
        "passed": int(status_counts.get("fail", 0)) == 0,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "new_result_dir": str(new_result_dir),
        "reference_result_dir": str(reference_result_dir),
        "reference_lmp_result_dir": str(reference_lmp_result_dir),
        "output_dir": str(output_dir),
        "row_limit": row_limit,
        "only_new_files": bool(only_new_files),
        "atol": atol,
        "rtol": rtol,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reproduced sparse results.")
    parser.add_argument("--new_result_dir", default=PROJECT_ROOT / "data" / "runs" / "main_experiment" / "result")
    parser.add_argument("--reference_result_dir", default=REFERENCE_ROOT / "result")
    parser.add_argument("--reference_lmp_result_dir", default=REFERENCE_ROOT / "LMP_result")
    parser.add_argument("--output_dir", default=PROJECT_ROOT / "data" / "runs" / "main_experiment" / "validation")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--row_limit", type=int, default=None)
    parser.add_argument(
        "--only_new_files",
        action="store_true",
        help="Validate only reproduced files that already exist in the new result directory.",
    )
    args = parser.parse_args()

    summary = compare_result_dirs(
        new_result_dir=resolve_path(args.new_result_dir),
        reference_result_dir=resolve_path(args.reference_result_dir),
        reference_lmp_result_dir=resolve_path(args.reference_lmp_result_dir),
        output_dir=resolve_path(args.output_dir),
        atol=float(args.atol),
        rtol=float(args.rtol),
        row_limit=args.row_limit,
        only_new_files=bool(args.only_new_files),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
