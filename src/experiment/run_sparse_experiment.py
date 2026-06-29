#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sparse experiment runner for InFCD_MSDG.

The runner exposes the sparse placement strategies used by this repository and
keeps the result schema compatible with the TTFT and summary scripts by writing
one `{strategy}_deployment_list` family of columns per strategy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
from tqdm import tqdm

from src.experiment import sparse_experiment_common as common
from src.algorithm.single_chain_real_net.multi_all_node_KV_optimizer_sparse import (
    MultiFunctionOptimizerSparse,
)
from src.algorithm.single_chain_real_net.current_module_greedy_multi_all_node_KV_optimizer_sparse import (
    CurrentModuleGreedyMultiFunctionOptimizerSparse,
)
from src.algorithm.single_chain_real_net.brute_force_single_chain_optimizer_sparse import (
    BruteForceSingleChainOptimizerSparse,
)
from src.algorithm.single_chain_real_net.shortest_path_all_node_KV_optimizer_sparse import (
    ShortestPathOptimizerSparse,
)

DEFAULT_RUN_NAME = "optimal_solutions_compare_module"
DEFAULT_RESULT_SUBDIR = "result"
PUBLIC_STRATEGIES = frozenset({
    "multi_max_profit",
    "lmp",
    "multi_max_user",
    "brute",
    "shortest_path",
})


def resolve_abs_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def resolve_run_dir(run_name: str, run_dir: Optional[str]) -> str:
    if run_dir:
        resolved = resolve_abs_path(run_dir, common.project_root)
        if resolved is None:
            raise ValueError("run_dir resolved to None")
        return resolved
    return os.path.join(common.DEFAULT_RUNS_DIR, run_name)


def parse_enabled_strategies(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set(PUBLIC_STRATEGIES)
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    invalid = requested - PUBLIC_STRATEGIES
    if invalid:
        raise SystemExit(
            "Unknown strategies: "
            + ",".join(sorted(invalid))
            + "\nValid strategies: "
            + ",".join(sorted(PUBLIC_STRATEGIES))
        )
    if not requested:
        raise SystemExit("No valid strategies requested.")
    return requested


def metric_context(test_data: Dict[str, Any]) -> Dict[str, float]:
    resource_demands = test_data.get("resource_demands", []) or []
    computation_capacity = test_data.get("computation_capacity", []) or []
    global_compute_per_user = sum(float(x[0]) for x in resource_demands if len(x) > 0)
    kv_per_user_gb = float(test_data.get("kv_per_user_gb", 0.0) or 0.0)
    if kv_per_user_gb <= 0.0:
        kv_per_user_gb = sum(float(x[1]) for x in resource_demands if len(x) > 1)
    return {
        "global_compute_per_user": global_compute_per_user,
        "total_compute_capacity": sum(float(x[0]) for x in computation_capacity if len(x) > 0),
        "total_memory_capacity": sum(float(x[1]) for x in computation_capacity if len(x) > 1),
        "total_weights_gb": float(test_data.get("total_weights_gb", 0.0) or 0.0),
        "kv_per_user_gb": kv_per_user_gb,
    }


def base_result(row: Dict[str, Any], test_data: Dict[str, Any]) -> Dict[str, Any]:
    topology_params = test_data.get("topology_params", {}) or {}
    return {
        "test_data_id": int(test_data.get("test_data_id", row.get("test_data_id", 0)) or 0),
        "topology_name": test_data.get("topology_name", row.get("topology_name", "")),
        "topology_type": test_data.get("topology_type", row.get("topology_type", "")),
        "topology_params": json.dumps(topology_params, ensure_ascii=True),
        "topology_node_count": int(test_data.get("node_count", row.get("node_count", 0)) or 0),
        "model_name": test_data.get("model_name", row.get("model_name", "")),
        "module_count": int(test_data.get("module_count", row.get("K_segments", 0)) or 0),
        "seq_len": int(float(row.get("seq_len", 0) or 0)),
        "profit_per_user": float(test_data.get("profit_per_user", row.get("user_price_per_month", 0.0)) or 0.0),
        "target_total_users": int(row.get("target_total_users", 0) or 0),
        "gpu_set": test_data.get("gpu_set", row.get("gpu_set", "")),
    }


def write_plan(
    result: Dict[str, Any],
    prefix: str,
    plan: Any,
    optimizer: Any,
    context: Dict[str, float],
) -> None:
    common._write_plan_result_fields(
        result=result,
        prefix=prefix,
        plan=plan,
        optimizer=optimizer,
        **context,
    )


def shortest_path_plan(sp_result: Optional[Dict[str, Any]]) -> Optional[tuple]:
    if not sp_result:
        return None
    return (
        sp_result["total_cost"],
        sp_result["total_deploy_cost"],
        sp_result["total_comm_cost"],
        sp_result["total_profit"],
        sp_result["total_users"],
        sp_result["used_nodes"],
        sp_result["avg_modules_per_node"],
        sp_result["chain_count"],
        sp_result["chain_len_list"],
        sp_result["chain_avg_modules_list"],
        sp_result["chain_used_nodes_list"],
        sp_result.get("chain_capacity_users_list", []),
        sp_result.get("chain_served_users_list", []),
        sp_result.get("deployment_list", []),
        sp_result.get("edge_traffic_list", []),
        sp_result.get("external_traffic_list", []),
        sp_result.get("chain_time_list", []),
        sp_result.get("avg_chain_time", 0.0),
        sp_result.get("total_deploy_time", 0.0),
    )


def process_test_case(row: Dict[str, Any], enabled_strategies: Iterable[str]) -> Dict[str, Any]:
    enabled = set(enabled_strategies)
    test_data = common.build_test_data_from_row(row)
    result = base_result(row, test_data)
    context = metric_context(test_data)

    try:
        if {"multi_max_profit", "multi_max_user"} & enabled:
            optimizer = MultiFunctionOptimizerSparse(test_data)
            min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan = optimizer.optimize_for_profit()
            if "multi_max_profit" in enabled:
                write_plan(result, "multi_max_profit", max_profit_plan, optimizer, context)
            if "multi_max_user" in enabled:
                write_plan(result, "multi_max_user", max_users_plan, optimizer, context)

        if "lmp" in enabled:
            optimizer = CurrentModuleGreedyMultiFunctionOptimizerSparse(test_data)
            _, max_profit_plan, _, _ = optimizer.optimize_for_profit()
            write_plan(result, "lmp", max_profit_plan, optimizer, context)

        if "brute" in enabled:
            optimizer = BruteForceSingleChainOptimizerSparse(test_data)
            _, max_profit_plan, _, _ = optimizer.optimize_for_profit()
            write_plan(result, "brute", max_profit_plan, optimizer, context)

        if "shortest_path" in enabled:
            optimizer = ShortestPathOptimizerSparse(test_data)
            write_plan(
                result,
                "shortest_path",
                shortest_path_plan(optimizer.shortest_path_deployment()),
                optimizer,
                context,
            )
    except Exception as exc:
        result["process_error"] = str(exc)
        traceback.print_exc()
    finally:
        try:
            topology = test_data.get("sparse_topology")
            if topology is not None:
                topology.clear_caches()
        except Exception:
            pass
        gc.collect()
    return result


def resolve_input_files(args: argparse.Namespace, run_dir: str) -> List[str]:
    default_input_dir = os.path.join(run_dir, "input")
    if args.input_dir:
        input_dir = resolve_abs_path(args.input_dir, common.project_root)
        if not input_dir or not os.path.isdir(input_dir):
            raise SystemExit(f"input_dir not found: {input_dir}")
        return common.list_csv_files(input_dir)
    if args.input:
        candidates = [args.input] if os.path.isabs(args.input) else [
            os.path.join(default_input_dir, args.input),
            os.path.join(common.project_root, args.input),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return [candidate]
        raise SystemExit("input file not found. Tried:\n  - " + "\n  - ".join(candidates))
    if not os.path.isdir(default_input_dir):
        raise SystemExit(f"input_dir not found: {default_input_dir}")
    return common.list_csv_files(default_input_dir)


def run_file(input_file: str, output_file: str, enabled: Set[str], limit: Optional[int], processes: int) -> None:
    df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
    if "test_data_id" not in df.columns:
        df.insert(0, "test_data_id", range(len(df)))
    if limit is not None:
        df = df.head(limit)
    records = df.to_dict("records")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    results: List[Dict[str, Any]] = []
    start = time.time()
    if processes <= 1:
        for record in tqdm(records, desc=os.path.basename(input_file)):
            results.append(process_test_case(record, enabled))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
            futures = [executor.submit(process_test_case, record, enabled) for record in records]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=os.path.basename(input_file)):
                results.append(future.result())
    common.append_results(results, output_file)
    elapsed = time.time() - start
    print(f"[OK] {len(results)} rows -> {output_file} ({elapsed:.2f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sparse placement strategies.")
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output", default="result_{input_name}.csv")
    parser.add_argument("--strategies", default=None, help="Comma-separated strategy list.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    enabled = parse_enabled_strategies(args.strategies)
    run_dir = resolve_run_dir(args.run_name, args.run_dir)
    output_dir = resolve_abs_path(args.output_dir, common.project_root) if args.output_dir else os.path.join(run_dir, DEFAULT_RESULT_SUBDIR)
    if output_dir is None:
        raise SystemExit("output_dir resolved to None")
    input_files = resolve_input_files(args, run_dir)
    print(f"[INFO] Enabled strategies: {sorted(enabled)}")
    for input_file in input_files:
        stem = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, args.output.replace("{input_name}", stem))
        run_file(input_file, output_file, enabled, args.limit, max(1, int(args.processes)))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
