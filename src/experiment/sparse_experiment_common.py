#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared data conversion and result-writing helpers for sparse experiments."""

import os
import sys
import time
import gc
import json
import traceback
import argparse
import multiprocessing
import concurrent.futures
from collections import deque

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.tools.real_net_topology_from_spec import build_real_net_topology_bundle_from_row
from src.tools.sparse_topology import SparseTopology

LARGE_GRAPH_NODE_THRESHOLD = 5000

PATH_SOURCE_CACHE_MAX = 256

OPTIMIZER_CANDIDATE_LIMIT_GLOBAL = 96
OPTIMIZER_CANDIDATE_LIMIT_POD = 48
OPTIMIZER_CANDIDATE_LIMIT_EDGE = 12

SP_BEAM_WIDTH = 32

MAX_INFLIGHT_FACTOR = 2
CHECKPOINT_FLUSH_INTERVAL = 1

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# /src/experiment/test_experiment1.py
current_dir = os.path.dirname(os.path.abspath(__file__))    # .../src/experiment
src_dir = os.path.dirname(current_dir)                      # .../src
project_root = os.path.dirname(src_dir)                     # .../

algorithm_dir = os.path.join(src_dir, 'algorithm')          # /src/algorithm

sys.path.append(src_dir)
sys.path.append(algorithm_dir)
sys.path.append(project_root)

DEFAULT_RUN_NAME = "experiment5"
DEFAULT_RUNS_DIR = os.path.join(project_root, "data", "runs")

def resolve_abs_path(path: str, base_dir: str) -> str:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def resolve_run_dir(run_name: str, run_dir: str) -> str:
    if run_dir:
        return resolve_abs_path(run_dir, project_root)
    return os.path.join(DEFAULT_RUNS_DIR, run_name)


def list_csv_files(input_dir: str):
    files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".csv"):
            files.append(os.path.join(input_dir, filename))
    return sorted(files)

# /data/analysis/table

def _degree_stats_from_neighbors(neighbors: List[List[int]]) -> Tuple[float, float, float]:
    deg = np.array([len(vs) for vs in neighbors], dtype=float)
    if deg.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.std(deg)), float(np.min(deg)), float(np.max(deg))


def _sample_shortest_path_stats(
    neighbors: List[List[int]],
    node_count: int,
    max_sources: int = 32,
) -> Tuple[float, float, float]:
    """
    10w

    """
    return 0.0, 0.0, 0.0


def _dense_network_matrices_from_sparse(
    node_count: int,
    neighbors: List[List[int]],
    bandwidth_dict_mb: Dict,
) -> Tuple[List[List[float]], List[List[float]]]:
    bandwidth_matrix = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    distance_matrix = [[0.0 if i == j else float("inf") for j in range(node_count)] for i in range(node_count)]

    for u in range(node_count):
        for v in neighbors[u]:
            bw = bandwidth_dict_mb.get((u, v), bandwidth_dict_mb.get((v, u), 0.0))
            bandwidth_matrix[u][v] = float(bw)
            distance_matrix[u][v] = 1.0

    for src in range(node_count):
        dist = distance_matrix[src]
        q = deque([src])
        while q:
            u = q.popleft()
            next_dist = dist[u] + 1.0
            for v in neighbors[u]:
                if next_dist < dist[v]:
                    dist[v] = next_dist
                    q.append(v)
        for dst in range(node_count):
            if not np.isfinite(dist[dst]):
                dist[dst] = 0.0

    return bandwidth_matrix, distance_matrix


def _link_bandwidth_stats_from_dict(
    bandwidth_dict_mb: Dict[Tuple[int, int], float]
) -> Tuple[float, float, float, float]:
    """
     Gbps
    """
    vals_gbps: List[float] = []
    seen = set()

    for (u, v), bw_mb in bandwidth_dict_mb.items():
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        vals_gbps.append(float(bw_mb) / 125.0)

    if not vals_gbps:
        return 0.0, 0.0, 0.0, 0.0

    arr = np.array(vals_gbps, dtype=float)
    return (
        float(np.mean(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.var(arr)),
    )


# ----------------------------------------------------------------------
#  CSV  test_data
# ----------------------------------------------------------------------
def build_test_data_from_row(row):
    """
     CSV  test_data
    10w
    -  sparse
    -
    -  fat_tree / sparse
    """
    test_id = int(row.get("test_data_id", 0))
    tokens_per_user = float(row["tokens_per_user"])
    network_total_bandwidth_gbps = float(row["network_total_bandwidth_gbps"])
    link_price_per_gbps_month = float(row["link_price_per_gbps_month"])
    user_price_per_month = float(row["user_price_per_month"])

    bundle = build_real_net_topology_bundle_from_row(row)
    node_count = int(bundle["node_count"])
    neighbors = bundle["neighbors"]
    bandwidth_dict_mb = bundle["bandwidth_dict_mb"]
    per_node_gpu = bundle["per_node_gpu"]
    topo_meta = bundle.get("topo_meta", {}) or {}
    gpu_ratio_config_json = bundle.get("gpu_ratio_config_json", "{}")
    gpu_ratio_detail = bundle.get("gpu_ratio_detail", "")
    gpu_set_display = bundle.get("gpu_set_display", row.get("gpu_set", ""))

    segments_layers = json.loads(row["segments_layers_json"])
    segments_detail = json.loads(row["segments_detail_json"])
    segments_summary = json.loads(row["segments_summary_json"])

    module_count = len(segments_detail)

    degree_std, degree_min, degree_max = _degree_stats_from_neighbors(neighbors)
    avg_shortest_path_len, shortest_path_len_std, shortest_path_len_max = 0.0, 0.0, 0.0
    link_bandwidth_mean, link_bandwidth_min, link_bandwidth_max, link_bandwidth_var = \
        _link_bandwidth_stats_from_dict(bandwidth_dict_mb)

    compute_utilization_factor = 0.4
    computation_capacity = []
    node_costs = []
    total_compute_cap = 0.0
    total_mem_cap = 0.0

    for gpu_info in per_node_gpu:
        g_tflops = float(gpu_info["G_TFLOPS"])
        vram_bytes = float(gpu_info["VRAM_bytes"])
        cost_per_gb_month = float(gpu_info["cost_per_GB_month"])

        compute_cap = g_tflops * compute_utilization_factor
        mem_cap_gb = vram_bytes / (1024.0 ** 3)

        computation_capacity.append([compute_cap, mem_cap_gb])
        node_costs.append([0.0, cost_per_gb_month])

        total_compute_cap += compute_cap
        total_mem_cap += mem_cap_gb

    avg_compute_ability = total_compute_cap / node_count if node_count > 0 else 0.0
    avg_memory_ability = total_mem_cap / node_count if node_count > 0 else 0.0

    resource_demands = []
    weight_memory_demands = []
    data_sizes = []

    for idx, seg in enumerate(segments_detail):
        comp_tok = float(seg.get("compute_tflops_per_token", 0.0))
        comp_user = seg.get("compute_tflops_per_user_per_sec", None)
        if comp_user is None:
            comp_user = seg.get("compute_tflops_per_user", None)
        if comp_user is None:
            comp_user = comp_tok * tokens_per_user
        comp_user = float(comp_user)

        kv_gb = seg.get("kv_gb", None)
        if kv_gb is None:
            kv_gb = seg.get("memory_gb", 0.0)
        kv_gb = float(kv_gb)

        w_gb = float(seg.get("weights_gb", 0.0))

        resource_demands.append([comp_user, kv_gb])
        weight_memory_demands.append(w_gb)

        if idx < module_count - 1:
            boundary_mb = float(seg.get("boundary_data_mb", 0.0))
            data_sizes.append(boundary_mb)

    total_weights_gb = float(sum(weight_memory_demands))
    kv_per_user_gb = float(sum(x[1] for x in resource_demands))

    bandwidth_cost = link_price_per_gbps_month / 125.0
    profit_per_user = user_price_per_month

    memory_cost = (
        sum(cost[1] for cost in node_costs) / len(node_costs)
        if node_costs else 0.0
    )

    degrees = [len(vs) for vs in neighbors]
    topology_degree = float(sum(degrees) / len(degrees)) if degrees else 0.0

    topology_type = str(row.get("topology_type", "")).lower().replace("-", "_")
    if not topology_type:
        topo_spec = json.loads(row.get("topology_spec_json", "{}"))
        topology_type = str(topo_spec.get("type", "")).lower().replace("-", "_")

    topology_params = json.loads(row.get("topology_spec_json", "{}")).get("params", {}) or {}

    layer_sizes = topo_meta.get("layer_sizes", [node_count])
    node_layers = topo_meta.get("node_layers", [0] * node_count)
    gpu_nodes = topo_meta.get("gpu_nodes", [])
    switch_nodes = topo_meta.get("switch_nodes", [])

    gateway_uplink_nodes = []
    gateway_downlink_nodes = []

    if topology_type == "fat_tree":
        core_switch_count = int(topology_params.get("core_switch_count", 0))
        gateway_uplink_nodes = list(range(core_switch_count))
        gateway_downlink_nodes = list(range(core_switch_count))
    else:
        raise ValueError("10w  fat_tree ")

    user_uplink_mb_per_user = float(row.get("user_uplink_mb_per_user", 0.0))
    user_downlink_mb_per_user = float(row.get("user_downlink_mb_per_user", 0.0))
    return_to_gateway_mb_per_user = float(row.get("return_to_gateway_mb_per_user", 0.0))

    sparse_topology = SparseTopology(
        node_count=node_count,
        neighbors=neighbors,
        bandwidth_mb_per_edge=bandwidth_dict_mb,
        source_cache_max=PATH_SOURCE_CACHE_MAX,
    )
    bandwidth_matrix, distance_matrix = _dense_network_matrices_from_sparse(
        node_count,
        neighbors,
        bandwidth_dict_mb,
    )

    gateway_node_list = sorted(set(gateway_uplink_nodes + gateway_downlink_nodes))
    gateway_node = int(gateway_node_list[0]) if gateway_node_list else -1

    fat_tree_num_pods = int(topology_params.get("num_pods", 0))
    fat_tree_core_switch_count = int(topology_params.get("core_switch_count", 0))
    fat_tree_agg_per_pod = int(topology_params.get("agg_per_pod", 0))
    fat_tree_edge_per_pod = int(topology_params.get("edge_per_pod", 0))
    fat_tree_gpus_per_edge = int(topology_params.get("gpus_per_edge", 0))
    fat_tree_gpu_start = int(sum(layer_sizes[:3])) if len(layer_sizes) >= 4 else -1

    test_data = {
        "test_data_id": test_id,
        "topology_name": row.get("topology_name", ""),
        "node_count": node_count,
        "module_count": module_count,
        "model_name": row.get("model_name", ""),
        "G_groups": int(row.get("G_groups", len(segments_layers))),
        "K_segments": int(row.get("K_segments", module_count)),
        "partition_index": int(row.get("partition_index", 0)),
        "gpu_map_mode": row.get("gpu_map_mode", ""),
        "gpu_set": gpu_set_display,
        "gpu_ratio_detail": gpu_ratio_detail,
        "gpu_ratio_config_json": gpu_ratio_config_json,

        "model_num_layers": int(row.get("model_num_layers", 0)),
        "model_total_params": float(row.get("model_total_params", 0.0)),
        "model_hidden_size": int(row.get("model_hidden_size", 0)),
        "model_num_heads": int(row.get("model_num_heads", 0)),
        "model_num_kv_heads": int(row.get("model_num_kv_heads", 0)),

        "model_size": float(row.get("model_total_params", 0.0)),
        "network_total_bandwidth_gbps": network_total_bandwidth_gbps,
        "topology_degree": topology_degree,
        "computation_ability": avg_compute_ability,
        "memory_ability": avg_memory_ability,

        "degree_std": degree_std,
        "degree_min": degree_min,
        "degree_max": degree_max,
        "avg_shortest_path_len": avg_shortest_path_len,
        "shortest_path_len_std": shortest_path_len_std,
        "shortest_path_len_max": shortest_path_len_max,
        "link_bandwidth_mean": link_bandwidth_mean,
        "link_bandwidth_min": link_bandwidth_min,
        "link_bandwidth_max": link_bandwidth_max,
        "link_bandwidth_var": link_bandwidth_var,
        "link_avg_bandwidth": link_bandwidth_mean,
        "link_bandwidth_var_gbps": link_bandwidth_var,

        "computation_capacity": computation_capacity,
        "resource_demands": resource_demands,
        "weight_memory_demands": weight_memory_demands,
        "data_sizes": data_sizes,
        "total_weights_gb": total_weights_gb,
        "kv_per_user_gb": kv_per_user_gb,

        "gpu_cost": 0.0,
        "memory_cost": memory_cost,
        "bandwidth_cost": bandwidth_cost,
        "profit_per_user": profit_per_user,
        "tokens_per_user": tokens_per_user,
        "target_total_users": int(row.get("target_total_users", 0) or 0),
        "node_costs": node_costs,
        "bandwidth_matrix": bandwidth_matrix,
        "distance_matrix": distance_matrix,

        "topology_type": topology_type,
        "topology_params": topology_params,
        "layer_sizes": layer_sizes,
        "node_layers": node_layers,

        "gpu_nodes": gpu_nodes,
        "switch_nodes": switch_nodes,
        "per_node_gpu": per_node_gpu,
        "per_node_gpu_json": json.dumps(per_node_gpu, ensure_ascii=False),

        "gateway_node": gateway_node,
        "gateway_node_list": gateway_node_list,
        "gateway_uplink_nodes": gateway_uplink_nodes,
        "gateway_downlink_nodes": gateway_downlink_nodes,

        "user_uplink_mb_per_user": user_uplink_mb_per_user,
        "user_downlink_mb_per_user": user_downlink_mb_per_user,
        "return_to_gateway_mb_per_user": return_to_gateway_mb_per_user,

        "fat_tree_num_pods": fat_tree_num_pods,
        "fat_tree_core_switch_count": fat_tree_core_switch_count,
        "fat_tree_agg_per_pod": fat_tree_agg_per_pod,
        "fat_tree_edge_per_pod": fat_tree_edge_per_pod,
        "fat_tree_gpus_per_edge": fat_tree_gpus_per_edge,
        "fat_tree_gpu_start": fat_tree_gpu_start,

        "optimizer_candidate_limit_global": OPTIMIZER_CANDIDATE_LIMIT_GLOBAL,
        "optimizer_candidate_limit_pod": OPTIMIZER_CANDIDATE_LIMIT_POD,
        "optimizer_candidate_limit_edge": OPTIMIZER_CANDIDATE_LIMIT_EDGE,
        "sp_beam_width": SP_BEAM_WIDTH,

        "sparse_topology": sparse_topology,
    }

    return test_data

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def _json_dumps_safe(obj):
    if isinstance(obj, dict):
        obj = {str(k): v for k, v in obj.items()}
    return json.dumps(obj, ensure_ascii=False)

def _safe_div(numerator, denominator):
    """<=0  0.0"""
    try:
        denominator = float(denominator)
        if denominator <= 0:
            return 0.0
        return float(numerator) / denominator
    except Exception :
        return 0.0
def _summarize_edge_traffic_list(edge_traffic_list):
    """

        : [
            {
                "from_node": ...,
                "to_node": ...,
                "used_bandwidth": ...,
                "initial_bandwidth": ...
            },
            ...
        ]
         result
    """
    if not edge_traffic_list:
        return {
            "edge_traffic_total_bw": 0.0,
            "edge_traffic_edge_count": 0,
            "edge_traffic_avg_bw": 0.0,
            "edge_traffic_max_bw": 0.0,
            "edge_traffic_min_bw": 0.0,
            "edge_traffic_std_bw": 0.0,
            "edge_traffic_total_util": 0.0,
            "edge_traffic_avg_util": 0.0,
            "edge_traffic_max_util": 0.0,
        }
    used_bw_list = []
    util_list = []

    for e in edge_traffic_list:
        used_bw = float(e.get("used_bandwidth", 0.0) or 0.0)
        init_bw = e.get("initial_bandwidth", None)

        if used_bw > 0:
            used_bw_list.append(used_bw)

        if init_bw is not None:
            try:
                init_bw = float(init_bw)
                if init_bw > 0 and used_bw > 0:
                    util_list.append(used_bw / init_bw)
            except Exception:
                pass
    if not used_bw_list:
        return {
            "edge_traffic_total_bw": 0.0,
            "edge_traffic_edge_count": 0,
            "edge_traffic_avg_bw": 0.0,
            "edge_traffic_max_bw": 0.0,
            "edge_traffic_min_bw": 0.0,
            "edge_traffic_std_bw": 0.0,
            "edge_traffic_total_util": 0.0,
            "edge_traffic_avg_util": 0.0,
            "edge_traffic_max_util": 0.0,
        }
    return {
        "edge_traffic_total_bw": float(np.sum(used_bw_list)),
        "edge_traffic_edge_count": int(len(used_bw_list)),
        "edge_traffic_avg_bw": float(np.mean(used_bw_list)),
        "edge_traffic_max_bw": float(np.max(used_bw_list)),
        "edge_traffic_min_bw": float(np.min(used_bw_list)),
        "edge_traffic_std_bw": float(np.std(used_bw_list)),
        "edge_traffic_total_util": float(np.sum(util_list)) if util_list else 0.0,
        "edge_traffic_avg_util": float(np.mean(util_list)) if util_list else 0.0,
        "edge_traffic_max_util": float(np.max(util_list)) if util_list else 0.0,
    }

def _summarize_external_traffic_list(external_traffic_list):
    """

        : [
            {
                "from_node": ...,
                "to_node": ...,
                "used_bandwidth": ...,
                "initial_bandwidth": ...,
                "traffic_type": "external_uplink" / "external_downlink" / ...
            },
            ...
        ]
    """
    if not external_traffic_list:
        return {
            "external_traffic_total_bw": 0.0,
            "external_traffic_edge_count": 0,
            "external_traffic_avg_bw": 0.0,
            "external_traffic_max_bw": 0.0,
            "external_uplink_bw": 0.0,
            "external_downlink_bw": 0.0,
        }

    used_bw_list = []
    uplink_bw = 0.0
    downlink_bw = 0.0

    for e in external_traffic_list:
        used_bw = float(e.get("used_bandwidth", 0.0) or 0.0)
        traffic_type = str(e.get("traffic_type", "") or "")

        if used_bw > 0:
            used_bw_list.append(used_bw)

        if traffic_type == "external_uplink":
            uplink_bw += used_bw
        elif traffic_type == "external_downlink":
            downlink_bw += used_bw

    if not used_bw_list:
        return {
            "external_traffic_total_bw": 0.0,
            "external_traffic_edge_count": 0,
            "external_traffic_avg_bw": 0.0,
            "external_traffic_max_bw": 0.0,
            "external_uplink_bw": 0.0,
            "external_downlink_bw": 0.0,
        }

    return {
        "external_traffic_total_bw": float(np.sum(used_bw_list)),
        "external_traffic_edge_count": int(len(used_bw_list)),
        "external_traffic_avg_bw": float(np.mean(used_bw_list)),
        "external_traffic_max_bw": float(np.max(used_bw_list)),
        "external_uplink_bw": float(uplink_bw),
        "external_downlink_bw": float(downlink_bw),
    }

def _write_plan_derived_metrics(result: dict, prefix: str, total_cost, total_profit, total_users,
                                edge_traffic_list, external_traffic_list):
    """

    1)  /
    2) /
    3) /
    4)
    """
    total_cost = float(total_cost)
    total_profit = float(total_profit)
    total_users = float(total_users)

    edge_stats = _summarize_edge_traffic_list(edge_traffic_list)
    ext_stats = _summarize_external_traffic_list(external_traffic_list)

    result[f"{prefix}_avg_cost_per_user"] = _safe_div(total_cost, total_users)
    result[f"{prefix}_avg_profit_per_user"] = _safe_div(total_profit, total_users)

    for k, v in edge_stats.items():
        result[f"{prefix}_{k}"] = v

    for k, v in ext_stats.items():
        result[f"{prefix}_{k}"] = v

    edge_total_bw = edge_stats["edge_traffic_total_bw"]
    ext_total_bw = ext_stats["external_traffic_total_bw"]

    result[f"{prefix}_edge_traffic_per_user"] = _safe_div(edge_total_bw, total_users)
    result[f"{prefix}_external_traffic_per_user"] = _safe_div(ext_total_bw, total_users)
    result[f"{prefix}_internal_external_bw_ratio"] = _safe_div(edge_total_bw, ext_total_bw)

def _write_aggregated_node_memory_metrics(
        result: dict,
        prefix: str,
        optimizer,
        deployment_list,
        served_users_list
):
    """

    """
    if deployment_list and served_users_list:
        node_memory_used_gb, node_memory_util = optimizer.calculate_aggregated_node_memory_stats(
            deployment_list,
            served_users_list
        )
        result[f'{prefix}_node_memory_used_gb_json'] = _json_dumps_safe(node_memory_used_gb)
        result[f'{prefix}_node_memory_util_json'] = _json_dumps_safe(node_memory_util)
    else:
        result[f'{prefix}_node_memory_used_gb_json'] = _json_dumps_safe({})
        result[f'{prefix}_node_memory_util_json'] = _json_dumps_safe({})

def _write_plan_result_fields(
        result: dict,
        prefix: str,
        plan,
        optimizer,
        global_compute_per_user: float,
        total_compute_capacity: float,
        total_weights_gb: float,
        kv_per_user_gb: float,
        total_memory_capacity: float,
):
    if plan is None:
        result[f'{prefix}_error'] = "no feasible plan"
        return

    result[f'{prefix}_error'] = ""
    result[f'{prefix}_cost'] = plan[0]
    result[f'{prefix}_deploy_cost'] = plan[1]
    result[f'{prefix}_comm_cost'] = plan[2]
    result[f'{prefix}_profit'] = plan[3]
    result[f'{prefix}_users'] = plan[4]
    result[f'{prefix}_nodes'] = plan[5]
    result[f'{prefix}_avg_modules'] = plan[6]
    result[f'{prefix}_chain_count'] = plan[7]
    result[f'{prefix}_chain_len_list'] = json.dumps(plan[8], ensure_ascii=False)
    result[f'{prefix}_chain_avg_modules_list'] = json.dumps(plan[9], ensure_ascii=False)
    result[f'{prefix}_chain_used_nodes_list'] = json.dumps(plan[10], ensure_ascii=False)
    result[f'{prefix}_chain_capacity_users_list'] = _json_dumps_safe(plan[11])
    result[f'{prefix}_chain_served_users_list'] = _json_dumps_safe(plan[12])
    result[f'{prefix}_deployment_list'] = _json_dumps_safe(plan[13])
    result[f'{prefix}_edge_traffic_list'] = _json_dumps_safe(plan[14])
    result[f'{prefix}_external_traffic_list'] = _json_dumps_safe(
        plan[15] if len(plan) > 15 else []
    )

    _write_plan_derived_metrics(
        result=result,
        prefix=prefix,
        total_cost=plan[0],
        total_profit=plan[3],
        total_users=plan[4],
        edge_traffic_list=plan[14],
        external_traffic_list=plan[15] if len(plan) > 15 else [],
    )
    _write_aggregated_node_memory_metrics(
        result=result,
        prefix=prefix,
        optimizer=optimizer,
        deployment_list=plan[13],
        served_users_list=plan[12],
    )

    total_users = plan[4]
    used_nodes = plan[5]
    chain_count = plan[7]
    result[f'{prefix}_avg_users_per_chain'] = (
        float(total_users) / chain_count if chain_count > 0 else 0.0
    )
    result[f'{prefix}_avg_nodes_per_chain'] = (
        float(used_nodes) / chain_count if chain_count > 0 else 0.0
    )
    result[f'{prefix}_compute_util'] = (
        (global_compute_per_user * total_users) / total_compute_capacity
        if total_compute_capacity > 0 and global_compute_per_user > 0
        else 0.0
    )
    result[f'{prefix}_memory_util'] = (
        (total_weights_gb + kv_per_user_gb * total_users) / total_memory_capacity
        if total_memory_capacity > 0 and kv_per_user_gb > 0
        else 0.0
    )

    timing = plan[16] if len(plan) > 16 else []
    result[f'{prefix}_chain_time_list'] = json.dumps(timing, ensure_ascii=False)
    result[f'{prefix}_avg_chain_time'] = plan[17] if len(plan) > 17 else 0.0
    result[f'{prefix}_total_deploy_time'] = plan[18] if len(plan) > 18 else 0.0

    if isinstance(timing, (list, tuple)) and len(timing) >= 2:
        result[f'{prefix}_multi_time'] = float(timing[0])


def append_results(new_results, output_file):
    if not new_results:
        return
    df_new = pd.DataFrame(new_results)
    file_exists = os.path.exists(output_file)
    df_new.to_csv(
        output_file,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig",
    )


def save_checkpoint_ids(processed_ids, checkpoint_file):
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(sorted(int(x) for x in processed_ids), f, ensure_ascii=False)


def load_checkpoint(checkpoint_file):
    """ID"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                processed_ids = json.load(f)
            print(f" {len(processed_ids)} ")
            return processed_ids
        except Exception as e:
            print(f": {e}")
    return []


# ----------------------------------------------------------------------
