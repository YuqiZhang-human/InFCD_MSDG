#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 CSV  topology_spec_json + /GPU
- neighbors: List[List[int]]
- bandwidth_dict_mb: Dict[(min,max), float]   # MB/s
- per_node_gpu: List[dict]

 dense
 adjacency
 bandwidth

 fat_tree
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


GBPS_TO_MBPS = 125.0
_GEN_MODULE = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_sparse_generator_module():
    """
     sparse  GPU_CATALOG /
    assign_gpus_by_degree / assign_gpus_by_ratio
    """
    global _GEN_MODULE
    if _GEN_MODULE is not None:
        return _GEN_MODULE

    root = _project_root()
    path = root / "data" / "src" / "generate" / "experiment_data_generator.py"
    if not path.is_file():
        raise FileNotFoundError(f"Sparse generator not found: {path}")

    spec = importlib.util.spec_from_file_location("exp1_gen_sparse_real_net", path)
    mod = importlib.util.module_from_spec(spec)

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _GEN_MODULE = mod
    return mod


def _clean_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def _parse_ratio_config(s: str) -> Dict[str, float]:
    """
    :
        "A6000_48GB:0.3,H100_256GB:0.7"
    """
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, r = part.split(":", 1)
        name = name.strip()
        out[name] = float(r.strip())
    return out


def _parse_ratio_config_json(raw_value: Any) -> Dict[str, float]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        src = raw_value
    else:
        s = _clean_str(raw_value)
        if not s:
            return {}
        try:
            src = json.loads(s)
        except Exception:
            return {}
    if not isinstance(src, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in src.items():
        name = _clean_str(k)
        if not name:
            continue
        out[name] = float(v)
    return out


def _format_ratio_detail(ratio_config: Dict[str, float]) -> str:
    if not ratio_config:
        return ""
    return ",".join(
        f"{name}:{ratio_config[name]}"
        for name in sorted(ratio_config.keys())
    )


def _format_gpu_set_display(ratio_config: Dict[str, float]) -> str:
    if not ratio_config:
        return ""
    return ",".join(sorted(ratio_config.keys()))


def _load_json_if_present(raw_value: Any):
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, dict)):
        return raw_value
    s = _clean_str(raw_value)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _build_neighbors_and_bandwidth_from_dense(
    adjacency: List[List[Any]],
    bandwidth_gbps_matrix: List[List[Any]],
) -> Tuple[List[List[int]], Dict[Tuple[int, int], float]]:
    node_count = len(adjacency)
    neighbors: List[List[int]] = [[] for _ in range(node_count)]
    bandwidth_dict_mb: Dict[Tuple[int, int], float] = {}

    for u in range(node_count):
        for v in range(u + 1, node_count):
            if int(adjacency[u][v]) == 0:
                continue
            neighbors[u].append(v)
            neighbors[v].append(u)
            bandwidth_dict_mb[_edge_key(u, v)] = float(bandwidth_gbps_matrix[u][v]) * GBPS_TO_MBPS

    return neighbors, bandwidth_dict_mb


def _infer_topo_meta_from_row(
    spec: Dict[str, Any],
    node_count: int,
    per_node_gpu: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gpu_nodes: List[int] = []
    switch_nodes: List[int] = []
    for i, gpu_info in enumerate(per_node_gpu):
        gpu_type = _clean_str(gpu_info.get("gpu_type", ""))
        g_tflops = float(gpu_info.get("G_TFLOPS", 0.0) or 0.0)
        if gpu_type and gpu_type.lower() != "none" and g_tflops > 0.0:
            gpu_nodes.append(i)
        else:
            switch_nodes.append(i)

    topo_type = _clean_str(spec.get("type", "")).lower()
    params = spec.get("params", {}) or {}

    if topo_type == "fat_tree":
        core_switch_count = int(params.get("core_switch_count", 0))
        num_pods = int(params.get("num_pods", 0))
        agg_per_pod = int(params.get("agg_per_pod", 0))
        edge_per_pod = int(params.get("edge_per_pod", 0))
        gpus_per_edge = int(params.get("gpus_per_edge", 0))
        agg_count = num_pods * agg_per_pod
        edge_count = num_pods * edge_per_pod
        gpu_count = num_pods * edge_per_pod * gpus_per_edge
        layer_sizes = [core_switch_count, agg_count, edge_count, gpu_count]
    elif topo_type == "clos":
        layer_sizes = params.get("layer_sizes", []) or params.get("layer_config", []) or [node_count]
    else:
        layer_sizes = [node_count]

    node_layers: List[int] = []
    current_layer = 0
    nodes_in_current_layer = 0
    for _ in range(node_count):
        if current_layer < len(layer_sizes) and nodes_in_current_layer >= int(layer_sizes[current_layer]):
            current_layer += 1
            nodes_in_current_layer = 0
        node_layers.append(min(current_layer, max(len(layer_sizes) - 1, 0)))
        nodes_in_current_layer += 1

    return {
        "topology_type": topo_type,
        "layer_sizes": layer_sizes,
        "node_layers": node_layers,
        "gpu_nodes": gpu_nodes,
        "switch_nodes": switch_nodes,
    }


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _add_undirected_edge(
    neighbors: List[List[int]],
    edge_list: List[Tuple[int, int]],
    u: int,
    v: int,
) -> None:
    if u == v:
        return
    neighbors[u].append(v)
    neighbors[v].append(u)
    edge_list.append(_edge_key(u, v))


def _build_fat_tree_sparse_from_spec(
    spec: Dict[str, Any]
) -> Tuple[int, List[List[int]], Dict[str, Any]]:
    """
     fat_tree  topo_meta


    [core][agg][edge][gpu]
    """
    params = spec.get("params", {}) or {}

    num_pods = int(params.get("num_pods", 0))
    core_switch_count = int(params.get("core_switch_count", 0))
    agg_per_pod = int(params.get("agg_per_pod", 0))
    edge_per_pod = int(params.get("edge_per_pod", 0))
    gpus_per_edge = int(params.get("gpus_per_edge", 0))

    if (
        num_pods <= 0
        or core_switch_count <= 0
        or agg_per_pod <= 0
        or edge_per_pod <= 0
        or gpus_per_edge <= 0
    ):
        raise ValueError(
            f"Invalid fat_tree params: {params}. "
            f"All of num_pods/core_switch_count/agg_per_pod/edge_per_pod/gpus_per_edge must be > 0."
        )

    agg_count = num_pods * agg_per_pod
    edge_count = num_pods * edge_per_pod
    gpu_count = num_pods * edge_per_pod * gpus_per_edge
    node_count = core_switch_count + agg_count + edge_count + gpu_count

    core_start = 0
    agg_start = core_start + core_switch_count
    edge_start = agg_start + agg_count
    gpu_start = edge_start + edge_count

    neighbors: List[List[int]] = [[] for _ in range(node_count)]

    tier_core_agg: List[Tuple[int, int]] = []
    tier_agg_edge: List[Tuple[int, int]] = []
    tier_edge_gpu: List[Tuple[int, int]] = []

    def agg_idx(pod: int, a: int) -> int:
        return agg_start + pod * agg_per_pod + a

    def edge_idx(pod: int, e: int) -> int:
        return edge_start + pod * edge_per_pod + e

    def gpu_idx(pod: int, e: int, g: int) -> int:
        return gpu_start + (pod * edge_per_pod + e) * gpus_per_edge + g

    # core <-> agg pod  agg  core
    for pod in range(num_pods):
        for a in range(agg_per_pod):
            a_node = agg_idx(pod, a)
            for c in range(core_switch_count):
                _add_undirected_edge(neighbors, tier_core_agg, c, a_node)

    # agg <-> edge pod
    for pod in range(num_pods):
        for a in range(agg_per_pod):
            a_node = agg_idx(pod, a)
            for e in range(edge_per_pod):
                e_node = edge_idx(pod, e)
                _add_undirected_edge(neighbors, tier_agg_edge, a_node, e_node)

    # edge <-> gpu
    for pod in range(num_pods):
        for e in range(edge_per_pod):
            e_node = edge_idx(pod, e)
            for g in range(gpus_per_edge):
                g_node = gpu_idx(pod, e, g)
                _add_undirected_edge(neighbors, tier_edge_gpu, e_node, g_node)

    node_layers = (
        [0] * core_switch_count
        + [1] * agg_count
        + [2] * edge_count
        + [3] * gpu_count
    )

    topo_meta: Dict[str, Any] = {
        "topology_type": "fat_tree",
        "layer_sizes": [core_switch_count, agg_count, edge_count, gpu_count],
        "node_layers": node_layers,
        "gpu_nodes": list(range(gpu_start, node_count)),
        "switch_nodes": list(range(0, gpu_start)),
        "edge_tiers": {
            "core_agg": tier_core_agg,
            "agg_edge": tier_agg_edge,
            "edge_gpu": tier_edge_gpu,
        },
    }

    return node_count, neighbors, topo_meta


def _build_fat_tree_bandwidth_dict_mb(
    topo_meta: Dict[str, Any],
    network_total_bandwidth_gbps: float,
) -> Dict[Tuple[int, int], float]:
    """

    fat-tree  B_total  tier
    tier
    """
    edge_tiers = topo_meta.get("edge_tiers", {}) or {}

    non_empty_tiers: List[List[Tuple[int, int]]] = [
        edges for _, edges in edge_tiers.items() if edges
    ]
    if not non_empty_tiers or network_total_bandwidth_gbps <= 0:
        return {}

    per_tier_total_gbps = float(network_total_bandwidth_gbps) / float(len(non_empty_tiers))
    out: Dict[Tuple[int, int], float] = {}

    for edges in non_empty_tiers:
        per_link_gbps = per_tier_total_gbps / float(len(edges))
        per_link_mb = per_link_gbps * GBPS_TO_MBPS
        for u, v in edges:
            out[_edge_key(u, v)] = per_link_mb

    return out


def build_real_net_topology_bundle_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
     CSV

    Returns:
        neighbors: List[List[int]]
        bandwidth_dict_mb: Dict[(min,max), float]  # MB/s
        per_node_gpu: List[dict]
        topo_meta: dict
        node_count: int
    """
    gen = _load_sparse_generator_module()

    raw_spec = row.get("topology_spec_json", "{}")
    if isinstance(raw_spec, dict):
        spec = dict(raw_spec)
    else:
        spec = json.loads(raw_spec or "{}")

    topo_type = _clean_str(spec.get("type", "")).lower()

    dense_adjacency = _load_json_if_present(row.get("adjacency_json"))
    dense_bandwidth = _load_json_if_present(row.get("bandwidth_json"))
    dense_per_node_gpu = _load_json_if_present(row.get("per_node_gpu_json"))

    if dense_adjacency and dense_bandwidth and dense_per_node_gpu:
        node_count = len(dense_adjacency)
        neighbors, bandwidth_dict_mb = _build_neighbors_and_bandwidth_from_dense(
            adjacency=dense_adjacency,
            bandwidth_gbps_matrix=dense_bandwidth,
        )
        topo_meta = _infer_topo_meta_from_row(
            spec=spec,
            node_count=node_count,
            per_node_gpu=dense_per_node_gpu,
        )
        return {
            "neighbors": neighbors,
            "bandwidth_dict_mb": bandwidth_dict_mb,
            "per_node_gpu": dense_per_node_gpu,
            "topo_meta": topo_meta,
            "node_count": node_count,
            "gpu_ratio_config": _parse_ratio_config_json(row.get("gpu_ratio_config_json", "")),
            "gpu_ratio_config_json": _clean_str(row.get("gpu_ratio_config_json", "")) or "{}",
            "gpu_ratio_detail": _clean_str(row.get("gpu_ratio_detail", "")),
            "gpu_set_display": _clean_str(row.get("gpu_set", "")),
        }

    if topo_type != "fat_tree":
        raise ValueError(
            f"Unsupported topology_type in sparse runtime builder: {topo_type}. "
            f"This file now supports fat_tree only."
        )

    network_total_bandwidth_gbps = float(row["network_total_bandwidth_gbps"])
    rank_order = _clean_str(row.get("gpu_rank_order", "desc")) or "desc"
    map_mode = _clean_str(row.get("gpu_map_mode", "high_to_high")) or "high_to_high"
    gpu_ratio_config = _parse_ratio_config_json(row.get("gpu_ratio_config_json", ""))
    gpu_ratio_detail = _clean_str(row.get("gpu_ratio_detail", ""))
    gpu_set_str = _clean_str(row.get("gpu_set", ""))

    node_count, neighbors, topo_meta = _build_fat_tree_sparse_from_spec(spec)
    bandwidth_dict_mb = _build_fat_tree_bandwidth_dict_mb(
        topo_meta=topo_meta,
        network_total_bandwidth_gbps=network_total_bandwidth_gbps,
    )

    degree_list = [len(vs) for vs in neighbors]
    gpu_nodes = topo_meta.get("gpu_nodes", list(range(node_count)))

    if not gpu_ratio_config and gpu_ratio_detail:
        # Backward-compatible fallback for older sparse CSVs.
        gpu_ratio_config = _parse_ratio_config(gpu_ratio_detail)

    if gpu_ratio_config:
        normalized_ratio_detail = _format_ratio_detail(gpu_ratio_config)
        gpu_set_display = _format_gpu_set_display(gpu_ratio_config)

        per_node_gpu = gen.assign_gpus_by_ratio(
            degree_list=degree_list,
            ratio_config=gpu_ratio_config,
            rank_order=rank_order,
            map_mode=map_mode,
            gpu_nodes=gpu_nodes,
        )
    else:
        gpu_names = [x.strip() for x in gpu_set_str.split(",") if x.strip()]
        if not gpu_names:
            raise ValueError("gpu_set is empty and not in ratio mode")

        per_node_gpu = gen.assign_gpus_by_degree(
            degree_list=degree_list,
            gpu_set_names=gpu_names,
            rank_order=rank_order,
            map_mode=map_mode,
            gpu_nodes=gpu_nodes,
        )
        normalized_ratio_detail = ""
        gpu_set_display = gpu_set_str

    return {
        "neighbors": neighbors,
        "bandwidth_dict_mb": bandwidth_dict_mb,
        "per_node_gpu": per_node_gpu,
        "topo_meta": topo_meta,
        "node_count": node_count,
        "gpu_ratio_config": gpu_ratio_config,
        "gpu_ratio_config_json": json.dumps(gpu_ratio_config, ensure_ascii=False, sort_keys=True),
        "gpu_ratio_detail": normalized_ratio_detail,
        "gpu_set_display": gpu_set_display,
    }
