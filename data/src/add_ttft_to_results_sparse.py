#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add TTFT and TPOT estimates to sparse experiment result CSV files.

This script follows the corrected serving model discussed for distributed LLM
inference:

* TTFT ends when the first output token is sampled and flushed. The first token
  is produced by the prefill logits, so single-token decode is not included in
  TTFT.
* TPOT is the steady-state per-output-token interval of the decode pipeline.
* Oversubscription is modeled as stage/module-level pipeline pressure. A later
  wave may enter an earlier GPU as soon as that stage is free, but it may wait
  again before downstream stages. The queue term is therefore produced by a
  pipeline recurrence, not by a request-level "wait until full generation
  completes" residence time.

For each chain and each load factor alpha:

    incoming_users = alpha * chain_capacity_users

Incoming users are split into admission waves of at most chain_capacity_users.
Capacity decides how many requests may enter the chain at once, while
prefill_microbatch_size controls the representative per-request prefill service
time. Prefill waves traverse the deployed module pipeline with this recurrence:

    start[w][i] = max(finish[w][i-1] + comm[w][i-1], finish[w-1][i])
    finish[w][i] = start[w][i] + prefill_compute[w][i]

The chain TTFT is the incoming-user-weighted mean over wave TTFT values:

    TTFT =
        request_network_ms
      + tokenize/api_ms
      + schedule_ms
      + prefill_pipeline_finish_ms
      + first_token_sample/detokenize/serialize_ms
      + response_network_ms

The chain TPOT is the decode pipeline interval under the active decode batch:

    TPOT =
        decode_schedule_ms
      + max(stage_decode_compute_ms, boundary_decode_comm_ms)
      + per-token sample/detokenize/stream_ms

The strategy-level values are weighted by incoming users across chains. Baseline
columns use load factor 1.0, and additional columns are emitted for the default
0.1x, 0.2x, ..., 2.0x oversubscription sweep.

Usage:

    python -m data.src.add_ttft_to_results_sparse ^
      --input_dir data/runs/TTFT_TPOT_analysis_experiment/input ^
      --result_dir data/runs/TTFT_TPOT_analysis_experiment/results ^
      --output_dir data/runs/TTFT_TPOT_analysis_experiment/results_ttft

Optional custom parameters:

    python -m data.src.add_ttft_to_results_sparse --params my_ttft_params.json ...

You can create a template parameter file with:

    python -m data.src.add_ttft_to_results_sparse --write_default_params ttft_params.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.real_net_topology_from_spec import build_real_net_topology_bundle_from_row
from src.tools.sparse_topology import SparseTopology


DEFAULT_PARAMS: Dict[str, Any] = {
    "prompt_tokens": None,
    "fallback_prompt_tokens": 1024,
    "prefill_gpu_efficiency": 0.9,
    "decode_gpu_efficiency": 0.9,
    "prefill_microbatch_size": 2.0,
    "decode_active_user_cap": None,
    "load_factors": [round(i / 10.0, 1) for i in range(1, 21)],
    "baseline_load_factor": 1.0,
    "incoming_user_basis": "capacity",
    "schedule_base_ms": 0.50,
    "schedule_ms_per_concurrent_user": 0.02,
    "decode_schedule_base_ms": 0.05,
    "decode_schedule_ms_per_concurrent_user": 0.001,
    "api_overhead_ms": 5.0,
    "tokenization_ms_per_1k_tokens": 4.0,
    "sample_first_token_ms": 0.20,
    "sample_decode_token_ms": 0.10,
    "detokenization_ms_per_token": 0.2,
    "serialize_first_token_ms": 0.10,
    "stream_token_ms": 0.05,
    "client_rtt_ms": 20.0,
    "client_bandwidth_MBps": 100.0,
    "prompt_text_bytes_per_token": 1.5,
    "first_token_bytes": 1.5,
    "stream_token_bytes": 1.5,
    "fabric_per_hop_latency_ms": 0.02,
    "network_effective_bandwidth_fraction": 1,
    "module_kernel_overhead_ms": 0.03,
    "empty_or_missing_strategy_ttft": np.nan,
    "empty_or_missing_strategy_tpot": np.nan,
    "include_chain_details_json": True,
    "include_load_scenario_details_json": False,
    "ttft_slo_ms": None,
    "tpot_slo_ms": None,
}


_NAT_SPLIT = re.compile(r"(\d+)")


def natural_key(path: Path) -> List[Any]:
    out: List[Any] = []
    for part in _NAT_SPLIT.split(path.name):
        out.append(int(part) if part.isdigit() else part.lower())
    return out


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def list_csv_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"],
        key=natural_key,
    )


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_params(path: Optional[str]) -> Dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    if not path:
        return params
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        custom = json.load(f)
    return deep_update(params, custom)


def parse_ms_value(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    s = str(value).strip().lower()
    if not s:
        return None
    if s.endswith("ms"):
        s = s[:-2].strip()
    return float(s)


def parse_param_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    lower = text.lower()
    if lower.endswith("ms"):
        return parse_ms_value(text)
    if lower in {"none", "null"}:
        return None
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return raw


def apply_param_overrides(params: Dict[str, Any], overrides: Optional[List[str]]) -> None:
    if not overrides:
        return
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set_param expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set_param has empty key: {item}")
        params[key] = parse_param_value(raw)


def slo_violation(value: Any, slo_ms: Optional[float]) -> float:
    if slo_ms is None:
        return float("nan")
    v = as_float(value, np.nan)
    if not math.isfinite(v):
        return float("nan")
    return float(v > slo_ms)


def add_slo_flags(out: Dict[str, Any], prefix: str, suffix: str, row: Dict[str, Any], params: Dict[str, Any]) -> None:
    ttft_slo = parse_ms_value(params.get("ttft_slo_ms"))
    tpot_slo = parse_ms_value(params.get("tpot_slo_ms"))
    ttft_bad = slo_violation(row.get("ttft_ms"), ttft_slo)
    tpot_bad = slo_violation(row.get("tpot_ms"), tpot_slo)
    out[f"{prefix}_ttft_slo_ms{suffix}"] = np.nan if ttft_slo is None else ttft_slo
    out[f"{prefix}_tpot_slo_ms{suffix}"] = np.nan if tpot_slo is None else tpot_slo
    out[f"{prefix}_ttft_slo_violation{suffix}"] = ttft_bad
    out[f"{prefix}_tpot_slo_violation{suffix}"] = tpot_bad
    if math.isfinite(ttft_bad) and math.isfinite(tpot_bad):
        slo_bad = float(bool(ttft_bad) or bool(tpot_bad))
        out[f"{prefix}_slo_violation{suffix}"] = slo_bad
        out[f"{prefix}_slo_attained{suffix}"] = float(not bool(slo_bad))
    else:
        out[f"{prefix}_slo_violation{suffix}"] = float("nan")
        out[f"{prefix}_slo_attained{suffix}"] = float("nan")


def write_default_params(path: str) -> None:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in DEFAULT_PARAMS.items()
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def parse_obj(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, (list, dict, tuple)):
        return value
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return default
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(s)
        except Exception:
            pass
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def strategy_prefixes_from_result_columns(columns: Iterable[str]) -> List[str]:
    suffix = "_deployment_list"
    prefixes = []
    for col in columns:
        if col.endswith(suffix) and not col.endswith("_edge_traffic_list"):
            prefixes.append(col[: -len(suffix)])
    # Longest first avoids max_profit owning max_profit_dynamic, but output sorted
    return sorted(set(prefixes))


def ensure_input_test_ids(input_df: pd.DataFrame) -> pd.DataFrame:
    out = input_df.copy()
    if "test_data_id" not in out.columns:
        out.insert(0, "test_data_id", range(len(out)))
    return out


def merge_input_row(result_row: pd.Series, input_lookup: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    test_id = int(result_row.get("test_data_id", 0))
    input_row = dict(input_lookup.get(test_id, {}))
    merged = input_row.copy()
    for key, value in result_row.items():
        if key not in merged or key in {
            "test_data_id",
            "topology_spec_json",
            "per_node_gpu_json",
            "gpu_ratio_config_json",
            "gpu_ratio_detail",
            "gpu_set",
        }:
            merged[key] = value
    return merged


def make_topology(row: Dict[str, Any]) -> SparseTopology:
    bundle = build_real_net_topology_bundle_from_row(row)
    return SparseTopology(
        node_count=int(bundle["node_count"]),
        neighbors=bundle["neighbors"],
        bandwidth_mb_per_edge=bundle["bandwidth_dict_mb"],
        source_cache_max=256,
    )


def gpu_info_by_node(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    raw = row.get("per_node_gpu_json", "[]")
    items = parse_obj(raw, [])
    out: Dict[int, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            node = int(item.get("node_index", len(out)))
            out[node] = item
    return out


def get_node_tflops(node: int, gpu_map: Dict[int, Dict[str, Any]], fallback: float) -> float:
    item = gpu_map.get(int(node), {})
    val = as_float(item.get("G_TFLOPS", fallback), fallback)
    return val if val > 0 else fallback


def prompt_tokens_for_row(row: Dict[str, Any], params: Dict[str, Any]) -> int:
    configured = params.get("prompt_tokens")
    if configured is not None:
        return max(1, int(configured))
    from_input = as_float(row.get("seq_len", np.nan), np.nan)
    if math.isfinite(from_input) and from_input > 0:
        return max(1, int(round(from_input)))
    return max(1, int(params["fallback_prompt_tokens"]))


def path_hops_and_bw(topology: SparseTopology, src: int, dst: int) -> Tuple[int, float]:
    if int(src) == int(dst):
        return 0, float("inf")
    path = topology.get_path_nodes(int(src), int(dst))
    if not path or len(path) < 2:
        return 0, 0.0
    bw = float("inf")
    for idx in range(len(path) - 1):
        bw = min(bw, topology.get_initial_bandwidth(path[idx], path[idx + 1]))
    if bw == float("inf"):
        bw = 0.0
    return len(path) - 1, bw


def boundary_mb_per_token(segment: Dict[str, Any], tokens_per_user: float) -> float:
    if tokens_per_user <= 0:
        return 0.0
    return as_float(segment.get("boundary_data_mb", 0.0), 0.0) / tokens_per_user


def load_factors_from_params(params: Dict[str, Any]) -> List[float]:
    raw = params.get("load_factors", DEFAULT_PARAMS["load_factors"])
    if isinstance(raw, str):
        items = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        items = list(raw or [])
    out: List[float] = []
    for item in items:
        try:
            value = float(item)
        except Exception:
            continue
        if value > 0:
            out.append(round(value, 6))
    return sorted(set(out)) or [1.0]


def load_tag(load_factor: float) -> str:
    return f"{int(round(float(load_factor) * 100)):03d}"


def positive_capacity(value: Any, fallback: float) -> float:
    cap = as_float(value, 0.0)
    if cap > 0:
        return cap
    return max(0.0, float(fallback))


def prefill_microbatch_limit(capacity_users: float, params: Dict[str, Any]) -> float:
    raw = params.get("prefill_microbatch_size", 1.0)
    if raw is None:
        return max(1e-9, float(capacity_users))
    limit = as_float(raw, 1.0)
    if limit <= 0:
        return max(1e-9, float(capacity_users))
    return max(1e-9, min(float(capacity_users), limit))


def split_prefill_waves(
    incoming_users: float,
    capacity_users: float,
    params: Dict[str, Any],
) -> List[float]:
    incoming = max(0.0, float(incoming_users))
    capacity = max(1e-9, float(capacity_users))
    waves: List[float] = []
    remaining = incoming
    while remaining > 1e-9:
        batch = min(capacity, remaining)
        waves.append(batch)
        remaining -= batch
    return waves


def prefill_service_batch_users(wave_users: float, capacity_users: float, params: Dict[str, Any]) -> float:
    if wave_users <= 0:
        return 0.0
    return min(max(1.0, float(wave_users)), prefill_microbatch_limit(capacity_users, params))


def active_decode_users(incoming_users: float, capacity_users: float, params: Dict[str, Any]) -> float:
    active = min(max(0.0, float(incoming_users)), max(0.0, float(capacity_users)))
    cap = params.get("decode_active_user_cap")
    if cap is not None:
        active = min(active, max(0.0, as_float(cap, active)))
    return active


def incoming_users_for_load(
    served_users: float,
    capacity_users: float,
    load_factor: float,
    params: Dict[str, Any],
) -> float:
    basis = str(params.get("incoming_user_basis", "capacity")).strip().lower()
    if basis in {"served", "served_users", "actual"}:
        base = max(0.0, float(served_users))
    elif basis in {"capacity", "capacity_users", "cap"}:
        base = max(0.0, float(capacity_users))
    else:
        raise ValueError(
            "incoming_user_basis must be one of: capacity, served"
        )
    return max(0.0, float(load_factor) * base)


def compute_client_network_components(
    prompt_tokens: int,
    params: Dict[str, Any],
) -> Tuple[float, float]:
    prompt_mb = (
        prompt_tokens * float(params["prompt_text_bytes_per_token"]) / (1024.0 * 1024.0)
    )
    first_token_mb = float(params["first_token_bytes"]) / (1024.0 * 1024.0)
    client_bw = max(1e-12, float(params["client_bandwidth_MBps"]))
    half_rtt_ms = float(params["client_rtt_ms"]) / 2.0
    request_ms = half_rtt_ms + prompt_mb / client_bw * 1000.0
    response_ms = half_rtt_ms + first_token_mb / client_bw * 1000.0
    return request_ms, response_ms


def compute_api_tokenization_ms(prompt_tokens: int, params: Dict[str, Any]) -> float:
    return (
        float(params["api_overhead_ms"])
        + float(params["tokenization_ms_per_1k_tokens"]) * prompt_tokens / 1000.0
    )


def first_token_postprocess_ms(params: Dict[str, Any]) -> float:
    return (
        float(params["sample_first_token_ms"])
        + float(params["detokenization_ms_per_token"])
        + float(params["serialize_first_token_ms"])
    )


def decode_token_postprocess_ms(params: Dict[str, Any]) -> float:
    stream_token_mb = float(params["stream_token_bytes"]) / (1024.0 * 1024.0)
    client_bw = max(1e-12, float(params["client_bandwidth_MBps"]))
    stream_network_ms = stream_token_mb / client_bw * 1000.0
    return (
        float(params["sample_decode_token_ms"])
        + float(params["detokenization_ms_per_token"])
        + float(params["stream_token_ms"])
        + stream_network_ms
    )


def schedule_ms(incoming_users: float, params: Dict[str, Any]) -> float:
    return (
        float(params["schedule_base_ms"])
        + float(params["schedule_ms_per_concurrent_user"]) * max(0.0, incoming_users)
    )


def decode_schedule_ms(active_users: float, params: Dict[str, Any]) -> float:
    return (
        float(params["decode_schedule_base_ms"])
        + float(params["decode_schedule_ms_per_concurrent_user"]) * max(0.0, active_users)
    )


def get_decode_efficiency(params: Dict[str, Any]) -> float:
    if "decode_gpu_efficiency" in params:
        return float(params["decode_gpu_efficiency"])
    return float(params.get("first_decode_gpu_efficiency", 0.20))


def build_pipeline_stages(
    deployment: List[int],
    segments: List[Any],
    topology: SparseTopology,
    tokens_per_user: float,
) -> List[Dict[str, Any]]:
    """
    Consecutive modules placed on the same node are coalesced into one physical
    GPU stage. This prevents a single GPU from being treated as multiple
    simultaneously available pipeline stages.
    """
    n = min(len(deployment), len(segments))
    if n <= 0:
        return []

    stages: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for module_idx in range(n):
        node = int(deployment[module_idx])
        seg = segments[module_idx] if isinstance(segments[module_idx], dict) else {}
        tflops_per_token = as_float(seg.get("compute_tflops_per_token", 0.0), 0.0)

        if current is None or int(current["node"]) != node:
            if current is not None:
                stages.append(current)
            current = {
                "stage_index": len(stages),
                "node": node,
                "module_indices": [],
                "compute_tflops_per_token": 0.0,
                "module_count": 0,
                "outgoing_boundary_idx": None,
                "outgoing_hops": 0,
                "outgoing_bw_MBps": 0.0,
                "outgoing_mb_per_token": 0.0,
            }

        current["module_indices"].append(module_idx)
        current["compute_tflops_per_token"] += tflops_per_token
        current["module_count"] += 1

        next_node = int(deployment[module_idx + 1]) if module_idx + 1 < n else None
        if next_node is not None and next_node != node:
            hops, bw = path_hops_and_bw(topology, node, next_node)
            current["outgoing_boundary_idx"] = module_idx
            current["outgoing_hops"] = hops
            current["outgoing_bw_MBps"] = bw
            current["outgoing_mb_per_token"] = boundary_mb_per_token(seg, tokens_per_user)

    if current is not None:
        stages.append(current)

    for idx, stage in enumerate(stages):
        stage["stage_index"] = idx
    return stages


def stage_compute_ms(
    stage: Dict[str, Any],
    batch_users: float,
    prompt_tokens: int,
    node_tflops: float,
    efficiency: float,
    module_overhead_ms: float,
) -> Tuple[float, float]:
    work_tflops = (
        float(stage["compute_tflops_per_token"]) * max(0.0, batch_users) * prompt_tokens
    )
    if node_tflops <= 0 or efficiency <= 0:
        return 0.0, work_tflops
    overhead = module_overhead_ms * int(stage["module_count"]) if batch_users > 0 else 0.0
    return (work_tflops / (node_tflops * efficiency)) * 1000.0 + overhead, work_tflops


def stage_comm_ms(
    stage: Dict[str, Any],
    batch_users: float,
    token_count: int,
    params: Dict[str, Any],
) -> Tuple[float, float]:
    hops = int(stage.get("outgoing_hops", 0))
    bw = float(stage.get("outgoing_bw_MBps", 0.0))
    mb_per_token = float(stage.get("outgoing_mb_per_token", 0.0))
    payload_mb = mb_per_token * max(0.0, batch_users) * token_count
    if hops <= 0 or bw <= 0 or payload_mb <= 0:
        return 0.0, payload_mb
    effective_bw = bw * float(params["network_effective_bandwidth_fraction"])
    if effective_bw <= 0:
        return 0.0, payload_mb
    return (
        (payload_mb / effective_bw) * 1000.0
        + hops * float(params["fabric_per_hop_latency_ms"])
    ), payload_mb


def simulate_prefill_pipeline(
    stages: List[Dict[str, Any]],
    wave_users: List[float],
    prompt_tokens: int,
    gpu_map: Dict[int, Dict[str, Any]],
    avg_tflops: float,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if not stages or not wave_users:
        return {
            "prefill_pipeline_ms": float("nan"),
            "pipeline_queue_ms": float("nan"),
            "prefill_compute_ms": float("nan"),
            "inter_stage_network_prefill_ms": float("nan"),
            "inter_stage_prefill_payload_mb": float("nan"),
            "prefill_tflops": float("nan"),
            "prefill_wave_details": [],
        }

    stage_available = [0.0 for _ in stages]
    waves: List[Dict[str, Any]] = []
    total_users = sum(wave_users)
    module_overhead = float(params["module_kernel_overhead_ms"])
    prefill_efficiency = float(params["prefill_gpu_efficiency"])

    for wave_idx, batch in enumerate(wave_users):
        service_batch = prefill_service_batch_users(batch, batch, params)
        arrival_ms = 0.0
        no_queue_path_ms = 0.0
        wave_compute_ms = 0.0
        wave_comm_ms = 0.0
        wave_payload_mb = 0.0
        wave_tflops = 0.0
        stage_rows: List[Dict[str, Any]] = []

        for stage_idx, stage in enumerate(stages):
            node = int(stage["node"])
            node_tflops = get_node_tflops(node, gpu_map, avg_tflops)
            compute_ms, work_tflops = stage_compute_ms(
                stage,
                service_batch,
                prompt_tokens,
                node_tflops,
                prefill_efficiency,
                module_overhead,
            )
            start_ms = max(arrival_ms, stage_available[stage_idx])
            finish_ms = start_ms + compute_ms
            comm_ms, payload_mb = stage_comm_ms(stage, service_batch, prompt_tokens, params)

            stage_available[stage_idx] = finish_ms
            arrival_ms = finish_ms + comm_ms
            no_queue_path_ms += compute_ms + comm_ms
            wave_compute_ms += compute_ms
            wave_comm_ms += comm_ms
            wave_payload_mb += payload_mb
            wave_tflops += work_tflops

            stage_rows.append({
                "stage_index": int(stage_idx),
                "node": node,
                "module_indices": list(stage["module_indices"]),
                "batch_users": float(batch),
                "service_batch_users": float(service_batch),
                "start_ms": float(start_ms),
                "finish_ms": float(finish_ms),
                "compute_ms": float(compute_ms),
                "outgoing_comm_ms": float(comm_ms),
                "outgoing_payload_mb": float(payload_mb),
            })

        pipeline_finish_ms = stage_available[-1]
        queue_ms = max(0.0, pipeline_finish_ms - no_queue_path_ms)
        waves.append({
            "wave_index": int(wave_idx),
            "wave_users": float(batch),
            "service_batch_users": float(service_batch),
            "prefill_pipeline_ms": float(pipeline_finish_ms),
            "pipeline_queue_ms": float(queue_ms),
            "prefill_no_queue_path_ms": float(no_queue_path_ms),
            "prefill_compute_ms": float(wave_compute_ms),
            "inter_stage_network_prefill_ms": float(wave_comm_ms),
            "inter_stage_prefill_payload_mb": float(wave_payload_mb),
            "prefill_tflops": float(wave_tflops),
            "stage_details": stage_rows,
        })

    def wmean(key: str) -> float:
        if total_users <= 0:
            return float("nan")
        return float(sum(w[key] * w["wave_users"] for w in waves) / total_users)

    return {
        "prefill_pipeline_ms": wmean("prefill_pipeline_ms"),
        "pipeline_queue_ms": wmean("pipeline_queue_ms"),
        "prefill_compute_ms": wmean("prefill_compute_ms"),
        "inter_stage_network_prefill_ms": wmean("inter_stage_network_prefill_ms"),
        "inter_stage_prefill_payload_mb": wmean("inter_stage_prefill_payload_mb"),
        "prefill_tflops": wmean("prefill_tflops"),
        "prefill_wave_details": waves,
    }


def compute_decode_wave(
    stages: List[Dict[str, Any]],
    batch_users: float,
    gpu_map: Dict[int, Dict[str, Any]],
    avg_tflops: float,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    module_overhead = float(params["module_kernel_overhead_ms"])
    decode_efficiency = get_decode_efficiency(params)
    compute_stage_ms: List[float] = []
    comm_stage_ms: List[float] = []
    stage_rows: List[Dict[str, Any]] = []
    total_compute_ms = 0.0
    total_comm_ms = 0.0
    total_payload_mb = 0.0
    total_tflops = 0.0

    for stage_idx, stage in enumerate(stages):
        node = int(stage["node"])
        node_tflops = get_node_tflops(node, gpu_map, avg_tflops)
        compute_ms, work_tflops = stage_compute_ms(
            stage,
            batch_users,
            1,
            node_tflops,
            decode_efficiency,
            module_overhead,
        )
        comm_ms, payload_mb = stage_comm_ms(stage, batch_users, 1, params)
        compute_stage_ms.append(compute_ms)
        comm_stage_ms.append(comm_ms)
        total_compute_ms += compute_ms
        total_comm_ms += comm_ms
        total_payload_mb += payload_mb
        total_tflops += work_tflops
        stage_rows.append({
            "stage_index": int(stage_idx),
            "node": node,
            "module_indices": list(stage["module_indices"]),
            "batch_users": float(batch_users),
            "decode_compute_ms": float(compute_ms),
            "decode_outgoing_comm_ms": float(comm_ms),
            "decode_outgoing_payload_mb": float(payload_mb),
        })

    pipeline_interval_ms = max(compute_stage_ms + comm_stage_ms + [0.0])
    sched_ms = decode_schedule_ms(batch_users, params)
    post_ms = decode_token_postprocess_ms(params)
    tpot_ms = sched_ms + pipeline_interval_ms + post_ms
    return {
        "batch_users": float(batch_users),
        "tpot_ms": float(tpot_ms),
        "decode_schedule_ms": float(sched_ms),
        "decode_pipeline_interval_ms": float(pipeline_interval_ms),
        "decode_compute_ms": float(total_compute_ms),
        "inter_stage_network_decode_ms": float(total_comm_ms),
        "inter_stage_decode_payload_mb": float(total_payload_mb),
        "decode_tflops": float(total_tflops),
        "decode_token_postprocess_ms": float(post_ms),
        "decode_stage_details": stage_rows,
    }


def weighted_mean_rows(rows: List[Dict[str, Any]], key: str, weight_key: str) -> float:
    total_weight = sum(as_float(row.get(weight_key, 0.0), 0.0) for row in rows)
    if total_weight <= 0:
        return float("nan")
    return float(
        sum(as_float(row.get(key, np.nan), np.nan) * as_float(row.get(weight_key, 0.0), 0.0)
            for row in rows)
        / total_weight
    )


def weighted_percentile(values: List[float], weights: List[float], percentile: float) -> float:
    pairs = sorted(
        [(float(v), float(w)) for v, w in zip(values, weights) if w > 0 and math.isfinite(v)],
        key=lambda x: x[0],
    )
    if not pairs:
        return float("nan")
    total = sum(w for _, w in pairs)
    threshold = total * percentile
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= threshold:
            return value
    return pairs[-1][0]


def compute_chain_latency_for_load(
    row: Dict[str, Any],
    chain_idx: int,
    deployment: List[int],
    served_users: float,
    capacity_users: float,
    load_factor: float,
    topology: SparseTopology,
    gpu_map: Dict[int, Dict[str, Any]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    segments = parse_obj(row.get("segments_detail_json"), [])
    if not isinstance(segments, list):
        segments = []
    tokens_per_user = as_float(row.get("tokens_per_user", 0.0), 0.0)
    prompt_tokens = prompt_tokens_for_row(row, params)
    capacity = positive_capacity(capacity_users, served_users)
    incoming_users = incoming_users_for_load(
        served_users,
        capacity,
        load_factor,
        params,
    )
    waves = split_prefill_waves(incoming_users, capacity, params)
    decode_active_users = active_decode_users(incoming_users, capacity, params)
    stages = build_pipeline_stages(deployment, segments, topology, tokens_per_user)

    avg_tflops = as_float(row.get("computation_ability", 1.0), 1.0)
    prefill = simulate_prefill_pipeline(
        stages,
        waves,
        prompt_tokens,
        gpu_map,
        avg_tflops,
        params,
    )
    decode_rows = []
    if stages and decode_active_users > 0:
        decode_rows.append(
            compute_decode_wave(stages, decode_active_users, gpu_map, avg_tflops, params)
        )

    request_network_ms, response_network_ms = compute_client_network_components(
        prompt_tokens,
        params,
    )
    tokenize_ms = compute_api_tokenization_ms(prompt_tokens, params)
    sched_ms = schedule_ms(incoming_users, params)
    first_post_ms = first_token_postprocess_ms(params)
    client_network_ms = request_network_ms + response_network_ms

    ttft_ms = (
        request_network_ms
        + tokenize_ms
        + sched_ms
        + as_float(prefill["prefill_pipeline_ms"], 0.0)
        + first_post_ms
        + response_network_ms
    )

    total_prefill_wave_users = sum(waves)
    if decode_rows:
        decode_row = decode_rows[0]
        tpot_ms = as_float(decode_row.get("tpot_ms"), np.nan)
        decode_schedule_avg = as_float(decode_row.get("decode_schedule_ms"), np.nan)
        decode_interval_avg = as_float(decode_row.get("decode_pipeline_interval_ms"), np.nan)
        decode_compute_avg = as_float(decode_row.get("decode_compute_ms"), np.nan)
        decode_comm_avg = as_float(decode_row.get("inter_stage_network_decode_ms"), np.nan)
        decode_payload_avg = as_float(decode_row.get("inter_stage_decode_payload_mb"), np.nan)
        decode_tflops_avg = as_float(decode_row.get("decode_tflops"), np.nan)
        decode_post_avg = as_float(decode_row.get("decode_token_postprocess_ms"), np.nan)
    else:
        tpot_ms = float("nan")
        decode_schedule_avg = float("nan")
        decode_interval_avg = float("nan")
        decode_compute_avg = float("nan")
        decode_comm_avg = float("nan")
        decode_payload_avg = float("nan")
        decode_tflops_avg = float("nan")
        decode_post_avg = float("nan")

    p95_wave_ttft = weighted_percentile(
        [
            request_network_ms
            + tokenize_ms
            + sched_ms
            + as_float(w.get("prefill_pipeline_ms"), 0.0)
            + first_post_ms
            + response_network_ms
            for w in prefill["prefill_wave_details"]
        ],
        [as_float(w.get("wave_users"), 0.0) for w in prefill["prefill_wave_details"]],
        0.95,
    )

    return {
        "chain_index": int(chain_idx),
        "served_users": float(served_users),
        "capacity_users": float(capacity),
        "load_factor": float(load_factor),
        "incoming_users": float(incoming_users),
        "decode_active_users": float(decode_active_users),
        "prefill_microbatch_size": float(prefill_microbatch_limit(capacity, params)),
        "wave_count": int(len(waves)),
        "avg_wave_users": float(total_prefill_wave_users / len(waves)) if waves else 0.0,
        "max_wave_users": float(max(waves)) if waves else 0.0,
        "ttft_ms": float(ttft_ms),
        "ttft_p95_ms": float(p95_wave_ttft),
        "tpot_ms": float(tpot_ms),
        "client_network_ms": float(client_network_ms),
        "request_network_ms": float(request_network_ms),
        "response_network_ms": float(response_network_ms),
        "api_tokenization_ms": float(tokenize_ms),
        "schedule_ms": float(sched_ms),
        "first_token_postprocess_ms": float(first_post_ms),
        "queue_ms": float(prefill["pipeline_queue_ms"]),
        "pipeline_queue_ms": float(prefill["pipeline_queue_ms"]),
        "prefill_pipeline_ms": float(prefill["prefill_pipeline_ms"]),
        "prefill_compute_ms": float(prefill["prefill_compute_ms"]),
        "inter_stage_network_prefill_ms": float(prefill["inter_stage_network_prefill_ms"]),
        "inter_stage_prefill_payload_mb": float(prefill["inter_stage_prefill_payload_mb"]),
        "prefill_tflops": float(prefill["prefill_tflops"]),
        "decode_schedule_ms": float(decode_schedule_avg),
        "decode_pipeline_interval_ms": float(decode_interval_avg),
        "decode_compute_ms": float(decode_compute_avg),
        "inter_stage_network_decode_ms": float(decode_comm_avg),
        "inter_stage_decode_payload_mb": float(decode_payload_avg),
        "decode_tflops": float(decode_tflops_avg),
        "decode_token_postprocess_ms": float(decode_post_avg),
        "stage_count": int(len(stages)),
        "stage_nodes": [int(s["node"]) for s in stages],
        "stage_module_indices": [list(s["module_indices"]) for s in stages],
        "prefill_wave_details": prefill["prefill_wave_details"],
        "decode_wave_details": decode_rows,
    }


def add_strategy_ttft_tpot(
    result_row: pd.Series,
    merged_row: Dict[str, Any],
    prefix: str,
    topology: SparseTopology,
    gpu_map: Dict[int, Dict[str, Any]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    deployment_list = parse_obj(result_row.get(f"{prefix}_deployment_list"), [])
    served_users_list = parse_obj(result_row.get(f"{prefix}_chain_served_users_list"), [])
    capacity_users_list = parse_obj(result_row.get(f"{prefix}_chain_capacity_users_list"), [])
    if not deployment_list or not served_users_list:
        return {}
    if not isinstance(deployment_list, list) or not isinstance(served_users_list, list):
        return {}

    out: Dict[str, Any] = {}
    scenario_details: List[Dict[str, Any]] = []
    load_factors = load_factors_from_params(params)
    baseline_load = float(params.get("baseline_load_factor", 1.0))
    include_chain_details = bool(params.get("include_chain_details_json", True))
    include_load_details = bool(params.get("include_load_scenario_details_json", False))

    scalar_keys = [
        "ttft_ms",
        "tpot_ms",
        "client_network_ms",
        "request_network_ms",
        "response_network_ms",
        "api_tokenization_ms",
        "schedule_ms",
        "first_token_postprocess_ms",
        "queue_ms",
        "pipeline_queue_ms",
        "prefill_pipeline_ms",
        "prefill_compute_ms",
        "inter_stage_network_prefill_ms",
        "inter_stage_prefill_payload_mb",
        "prefill_tflops",
        "decode_schedule_ms",
        "decode_pipeline_interval_ms",
        "decode_compute_ms",
        "inter_stage_network_decode_ms",
        "inter_stage_decode_payload_mb",
        "decode_tflops",
        "decode_token_postprocess_ms",
        "incoming_users",
        "decode_active_users",
        "capacity_users",
        "prefill_microbatch_size",
        "wave_count",
        "avg_wave_users",
        "max_wave_users",
        "stage_count",
    ]

    baseline_rows: List[Dict[str, Any]] = []
    for load_factor in load_factors:
        chain_rows: List[Dict[str, Any]] = []
        for idx, deployment in enumerate(deployment_list):
            if not isinstance(deployment, list):
                continue
            served = as_float(served_users_list[idx], 0.0) if idx < len(served_users_list) else 0.0
            cap_raw = (
                capacity_users_list[idx]
                if isinstance(capacity_users_list, list) and idx < len(capacity_users_list)
                else served
            )
            capacity = positive_capacity(cap_raw, served)
            if capacity <= 0 and served <= 0:
                continue
            chain_rows.append(
                compute_chain_latency_for_load(
                    merged_row,
                    idx,
                    [int(x) for x in deployment],
                    served,
                    capacity,
                    float(load_factor),
                    topology,
                    gpu_map,
                    params,
                )
            )

        if not chain_rows:
            continue

        weights = [row["incoming_users"] for row in chain_rows]
        total_weight = sum(weights)

        def wmean(key: str) -> float:
            if total_weight <= 0:
                return float("nan")
            return float(sum(row[key] * row["incoming_users"] for row in chain_rows) / total_weight)

        scenario: Dict[str, Any] = {
            "load_factor": float(load_factor),
            "chain_count": len(chain_rows),
            "total_incoming_users": float(total_weight),
        }
        for key in scalar_keys:
            if key in chain_rows[0]:
                scenario[key] = wmean(key)
        scenario["ttft_p95_ms"] = weighted_percentile(
            [row["ttft_p95_ms"] for row in chain_rows],
            weights,
            0.95,
        )
        scenario["ttft_max_ms"] = max(row["ttft_ms"] for row in chain_rows)
        scenario["tpot_p95_ms"] = weighted_percentile(
            [row["tpot_ms"] for row in chain_rows],
            weights,
            0.95,
        )
        scenario["tpot_max_ms"] = max(row["tpot_ms"] for row in chain_rows)
        scenario_for_export = dict(scenario)
        if include_load_details:
            scenario_for_export["chain_details"] = chain_rows
        scenario_details.append(scenario_for_export)

        tag = load_tag(load_factor)
        for key in scalar_keys + ["ttft_p95_ms", "ttft_max_ms", "tpot_p95_ms", "tpot_max_ms"]:
            if key in scenario:
                out[f"{prefix}_{key}_load_{tag}"] = scenario[key]
        out[f"{prefix}_chain_count_load_{tag}"] = len(chain_rows)
        out[f"{prefix}_total_incoming_users_load_{tag}"] = float(total_weight)
        add_slo_flags(out, prefix, f"_load_{tag}", scenario, params)

        if abs(float(load_factor) - baseline_load) < 1e-9:
            baseline_rows = chain_rows
            for key in scalar_keys:
                if key in scenario:
                    out[f"{prefix}_{key}"] = scenario[key]
            out[f"{prefix}_ttft_p95_ms"] = scenario["ttft_p95_ms"]
            out[f"{prefix}_ttft_max_ms"] = scenario["ttft_max_ms"]
            out[f"{prefix}_tpot_p95_ms"] = scenario["tpot_p95_ms"]
            out[f"{prefix}_tpot_max_ms"] = scenario["tpot_max_ms"]
            out[f"{prefix}_ttft_chain_count"] = len(chain_rows)
            out[f"{prefix}_tpot_chain_count"] = len(chain_rows)
            add_slo_flags(out, prefix, "", scenario, params)
            if include_chain_details:
                out[f"{prefix}_ttft_chain_details_json"] = json.dumps(chain_rows, ensure_ascii=False)
                out[f"{prefix}_tpot_chain_details_json"] = json.dumps(chain_rows, ensure_ascii=False)

    if not out:
        return {}

    if not baseline_rows and scenario_details:
        nearest = min(
            scenario_details,
            key=lambda item: abs(float(item["load_factor"]) - baseline_load),
        )
        for key in scalar_keys:
            if key in nearest:
                out[f"{prefix}_{key}"] = nearest[key]
        out[f"{prefix}_ttft_p95_ms"] = nearest["ttft_p95_ms"]
        out[f"{prefix}_ttft_max_ms"] = nearest["ttft_max_ms"]
        out[f"{prefix}_tpot_p95_ms"] = nearest["tpot_p95_ms"]
        out[f"{prefix}_tpot_max_ms"] = nearest["tpot_max_ms"]
        out[f"{prefix}_ttft_chain_count"] = int(nearest["chain_count"])
        out[f"{prefix}_tpot_chain_count"] = int(nearest["chain_count"])
        add_slo_flags(out, prefix, "", nearest, params)

    out[f"{prefix}_ttft_prompt_tokens"] = int(prompt_tokens_for_row(merged_row, params))
    if include_load_details:
        out[f"{prefix}_ttft_tpot_load_scenarios_json"] = json.dumps(
            scenario_details,
            ensure_ascii=False,
        )
    return out


def process_file(
    result_file: Path,
    input_lookup: Dict[int, Dict[str, Any]],
    output_file: Path,
    params: Dict[str, Any],
) -> None:
    df = pd.read_csv(result_file, encoding="utf-8-sig", low_memory=False)
    prefixes = strategy_prefixes_from_result_columns(df.columns)
    if not prefixes:
        raise RuntimeError(f"No strategy deployment columns found in {result_file}")

    additions: List[Dict[str, Any]] = []
    input_context_rows: List[Dict[str, Any]] = []
    topology_cache: Dict[int, SparseTopology] = {}
    gpu_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}
    existing_cols = set(df.columns)

    for _, row in df.iterrows():
        test_id = int(row.get("test_data_id", 0))
        input_context = {
            key: value
            for key, value in input_lookup.get(test_id, {}).items()
            if key not in existing_cols
        }
        input_context_rows.append(input_context)

        merged = merge_input_row(row, input_lookup)
        if test_id not in topology_cache:
            topology_cache[test_id] = make_topology(merged)
            gpu_cache[test_id] = gpu_info_by_node(merged)
        topology = topology_cache[test_id]
        gpu_map = gpu_cache[test_id]

        out: Dict[str, Any] = {}
        for prefix in prefixes:
            out.update(add_strategy_ttft_tpot(row, merged, prefix, topology, gpu_map, params))
        additions.append(out)

    input_context_df = pd.DataFrame(input_context_rows)
    add_df = pd.DataFrame(additions)
    out_df = pd.concat(
        [
            df.reset_index(drop=True),
            input_context_df.reset_index(drop=True),
            add_df.reset_index(drop=True),
        ],
        axis=1,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Single generated input CSV")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of generated input CSVs")
    parser.add_argument("--result", type=str, default=None, help="Single result CSV")
    parser.add_argument("--result_dir", type=str, default=None, help="Directory of result CSVs")
    parser.add_argument("--output", type=str, default=None, help="Single output CSV")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for result copies")
    parser.add_argument("--params", type=str, default=None, help="Optional TTFT/TPOT parameter JSON")
    parser.add_argument("--TTFT_SLO", type=str, default=None, help="TTFT SLO threshold in ms, e.g. 450ms")
    parser.add_argument("--TPOT_SLO", type=str, default=None, help="TPOT SLO threshold in ms, e.g. 40ms")
    parser.add_argument(
        "--set_param",
        action="append",
        default=None,
        help="Override one TTFT/TPOT model parameter as key=value. May be repeated.",
    )
    parser.add_argument(
        "--load_factors",
        type=str,
        default=None,
        help="Optional comma-separated load factors, e.g. 0.5,1.0,1.5,2.0",
    )
    parser.add_argument("--write_default_params", type=str, default=None)
    args = parser.parse_args()

    if args.write_default_params:
        write_default_params(args.write_default_params)
        print(f"[OK] wrote default params to {resolve_path(args.write_default_params)}")
        return

    if not (args.input or args.input_dir):
        raise SystemExit("--input or --input_dir is required")
    if not (args.result or args.result_dir):
        raise SystemExit("--result or --result_dir is required")

    params = load_params(args.params)
    if args.load_factors:
        params["load_factors"] = [float(x.strip()) for x in args.load_factors.split(",") if x.strip()]
    if args.TTFT_SLO is not None:
        params["ttft_slo_ms"] = parse_ms_value(args.TTFT_SLO)
    if args.TPOT_SLO is not None:
        params["tpot_slo_ms"] = parse_ms_value(args.TPOT_SLO)
    apply_param_overrides(params, args.set_param)

    input_paths = list_csv_files(resolve_path(args.input or args.input_dir))
    input_dfs = []
    for path in input_paths:
        input_dfs.append(ensure_input_test_ids(pd.read_csv(path, encoding="utf-8-sig", low_memory=False)))
    input_df = pd.concat(input_dfs, ignore_index=True)
    input_lookup = {
        int(row["test_data_id"]): dict(row)
        for _, row in input_df.iterrows()
    }

    result_paths = list_csv_files(resolve_path(args.result or args.result_dir))
    if args.output and len(result_paths) != 1:
        raise SystemExit("--output can only be used with a single --result file")

    if args.output:
        output_paths = [resolve_path(args.output)]
    else:
        if not args.output_dir:
            raise SystemExit("--output_dir is required when processing a result directory")
        out_dir = resolve_path(args.output_dir)
        output_paths = [out_dir / p.name for p in result_paths]

    for result_file, output_file in zip(result_paths, output_paths):
        print(f"[INFO] processing {result_file}")
        process_file(result_file, input_lookup, output_file, params)
        print(f"[OK] wrote {output_file}")


if __name__ == "__main__":
    main()
