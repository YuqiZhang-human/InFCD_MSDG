

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
experiment1_data_generator_new_B_total_Stable_BFS_real_net_sparse.py

 **** `experiment1_data_generator_new_B_total_Stable_BFS_real_net.py`
 **** `adjacency_json` / `distance_json` / `bandwidth_json` / `per_node_gpu_json`
 `topology_spec_json`

():

1. ""();
2.  GPU "",, & ;
3.  model_name  G, L  G ,
    G ""( K  G );
4. ,:
   -  token (TFLOPs/token)
   - ( + KV)(GB)
   - (boundary_data_mb,:MB/s/,Activation + KV)
5. :
   -  topology_name
   -  network_total_bandwidth_gbps
   -  link_price_per_gbps_month
   - GPU  gpu_set
   - GPU  map_mode(/)
   -  K
   -  seq_len
   -  pricing_profile: {user_price_per_month, tokens_per_user}
     - user_price_per_month: ($/month)
     - tokens_per_user: (token/sec)
6.  model_name  G ,,
7.  +  CSV
8. fat-treeclos
9. ,,

" CONFIG",,
"""

import os
import sys
import argparse
import json
import time
import itertools
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict

import numpy as np
import pandas as pd

# Add project root to Python path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SRC_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.dirname(DATA_SRC_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.tools.topology_generator_with_switches_sparse import TopologyGeneratorWithSwitches
from src.tools.gpu_distribution_generator import GPUDistributionGenerator

# ============================================================
# ============================================================
# Path variables already defined above for imports
DEFAULT_RUN_NAME = "optimal_solutions_compare_module"
DEFAULT_RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "runs")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_RUNS_DIR, DEFAULT_RUN_NAME, "input")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_FILE_PREFIX = "input"

# ============================================================
# ()
# ============================================================

# 1.
# (--run_name/--run_dir/--output_dir/--chunk_size)
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
BATCH_SIZE = DEFAULT_CHUNK_SIZE
FILE_PREFIX = DEFAULT_FILE_PREFIX


# 3. GPU (TFLOPs /  / )
GPU_CATALOG: Dict[str, Dict[str, Any]] = {
    "V100": {
        "name": "V100",
        "G_TFLOPS": 14,  # FP32 CUDA Core(V100,FP32)
        "VRAM_bytes": int(16 * 1024**3),
        "cost_per_GB_month": 14.31
    },
    "5090_32GB": {
        "name": "5090_32GB",
        "G_TFLOPS": 109.7,  # Turing FP16(8x speedup)
        "VRAM_bytes": int(32 * 1024**3),
        "cost_per_GB_month": 14.96
    },
    "A6000_48GB": {
        "name": "A6000_48GB",
        "G_TFLOPS": 38.71,  # Ada FP16 Tensor Core
        "VRAM_bytes": int(48 * 1024**3),
        "cost_per_GB_month": 8.52
    },
    "A40": {
        "name": "A40",
        "G_TFLOPS": 250.0,  # Ampere FP16 Tensor Core
        "VRAM_bytes": int(48 * 1024**3),
        "cost_per_GB_month": 6.63
    },
    "A100_80GB": {
        "name": "A100_80GB",
        "G_TFLOPS": 19.5,  # Ampere FP16 Tensor Core
        "VRAM_bytes": int(80 * 1024**3),
        "cost_per_GB_month": 10.62
    },
    "H100_256GB": {
        "name": "H100_256GB",
        "G_TFLOPS": 183.0,  # Hopper FP16 Tensor Core(PCIe)
        "VRAM_bytes": int(256* 1024**3),
        "cost_per_GB_month": 8.20
    },
    "Pro6000_96GB": {
        "name": "Pro6000_96GB",
        "G_TFLOPS": 126.0,  # Hopper FP16 Tensor Core
        "VRAM_bytes": int(96 * 1024**3),
        "cost_per_GB_month": 4.99
    }
}

# 4. GPU (rank_order ,)
GPU_ASSIGNMENT_CONFIG: Dict[str, Any] = {
    # :desc = ;asc =
    "rank_order": "desc",
}

# 5. (),
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "Llama-3.1-13B-Instruct": {
        "name": "Llama-3.1-13B-Instruct",
        "num_layers": 56,
        "total_params": float(13e9),
        "bytes_per_param": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "flops_per_token_per_layer": float(24 * (4096 ** 2)),  # ~402M FLOPs//token
        "activation_mb_per_boundary": 32.0,
    },
    "Llama-3.1-70B-Instruct": {
        "name": "Llama-3.1-70B-Instruct",
        "num_layers": 80,
        "total_params": float(70e9),
        "bytes_per_param": 2, # KV
        "hidden_size": 8192, # KV
        "num_attention_heads": 64, # KV
        "num_key_value_heads": 8, # KV
        "flops_per_token_per_layer": float(24 * (8192 ** 2)),  # ~1.61G FLOPs//token
        "activation_mb_per_boundary": 64.0,
    },
    "Qwen2.5-7B": {
        "name": "Qwen2.5-7B",
        "num_layers": 28,
        "total_params": float(7e9),
        "bytes_per_param": 2,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "flops_per_token_per_layer": float(24 * (3584 ** 2)),  # ~402M FLOPs//token
        "activation_mb_per_boundary": 28,
    },
    "Mistral-7B": {
      "name": "Mistral-7B",
      "num_layers": 32,
      "total_params": float(7e9),
      "bytes_per_param": 2,
      "hidden_size": 4096,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "flops_per_token_per_layer": float(24 * (4096 ** 2)),  # ~402M FLOPs//token
      "activation_mb_per_boundary": 32.0,
    },
    "Gemma-2-27B": {
        "name": "Gemma-2-27B",
        "num_layers": 46,
        "total_params": float(27e9),
        "bytes_per_param": 2,
        "hidden_size": 4608,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "flops_per_token_per_layer": float(24 * (4608 ** 2)),   # ~1.01G FLOPs//token
        "activation_mb_per_boundary": 36.0,
    },
    "DeepSeek-V2": {
      "name": "DeepSeek-V2",
      "num_layers": 60,
      "total_params": float(236e9),
      "bytes_per_param": 2,
      "hidden_size": 5120,
      "num_attention_heads": 128,
      "num_key_value_heads": 128,
      "flops_per_token_per_layer": float(24 * (5120 ** 2)),  # ~1.58G FLOPs//token
      "activation_mb_per_boundary": 40.0,
  }
}

# 6. (KV / Activation ),
RUN_CONFIG: Dict[str, Any] = {
    "batch_size": 2,   #
    # tokens_per_user , pricing_profiles
    # seq_len , GENERATION_CONFIG["seq_len_list"]
}

# 7. ()
GENERATION_CONFIG: Dict[str, Any] = {
    "seq_len_list": [2048],

    # (1):  topology_spec  network_profiles
    # Keep every edge->GPU link at the 8.333333 Gbps implied by the
    # p4/c2/a2/e2/g2 baseline: B_total = 25 * num_pods * edge_per_pod * gpus_per_edge.
    # Unit price decreases mildly with capacity: 40 * (B_total / 400) ** -0.1.
    # Total monthly cost still increases with capacity, proportional to B_total ** 0.9.
    "topology_specs": [
        {"type": "fat_tree", "params": {"num_pods": 2,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 2},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 200,  "link_price_per_gbps_month": 42.9}]},
        #{"type": "fat_tree", "params": {"num_pods": 3,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 2},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 300,  "link_price_per_gbps_month": 41.2}]},
        #{"type": "fat_tree", "params": {"num_pods": 4,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 2},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 400,  "link_price_per_gbps_month": 40.0}]},
        #{"type": "fat_tree", "params": {"num_pods": 5,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 2},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 500,  "link_price_per_gbps_month": 39.1}]},
        #{"type": "fat_tree", "params": {"num_pods": 6,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 2},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 600,  "link_price_per_gbps_month": 38.4}]},
        #{"type": "fat_tree", "params": {"num_pods": 7,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 2},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 700,  "link_price_per_gbps_month": 37.8}]},

        #{"type": "fat_tree", "params": {"num_pods": 2,   "core_switch_count": 3,   "agg_per_pod": 3,  "edge_per_pod": 3,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 450,  "link_price_per_gbps_month": 39.5}]},
        #{"type": "fat_tree", "params": {"num_pods": 3,   "core_switch_count": 3,   "agg_per_pod": 3,  "edge_per_pod": 3,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 675,  "link_price_per_gbps_month": 38.0}]},
        #{"type": "fat_tree", "params": {"num_pods": 4,   "core_switch_count": 3,   "agg_per_pod": 3,  "edge_per_pod": 3,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 900,  "link_price_per_gbps_month": 36.9}]},
        #{"type": "fat_tree", "params": {"num_pods": 5,   "core_switch_count": 3,   "agg_per_pod": 3,  "edge_per_pod": 3,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1125, "link_price_per_gbps_month": 36.1}]},
        #{"type": "fat_tree", "params": {"num_pods": 6,   "core_switch_count": 3,   "agg_per_pod": 3,  "edge_per_pod": 3,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1350, "link_price_per_gbps_month": 35.4}]},
        #{"type": "fat_tree", "params": {"num_pods": 7,   "core_switch_count": 3,   "agg_per_pod": 3,  "edge_per_pod": 3,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1575, "link_price_per_gbps_month": 34.9}]},

        # {"type": "fat_tree", "params": {"num_pods": 2,   "core_switch_count": 4,   "agg_per_pod": 4,  "edge_per_pod": 4,  "gpus_per_edge": 4},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 800,  "link_price_per_gbps_month": 37.3}]},
        # {"type": "fat_tree", "params": {"num_pods": 3,   "core_switch_count": 4,   "agg_per_pod": 4,  "edge_per_pod": 4,  "gpus_per_edge": 4},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1200, "link_price_per_gbps_month": 35.8}]},
        # {"type": "fat_tree", "params": {"num_pods": 4,   "core_switch_count": 4,   "agg_per_pod": 4,  "edge_per_pod": 4,  "gpus_per_edge": 4},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1600, "link_price_per_gbps_month": 34.8}]},
        # {"type": "fat_tree", "params": {"num_pods": 5,   "core_switch_count": 4,   "agg_per_pod": 4,  "edge_per_pod": 4,  "gpus_per_edge": 4},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 2000, "link_price_per_gbps_month": 34.1}]},
        # {"type": "fat_tree", "params": {"num_pods": 6,   "core_switch_count": 4,   "agg_per_pod": 4,  "edge_per_pod": 4,  "gpus_per_edge": 4},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 2400, "link_price_per_gbps_month": 33.4}]},

        # {"type": "fat_tree", "params": {"num_pods": 2,   "core_switch_count": 5,   "agg_per_pod": 5,  "edge_per_pod": 5,  "gpus_per_edge": 5},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1250, "link_price_per_gbps_month": 35.7}]},
        # {"type": "fat_tree", "params": {"num_pods": 3,   "core_switch_count": 5,   "agg_per_pod": 5,  "edge_per_pod": 5,  "gpus_per_edge": 5},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1875, "link_price_per_gbps_month": 34.3}]},
        # {"type": "fat_tree", "params": {"num_pods": 4,   "core_switch_count": 5,   "agg_per_pod": 5,  "edge_per_pod": 5,  "gpus_per_edge": 5},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 2500, "link_price_per_gbps_month": 33.3}]},
        # {"type": "fat_tree", "params": {"num_pods": 5,   "core_switch_count": 5,   "agg_per_pod": 5,  "edge_per_pod": 5,  "gpus_per_edge": 5},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 3125, "link_price_per_gbps_month": 32.6}]},
        # {"type": "fat_tree", "params": {"num_pods": 6,   "core_switch_count": 5,   "agg_per_pod": 5,  "edge_per_pod": 5,  "gpus_per_edge": 5},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 3750, "link_price_per_gbps_month": 32.0}]},

        # {"type": "fat_tree", "params": {"num_pods": 2,   "core_switch_count": 6,   "agg_per_pod": 6,  "edge_per_pod": 6,  "gpus_per_edge": 6},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1800, "link_price_per_gbps_month": 34.4}]},
        # {"type": "fat_tree", "params": {"num_pods": 3,   "core_switch_count": 6,   "agg_per_pod": 6,  "edge_per_pod": 6,  "gpus_per_edge": 6},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 2700, "link_price_per_gbps_month": 33.0}]},
        # {"type": "fat_tree", "params": {"num_pods": 4,   "core_switch_count": 6,   "agg_per_pod": 6,  "edge_per_pod": 6,  "gpus_per_edge": 6},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 3600, "link_price_per_gbps_month": 32.1}]},
        # {"type": "fat_tree", "params": {"num_pods": 5,   "core_switch_count": 6,   "agg_per_pod": 6,  "edge_per_pod": 6,  "gpus_per_edge": 6},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 4500, "link_price_per_gbps_month": 31.4}]},
        # {"type": "fat_tree", "params": {"num_pods": 6,   "core_switch_count": 6,   "agg_per_pod": 6,  "edge_per_pod": 6,  "gpus_per_edge": 6},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 5400, "link_price_per_gbps_month": 30.8}]},

        # {"type": "fat_tree", "params": {"num_pods": 4,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 480,  "link_price_per_gbps_month": 38}]},   # 28
        #  {"type": "fat_tree", "params": {"num_pods": 6,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 720,  "link_price_per_gbps_month": 40}]},   # 40
        #  {"type": "fat_tree", "params": {"num_pods": 8,   "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 960,  "link_price_per_gbps_month": 42}]},   # 52
        #  {"type": "fat_tree", "params": {"num_pods": 10,  "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 960,  "link_price_per_gbps_month": 44}]},   # 64

        #  {"type": "fat_tree", "params": {"num_pods": 12,  "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1200, "link_price_per_gbps_month": 46}]},   # 76
        #  {"type": "fat_tree", "params": {"num_pods": 16,  "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1920, "link_price_per_gbps_month": 48}]},   # 100
        # {"type": "fat_tree", "params": {"num_pods": 20,  "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 3},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 2400, "link_price_per_gbps_month": 50}]},   # 124
        #  {"type": "fat_tree", "params": {"num_pods": 32,  "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 4},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 6144, "link_price_per_gbps_month": 22}]},   # 386
        #  {"type": "fat_tree", "params": {"num_pods": 32,  "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 6},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 9216, "link_price_per_gbps_month": 22}]},   # 514
        #  {"type": "fat_tree", "params": {"num_pods": 48,  "core_switch_count": 2,   "agg_per_pod": 2,  "edge_per_pod": 2,  "gpus_per_edge": 6},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 13824, "link_price_per_gbps_month": 22}]},   # 770

        #  {"type": "fat_tree", "params": {"num_pods": 9,   "core_switch_count": 100, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 8},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 17280, "link_price_per_gbps_month": 22}]},   # 1000
        #  {"type": "fat_tree", "params": {"num_pods": 13,  "core_switch_count": 100, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 8},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 24960, "link_price_per_gbps_month": 22}]},   # 1400
        #  {"type": "fat_tree", "params": {"num_pods": 19,  "core_switch_count": 100, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 8},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 36480, "link_price_per_gbps_month": 22}]},   # 2000
        #  {"type": "fat_tree", "params": {"num_pods": 27,  "core_switch_count": 100, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 8},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 51840, "link_price_per_gbps_month": 22}]},   # 2800
        #  {"type": "fat_tree", "params": {"num_pods": 39,  "core_switch_count": 100, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 8},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 74880, "link_price_per_gbps_month": 22}]},   # 4000
        #  {"type": "fat_tree", "params": {"num_pods": 55,  "core_switch_count": 100, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 8},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 105600, "link_price_per_gbps_month": 22}]},   # 5600
        #  {"type": "fat_tree", "params": {"num_pods": 79,  "core_switch_count": 100, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 8},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 151680, "link_price_per_gbps_month": 22}]},   # 8000
        #  {"type": "fat_tree", "params": {"num_pods": 99,  "core_switch_count": 100, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 8},  "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 190080, "link_price_per_gbps_month": 22}]},   # 10000

        #  {"type": "fat_tree", "params": {"num_pods": 69,  "core_switch_count": 200, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 18}, "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 298080, "link_price_per_gbps_month": 22}]},   # 14000
        #  {"type": "fat_tree", "params": {"num_pods": 99,  "core_switch_count": 200, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 18}, "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 427680, "link_price_per_gbps_month": 22}]},   # 20000
        #  {"type": "fat_tree", "params": {"num_pods": 139, "core_switch_count": 200, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 18}, "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 600480, "link_price_per_gbps_month": 22}]},   # 28000
        #  {"type": "fat_tree", "params": {"num_pods": 199, "core_switch_count": 200, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 18}, "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 859680, "link_price_per_gbps_month": 22}]},   # 40000
        #  {"type": "fat_tree", "params": {"num_pods": 279, "core_switch_count": 200, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 18}, "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1205280, "link_price_per_gbps_month": 22}]},   # 56000
        #  {"type": "fat_tree", "params": {"num_pods": 399, "core_switch_count": 200, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 18}, "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 1723680, "link_price_per_gbps_month": 22}]},   # 80000
        #  {"type": "fat_tree", "params": {"num_pods": 499, "core_switch_count": 200, "agg_per_pod": 10, "edge_per_pod": 10, "gpus_per_edge": 18}, "seed": 42, "network_profiles": [{"network_total_bandwidth_gbps": 2155680, "link_price_per_gbps_month": 22}]},   # 100000
    ],

    #  topology_specs
    "topology_names": [],



    # (3) G(,; G )
    "G": 10,

    # (4) K()
    "K_list": [3,4,5,6,7,8],

        # (7)GPU ()---- GPU
    # (8)GPU  map_mode()
    #     - "high_to_high":    gpu_set ""
    #     - "high_to_low":     gpu_set ""
    # (10) profiles()
    #      {user_price_per_month, tokens_per_user}
    #     - user_price_per_month: (:$/month)
    #     - tokens_per_user: (:token/sec)
    # Use a heavier profile so SP is penalized by single-node memory limits,
    # while slightly lower tokens/user and cheaper links give LMU room to pull ahead on users.
    "service_profiles": [
      # ========== 7B (,) ==========
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 100,  "user_price_per_month": 7.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 100,  "user_price_per_month": 8.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 100,  "user_price_per_month": 9.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 100,  "user_price_per_month": 10.0},
       {"model_name": "Qwen2.5-7B", "tokens_per_user": 100,  "user_price_per_month": 20.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 100,  "user_price_per_month": 30.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 150,  "user_price_per_month": 40.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 100,  "user_price_per_month": 50.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 100,  "user_price_per_month": 60.0},
    #   {"model_name": "Qwen2.5-7B", "tokens_per_user": 80,  "user_price_per_month": 12.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 100, "user_price_per_month": 18.0},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 150, "user_price_per_month": 36},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 150, "user_price_per_month": 40},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 250, "user_price_per_month": 18},
    #   {"model_name": "Qwen2.5-7B", "tokens_per_user": 300, "user_price_per_month": 96},
    #    {"model_name": "Qwen2.5-7B", "tokens_per_user": 400, "user_price_per_month": 144},

      # {"model_name": "Mistral-7B", "tokens_per_user": 60,  "user_price_per_month": 10.0},
      # {"model_name": "Mistral-7B", "tokens_per_user": 80,  "user_price_per_month": 15.0},
      # {"model_name": "Mistral-7B", "tokens_per_user": 100, "user_price_per_month": 22.0},

    #   # ========== 13B (,) ==========
    #   {"model_name": "Llama-3.1-13B-Instruct", "tokens_per_user": 55,  "user_price_per_month": 40.0},
    #   {"model_name": "Llama-3.1-13B-Instruct", "tokens_per_user": 80,  "user_price_per_month": 25.0},
    #    {"model_name": "Llama-3.1-13B-Instruct", "tokens_per_user": 100, "user_price_per_month": 20.0},
    #   {"model_name":"Llama-3.1-13B-Instruct", "tokens_per_user": 60, "user_price_per_month": 30.0}
      # ========== 27B () ==========
      # {"model_name": "Gemma-2-27B", "tokens_per_user": 60,  "user_price_per_month": 30.0},
      # {"model_name": "Gemma-2-27B", "tokens_per_user": 80,  "user_price_per_month": 45.0},
      #  {"model_name": "Gemma-2-27B", "tokens_per_user": 50, "user_price_per_month": 20.0},
      #  # ========== 70B () ==========
      #  {"model_name": "Llama-3.1-70B-Instruct", "tokens_per_user": 60,  "user_price_per_month": 50.0},
      #  {"model_name": "Llama-3.1-70B-Instruct", "tokens_per_user": 80,  "user_price_per_month": 70.0},
      #  {"model_name": "Llama-3.1-70B-Instruct", "tokens_per_user": 100, "user_price_per_month": 95.0},

    #    # ========== 236B (,) ==========
    #    {"model_name": "DeepSeek-V2", "tokens_per_user": 60,  "user_price_per_month": 100.0},
    #    {"model_name": "DeepSeek-V2", "tokens_per_user": 80,  "user_price_per_month": 140.0},
    #    {"model_name": "DeepSeek-V2", "tokens_per_user": 100, "user_price_per_month": 180.0},
  ],


}

DEFAULT_TARGET_TOTAL_USERS: List[int] = [30, 60, 90, 120, 150, 180]

# Single GPU distribution entrypoint. Defaults match the ratio path that was
# effectively active in sparse_bound_profiles.py.
GPU_DISTRIBUTION_CONFIG: Dict[str, Any] = {
    "map_mode_list": ["high_to_low"],
    "ratios": [
        {"A6000_48GB":0.3,"H100_256GB":0.7}
    ],
}

# ============================================================
# : & GPU
# ============================================================
def build_topology_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    network_profiles = spec.get("network_profiles", [])
    topology_spec = {k: v for k, v in spec.items() if k != "network_profiles"}
    topo_name = spec.get("name", "")
    topo_type = spec.get("type", "")
    params = spec.get("params", {}) or {}
    seed = spec.get("seed", 42)

    if not topo_name:
        if topo_type == "fat_tree":
            topo_name = (
                f"fat_tree_p{params.get('num_pods', 0)}_"
                f"c{params.get('core_switch_count', 0)}_"
                f"a{params.get('agg_per_pod', 0)}_"
                f"e{params.get('edge_per_pod', 0)}_"
                f"g{params.get('gpus_per_edge', 0)}"
            )
        elif topo_type == "clos":
            topo_name = f"clos_{'_'.join(map(str, params.get('layer_sizes', params.get('layer_config', []))))}"
        else:
            topo_name = "topology"

    topo_cfg = {
        "topology_type": topo_type,
        "random_seed": seed,
        "fat_tree": {
            "num_pods": int(params.get("num_pods", 0)),
            "core_switch_count": int(params.get("core_switch_count", 0)),
            "agg_per_pod": int(params.get("agg_per_pod", 0)),
            "edge_per_pod": int(params.get("edge_per_pod", 0)),
            "gpus_per_edge": int(params.get("gpus_per_edge", 0)),
        },
        "clos": {
            "layer_config": params.get("layer_sizes", params.get("layer_config", [])),
        },
    }

    gen = TopologyGeneratorWithSwitches(topo_cfg)
    metadata = gen.generate()

    node_count = int(metadata.total_nodes)
    degree_list = [int(x) for x in metadata.degree_list]
    gpu_nodes = [int(x) for x in metadata.gpu_nodes]

    return {
        "name": topo_name,
        "type": topo_type,
        "node_count": node_count,
        "degree_list": degree_list,
        "gpu_nodes": gpu_nodes,
        "network_profiles": network_profiles,
        "spec": topology_spec,
        "spec_json": json.dumps(topology_spec, ensure_ascii=False),
    }

def iter_topology_instances():
    specs = GENERATION_CONFIG.get("topology_specs", [])
    if not specs:
        raise ValueError("GENERATION_CONFIG['topology_specs'] ")

    for spec in specs:
        if not spec.get("network_profiles"):
            raise ValueError("Each topology spec must bind a non-empty 'network_profiles' list.")
        yield build_topology_from_spec(spec)



def assign_gpus_by_degree(
    degree_list: List[int],
    gpu_set_names: List[str],
    rank_order: str,
    map_mode: str,
    gpu_nodes: List[int] = None,
) -> List[Dict[str, Any]]:
    n = len(degree_list)
    reverse = (rank_order == "desc")

    node_indices = list(gpu_nodes) if gpu_nodes is not None else list(range(n))
    node_indices_sorted = sorted(
        node_indices,
        key=lambda i: (degree_list[i], i),
        reverse=reverse
    )

    S = len(gpu_set_names)
    allocation_count = len(node_indices)

    base = allocation_count // S
    rem = allocation_count % S
    counts = [base] * S
    for i in range(rem):
        counts[i] += 1

    if map_mode == "high_to_high":
        band_indices = list(reversed(range(S)))
    elif map_mode == "high_to_low":
        band_indices = list(range(S))
    else:
        raise ValueError(f"Unknown map_mode: {map_mode}")

    assignment_map: Dict[int, str] = {}
    pos = 0
    for band_idx in band_indices:
        c = counts[band_idx]
        gpu_name = gpu_set_names[band_idx]
        for _ in range(c):
            if pos >= len(node_indices_sorted):
                break
            node_index = node_indices_sorted[pos]
            pos += 1
            assignment_map[node_index] = gpu_name

    result = []
    for i in range(n):
        gpu_assignment = assignment_map.get(i, None)
        if gpu_assignment is not None:
            gpu_info = GPU_CATALOG[gpu_assignment]
            result.append({
                "node_index": i,
                "gpu_type": gpu_info["name"],
                "G_TFLOPS": gpu_info["G_TFLOPS"],
                "VRAM_bytes": gpu_info["VRAM_bytes"],
                "cost_per_GB_month": gpu_info["cost_per_GB_month"],
                "degree": int(degree_list[i]),
            })
        else:
            result.append({
                "node_index": i,
                "gpu_type": "None",
                "G_TFLOPS": 0.0,
                "VRAM_bytes": 0,
                "cost_per_GB_month": 0.0,
                "degree": int(degree_list[i]),
            })
    return result

def assign_gpus_by_ratio(
    degree_list: List[int],
    ratio_config: Dict[str, float],
    rank_order: str,
    map_mode: str,
    gpu_nodes: List[int] = None,
) -> List[Dict[str, Any]]:
    n = len(degree_list)
    reverse = (rank_order == "desc")

    node_indices = list(gpu_nodes) if gpu_nodes is not None else list(range(n))
    node_indices_sorted = sorted(
        node_indices,
        key=lambda i: (degree_list[i], i),
        reverse=reverse
    )

    gpu_names = list(ratio_config.keys())
    ratio_values = [max(0.0, float(ratio_config[g])) for g in gpu_names]
    ratio_sum = sum(ratio_values)
    if ratio_sum <= 0.0:
        raise ValueError(f"Invalid ratio_config: {ratio_config}")

    allocation_count = len(node_indices)
    raw_counts = [(v / ratio_sum) * allocation_count for v in ratio_values]
    counts = [int(x) for x in raw_counts]

    remain = allocation_count - sum(counts)
    frac_order = sorted(
        range(len(gpu_names)),
        key=lambda i: (raw_counts[i] - counts[i], -i),
        reverse=True
    )
    for i in frac_order[:remain]:
        counts[i] += 1

    if map_mode == "high_to_high":
        band_indices = list(reversed(range(len(gpu_names))))
    elif map_mode == "high_to_low":
        band_indices = list(range(len(gpu_names)))
    else:
        raise ValueError(f"Unknown map_mode: {map_mode}")

    assignment_map: Dict[int, str] = {}
    pos = 0
    for band_idx in band_indices:
        c = counts[band_idx]
        gpu_name = gpu_names[band_idx]
        for _ in range(c):
            if pos >= len(node_indices_sorted):
                break
            node_index = node_indices_sorted[pos]
            pos += 1
            assignment_map[node_index] = gpu_name

    result = []
    for i in range(n):
        gpu_assignment = assignment_map.get(i, None)
        if gpu_assignment is not None:
            gpu_info = GPU_CATALOG[gpu_assignment]
            result.append({
                "node_index": i,
                "gpu_type": gpu_info["name"],
                "G_TFLOPS": gpu_info["G_TFLOPS"],
                "VRAM_bytes": gpu_info["VRAM_bytes"],
                "cost_per_GB_month": gpu_info["cost_per_GB_month"],
                "degree": int(degree_list[i]),
            })
        else:
            result.append({
                "node_index": i,
                "gpu_type": "None",
                "G_TFLOPS": 0.0,
                "VRAM_bytes": 0,
                "cost_per_GB_month": 0.0,
                "degree": int(degree_list[i]),
            })
    return result

# ============================================================
# :G  +  +
# ============================================================

def group_layers_evenly(num_layers: int, G: int) -> List[int]:
    """
     num_layers , G
     L=80, G=10 -> [8,8,8,8,8,8,8,8,8,8];
    L=82, G=10 -> [9,9,9,9,8,8,8,8,8,8]
    """
    base = num_layers // G
    rem = num_layers % G
    groups = [base + (1 if i < rem else 0) for i in range(G)]
    assert sum(groups) == num_layers
    return groups


def enumerate_compositions(G: int, K: int) -> List[List[int]]:
    """
     G  K ()
     G=5,K=2:
      [1,4], [2,3], [3,2], [4,1]
    """
    if K < 1 or K > G:
        raise ValueError(f"Invalid composition: G={G}, K={K}")
    if K == 1:
        return [[G]]
    positions = range(1, G)
    compositions = []
    for bars in itertools.combinations(positions, K - 1):
        prev = 0
        segs = []
        for b in bars:
            segs.append(b - prev)
            prev = b
        segs.append(G - prev)
        compositions.append(segs)
    return compositions


def expand_groups_to_layers(group_sizes: List[int], comp: List[int]) -> List[int]:
    """
     group_sizes[g],"" composition comp,
    ""
    """
    layers = []
    idx = 0
    for g_count in comp:
        s = sum(group_sizes[idx: idx + g_count])
        layers.append(s)
        idx += g_count
    return layers


def estimate_partition_resources(
    model_cfg: Dict[str, Any],
    layers_per_segment: List[int],
    run_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
     & ,:
    - compute_tflops_per_token_i
    - memory_gb_i( + KV)
    - boundary_data_mb_i:(MB/s/user)

     tokens_per_user "(token/sec)"

     i -> i+1:
      - Activation: token,  act_bytes_per_token * tokens_per_user (bytes/sec)
      - KV: token,"" KV
            per-token KV bytes  kv_bytes_per_token_per_layer * layers_before
             tokens_per_user  KV bytes/sec
    """
    L = int(model_cfg["num_layers"])
    total_params = float(model_cfg["total_params"])
    bytes_per_param = int(model_cfg["bytes_per_param"])
    hidden_size = int(model_cfg["hidden_size"])
    num_heads = int(model_cfg["num_attention_heads"])
    num_kv_heads = int(model_cfg["num_key_value_heads"])
    flops_per_token_per_layer = float(model_cfg["flops_per_token_per_layer"])

    seq_len = int(run_cfg["seq_len"])
    batch_size = int(run_cfg["batch_size"])
    tokens_per_user = int(run_cfg["tokens_per_user"])  # token/sec

    bytes_per_act = 2  # /KV  FP16

    # 1.
    total_weights_bytes = total_params * bytes_per_param
    total_weights_gb = total_weights_bytes / (1024**3)

    # 2. KV: token  KV ()
    head_dim = hidden_size / num_heads
    kv_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * bytes_per_param  # K+V

    # 3. Activation: token
    act_bytes_per_token = hidden_size * bytes_per_act

    segments = []
    K = len(layers_per_segment)

    #  prefix_layers[i] =  i
    prefix_layers = [0]
    for li in layers_per_segment:
        prefix_layers.append(prefix_layers[-1] + li)

    for i, li in enumerate(layers_per_segment):
        # 3.1 :flops_per_token_per_layer  li
        compute_tflops_per_token = flops_per_token_per_layer * li / (10**12)

        # 3.2 :
        frac = li / L
        weights_gb_i = total_weights_gb * frac

        # 3.3 KV (): KV  seq_len  batch_size
        kv_bytes_i = kv_bytes_per_token_per_layer * seq_len * batch_size * li
        kv_gb_batch = kv_bytes_i / (1024 ** 3)
        kv_gb_per_user = kv_gb_batch / batch_size  #  batch_size>0()


        # 3.4 (Activation + KV), i < K-1
        if i < K - 1:
            # Activation : token  act   bytes/sec
            act_bytes_per_sec = act_bytes_per_token * tokens_per_user

            # KV :
            #  token, KV
            layers_before = prefix_layers[i + 1]  #  i+1
            kv_bytes_per_token_cross = kv_bytes_per_token_per_layer * layers_before
            kv_bytes_per_sec = kv_bytes_per_token_cross * tokens_per_user

            boundary_bytes_per_sec = act_bytes_per_sec + kv_bytes_per_sec
            boundary_mb = boundary_bytes_per_sec / (1024**2)  # MB/sec per user
        else:
            boundary_mb = 0.0

        segments.append({
            "segment_index": i,
            "layers": li,
            "compute_tflops_per_token": float(round(compute_tflops_per_token, 6)),
            "compute_tflops_per_user_per_sec": float(round(compute_tflops_per_token * tokens_per_user, 6)),
            "weights_gb": float(round(weights_gb_i, 6)),       #
            "kv_gb": float(round(kv_gb_per_user, 6)),          # per-user KV
            # :(MB/s)
            "boundary_data_mb": float(round(boundary_mb, 6)),
        })

    summary = {
        "total_compute_tflops_per_token": float(round(sum(s["compute_tflops_per_token"] for s in segments), 6)),
        "total_weights_gb": float(round(sum(s["weights_gb"] for s in segments), 6)),  # ()
        "total_kv_gb": float(round(sum(s["kv_gb"] for s in segments), 6)),
        "max_boundary_data_mb": float(round(max(s["boundary_data_mb"] for s in segments), 6)),
    }

    return {
        "segments_layers": layers_per_segment,
        "segments": segments,
        "summary": summary,
    }


# ============================================================
# ( +  + )
# ============================================================

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def resolve_run_dir(run_name: str, run_dir: str) -> str:
    if run_dir:
        return run_dir if os.path.isabs(run_dir) else os.path.join(PROJECT_ROOT, run_dir)
    return os.path.join(DEFAULT_RUNS_DIR, run_name)


def resolve_output_dir(run_dir: str, output_dir: str) -> str:
    if output_dir:
        return output_dir if os.path.isabs(output_dir) else os.path.join(PROJECT_ROOT, output_dir)
    return os.path.join(run_dir, "input")


def write_chunk(rows: List[Dict[str, Any]], output_dir: str, file_prefix: str, chunk_index: int) -> str:
    filename = f"{file_prefix}_{chunk_index:04d}.csv"
    path = os.path.join(output_dir, filename)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _validate_ratio_dict(ratio: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(ratio, dict) or not ratio:
        raise argparse.ArgumentTypeError("Each GPU ratio must be a non-empty dict.")

    normalized: Dict[str, float] = {}
    total = 0.0
    for gpu_type, value in ratio.items():
        gpu_name = str(gpu_type).strip()
        if not gpu_name:
            raise argparse.ArgumentTypeError("GPU type names must be non-empty strings.")
        if gpu_name not in GPU_CATALOG:
            raise argparse.ArgumentTypeError(
                f"Unknown GPU type '{gpu_name}'. Available GPUs: {', '.join(GPU_CATALOG.keys())}"
            )
        try:
            ratio_value = float(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid ratio value for GPU type '{gpu_name}': {value}"
            ) from exc
        if ratio_value < 0.0 or ratio_value > 1.0:
            raise argparse.ArgumentTypeError(
                f"Ratio for GPU type '{gpu_name}' must be between 0 and 1, got {ratio_value}."
            )
        normalized[gpu_name] = ratio_value
        total += ratio_value

    if abs(total - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"GPU ratio values must sum to 1.0, got {total:.6f}.")
    return normalized


def parse_gpu_ratios_arg(raw: str) -> List[Dict[str, float]]:
    """
    Supports JSON dict/list or compact syntax:
      A6000_48GB=0.8,A100_80GB=0.2
      A6000_48GB=0.8,A100_80GB=0.2;A6000_48GB=0.6,A100_80GB=0.4
    """
    raw = str(raw or "").strip()
    if not raw:
        raise argparse.ArgumentTypeError("gpu_ratios cannot be empty.")

    if raw.startswith("{") or raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(f"Invalid JSON for gpu_ratios: {exc}") from exc
        if isinstance(parsed, dict):
            return [_validate_ratio_dict(parsed)]
        if isinstance(parsed, list):
            return [_validate_ratio_dict(item) for item in parsed]
        raise argparse.ArgumentTypeError("gpu_ratios JSON must be a dict or a list of dicts.")

    ratio_list: List[Dict[str, float]] = []
    for group in raw.split(";"):
        group = group.strip()
        if not group:
            continue
        ratio_dict: Dict[str, float] = {}
        for pair in group.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise argparse.ArgumentTypeError(f"Invalid GPU ratio pair '{pair}'. Expected GPU=value.")
            gpu_type, value = pair.split("=", 1)
            try:
                ratio_dict[gpu_type.strip()] = float(value)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid GPU ratio value in '{pair}'.") from exc
        ratio_list.append(_validate_ratio_dict(ratio_dict))

    if not ratio_list:
        raise argparse.ArgumentTypeError("gpu_ratios did not contain any valid ratio config.")
    return ratio_list


def parse_target_total_users_arg(raw: str) -> List[int]:
    values: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid target_total_users value '{token}'. Expected integer."
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("target_total_users values must be positive integers.")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("target_total_users cannot be empty.")
    return values


def _get_gpu_distribution_config(
    gpu_ratios: Optional[List[Dict[str, float]]] = None,
    gpu_map_mode: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, float]]]:
    map_modes = [str(x) for x in GPU_DISTRIBUTION_CONFIG["map_mode_list"]]
    ratios = GPU_DISTRIBUTION_CONFIG["ratios"]

    if gpu_map_mode is not None:
        gpu_map_mode = str(gpu_map_mode).strip()
        if gpu_map_mode not in ("high_to_high", "high_to_low"):
            raise argparse.ArgumentTypeError(
                "gpu_map_mode must be one of: high_to_high, high_to_low."
            )
        map_modes = [gpu_map_mode]
    if gpu_ratios is not None:
        ratios = gpu_ratios

    return map_modes, ratios


def generate_experiment_data(
    output_dir: str,
    chunk_size: int,
    file_prefix: str,
    with_target_users: bool = False,
    target_total_users: Optional[List[int]] = None,
    gpu_ratios: Optional[List[Dict[str, float]]] = None,
    gpu_map_mode: Optional[str] = None,
) -> None:
    """

    -  topology_specs
    -  GPU ratio
    -  sparse_bound_profiles.py  sparse schema
    - with_target_users=True  target_total_users
    """
    ensure_dir(output_dir)

    buffer: List[Dict[str, Any]] = []
    batch_idx = 0
    output_files: List[str] = []

    topology_instances = list(iter_topology_instances())

    G = GENERATION_CONFIG["G"]
    K_list = GENERATION_CONFIG["K_list"]
    seq_len_list = GENERATION_CONFIG["seq_len_list"]
    rank_order = GPU_ASSIGNMENT_CONFIG["rank_order"]
    service_profiles = GENERATION_CONFIG["service_profiles"]
    map_modes, ratio_configs = _get_gpu_distribution_config(
        gpu_ratios=gpu_ratios,
        gpu_map_mode=gpu_map_mode,
    )
    target_values: List[Optional[int]] = (
        list(target_total_users or DEFAULT_TARGET_TOTAL_USERS) if with_target_users else [None]
    )

    print("=== 1 ===")
    start_time = time.time()

    approx_total_combos = 0
    for topo in topology_instances:
        approx_total_combos += (
            len(K_list)
            * len(topo["network_profiles"])
            * len(map_modes)
            * len(ratio_configs)
            * len(seq_len_list)
            * len(service_profiles)
            * len(target_values)
        )

    print(f"( G->K ):{approx_total_combos}")

    for topo in topology_instances:
        n_nodes = int(topo["node_count"])
        degree_list = topo["degree_list"]
        gpu_nodes = topo.get("gpu_nodes", None)
        topo_network_profiles = topo["network_profiles"]

        for K, net_profile, map_mode, gpu_ratio, seq_len, service, target_value in itertools.product(
            K_list,
            topo_network_profiles,
            map_modes,
            ratio_configs,
            seq_len_list,
            service_profiles,
            target_values,
        ):
            net_bw_total = float(net_profile["network_total_bandwidth_gbps"])
            link_price = float(net_profile["link_price_per_gbps_month"])

            model_name = service["model_name"]
            user_price_per_month = float(service["user_price_per_month"])
            tokens_per_user = int(service["tokens_per_user"])

            model_cfg = MODEL_CATALOG[model_name]
            L = int(model_cfg["num_layers"])
            if G > L:
                raise RuntimeError(f"G={G}  {model_name}  L={L}")

            per_node_gpu = assign_gpus_by_ratio(
                degree_list=degree_list,
                ratio_config=gpu_ratio,
                rank_order=rank_order,
                map_mode=map_mode,
                gpu_nodes=gpu_nodes,
            )

            gpu_ratio_str = ",".join([f"{k}:{v}" for k, v in sorted(gpu_ratio.items())])
            gpu_set_str = ",".join(sorted(gpu_ratio.keys()))
            gpu_map_mode = map_mode
            gpu_ratio_detail = gpu_ratio_str
            gpu_ratio_config_json = json.dumps(
                {k: gpu_ratio[k] for k in sorted(gpu_ratio.keys())},
                ensure_ascii=False,
            )

            group_sizes = group_layers_evenly(L, G)
            compositions = enumerate_compositions(G, K)

            run_cfg_base = dict(RUN_CONFIG)
            run_cfg_base["seq_len"] = int(seq_len)
            run_cfg_base["tokens_per_user"] = tokens_per_user

            for part_idx, comp in enumerate(compositions):
                layers_per_seg = expand_groups_to_layers(group_sizes, comp)
                part_res = estimate_partition_resources(model_cfg, layers_per_seg, run_cfg_base)

                summary = part_res["summary"]
                max_boundary_data_mb = float(summary.get("max_boundary_data_mb", 0.0))

                user_uplink_mb_per_user = 0.1 * max_boundary_data_mb
                user_downlink_mb_per_user = 0.05 * max_boundary_data_mb
                return_to_gateway_mb_per_user = user_downlink_mb_per_user

                if str(topo["type"]).lower() == "fat_tree":
                    topo_params = topo.get("spec", {}).get("params", {}) if isinstance(topo.get("spec", {}), dict) else {}
                    if not topo_params:
                        try:
                            topo_params = json.loads(topo["spec_json"]).get("params", {})
                        except Exception:
                            topo_params = {}
                    core_switch_count = int(topo_params.get("core_switch_count", 0))
                    gateway_node_list = list(range(core_switch_count))
                else:
                    gateway_node_list = []

                row = {
                    "topology_name": topo["name"],
                    "node_count": n_nodes,
                    "model_name": model_name,
                    "G_groups": G,
                    "K_segments": K,
                    "partition_index": part_idx,
                    "network_total_bandwidth_gbps": net_bw_total,
                    "link_price_per_gbps_month": link_price,
                    "network_bandwidth_cost_per_month": net_bw_total * link_price,
                    "topology_type": topo["type"],
                    "topology_spec_json": topo["spec_json"],
                    "gateway_node_list": json.dumps(gateway_node_list, ensure_ascii=False),
                }
                if target_value is not None:
                    row["target_total_users"] = int(target_value)
                row.update({
                    "user_uplink_mb_per_user": float(user_uplink_mb_per_user),
                    "user_downlink_mb_per_user": float(user_downlink_mb_per_user),
                    "return_to_gateway_mb_per_user": float(return_to_gateway_mb_per_user),
                    "gpu_set": gpu_set_str,
                    "gpu_map_mode": gpu_map_mode,
                    "gpu_rank_order": rank_order,
                    "gpu_ratio_detail": gpu_ratio_detail,
                    "gpu_ratio_config_json": gpu_ratio_config_json,
                    "per_node_gpu_json": json.dumps(per_node_gpu, ensure_ascii=False),
                    "user_price_per_month": user_price_per_month,
                    "tokens_per_user": tokens_per_user,
                    "seq_len": int(seq_len),
                    "batch_size": RUN_CONFIG["batch_size"],
                    "model_num_layers": L,
                    "model_total_params": model_cfg["total_params"],
                    "model_hidden_size": model_cfg["hidden_size"],
                    "model_num_heads": model_cfg["num_attention_heads"],
                    "model_num_kv_heads": model_cfg["num_key_value_heads"],
                    "segments_layers_json": json.dumps(part_res["segments_layers"], ensure_ascii=False),
                    "segments_detail_json": json.dumps(part_res["segments"], ensure_ascii=False),
                    "segments_summary_json": json.dumps(part_res["summary"], ensure_ascii=False),
                })

                buffer.append(row)

                if len(buffer) >= chunk_size:
                    batch_path = write_chunk(buffer, output_dir, file_prefix, batch_idx)
                    print(f"[] {batch_path} ({len(buffer)} )")
                    buffer.clear()
                    output_files.append(batch_path)
                    batch_idx += 1

    if buffer:
        batch_path = write_chunk(buffer, output_dir, file_prefix, batch_idx)
        print(f"[] {batch_path} ({len(buffer)} )")
        buffer.clear()
        output_files.append(batch_path)

    if not output_files:
        print("[] ,")

    elapsed = time.time() - start_time
    print(f"=== 1, {elapsed:.2f}  ===")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--chunk_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--file_prefix", type=str, default=FILE_PREFIX)
    ap.add_argument(
        "--with_target_users",
        action="store_true",
        help="Generate target_total_users as an additional dimension.",
    )
    ap.add_argument(
        "--target_total_users",
        type=parse_target_total_users_arg,
        default=DEFAULT_TARGET_TOTAL_USERS,
        help="Comma-separated target user counts, e.g. 30,60,90,120,150,180.",
    )
    ap.add_argument(
        "--gpu_ratios",
        type=parse_gpu_ratios_arg,
        default=None,
        help=(
            "Override GPU ratios. Supports JSON dict/list or compact syntax, "
            "e.g. A6000_48GB=0.8,A100_80GB=0.2"
        ),
    )
    ap.add_argument(
        "--gpu_map_mode",
        type=str,
        default=None,
        choices=["high_to_high", "high_to_low"],
        help="Override GPU assignment map mode.",
    )
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_name, args.run_dir)
    output_dir = resolve_output_dir(run_dir, args.output_dir)
    generate_experiment_data(
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        file_prefix=args.file_prefix,
        with_target_users=args.with_target_users,
        target_total_users=args.target_total_users,
        gpu_ratios=args.gpu_ratios,
        gpu_map_mode=args.gpu_map_mode,
    )


if __name__ == "__main__":
    main()
