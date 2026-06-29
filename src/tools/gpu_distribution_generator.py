#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU


1. GPU
2.
3. high_to_high/high_to_low


- GPU
- GPU

 experiment1_data_generator GPU
"""

import itertools
from typing import List, Dict, Any, Tuple
import numpy as np


class GPUDistributionGenerator:
    """
    GPU


    1. GPU
    2.
    """

    def __init__(self, gpu_catalog: Dict[str, Dict[str, Any]]):
        """
        GPU

        Args:
            gpu_catalog: GPU
                {
                    "H100_256GB": {"name": "H100_256GB", "G_TFLOPS": 183.0, ...},
                    "A6000_48GB": {"name": "A6000_48GB", "G_TFLOPS": 38.71, ...},
                    ...
                }
        """
        self.gpu_catalog = gpu_catalog

    def generate_ratio_combinations(
        self,
        gpu_types: List[str],
        total_nodes: int,
        step: float = 0.1,
        min_nodes_per_type: int = 1
    ) -> List[Dict[str, float]]:
        """
        GPU


        1. GPU1.0
        2. GPUmin_nodes_per_type
        3. step0.1


            gpu_types = ["H100_256GB", "A6000_48GB"], total_nodes=20, step=0.1

            {"H100_256GB": 0.5, "A6000_48GB": 0.5}  # 10 H100 + 10 A6000

        Args:
            gpu_types: GPU ["H100_256GB", "A6000_48GB", "A40"]
            total_nodes:
            step: 0.10.05
            min_nodes_per_type: GPU1

        Returns:
             {gpu_type: ratio}
        """
        num_types = len(gpu_types)

        if num_types < 2:
            raise ValueError(f"2GPU{num_types}")

        if total_nodes < num_types * min_nodes_per_type:
            raise ValueError(
                f"total_nodes={total_nodes}, "
                f"{num_types * min_nodes_per_type}"
            )

        # GPU
        min_ratio = min_nodes_per_type / total_nodes

        combinations = []

        def generate_integer_allocations(
            remaining_nodes: int,
            remaining_types: int,
            current_allocation: List[int]
        ) -> List[List[int]]:
            """"""
            if remaining_types == 1:
                # GPU
                allocation = current_allocation + [remaining_nodes]
                return [allocation]

            allocations = []
            # GPUmin  remaining - (remaining_types-1)*min
            max_nodes = remaining_nodes - (remaining_types - 1) * min_nodes_per_type

            for n in range(min_nodes_per_type, max_nodes + 1):
                result = generate_integer_allocations(
                    remaining_nodes - n,
                    remaining_types - 1,
                    current_allocation + [n]
                )
                allocations.extend(result)

            return allocations

        integer_allocations = generate_integer_allocations(total_nodes, num_types, [])

        # step
        exact_match_combinations = []  # step
        approximate_combinations = []   #

        for allocation in integer_allocations:
            ratios = [n / total_nodes for n in allocation]
            ratio_dict = {gpu_type: ratios[i] for i, gpu_type in enumerate(gpu_types)}

            exact_match = True
            for ratio in ratios:
                # step
                if abs(round(ratio / step) * step - ratio) > 1e-6:
                    exact_match = False
                    break

            if exact_match:
                exact_match_combinations.append(ratio_dict)
            else:
                approximate_combinations.append(ratio_dict)

        if exact_match_combinations:
            combinations = exact_match_combinations
        elif approximate_combinations:
            import warnings
            warnings.warn(
                f" {total_nodes}  {step}"
                f""
                f" {len(approximate_combinations)} "
                f"1)  manual_ratios "
                f" 2) 10 3) "
            )
            combinations = approximate_combinations
        else:
            combinations = []

        unique_combinations = []
        seen = set()

        for combo in combinations:
            key = tuple(round(combo[gt], 6) for gt in gpu_types)
            if key not in seen:
                seen.add(key)
                unique_combinations.append(combo)

        return unique_combinations

    def distribute_by_ratio(
        self,
        ratio_config: Dict[str, float],
        adjacency: np.ndarray,
        map_mode: str = "high_to_high",
        rank_order: str = "desc",
        gpu_nodes: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        GPU


        1.
        2. rank_order: desc/asc
        3. GPU
        4. map_modehigh_to_high: GPU


            ratio_config = {"H100_256GB": 0.3, "A6000_48GB": 0.7}, total_nodes=20
            6H100 + 14A6000

        Args:
            ratio_config: GPU {"H100_256GB": 0.3, "A6000_48GB": 0.7}
            adjacency:
            map_mode:
                - "high_to_high": GPU
                - "high_to_low": GPU
            rank_order:
                - "desc":
                - "asc":
            gpu_nodes: GPUNone

        Returns:
            GPU
            {
                "node_index": 0,
                "gpu_type": "H100_256GB",
                "G_TFLOPS": 183.0,
                "VRAM_bytes": 256*1024^3,
                "cost_per_GB_month": 8.20,
                "degree": 5,
                "ratio_in_config": 0.3
            }
        """
        # 1.0
        total_ratio = sum(ratio_config.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"GPU1.0{total_ratio:.6f}"
            )

        # GPUcatalog
        for gpu_type in ratio_config.keys():
            if gpu_type not in self.gpu_catalog:
                raise ValueError(f"GPU {gpu_type} GPU_CATALOG")

        n = adjacency.shape[0]

        # 1
        degrees = adjacency.sum(axis=1).astype(int)

        # 2
        reverse = (rank_order == "desc")

        # gpu_nodesGPUGPU
        if gpu_nodes is not None:
            node_indices = gpu_nodes  # GPU
        else:
            node_indices = list(range(n))  # GPU

        node_indices_sorted = sorted(
            node_indices,
            key=lambda i: (degrees[i], i),
            reverse=reverse
        )

        # 3GPU
        gpu_types = list(ratio_config.keys())
        gpu_counts = {}
        # gpu_nodesGPU
        allocation_count = len(node_indices) if gpu_nodes is not None else n
        remaining = allocation_count

        # Use Hamilton apportionment so counts are based on the allocatable GPU
        # nodes instead of the full topology node count.
        raw_counts = []
        floor_counts = []
        for gpu_type in gpu_types:
            raw = ratio_config[gpu_type] * allocation_count
            raw_counts.append(raw)
            floor_counts.append(int(raw))

        remaining = allocation_count - sum(floor_counts)
        frac_order = sorted(
            range(len(gpu_types)),
            key=lambda i: (raw_counts[i] - floor_counts[i], -i),
            reverse=True,
        )

        for i, gpu_type in enumerate(gpu_types):
            gpu_counts[gpu_type] = floor_counts[i]

        for idx in frac_order[:remaining]:
            gpu_counts[gpu_types[idx]] += 1

        # 4map_modeGPU
        # GPUTFLOPS
        gpu_with_perf = [
            (gt, self.gpu_catalog[gt]["G_TFLOPS"])
            for gt in gpu_types
        ]
        gpu_sorted_by_perf = sorted(
            gpu_with_perf,
            key=lambda x: x[1],
            reverse=True
        )  #

        if map_mode == "high_to_high":
            #   GPU
            allocation_order = [gt for gt, _ in gpu_sorted_by_perf]
        elif map_mode == "high_to_low":
            #   GPU
            allocation_order = [
                gt for gt, _ in reversed(gpu_sorted_by_perf)
            ]
        else:
            raise ValueError(f"Unknown map_mode: {map_mode}")

        # 5-GPU
        assignment_pairs: List[Tuple[int, str]] = []
        pos = 0

        for gpu_type in allocation_order:
            count = gpu_counts[gpu_type]
            for _ in range(count):
                if pos >= len(node_indices_sorted):
                    break
                node_index = node_indices_sorted[pos]
                pos += 1
                assignment_pairs.append((node_index, gpu_type))

        # 6node_index
        result = []
        for i in range(n):
            # GPU
            gpu_assignment = next((name for idx, name in assignment_pairs if idx == i), None)

            if gpu_assignment is not None:
                # GPU
                gpu_info = self.gpu_catalog[gpu_assignment]
                result.append({
                    "node_index": i,
                    "gpu_type": gpu_info["name"],
                    "G_TFLOPS": gpu_info["G_TFLOPS"],
                    "VRAM_bytes": gpu_info["VRAM_bytes"],
                    "cost_per_GB_month": gpu_info["cost_per_GB_month"],
                    "degree": int(degrees[i]),
                    "ratio_in_config": ratio_config[gpu_assignment],
                })
            else:
                # GPU
                result.append({
                    "node_index": i,
                    "gpu_type": "None",
                    "G_TFLOPS": 0.0,
                    "VRAM_bytes": 0,
                    "cost_per_GB_month": 0.0,
                    "degree": int(degrees[i]),
                    "ratio_in_config": 0.0,
                })

        return result

    def get_gpu_summary(
        self,
        gpu_assignment: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        GPU

        Args:
            gpu_assignment: distribute_by_ratio()

        Returns:

        """
        from collections import Counter

        gpu_type_counts = Counter(
            item["gpu_type"] for item in gpu_assignment
        )

        total_nodes = len(gpu_assignment)

        summary = {
            "total_nodes": total_nodes,
            "gpu_types": list(gpu_type_counts.keys()),
            "gpu_counts": dict(gpu_type_counts),
            "gpu_ratios": {
                gt: count / total_nodes
                for gt, count in gpu_type_counts.items()
            },
            "total_tflops": sum(item["G_TFLOPS"] for item in gpu_assignment),
            "total_vram_gb": sum(item["VRAM_bytes"] for item in gpu_assignment) / (1024**3),
            "avg_degree": sum(item["degree"] for item in gpu_assignment) / total_nodes,
        }

        return summary


# ============================================================
# ============================================================

def create_sample_generator():
    """"""
    sample_catalog = {
        "H100_256GB": {
            "name": "H100_256GB",
            "G_TFLOPS": 183.0,
            "VRAM_bytes": int(256 * 1024**3),
            "cost_per_GB_month": 8.20
        },
        "A6000_48GB": {
            "name": "A6000_48GB",
            "G_TFLOPS": 38.71,
            "VRAM_bytes": int(48 * 1024**3),
            "cost_per_GB_month": 8.52
        },
        "A40": {
            "name": "A40",
            "G_TFLOPS": 250.0,
            "VRAM_bytes": int(48 * 1024**3),
            "cost_per_GB_month": 6.63
        }
    }
    return GPUDistributionGenerator(sample_catalog)


if __name__ == "__main__":
    import numpy as np

    print("=== GPU ===\n")

    generator = create_sample_generator()

    # 1
    print("12GPU200.1")
    combinations = generator.generate_ratio_combinations(
        gpu_types=["H100_256GB", "A6000_48GB"],
        total_nodes=20,
        step=0.1
    )
    print(f" {len(combinations)} ")
    for i, combo in enumerate(combinations[:5]):
        print(f"  {i+1}. {combo}")
    if len(combinations) > 5:
        print(f"  ... ({len(combinations)})")
    print()

    # 2
    print("2GPU")
    adjacency = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0]
    ])

    ratio_config = {"H100_256GB": 0.4, "A6000_48GB": 0.6}
    assignment = generator.distribute_by_ratio(
        ratio_config=ratio_config,
        adjacency=adjacency,
        map_mode="high_to_high",
        rank_order="desc"
    )

    print(f": {ratio_config}")
    print(":")
    for item in assignment:
        print(f"  {item['node_index']}: {item['gpu_type']} "
              f"(={item['degree']}, TFLOPS={item['G_TFLOPS']})")
    print()

    # 3
    print("3")
    summary = generator.get_gpu_summary(assignment)
    print(f": {summary['total_nodes']}")
    print(f"GPU: {summary['gpu_counts']}")
    print(f"GPU: {summary['gpu_ratios']}")
    print(f": {summary['total_tflops']:.2f} TFLOPS")
    print(f": {summary['avg_degree']:.2f}")
