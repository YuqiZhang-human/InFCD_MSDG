#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 /  dense


1.  GPU
   - fat_tree :  core / aggregation / edge / gpu  fat-tree
   - clos     : CLOS  GPU

2.
   -  /  / degree_list
   - gpu_nodes / switch_nodes
   - edge_tiers

3.
   - networkx
   -
   -
   - PNG
"""

import os
import json
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict


# ==========================
# ==========================

@dataclass
class NodeInfo:
    """"""
    node_id: int
    node_type: str  # "switch" or "gpu"
    layer: int      #
    label: str      #


@dataclass
class TopologyMetadata:
    """"""
    topology_type: str
    total_nodes: int
    switch_count: int
    gpu_count: int
    total_links: int

    layers: List[List[int]]                  #  ID
    layer_sizes: List[int]                   #
    node_layers: List[int]                   # node_id -> layer_idx
    degree_list: List[int]                   # node_id -> degree

    nodes: Dict[int, Dict[str, Any]]         # ID ->
    gpu_nodes: List[int]                     # GPUID
    switch_nodes: List[int]                  # ID
    edge_tiers: Dict[str, List[Tuple[int, int]]]  # tier_name -> [(u,v), ...]


# ==========================
# ==========================

CONFIG = {
    "topology_type": "fat_tree",
    "output_dir": "topology_output_with_switches",
    "random_seed": 42,

    "fat_tree": {
        "num_pods": 4,
        "core_switch_count": 4,
        "agg_per_pod": 2,
        "edge_per_pod": 2,
        "gpus_per_edge": 4,
    },

    "clos": {
        "layer_config": [2, 4, 3, 5, 4]
    }
}


# ==========================
# ==========================

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_metadata(metadata: TopologyMetadata, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _append_undirected_edge(
    edge_list: List[Tuple[int, int]],
    degree_list: List[int],
    u: int,
    v: int,
) -> None:
    if u == v:
        return
    a, b = _edge_key(u, v)
    edge_list.append((a, b))
    degree_list[a] += 1
    degree_list[b] += 1


# ==========================
# ==========================

class TopologyGeneratorWithSwitches:
    def __init__(self, config: Dict[str, Any]):
        self.config = self._normalize_config(config)
        seed = self.config.get("random_seed", None)
        if seed is not None:
            random.seed(seed)

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """

        """
        normalized = dict(config)
        topology_type = config.get("topology_type", "fat_tree")

        if topology_type == "fat_tree":
            fat_config = dict(config.get("fat_tree", {}))

            if "gpu_per_edge" in fat_config and "gpus_per_edge" not in fat_config:
                fat_config["gpus_per_edge"] = fat_config["gpu_per_edge"]
            if "pods" in fat_config and "num_pods" not in fat_config:
                fat_config["num_pods"] = fat_config["pods"]

            required_keys = [
                "num_pods",
                "core_switch_count",
                "agg_per_pod",
                "edge_per_pod",
                "gpus_per_edge",
            ]
            missing = [k for k in required_keys if k not in fat_config]
            if missing:
                raise ValueError(f"fat_tree : {missing}")

            normalized["fat_tree"] = fat_config

        elif topology_type == "clos":
            clos_config = dict(config.get("clos", {}))
            if "layer_sizes" in clos_config and "layer_config" not in clos_config:
                clos_config["layer_config"] = clos_config["layer_sizes"]

            if "layer_config" not in clos_config:
                raise ValueError("clos : layer_config")

            normalized["clos"] = clos_config

        else:
            raise ValueError(f": {topology_type}")

        return normalized

    def generate(self) -> TopologyMetadata:
        """
         G / adjacency / pos
        """
        ttype = self.config.get("topology_type", "fat_tree").lower()

        if ttype == "fat_tree":
            return self._generate_fat_tree()
        elif ttype == "clos":
            return self._generate_clos()
        else:
            raise ValueError(f": {ttype}")

    def _generate_fat_tree(self) -> TopologyMetadata:
        """
         fat-tree
        """
        params = self.config["fat_tree"]
        num_pods = int(params["num_pods"])
        core_switch_count = int(params["core_switch_count"])
        agg_per_pod = int(params["agg_per_pod"])
        edge_per_pod = int(params["edge_per_pod"])
        gpus_per_edge = int(params["gpus_per_edge"])

        if num_pods <= 0:
            raise ValueError("num_pods  > 0")
        if core_switch_count <= 0:
            raise ValueError("core_switch_count  > 0")
        if agg_per_pod <= 0:
            raise ValueError("agg_per_pod  > 0")
        if edge_per_pod <= 0:
            raise ValueError("edge_per_pod  > 0")
        if gpus_per_edge <= 0:
            raise ValueError("gpus_per_edge  > 0")

        agg_count = num_pods * agg_per_pod
        edge_count = num_pods * edge_per_pod
        gpu_count = num_pods * edge_per_pod * gpus_per_edge
        total_nodes = core_switch_count + agg_count + edge_count + gpu_count

        degree_list = [0] * total_nodes
        node_info: Dict[int, NodeInfo] = {}
        edge_tiers: Dict[str, List[Tuple[int, int]]] = {}

        current_node_id = 0

        # 1) core
        core_nodes: List[int] = []
        for i in range(core_switch_count):
            node_id = current_node_id
            current_node_id += 1
            core_nodes.append(node_id)
            node_info[node_id] = NodeInfo(
                node_id=node_id,
                node_type="switch",
                layer=0,
                label=f"C{i}"
            )

        # 2) pod  agg / edge
        agg_nodes_by_pod: List[List[int]] = []
        edge_nodes_by_pod: List[List[int]] = []

        for pod_idx in range(num_pods):
            pod_agg_nodes: List[int] = []
            pod_edge_nodes: List[int] = []

            for i in range(agg_per_pod):
                node_id = current_node_id
                current_node_id += 1
                pod_agg_nodes.append(node_id)
                node_info[node_id] = NodeInfo(
                    node_id=node_id,
                    node_type="switch",
                    layer=1,
                    label=f"A{pod_idx}-{i}"
                )

            for i in range(edge_per_pod):
                node_id = current_node_id
                current_node_id += 1
                pod_edge_nodes.append(node_id)
                node_info[node_id] = NodeInfo(
                    node_id=node_id,
                    node_type="switch",
                    layer=2,
                    label=f"E{pod_idx}-{i}"
                )

            agg_nodes_by_pod.append(pod_agg_nodes)
            edge_nodes_by_pod.append(pod_edge_nodes)

        # 3) edge  GPU
        gpu_nodes: List[int] = []
        for pod_idx in range(num_pods):
            for edge_local_idx, _edge_switch in enumerate(edge_nodes_by_pod[pod_idx]):
                for gpu_idx in range(gpus_per_edge):
                    gpu_id = current_node_id
                    current_node_id += 1
                    gpu_nodes.append(gpu_id)
                    node_info[gpu_id] = NodeInfo(
                        node_id=gpu_id,
                        node_type="gpu",
                        layer=3,
                        label=f"G{pod_idx}-{edge_local_idx}-{gpu_idx}"
                    )

        # 4)
        edge_tiers["tier_0_1"] = []
        all_agg_nodes = [n for pod in agg_nodes_by_pod for n in pod]
        for core in core_nodes:
            for agg in all_agg_nodes:
                _append_undirected_edge(edge_tiers["tier_0_1"], degree_list, core, agg)

        edge_tiers["tier_1_2"] = []
        for pod_idx in range(num_pods):
            for agg in agg_nodes_by_pod[pod_idx]:
                for edge in edge_nodes_by_pod[pod_idx]:
                    _append_undirected_edge(edge_tiers["tier_1_2"], degree_list, agg, edge)

        edge_tiers["tier_2_3"] = []
        gpu_ptr = core_switch_count + agg_count + edge_count
        for pod_idx in range(num_pods):
            for edge_local_idx, edge_switch in enumerate(edge_nodes_by_pod[pod_idx]):
                for _ in range(gpus_per_edge):
                    gpu_id = gpu_ptr
                    gpu_ptr += 1
                    _append_undirected_edge(edge_tiers["tier_2_3"], degree_list, edge_switch, gpu_id)

        agg_nodes_flat = [n for pod in agg_nodes_by_pod for n in pod]
        edge_nodes_flat = [n for pod in edge_nodes_by_pod for n in pod]
        layers = [core_nodes, agg_nodes_flat, edge_nodes_flat, gpu_nodes]
        layer_sizes = [len(layer) for layer in layers]

        node_layers = [0] * total_nodes
        for layer_idx, layer_nodes in enumerate(layers):
            for nid in layer_nodes:
                node_layers[nid] = layer_idx

        switch_nodes = core_nodes + agg_nodes_flat + edge_nodes_flat
        total_links = sum(len(v) for v in edge_tiers.values())

        return TopologyMetadata(
            topology_type="fat_tree",
            total_nodes=total_nodes,
            switch_count=len(switch_nodes),
            gpu_count=len(gpu_nodes),
            total_links=total_links,
            layers=layers,
            layer_sizes=layer_sizes,
            node_layers=node_layers,
            degree_list=degree_list,
            nodes={n: asdict(info) for n, info in node_info.items()},
            gpu_nodes=gpu_nodes,
            switch_nodes=switch_nodes,
            edge_tiers=edge_tiers,
        )

    def _generate_clos(self) -> TopologyMetadata:
        """
         CLOS
        """
        params = self.config["clos"]
        layer_config = [int(x) for x in params["layer_config"]]

        if len(layer_config) < 2:
            raise ValueError("layer_config 2")

        total_nodes = sum(layer_config)
        degree_list = [0] * total_nodes
        node_info: Dict[int, NodeInfo] = {}
        layers: List[List[int]] = []
        edge_tiers: Dict[str, List[Tuple[int, int]]] = {}

        current_node_id = 0

        for layer_idx, node_count in enumerate(layer_config):
            layer_nodes: List[int] = []
            is_switch_layer = (layer_idx == 0) or (layer_idx == len(layer_config) - 1)
            node_type = "switch" if is_switch_layer else "gpu"
            layer_label_prefix = "S" if is_switch_layer else "G"

            for i in range(node_count):
                node_id = current_node_id
                current_node_id += 1
                layer_nodes.append(node_id)
                node_info[node_id] = NodeInfo(
                    node_id=node_id,
                    node_type=node_type,
                    layer=layer_idx,
                    label=f"{layer_label_prefix}{layer_idx}-{i}"
                )

            layers.append(layer_nodes)

        for layer in range(len(layers) - 1):
            tier_name = f"tier_{layer}_{layer + 1}"
            edge_tiers[tier_name] = []
            current_layer_nodes = layers[layer]
            next_layer_nodes = layers[layer + 1]

            for u in current_layer_nodes:
                for v in next_layer_nodes:
                    _append_undirected_edge(edge_tiers[tier_name], degree_list, u, v)

        switch_nodes = layers[0] + layers[-1]
        gpu_nodes = [node for layer in layers[1:-1] for node in layer]
        total_links = sum(len(v) for v in edge_tiers.values())
        layer_sizes = [len(layer) for layer in layers]

        node_layers = [0] * total_nodes
        for layer_idx, layer_nodes in enumerate(layers):
            for nid in layer_nodes:
                node_layers[nid] = layer_idx

        return TopologyMetadata(
            topology_type="clos",
            total_nodes=total_nodes,
            switch_count=len(switch_nodes),
            gpu_count=len(gpu_nodes),
            total_links=total_links,
            layers=layers,
            layer_sizes=layer_sizes,
            node_layers=node_layers,
            degree_list=degree_list,
            nodes={n: asdict(info) for n, info in node_info.items()},
            gpu_nodes=gpu_nodes,
            switch_nodes=switch_nodes,
            edge_tiers=edge_tiers,
        )


# ==========================
# ==========================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, CONFIG["output_dir"])
    ensure_dir(output_dir)

    gen = TopologyGeneratorWithSwitches(CONFIG)
    metadata = gen.generate()

    print(f": {metadata.topology_type}")
    print(f": {metadata.total_nodes}")
    print(f"  - : {metadata.switch_count}")
    print(f"  - GPU: {metadata.gpu_count}")
    print(f": {metadata.total_links}")

    topo_name = f"{metadata.topology_type}_s{metadata.switch_count}_g{metadata.gpu_count}"

    meta_path = os.path.join(output_dir, f"{topo_name}_metadata.json")
    save_metadata(metadata, meta_path)
    print(f": {meta_path}")

    print("\n:")
    for i, layer in enumerate(metadata.layers):
        if not layer:
            continue
        first_node_id = layer[0]
        node_data = metadata.nodes.get(first_node_id) or metadata.nodes.get(str(first_node_id))
        first_node_type = node_data.get("node_type", "unknown") if node_data else "unknown"
        layer_name = "Switch Layer" if first_node_type == "switch" else "GPU Layer"
        print(f"  {layer_name} {i}: {len(layer)} nodes - {layer[:5]}{'...' if len(layer) > 5 else ''}")

    print("\n (edge_tiers):")
    for tier_name, edges in metadata.edge_tiers.items():
        print(f"  {tier_name}: {len(edges)} links")


if __name__ == "__main__":
    main()
