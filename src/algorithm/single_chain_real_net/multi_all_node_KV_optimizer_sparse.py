#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LMU, MSDG
"""

import json
import math
import copy
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.tools.sparse_topology import SparseTopology


class MultiFunctionOptimizerSparse:

    """ SparseTopology O(N^2) """

    def __init__(self, test_data: Dict[str, Any], verbose: bool = False) -> None:
        """
        Args:
            test_data:  sparse_topology: SparseTopology
                 MultiFunctionOptimizer
        """
        # test_data_id
        self.test_data_id = int(test_data.get("test_data_id", 0))
        self.verbose = bool(verbose)
        self.node_count = int(test_data["node_count"])
        self.module_count = int(test_data["module_count"])
        self.computation_capacity = self._parse_array(test_data["computation_capacity"])
        self.resource_demands = self._parse_array(test_data["resource_demands"])
        self.data_sizes = self._parse_array(test_data["data_sizes"])
        st = test_data.get("sparse_topology")
        if not isinstance(st, SparseTopology):
            raise TypeError(
                "MultiFunctionOptimizerSparse  test_data['sparse_topology']  SparseTopology"
            )
        self.topology: SparseTopology = st
        self.gpu_cost = float(test_data.get("gpu_cost", 0.0))
        self.memory_cost = float(test_data.get("memory_cost", 0.0))
        self.bandwidth_cost = float(test_data.get("bandwidth_cost", 0.0))
        self.profit_per_user = float(test_data.get("profit_per_user", 0.0))
        self.user_uplink_mb_per_user = float(test_data.get("user_uplink_mb_per_user", 0.0))
        self.user_downlink_mb_per_user = float(test_data.get("user_downlink_mb_per_user", 0.0))
        self.return_to_gateway_mb_per_user = float(
            test_data.get("return_to_gateway_mb_per_user", 0.0)
        )

        node_costs = test_data.get("node_costs")
        self.node_costs = self._parse_array(node_costs) if node_costs is not None else None
        self.distance_matrix = None

        if self.node_costs is None:
            self.node_costs = [[self.gpu_cost, self.memory_cost] for _ in range(self.node_count)]

        raw_weight_mem = test_data.get("weight_memory_demands")
        if raw_weight_mem is not None:
            parsed = self._parse_array(raw_weight_mem)
            if len(parsed) != self.module_count:
                raise ValueError("weight_memory_demands  module_count")
            self.weight_memory_demands = [float(x) for x in parsed]
        else:
            self.weight_memory_demands = [0.0 for _ in range(self.module_count)]

        #  /  I/O / gateway
        self.topology_type = (
            str(test_data.get("topology_type", ""))
            .lower()
            .replace("-", "_")
        )

        self.topology_params = self._parse_topology_params(
            test_data.get("topology_params", {})
        )
        self.topology_layers = self._parse_nested_int_list(
            test_data.get("layers", test_data.get("topology_layers", []))
        )

        self.gpu_nodes = self._parse_int_list(
            test_data.get("gpu_nodes", list(range(self.node_count)))
        )
        self.switch_nodes = self._parse_int_list(
            test_data.get("switch_nodes", [])
        )

        gateway_node_raw = test_data.get("gateway_node", test_data.get("gateway_node_list", []))
        gateway_nodes_from_raw = self._parse_int_list(gateway_node_raw)

        self.gateway_uplink_nodes = self._parse_int_list(
            test_data.get("gateway_uplink_nodes", [])
        )
        self.gateway_downlink_nodes = self._parse_int_list(
            test_data.get("gateway_downlink_nodes", [])
        )

        if not self.gateway_uplink_nodes and gateway_nodes_from_raw:
            self.gateway_uplink_nodes = gateway_nodes_from_raw[:]
        if not self.gateway_downlink_nodes and gateway_nodes_from_raw:
            self.gateway_downlink_nodes = gateway_nodes_from_raw[:]

        if self.topology_type == "fat_tree":
            inferred_core_nodes = self._infer_fat_tree_core_nodes()
            if not self.gateway_uplink_nodes:
                self.gateway_uplink_nodes = inferred_core_nodes[:]
            if not self.gateway_downlink_nodes:
                self.gateway_downlink_nodes = inferred_core_nodes[:]

        self.access_ingress_nodes = [int(x) for x in self.gateway_uplink_nodes]
        self.access_egress_nodes = [int(x) for x in self.gateway_downlink_nodes]

        #  gateway/
        if self.access_ingress_nodes:
            self.gateway_node = int(self.access_ingress_nodes[0])
        else:
            self.gateway_node = -1

        if self.topology_type == "fat_tree":
            if not self.access_ingress_nodes or not self.access_egress_nodes:
                raise ValueError(
                    "fat_tree requires gateway/core access nodes, but none were provided or inferred."
                )
            for gw in self.access_ingress_nodes + self.access_egress_nodes:
                if gw < 0 or gw >= self.node_count:
                    raise ValueError(
                        f"fat_tree gateway node {gw} out of range for node_count={self.node_count}"
                    )
        #  &  SparseTopology
        self.initial_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.remaining_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.modules_loaded_per_node: List[set] = [set() for _ in range(self.node_count)]

        self.global_candidate_limit = int(
            test_data.get("optimizer_candidate_limit_global", 96)
        )
        self.pod_candidate_limit = int(
            test_data.get("optimizer_candidate_limit_pod", 48)
        )
        self.edge_candidate_limit = int(
            test_data.get("optimizer_candidate_limit_edge", 12)
        )

        self.fat_tree_num_pods = int(test_data.get("fat_tree_num_pods", 0))
        self.fat_tree_core_switch_count = int(test_data.get("fat_tree_core_switch_count", 0))
        self.fat_tree_agg_per_pod = int(test_data.get("fat_tree_agg_per_pod", 0))
        self.fat_tree_edge_per_pod = int(test_data.get("fat_tree_edge_per_pod", 0))
        self.fat_tree_gpus_per_edge = int(test_data.get("fat_tree_gpus_per_edge", 0))
        self.fat_tree_gpu_start = int(test_data.get("fat_tree_gpu_start", -1))

        self._setup_candidate_pools()
        self._ingress_hop_sum_cache: Dict[int, Optional[int]] = {}
        self._egress_hop_sum_cache: Dict[int, Optional[int]] = {}
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def _fat_tree_selector_enabled(self) -> bool:
        return (
            self.topology_type == "fat_tree"
            and self.fat_tree_num_pods > 0
            and self.fat_tree_edge_per_pod > 0
            and self.fat_tree_gpus_per_edge > 0
            and self.fat_tree_gpu_start >= 0
        )

    def _fat_tree_gpu_local_index(self, node: int) -> int:
        return int(node) - self.fat_tree_gpu_start

    def _fat_tree_edge_id(self, node: int) -> int:
        local_idx = self._fat_tree_gpu_local_index(node)
        if local_idx < 0:
            return -1
        return local_idx // self.fat_tree_gpus_per_edge

    def _fat_tree_pod_id(self, node: int) -> int:
        edge_id = self._fat_tree_edge_id(node)
        if edge_id < 0:
            return -1
        return edge_id // self.fat_tree_edge_per_pod

    def _setup_candidate_pools(self) -> None:
        self._global_gpu_rank = sorted(
            self.gpu_nodes,
            key=lambda node: (
                -float(self.computation_capacity[node][0]),
                -float(self.computation_capacity[node][1]),
                int(node),
            ),
        )

        self._gpu_nodes_by_edge: Dict[int, List[int]] = {}
        self._gpu_nodes_by_pod: Dict[int, List[int]] = {}

        if not self._fat_tree_selector_enabled():
            return

        for node in self._global_gpu_rank:
            edge_id = self._fat_tree_edge_id(node)
            pod_id = self._fat_tree_pod_id(node)

            self._gpu_nodes_by_edge.setdefault(edge_id, []).append(node)
            self._gpu_nodes_by_pod.setdefault(pod_id, []).append(node)

    def _add_candidate_slice(
        self,
        seq: List[int],
        limit: int,
        out: List[int],
        seen: set,
    ) -> None:
        if limit <= 0:
            return
        for node in seq:
            if node in seen:
                continue
            seen.add(node)
            out.append(node)
            if len(out) >= limit:
                return

    def _build_candidate_nodes(self, module_idx: int, deployment: List[int]) -> List[int]:
        # Keep the sparse optimizer algorithmically equivalent to the dense
        # optimizer: only the topology storage/query layer should differ.
        return [int(node) for node in self.gpu_nodes]

    def _parse_array(self, data):
        """ JSON """
        if data is None:
            return None
        if isinstance(data, str):
            text = data.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                try:
                    return eval(text)
                except Exception:

                    raise ValueError(f": {data}")
        return data
    def _parse_int_list(self, value) -> List[int]:
        if value is None:
            return []
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    return [int(s)]
                except Exception:
                    return []
        else:
            parsed = value

        if isinstance(parsed, list):
            result = []
            for x in parsed:
                if x is None or x == "":
                    continue
                result.append(int(x))
            return result

        try:
            return [int(parsed)]
        except Exception:
            return []

    def _parse_nested_int_list(self, value) -> List[List[int]]:
        if value is None:
            return []
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            try:
                value = json.loads(s)
            except Exception:
                return []

        if not isinstance(value, list):
            return []

        result: List[List[int]] = []
        for group in value:
            if isinstance(group, list):
                result.append([int(x) for x in group if x is not None and x != ""])
        return result

    def _parse_topology_params(self, value) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return {}
            try:
                value = json.loads(s)
            except Exception:
                return {}
        return dict(value) if isinstance(value, dict) else {}

    def _infer_fat_tree_core_nodes(self) -> List[int]:
        if self.topology_layers and len(self.topology_layers) >= 1 and self.topology_layers[0]:
            return [int(x) for x in self.topology_layers[0]]

        core_switch_count = int(self.topology_params.get("core_switch_count", 0))
        if core_switch_count > 0:
            return list(range(core_switch_count))

        return []

    def get_node_capacity(self, node: int) -> Tuple[float, float]:
        """ [, ] """
        if node < 0 or node >= self.node_count:
            return 0.0, 0.0
        return (
            float(self.remaining_computation_capacity[node][0]),
            float(self.remaining_computation_capacity[node][1]),
        )

    def _get_fat_tree_gateway_internal_cost(
            self, deployment: List[int], user_count: int
    ) -> float:
        if self.topology_type != "fat_tree":
            return 0.0
        if user_count <= 0 or not deployment:
            return 0.0
        if not self.access_ingress_nodes or not self.access_egress_nodes:
            return float("inf")

        total = 0.0
        first_node = int(deployment[0])
        last_node = int(deployment[-1])
        ingress_hops = self._get_ingress_hop_sum(first_node)
        egress_hops = self._get_egress_hop_sum(last_node)
        if ingress_hops is None or egress_hops is None:
            return float("inf")

        if self.user_uplink_mb_per_user > 0:
            uplink_per_gw = self.user_uplink_mb_per_user / len(self.access_ingress_nodes)
            total += uplink_per_gw * self.bandwidth_cost * user_count * ingress_hops

        if self.return_to_gateway_mb_per_user > 0:
            downlink_per_gw = self.return_to_gateway_mb_per_user / len(self.access_egress_nodes)
            total += downlink_per_gw * self.bandwidth_cost * user_count * egress_hops

        return total

    def _get_clos_gateway_internal_cost(
            self, deployment: List[int], user_count: int
    ) -> float:
        """
        CLOS /
        1) gateway_uplink_nodes -> first_node
        2) last_node -> gateway_downlink_nodes
        """
        if self.topology_type != "clos":
            return 0.0
        if user_count <= 0 or not deployment:
            return 0.0

        total = 0.0
        first_node = int(deployment[0])
        last_node = int(deployment[-1])

        # gateway_uplink_nodes -> first_node
        if self.gateway_uplink_nodes and self.user_uplink_mb_per_user > 0:
            uplink_per_gw = self.user_uplink_mb_per_user / len(self.gateway_uplink_nodes)
            for gw_node in self.gateway_uplink_nodes:
                gw_node = int(gw_node)
                if gw_node == first_node or uplink_per_gw <= 0:
                    continue

                path = self.get_path_nodes(gw_node, first_node)
                if not path or len(path) < 2:
                    return float("inf")

                hop = len(path) - 1
                total += uplink_per_gw * self.bandwidth_cost * user_count * hop

        # last_node -> gateway_downlink_nodes
        if self.gateway_downlink_nodes and self.user_downlink_mb_per_user > 0:
            downlink_per_gw = self.user_downlink_mb_per_user / len(self.gateway_downlink_nodes)
            for gw_node in self.gateway_downlink_nodes:
                gw_node = int(gw_node)
                if gw_node == last_node or downlink_per_gw <= 0:
                    continue

                path = self.get_path_nodes(last_node, gw_node)
                if not path or len(path) < 2:
                    return float("inf")

                hop = len(path) - 1
                total += downlink_per_gw * self.bandwidth_cost * user_count * hop

        return total

    def _get_total_access_comm_cost(
            self, deployment: List[int], user_count: int
    ) -> float:
        """
         deployment  boundary
        - CLOS:
             __user__ -> ingress gateways
             egress gateways -> __user__
             ingress gateways -> first_node
             last_node -> egress gateways
        - fat-tree:
             __user__ -> gateway
             gateway -> __user__
             gateway -> first_node
             last_node -> gateway
        """
        if user_count <= 0 or not deployment:
            return 0.0

        access_cost = 0.0

        if self.topology_type == "clos":
            # /
            access_cost += (
                self.user_uplink_mb_per_user + self.user_downlink_mb_per_user
            ) * self.bandwidth_cost * user_count

            #  /
            extra_internal = self._get_clos_gateway_internal_cost(deployment, user_count)
            if math.isinf(extra_internal):
                return float("inf")
            access_cost += extra_internal

        elif self.topology_type == "fat_tree":
            #  <-> gateway
            access_cost += (
                self.user_uplink_mb_per_user + self.user_downlink_mb_per_user
            ) * self.bandwidth_cost * user_count

            #  gateway -> first_node, last_node -> gateway
            extra_internal = self._get_fat_tree_gateway_internal_cost(deployment, user_count)
            if math.isinf(extra_internal):
                return float("inf")
            access_cost += extra_internal

        return access_cost

    def get_module_demand(self, module: int) -> Tuple[float, float]:
        """
         [, KV (GB/user)]
         KV
        """
        if module < 0 or module >= self.module_count:
            return 0.0, 0.0
        return (
            float(self.resource_demands[module][0]),
            float(self.resource_demands[module][1]),
        )
    def get_module_weight_memory(self, module: int) -> float:
        """ (GB)"""
        if module < 0 or module >= self.module_count:
            return 0.0
        return float(self.weight_memory_demands[module])
    def get_link_bandwidth(self, from_node: int, to_node: int) -> float:
        """ (MB/s)"""
        if (
            from_node < 0
            or from_node >= self.node_count
            or to_node < 0
            or to_node >= self.node_count
        ):
            return 0.0
        return float(self.topology.get_link_bandwidth(from_node, to_node))
    def get_data_size(self, boundary_index: int) -> float:
        """ (MB/s)"""
        if boundary_index < 0 or boundary_index >= len(self.data_sizes):
            return 0.0
        return float(self.data_sizes[boundary_index])
    def get_path_nodes(self, s: int, t: int):
        """ s->t  None"""
        return self.topology.get_path_nodes(s, t)

    def get_path_hops(self, s: int, t: int) -> int:
        """ hop """
        return int(self.topology.get_path_hops(s, t))

    def _sum_hops_to_node(
            self,
            sources: List[int],
            node: int,
            cache: Dict[int, Optional[int]],
    ) -> Optional[int]:
        node = int(node)
        if node in cache:
            return cache[node]

        total = 0
        for src in sources:
            src = int(src)
            if src == node:
                continue
            hop = self.get_path_hops(src, node)
            if hop <= 0 or hop >= 10 ** 9:
                cache[node] = None
                return None
            total += hop

        cache[node] = total
        return total

    def _get_ingress_hop_sum(self, node: int) -> Optional[int]:
        return self._sum_hops_to_node(self.access_ingress_nodes, node, self._ingress_hop_sum_cache)

    def _get_egress_hop_sum(self, node: int) -> Optional[int]:
        return self._sum_hops_to_node(self.access_egress_nodes, node, self._egress_hop_sum_cache)
    def get_chain_total_hops(self, deployment: list[int]) -> int:
        """ hop  hop=0"""
        m = self.module_count
        if not deployment or len(deployment) != m:
            return 0
        total = 0
        for i in range(m - 1):
            a = deployment[i]
            b = deployment[i + 1]
            if a == b:
                continue
            hops = self.get_path_hops(a, b)  # hops: int
            #  1e9/INF
            if hops is None:
                continue
            if isinstance(hops, (int, float)) and hops >= 1e8:
                continue
            total += int(hops)
        return total
    def get_path_bottleneck_bw(self, s: int, t: int) -> float:
        """
         s->t min edge bw
         0
        """
        path = self.get_path_nodes(s, t)
        if not path:
            return 0.0
        if len(path) < 2:
            return float('inf')
        b = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            b = min(b, float(self.topology.get_link_bandwidth(u, v)))
        return 0.0 if b == float('inf') else float(b)
    # ----------------------------------------------------------------------
    #  + KV
    # ----------------------------------------------------------------------

    def calculate_node_memory_stats_for_deployment(
            self,
            deployment: List[int],
            user_count: int
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
         deployment + user_count
        1) node_memory_used_gb
        2) node_memory_util


        - weight_memory_demands[module]
        - resource_demands[module][1]  KV  / user user_count
        """
        node_weight_gb = {i: 0.0 for i in range(self.node_count)}
        node_kv_gb = {i: 0.0 for i in range(self.node_count)}

        for module_idx, node in enumerate(deployment):
            node = int(node)

            weight_gb = float(self.get_module_weight_memory(module_idx))
            node_weight_gb[node] += weight_gb

            # KV per-user * users
            _, kv_per_user_gb = self.get_module_demand(module_idx)
            node_kv_gb[node] += float(kv_per_user_gb) * float(user_count)

        node_memory_used_gb = {}
        node_memory_util = {}

        for node in range(self.node_count):
            used_gb = node_weight_gb[node] + node_kv_gb[node]
            mem_cap_gb = float(self.initial_computation_capacity[node][1])

            node_memory_used_gb[node] = used_gb
            if mem_cap_gb > 0:
                node_memory_util[node] = used_gb / mem_cap_gb
            else:
                node_memory_util[node] = 0.0

        return node_memory_used_gb, node_memory_util

    def calculate_aggregated_node_memory_stats(
            self,
            deployment_list: List[List[int]],
            served_users_list: List[int]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """

        -
        - KV  served_users
        """
        node_weight_gb = {i: 0.0 for i in range(self.node_count)}
        node_kv_gb = {i: 0.0 for i in range(self.node_count)}
        loaded_module_on_node = {i: set() for i in range(self.node_count)}

        if not deployment_list or not served_users_list:
            node_memory_util = {
                i: 0.0 for i in range(self.node_count)
            }
            return node_weight_gb, node_memory_util

        for deployment, user_count in zip(deployment_list, served_users_list):
            if not deployment or user_count <= 0:
                continue

            for module_idx, node in enumerate(deployment):
                node = int(node)

                if module_idx not in loaded_module_on_node[node]:
                    node_weight_gb[node] += float(self.get_module_weight_memory(module_idx))
                    loaded_module_on_node[node].add(module_idx)

                _, kv_per_user_gb = self.get_module_demand(module_idx)
                node_kv_gb[node] += float(kv_per_user_gb) * float(user_count)

        node_memory_used_gb = {}
        node_memory_util = {}

        for node in range(self.node_count):
            used_gb = node_weight_gb[node] + node_kv_gb[node]
            mem_cap_gb = float(self.initial_computation_capacity[node][1])

            node_memory_used_gb[node] = used_gb
            node_memory_util[node] = (used_gb / mem_cap_gb) if mem_cap_gb > 0 else 0.0

        return node_memory_used_gb, node_memory_util

    def _build_edge_flow_per_user_for_deployment(
            self,
            deployment: List[int]
    ) -> Dict[Tuple[int, int], float]:
        """
         deployment  per-user MB/s per user


        1)  boundary
        2) fat-tree: gateway -> first_node
        3) fat-tree: last_node -> gateway
        4) clos: gateway_uplink_nodes -> first_node
        5) clos: last_node -> gateway_downlink_nodes


        - external_traffic_list  __user__

        -
           (u, v) (v, u)

          - calculate_max_users_for_deployment
          - apply_chain_consumption
          - build_edge_traffic_list
        """
        edge_flow_per_user = defaultdict(float)
        #  boundary
        # len == module_count
        chain_len = len(deployment)

        # 1)  boundary
        for boundary_idx in range(chain_len - 1):
            from_node = int(deployment[boundary_idx])
            to_node = int(deployment[boundary_idx + 1])

            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            path = self.get_path_nodes(from_node, to_node)
            if not path or len(path) < 2:
                return {}

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_flow_per_user[(u, v)] += data_size

        # 2) fat-tree: gateway -> first_node
        # 3) fat-tree: last_node -> gateway
        if self.topology_type == "fat_tree" and deployment:
            first_node = int(deployment[0])
            last_node = int(deployment[-1])

            if self.access_ingress_nodes and self.user_uplink_mb_per_user > 0:
                uplink_per_gw = self.user_uplink_mb_per_user / len(self.access_ingress_nodes)
                for gw in self.access_ingress_nodes:
                    gw = int(gw)
                    if gw == first_node or uplink_per_gw <= 0:
                        continue
                    path = self.get_path_nodes(gw, first_node)
                    if not path or len(path) < 2:
                        return {}
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_flow_per_user[(u, v)] += uplink_per_gw

            if self.access_egress_nodes and self.return_to_gateway_mb_per_user > 0:
                downlink_per_gw = self.return_to_gateway_mb_per_user / len(self.access_egress_nodes)
                for gw in self.access_egress_nodes:
                    gw = int(gw)
                    if gw == last_node or downlink_per_gw <= 0:
                        continue
                    path = self.get_path_nodes(last_node, gw)
                    if not path or len(path) < 2:
                        return {}
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_flow_per_user[(u, v)] += downlink_per_gw

        # 4) & 5) clos: gateway_uplink_nodes -> first_node  last_node -> gateway_downlink_nodes
        if self.topology_type == "clos" and deployment:
            first_node = int(deployment[0])
            last_node = int(deployment[-1])

            # Uplink:  -> first_node
            if self.gateway_uplink_nodes and self.user_uplink_mb_per_user > 0:
                uplink_per_gw = self.user_uplink_mb_per_user / len(self.gateway_uplink_nodes)
                for gw_node in self.gateway_uplink_nodes:
                    if gw_node != first_node and uplink_per_gw > 0:
                        path = self.get_path_nodes(int(gw_node), first_node)
                        if path and len(path) >= 2:
                            for i in range(len(path) - 1):
                                u, v = path[i], path[i + 1]
                                edge_flow_per_user[(u, v)] += uplink_per_gw

            # Downlink: last_node ->
            if self.gateway_downlink_nodes and self.user_downlink_mb_per_user > 0:
                downlink_per_gw = self.user_downlink_mb_per_user / len(self.gateway_downlink_nodes)
                for gw_node in self.gateway_downlink_nodes:
                    if gw_node != last_node and downlink_per_gw > 0:
                        path = self.get_path_nodes(last_node, int(gw_node))
                        if path and len(path) >= 2:
                            for i in range(len(path) - 1):
                                u, v = path[i], path[i + 1]
                                edge_flow_per_user[(u, v)] += downlink_per_gw

        return edge_flow_per_user

    def calculate_max_users_for_deployment(self, deployment: List[int]) -> int:
        """
         +  self.modules_loaded_per_node

        -  / ( + KV)
        -
        """
        n = self.node_count
        m = self.module_count
        if len(deployment) != m:

            return 0

        # 1.  &
        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n  #
        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand
            weight_mem = self.get_module_weight_memory(module_idx)
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem
        limits: List[float] = [] #  float  floor
        #  &
        for node in range(n):
            comp_cap, mem_cap = self.get_node_capacity(node)
            comp_use = node_compute_per_user[node]
            kv_use = node_kv_per_user[node]
            new_weight = node_new_weight[node]
            if comp_use <= 0 and kv_use <= 0 and new_weight <= 0:
                continue
            if new_weight > mem_cap + 1e-9:
                return 0
            comp_limit = float("inf")
            mem_limit = float("inf")
            if comp_use > 0:
                if comp_cap <= 0:
                    return 0
                comp_limit = comp_cap / comp_use
            if kv_use > 0:
                avail_mem = mem_cap - new_weight
                if avail_mem <= 0:
                    return 0
                mem_limit = avail_mem / kv_use
            node_limit = min(comp_limit, mem_limit)
            if node_limit <= 0:
                return 0
            limits.append(node_limit)
        # 2.  flow
        edge_flow_per_user = self._build_edge_flow_per_user_for_deployment(deployment)
        if edge_flow_per_user is None:
            return 0

        undirected_edge_flow_per_user: Dict[Tuple[int, int], float] = defaultdict(float)
        for (u, v), flow_per_user in edge_flow_per_user.items():
            if flow_per_user <= 0:
                continue
            key = (u, v) if u < v else (v, u)
            undirected_edge_flow_per_user[key] += flow_per_user

        for (u, v), flow_per_user in undirected_edge_flow_per_user.items():
            if flow_per_user <= 0:
                continue

            remain_bw = min(
                self.get_link_bandwidth(u, v),
                self.get_link_bandwidth(v, u),
            )
            if remain_bw <= 0:
                return 0

            user_by_edge = remain_bw / flow_per_user
            if user_by_edge <= 0:
                return 0

            limits.append(user_by_edge)
        if not limits:
            return 0
        return max(0, int(math.floor(min(limits))))

    def calculate_costs_for_deployment(
        self, deployment: List[int], user_count: int
    ) -> Tuple[float, float, float, float]:
        """

        - total_cost
        - deploy_cost         +
        - communication_cost
        - profit
        """
        n = self.node_count
        m = self.module_count
        if user_count <= 0 or len(deployment) != m:
            return 0.0, 0.0, 0.0, 0.0
        # 1. per user +
        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n
        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand
            weight_mem = self.get_module_weight_memory(module_idx)
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem
        # 2. GPU +
        deploy_cost = 0.0
        for node in range(n):
            comp_use = node_compute_per_user[node] * user_count
            kv_use = node_kv_per_user[node] * user_count
            weight_use = node_new_weight[node]
            gpu_cost_node, mem_cost_node = self.node_costs[node]
            #  gpu_cost_node  0
            deploy_cost += comp_use * float(gpu_cost_node)
            deploy_cost += (kv_use + weight_use) * float(mem_cost_node)
        # 3.
        # 3.
        comm_cost = 0.0

        # 3.1  boundary
        for boundary_idx in range(m - 1):
            from_node = int(deployment[boundary_idx])
            to_node = int(deployment[boundary_idx + 1])

            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            hop = self.get_path_hops(from_node, to_node)
            if hop <= 0 or hop >= 10 ** 9:
                return float("inf"), float("inf"), float("inf"), float("-inf")
            comm_cost += data_size * self.bandwidth_cost * user_count * hop

        # 3.2  + gateway/
        access_comm_cost = self._get_total_access_comm_cost(deployment, user_count)
        if math.isinf(access_comm_cost):
            return float("inf"), float("inf"), float("inf"), float("-inf")
        comm_cost += access_comm_cost

        total_cost = deploy_cost + comm_cost
        profit = self.profit_per_user * user_count - total_cost
        return total_cost, deploy_cost, comm_cost, profit
    # ----------------------------------------------------------------------
    #  &
    # ----------------------------------------------------------------------

    def apply_chain_consumption(self, deployment: List[int], user_count: int) -> None:
        """

         modules_loaded_per_node
        """
        if user_count <= 0:
            return
        n = self.node_count
        m = self.module_count
        # 1.  + KV +
        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n
        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand
            weight_mem = self.get_module_weight_memory(module_idx)
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem
                self.modules_loaded_per_node[node].add(module_idx)
        for node in range(n):
            comp_cap, mem_cap = self.remaining_computation_capacity[node]
            comp_cap = float(comp_cap) - node_compute_per_user[node] * user_count
            mem_cap = (
                float(mem_cap)
                - node_kv_per_user[node] * user_count
                - node_new_weight[node]
            )
            if comp_cap < 0.0:
                comp_cap = 0.0
            if mem_cap < 0.0:
                mem_cap = 0.0
            self.remaining_computation_capacity[node][0] = comp_cap
            self.remaining_computation_capacity[node][1] = mem_cap
        # 2.  per-user
        # 2.  flow
        edge_flow_per_user = self._build_edge_flow_per_user_for_deployment(deployment)
        if edge_flow_per_user is None:
            return

        undirected_edge_flow_per_user = defaultdict(float)
        for (u, v), flow_per_user in edge_flow_per_user.items():
            if flow_per_user <= 0:
                continue
            key = (u, v) if u < v else (v, u)
            undirected_edge_flow_per_user[key] += flow_per_user

        for (u, v), flow_per_user in undirected_edge_flow_per_user.items():
            consume = flow_per_user * user_count
            self.topology.decrement_bandwidth(u, v, consume)
    # ----------------------------------------------------------------------
    #  &
    # ----------------------------------------------------------------------
    def _get_partial_access_comm_cost(
            self,
            partial_deployment: List[int],
            user_count: int
    ) -> float:
        """
         user_count


        - clos:
            *  __user__ -> first_node
            *  last_node -> __user__

        - fat-tree:
            *  __user__ -> gateway
            * gateway -> __user__

            * gateway -> first_node
            *  last_node -> gateway
               hop
        """
        if not partial_deployment or user_count <= 0:
            return 0.0

        first_node = int(partial_deployment[0])
        last_node = int(partial_deployment[-1])

        user_count_f = float(user_count)
        total = 0.0

        # 1) /clos / fat-tree
        if self.user_uplink_mb_per_user > 0:
            total += float(self.user_uplink_mb_per_user) * self.bandwidth_cost * user_count_f
        if self.user_downlink_mb_per_user > 0:
            total += float(self.user_downlink_mb_per_user) * self.bandwidth_cost * user_count_f

        # 2) fat-tree gateway  /  hop
        if self.topology_type == "fat_tree":
            ingress_hops = self._get_ingress_hop_sum(first_node)
            egress_hops = self._get_egress_hop_sum(last_node)
            if ingress_hops is None or egress_hops is None:
                return float("inf")

            if self.access_ingress_nodes and self.user_uplink_mb_per_user > 0:
                uplink_per_gw = float(self.user_uplink_mb_per_user) / len(self.access_ingress_nodes)
                total += uplink_per_gw * self.bandwidth_cost * user_count_f * ingress_hops

            if self.access_egress_nodes and self.return_to_gateway_mb_per_user > 0:
                downlink_per_gw = float(self.return_to_gateway_mb_per_user) / len(self.access_egress_nodes)
                total += downlink_per_gw * self.bandwidth_cost * user_count_f * egress_hops

        return total

    def _evaluate_prefix_chain(
            self, partial_deployment: List[int]
    ) -> Optional[Tuple[int, float, float]]:
        """


         DFS  0..len-1

        -  users/

        -  total_cost + boundary
          + /
        -  profit = profit_per_user * users - total_cost

         (users, total_cost, profit) None
         calculate_max_users_for_deployment / calculate_costs_for_deployment
        /
        """
        L = len(partial_deployment)
        if L == 0:
            return None

        n = self.node_count
        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n
        for module_idx, node in enumerate(partial_deployment):
            node = int(node)
            if node < 0 or node >= n:
                return None
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += self.get_module_weight_memory(module_idx)

        limits: List[float] = []
        for node in range(n):
            comp_cap, mem_cap = self.get_node_capacity(node)
            comp_use = node_compute_per_user[node]
            kv_use = node_kv_per_user[node]
            new_weight = node_new_weight[node]
            if comp_use <= 0 and kv_use <= 0 and new_weight <= 0:
                continue
            if new_weight > mem_cap + 1e-9:
                return None
            if comp_use > 0:
                if comp_cap <= 0:
                    return None
                limits.append(comp_cap / comp_use)
            if kv_use > 0:
                avail_mem = mem_cap - new_weight
                if avail_mem <= 0:
                    return None
                limits.append(avail_mem / kv_use)

        edge_flow_per_user = self._build_edge_flow_per_user_for_deployment(partial_deployment)
        undirected_edge_flow: Dict[Tuple[int, int], float] = defaultdict(float)
        for (u, v), flow_per_user in edge_flow_per_user.items():
            if flow_per_user <= 0:
                continue
            key = (u, v) if u < v else (v, u)
            undirected_edge_flow[key] += flow_per_user
        for (u, v), flow_per_user in undirected_edge_flow.items():
            remain_bw = min(self.get_link_bandwidth(u, v), self.get_link_bandwidth(v, u))
            if remain_bw <= 0:
                return None
            limits.append(remain_bw / flow_per_user)

        if not limits:
            users = 1
        else:
            users = int(math.floor(min(limits)))
            if users <= 0:
                return None

        deploy_cost = 0.0
        for node in range(n):
            gpu_cost_node, mem_cost_node = self.node_costs[node]
            deploy_cost += node_compute_per_user[node] * users * float(gpu_cost_node)
            deploy_cost += (
                node_kv_per_user[node] * users + node_new_weight[node]
            ) * float(mem_cost_node)

        comm_cost = 0.0
        for boundary_idx in range(L - 1):
            from_node = int(partial_deployment[boundary_idx])
            to_node = int(partial_deployment[boundary_idx + 1])
            if from_node == to_node:
                continue
            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue
            hop = self.get_path_hops(from_node, to_node)
            if hop <= 0 or hop >= 10 ** 9:
                return None
            comm_cost += data_size * self.bandwidth_cost * users * hop

        access_comm_cost = self._get_partial_access_comm_cost(partial_deployment, users)
        if not math.isfinite(access_comm_cost):
            return None
        comm_cost += access_comm_cost

        total_cost = deploy_cost + comm_cost
        profit = self.profit_per_user * users - total_cost
        return users, total_cost, profit

    def _deploy_single_chain_greedy(
            self, objective: str
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int], int]]:
        """
         +


        objective:
            - "min_cost"
            - "max_profit"
            - "min_profit"
            - "max_users"

        :
            (
                total_cost,
                deploy_cost,
                comm_cost,
                profit,
                max_users,
                used_nodes_count,
                avg_modules_per_node,
                deployment,
                chain_len,
            )
             None
        """
        n = self.node_count
        m = self.module_count
        if n <= 0 or m <= 0:
            return None

        deployment = [-1] * m

        for module_idx in range(m):
            best_candidate = None
            candidate_nodes = self._build_candidate_nodes(module_idx, deployment)

            for node in candidate_nodes:
                node = int(node)
                #  0..module_idx-1 +
                #  /  /
                partial_deployment = deployment[:module_idx] + [node]
                prefix_eval = self._evaluate_prefix_chain(partial_deployment)
                if prefix_eval is None:
                    continue
                users_prefix, total_cost_prefix, profit_prefix = prefix_eval

                if objective == "min_cost":
                    score = (
                        total_cost_prefix,
                        -profit_prefix,
                        -users_prefix,
                    )
                elif objective == "max_profit":
                    score = (
                        -profit_prefix,
                        -users_prefix,
                        total_cost_prefix,
                    )
                elif objective == "min_profit":
                    score = (
                        profit_prefix,
                        total_cost_prefix,
                        users_prefix,
                    )
                elif objective == "max_users":
                    score = (
                        -users_prefix,
                        total_cost_prefix,
                        -profit_prefix,
                    )
                else:
                    continue

                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (score, node)

            if best_candidate is None:
                return None

            chosen_node = best_candidate[1]
            deployment[module_idx] = chosen_node

        max_users = self.calculate_max_users_for_deployment(deployment)
        if max_users <= 0:
            return None

        total_cost, deploy_cost, comm_cost, profit = self.calculate_costs_for_deployment(
            deployment, max_users
        )

        used_nodes_count = len(set(deployment))
        avg_mods_per_node = (
            self.module_count / used_nodes_count if used_nodes_count > 0 else 0.0
        )
        chain_len = self.get_chain_total_hops(deployment)

        return (
            total_cost,
            deploy_cost,
            comm_cost,
            profit,
            max_users,
            used_nodes_count,
            avg_mods_per_node,
            deployment.copy(),
            chain_len,
        )

    def find_best_single_chain_for_objective(
            self, objective: str
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int], int]]:
        """


        """
        return self._deploy_single_chain_greedy(objective)
    # ----------------------------------------------------------------------
    #  objective
    # ----------------------------------------------------------------------
    def build_external_traffic_list_for_deployment(
            self,
            deployment: List[int],
            user_count: int
    ) -> List[Dict[str, Any]]:
        """
             deployment

             remaining_bandwidth_matrix
        """
        if not deployment or user_count <= 0:
            return []
        first_node = int(deployment[0])
        last_node = int(deployment[-1])
        external_edges: List[Dict[str, Any]] = []
        uplink_bw = float(self.user_uplink_mb_per_user) * float(user_count)
        downlink_bw = float(self.user_downlink_mb_per_user) * float(user_count)

        # gateway
        if self.topology_type == "fat_tree":
            access_ingress_nodes = [int(x) for x in self.access_ingress_nodes]
            access_egress_nodes = [int(x) for x in self.access_egress_nodes]
        elif self.gateway_uplink_nodes and self.gateway_downlink_nodes:
            # Clos: gateway
            access_ingress_nodes = self.gateway_uplink_nodes
            access_egress_nodes = self.gateway_downlink_nodes
        else:
            # gateway_nodes
            access_ingress_nodes = [first_node]
            access_egress_nodes = [last_node]

        # gateway
        uplink_bw_per_node = uplink_bw / len(access_ingress_nodes) if access_ingress_nodes else 0.0
        downlink_bw_per_node = downlink_bw / len(access_egress_nodes) if access_egress_nodes else 0.0

        # gatewayexternal edge
        for gw_node in access_ingress_nodes:
            if uplink_bw_per_node > 1e-12:
                external_edges.append({
                    "from_node": "__user__",
                    "to_node": int(gw_node),
                    "used_bandwidth": float(uplink_bw_per_node),
                    "initial_bandwidth": None,
                    "traffic_type": "external_uplink",
                })

        for gw_node in access_egress_nodes:
            if downlink_bw_per_node > 1e-12:
                external_edges.append({
                    "from_node": int(gw_node),
                    "to_node": "__user__",
                    "used_bandwidth": float(downlink_bw_per_node),
                    "initial_bandwidth": None,
                    "traffic_type": "external_downlink",
                })

        return external_edges

    def _merge_external_traffic_edges(
            self,
            acc: Dict[Tuple[str, str, str], Dict[str, Any]],
            new_edges: List[Dict[str, Any]]
    ) -> None:
        """
         new_edges  acc
        key = (from_node, to_node, traffic_type)
        """
        for e in new_edges:
            key = (
                str(e["from_node"]),
                str(e["to_node"]),
                str(e.get("traffic_type", "external"))
            )
            if key not in acc:
                acc[key] = {
                    "from_node": e["from_node"],
                    "to_node": e["to_node"],
                    "used_bandwidth": float(e["used_bandwidth"]),
                    "initial_bandwidth": e.get("initial_bandwidth", None),
                    "traffic_type": e.get("traffic_type", "external"),
                }
            else:
                acc[key]["used_bandwidth"] += float(e["used_bandwidth"])

    def build_edge_traffic_list(self) -> List[Dict[str, float]]:
        """

         edge_traffic_list


        -
        - i < j

        """
        edge_traffic_list: List[Dict[str, float]] = []
        for i, j, initial_bw, used_bw in self.topology.iter_edges_with_usage():
            edge_traffic_list.append({
                "from_node": int(i),
                "to_node": int(j),
                "used_bandwidth": float(used_bw),
                "initial_bandwidth": float(initial_bw),
                "traffic_type": "internal",
            })
        return edge_traffic_list

    def deploy_until_exhaustion(self, objective: str):
        """



            :(
                    total_cost,
                    total_deploy_cost,
                    total_comm_cost,
                    total_profit,
                    total_users,
                    used_nodes_count,
                    avg_modules_per_node,
                    chain_count,
                    chain_len_list,
                    chain_avg_modules_list,
                    chain_used_nodes_list,
                    chain_deployment_list,
                    edge_traffic_list,
                    external_traffic_list,
                ) None
        """

        self.remaining_computation_capacity = copy.deepcopy(self.initial_computation_capacity)
        self.topology.reset_bandwidth()
        self.modules_loaded_per_node = [set() for _ in range(self.node_count)]
        total_cost = 0.0
        total_deploy_cost = 0.0
        total_comm_cost = 0.0
        total_profit = 0.0
        total_users = 0
        used_nodes_union = set()
        chain_count = 0
        chain_len_list: list[int] = []
        chain_avg_modules_list: list[float] = []
        chain_used_nodes_list: list[int] = []
        chain_capacity_users_list: list[int] = []
        chain_served_users_list: list[int] = []
        chain_deployment_list: List[List[int]] = []
        external_traffic_accumulator: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        chain_time_list: list[float] = []
        while True:
            chain_start = time.perf_counter()
            best = self.find_best_single_chain_for_objective(objective)
            if best is None:
                break
            (
                cost,
                deploy_cost,
                comm_cost,
                profit,
                user_count,
                used_nodes_count,
                avg_mods_per_node,
                deployment,
                chain_len,
            ) = best
            # max_users = 0
            if user_count <= 0:
                break
            total_cost += cost
            total_deploy_cost += deploy_cost
            total_comm_cost += comm_cost
            total_profit += profit
            total_users += user_count
            used_nodes_union.update(deployment)
            chain_count += 1
            chain_len_list.append(int(chain_len))
            chain_avg_modules_list.append(float(avg_mods_per_node))
            chain_used_nodes_list.append(int(used_nodes_count))
            chain_capacity_users_list.append(int(user_count))
            chain_served_users_list.append(int(user_count))
            chain_deployment_list.append([int(x) for x in deployment])
            external_edges = self.build_external_traffic_list_for_deployment(
                deployment, user_count
            )
            self._merge_external_traffic_edges(external_traffic_accumulator, external_edges)
            self.apply_chain_consumption(deployment, user_count)
            chain_time_list.append(time.perf_counter() - chain_start)
            break
        if chain_count == 0:
            return None
        used_nodes_count = len(used_nodes_union)
        avg_modules_per_node = (
            self.module_count * chain_count / used_nodes_count
            if used_nodes_count > 0
            else 0.0
        )
        edge_traffic_list = self.build_edge_traffic_list()
        external_traffic_list = list(external_traffic_accumulator.values())
        avg_chain_time = sum(chain_time_list) / len(chain_time_list) if chain_time_list else 0.0
        total_deploy_time = sum(chain_time_list)
        return (
            total_cost,
            total_deploy_cost,
            total_comm_cost,
            total_profit,
            total_users,
            used_nodes_count,
            avg_modules_per_node,
            chain_count,
            chain_len_list,
            chain_avg_modules_list,
            chain_used_nodes_list,
            chain_capacity_users_list,
            chain_served_users_list,
            chain_deployment_list,
            edge_traffic_list,
            external_traffic_list,
            chain_time_list,
            avg_chain_time,
            total_deploy_time,
        )

    # ----------------------------------------------------------------------

    #  4

    # ----------------------------------------------------------------------

    def optimize_for_profit(self):
        """
         /

        1. min_cost    :
        2. max_profit  :
        3. min_profit  :
        4. max_users   :
        :(min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)
         plan  16
            (
                total_cost,
                total_deploy_cost,
                total_comm_cost,
                total_profit,
                total_users,
                used_nodes_count,
                avg_modules_per_node,
                chain_count,
                chain_len_list,
                chain_avg_modules_list,
                chain_used_nodes_list,
                chain_capacity_users_list,
                chain_served_users_list,
                chain_deployment_list,
                edge_traffic_list,
                external_traffic_list,
            )
        """
        min_cost_plan = self.deploy_until_exhaustion("min_cost")
        max_profit_plan = self.deploy_until_exhaustion("max_profit")
        min_profit_plan = self.deploy_until_exhaustion("min_profit")
        max_users_plan = self.deploy_until_exhaustion("max_users")
        return (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)
