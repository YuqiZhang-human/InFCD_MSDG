# shortest_path_all_node.py
# -*- coding: utf-8 -*-
"""
SPC

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import copy
import time
import json
from collections import defaultdict

from src.tools.sparse_topology import SparseTopology


@dataclass
class PathResult:
    """"""
    per_user_deploy_cost: float
    per_user_comm_cost: float
    per_user_total_cost: float
    capacity_users: int
    served_users: int
    deployment: List[int]

class ShortestPathOptimizerSparse:
    def __init__(self, test_case: Dict[str, Any]):
        """
        test_case

        - "test_data_id":  ID
        - "node_count":  N
        - "module_count":  M
        - "computation_capacity": List[[compute_cap, mem_cap], ...]
        - "resource_demands": List[[compute_demand_per_user, kv_mem_per_user], ...]
        - "weight_memory_demands": List[weight_mem_GB],  M 0
        - "data_sizes": List[boundary_data_mb_per_sec_per_user] M-1
        - "sparse_topology": SparseTopology
        - "node_costs": List[[gpu_cost, memory_cost], ...]
        - "bandwidth_cost": $/ (MB/s * month)
        - "profit_per_user": $/month
        """
        self.test_case = test_case
        self.test_data_id: int = int(test_case.get("test_data_id", -1))

        self.node_count: int = int(test_case["node_count"])
        self.module_count: int = int(test_case["module_count"])

        st = test_case.get("sparse_topology")
        if not isinstance(st, SparseTopology):
            raise TypeError(
                "ShortestPathOptimizerSparse  test_case['sparse_topology']  SparseTopology"
            )
        self.topology: SparseTopology = st

        # computation_capacity[i] = [compute_cap_i, mem_cap_i]
        self.initial_computation_capacity: List[List[float]] = [
            [float(c[0]), float(c[1])] for c in test_case["computation_capacity"]
        ]

        self.computation_capacity: List[List[float]] = copy.deepcopy(
            self.initial_computation_capacity
        )

        self.remaining_computation_capacity: List[List[float]] = copy.deepcopy(
            self.initial_computation_capacity
        )

        #  per-user  KV per user
        # resource_demands[i] = [compute_demand, kv_mem_demand]
        self.resource_demands: List[List[float]] = [
            [float(d[0]), float(d[1])] for d in test_case["resource_demands"]
        ]

        raw_weight_mem = test_case.get("weight_memory_demands")
        if raw_weight_mem is not None:
            self.weight_memory_demands: List[float] = [float(x) for x in raw_weight_mem]
            if len(self.weight_memory_demands) != self.module_count:
                raise ValueError("weight_memory_demands  module_count")
        else:
            self.weight_memory_demands = [0.0 for _ in range(self.module_count)]

        # GPUtest_data
        self.gpu_nodes = test_case.get("gpu_nodes", list(range(self.node_count)))
        self.switch_nodes = test_case.get("switch_nodes", [])
        self.gateway_uplink_nodes = test_case.get("gateway_uplink_nodes", [])
        self.gateway_downlink_nodes = test_case.get("gateway_downlink_nodes", [])

        # MB/s per user
        self.data_sizes: List[float] = [float(x) for x in test_case.get("data_sizes", [])]

        # gpu_cost, mem_cost
        # node_costs[i] = [gpu_cost_i, mem_cost_i]
        node_costs_raw = test_case.get("node_costs", [])
        self.node_gpu_costs: List[float] = []
        self.node_mem_costs: List[float] = []
        for c in node_costs_raw:
            #  dict
            if isinstance(c, dict):
                self.node_gpu_costs.append(float(c.get("gpu_cost", 0.0)))
                self.node_mem_costs.append(float(c.get("memory_cost", 0.0)))
            else:
                self.node_gpu_costs.append(float(c[0]))
                self.node_mem_costs.append(float(c[1]))

        self.bandwidth_cost: float = float(test_case.get("bandwidth_cost", 0.0))
        self.profit_per_user: float = float(test_case.get("profit_per_user", 0.0))

        #  /  I/O / gateway
        self.topology_type: str = (
            str(test_case.get("topology_type", ""))
            .lower()
            .replace("-", "_")
        )
        # gateway_nodeintJSON
        gateway_node_raw = test_case.get("gateway_node", -1)
        if isinstance(gateway_node_raw, str):
            try:
                gateway_node_parsed = json.loads(gateway_node_raw)
                if isinstance(gateway_node_parsed, list):
                    self.gateway_node = int(gateway_node_parsed[0]) if gateway_node_parsed else -1
                else:
                    self.gateway_node = int(gateway_node_parsed)
            except:
                self.gateway_node = int(gateway_node_raw) if gateway_node_raw.isdigit() else -1
        else:
            self.gateway_node = int(gateway_node_raw)

        # fat-tree/ gateway_node_list  gateway_node
        if self.topology_type == "fat_tree":
            if not self.gateway_uplink_nodes:
                gateway_node_list_raw = test_case.get("gateway_node_list", [])
                if gateway_node_list_raw:
                    self.gateway_uplink_nodes = [int(x) for x in gateway_node_list_raw]
                elif self.gateway_node >= 0:
                    self.gateway_uplink_nodes = [self.gateway_node]
            if not self.gateway_downlink_nodes:
                self.gateway_downlink_nodes = list(self.gateway_uplink_nodes)
            #  gateway_node
            if self.gateway_node < 0 and self.gateway_uplink_nodes:
                self.gateway_node = int(self.gateway_uplink_nodes[0])
            if self.gateway_node < 0 and not self.gateway_uplink_nodes:
                raise ValueError(
                    "fat_tree requires a valid gateway_node or gateway_uplink_nodes. "
                    "In this project gateway nodes are the core switch nodes."
                )
            if self.gateway_node >= 0 and self.gateway_node >= self.node_count:
                raise ValueError(
                    f"fat_tree gateway_node={self.gateway_node} out of range for node_count={self.node_count}"
                )
        self.user_uplink_mb_per_user: float = float(
            test_case.get("user_uplink_mb_per_user", 0.0)
        )
        self.user_downlink_mb_per_user: float = float(
            test_case.get("user_downlink_mb_per_user", 0.0)
        )
        self.return_to_gateway_mb_per_user: float = float(
            test_case.get("return_to_gateway_mb_per_user", 0.0)
        )

        self.modules_loaded_per_node: List[set] = [set() for _ in range(self.node_count)]
        self.global_candidate_limit = int(
            test_case.get("optimizer_candidate_limit_global", 96)
        )
        self.pod_candidate_limit = int(
            test_case.get("optimizer_candidate_limit_pod", 48)
        )
        self.edge_candidate_limit = int(
            test_case.get("optimizer_candidate_limit_edge", 12)
        )
        self.beam_width = int(test_case.get("sp_beam_width", 32))

        self.fat_tree_num_pods = int(test_case.get("fat_tree_num_pods", 0))
        self.fat_tree_core_switch_count = int(test_case.get("fat_tree_core_switch_count", 0))
        self.fat_tree_agg_per_pod = int(test_case.get("fat_tree_agg_per_pod", 0))
        self.fat_tree_edge_per_pod = int(test_case.get("fat_tree_edge_per_pod", 0))
        self.fat_tree_gpus_per_edge = int(test_case.get("fat_tree_gpus_per_edge", 0))
        self.fat_tree_gpu_start = int(test_case.get("fat_tree_gpu_start", -1))

        self._setup_candidate_pools()
        self._gateway_ingress_hop_sum_cache: Dict[int, Optional[int]] = {}
        self._gateway_egress_hop_sum_cache: Dict[int, Optional[int]] = {}


        self.debug: bool = False

    # ================================================================
    # ================================================================

    def build_external_traffic_list_for_deployment(
            self,
            deployment: List[int],
            user_count: int
    ) -> List[Dict[str, Any]]:
        """
             deployment

             SparseTopology
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
            # fat-tree:  gateway_uplink/downlink_nodes  Core
            access_ingress_nodes = [int(x) for x in self.gateway_uplink_nodes] if self.gateway_uplink_nodes else (
                [int(self.gateway_node)] if self.gateway_node >= 0 else []
            )
            access_egress_nodes = [int(x) for x in self.gateway_downlink_nodes] if self.gateway_downlink_nodes else (
                [int(self.gateway_node)] if self.gateway_node >= 0 else []
            )
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

    def _get_partial_access_comm_cost_per_user(
            self,
            partial_deployment: List[int]
    ) -> float:
        """
        ""


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
        if not partial_deployment:
            return 0.0

        first_node = int(partial_deployment[0])
        last_node = int(partial_deployment[-1])

        total = 0.0

        # 1) /clos / fat-tree
        if self.user_uplink_mb_per_user > 0:
            total += float(self.user_uplink_mb_per_user) * self.bandwidth_cost
        if self.user_downlink_mb_per_user > 0:
            total += float(self.user_downlink_mb_per_user) * self.bandwidth_cost

        # 2) fat-tree gateway  /  hop  Core
        if self.topology_type == "fat_tree":
            ingress_hops = self._get_gateway_ingress_hop_sum(first_node)
            egress_hops = self._get_gateway_egress_hop_sum(last_node)
            if ingress_hops is None or egress_hops is None:
                return float("inf")

            if self.gateway_uplink_nodes and self.user_uplink_mb_per_user > 0:
                uplink_per_gw = self.user_uplink_mb_per_user / len(self.gateway_uplink_nodes)
                total += uplink_per_gw * self.bandwidth_cost * ingress_hops

            if self.gateway_downlink_nodes and self.return_to_gateway_mb_per_user > 0:
                downlink_per_gw = self.return_to_gateway_mb_per_user / len(self.gateway_downlink_nodes)
                total += downlink_per_gw * self.bandwidth_cost * egress_hops

        # 3) clos gateway_uplink_nodes -> first_node  last_node -> gateway_downlink_nodes
        if self.topology_type == "clos":
            # Uplink:  -> first_node
            if self.gateway_uplink_nodes and self.user_uplink_mb_per_user > 0:
                uplink_per_gw = self.user_uplink_mb_per_user / len(self.gateway_uplink_nodes)
                for gw_node in self.gateway_uplink_nodes:
                    if gw_node != first_node and uplink_per_gw > 0:
                        path = self.get_path_nodes(int(gw_node), first_node)
                        if not path or len(path) < 2:
                            return float("inf")
                        hop = len(path) - 1
                        total += uplink_per_gw * self.bandwidth_cost * hop

            # Downlink: last_node ->
            if self.gateway_downlink_nodes and self.user_downlink_mb_per_user > 0:
                downlink_per_gw = self.user_downlink_mb_per_user / len(self.gateway_downlink_nodes)
                for gw_node in self.gateway_downlink_nodes:
                    if gw_node != last_node and downlink_per_gw > 0:
                        path = self.get_path_nodes(last_node, int(gw_node))
                        if not path or len(path) < 2:
                            return float("inf")
                        hop = len(path) - 1
                        total += downlink_per_gw * self.bandwidth_cost * hop

        return total

    def _get_distance(self, u: int, v: int) -> int:
        """
         u  v hop count

        Returns:
            hop  - 1 -1
        """
        path = self.get_path_nodes(u, v)
        if not path or len(path) < 2:
            return -1
        return len(path) - 1

    def _reconstruct_partial_path(self, prev_node: List[List[int]], layer: int, end_node: int) -> List[int]:
        """
         prev_node

        Args:
            prev_node: prev_node[layer][node]  layer  node
            layer:
            end_node:

        Returns:
             end_node
        """
        path = []
        current_layer = layer
        current_node = end_node

        while current_layer >= 0 and current_node >= 0:
            path.append(current_node)
            if current_layer == 0:
                break
            prev = prev_node[current_layer][current_node]
            if prev < 0:
                break
            current_node = prev
            current_layer -= 1

        path.reverse()
        return path

    def _build_edge_traffic_list(self) -> List[Dict[str, float]]:
        """
         edge_traffic_list
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

    def shortest_path_deployment(self) -> Optional[Dict[str, Any]]:
        """
         / DP

         mix_chain_shortest_path_all_node_KV_optimizer

        -
        -  chain_count  1
        """
        self.remaining_computation_capacity = copy.deepcopy(self.initial_computation_capacity)
        self.topology.reset_bandwidth()
        self.modules_loaded_per_node = [set() for _ in range(self.node_count)]

        total_cost = 0.0
        total_deploy_cost = 0.0
        total_comm_cost = 0.0
        total_profit = 0.0
        total_users = 0

        used_nodes_flags = [0] * self.node_count
        modules_assigned_count = [0] * self.node_count
        chain_count = 0

        chain_len_list: List[int] = []
        chain_avg_modules_list: List[float] = []
        chain_used_nodes_list: List[int] = []
        chain_capacity_users_list: List[int] = []
        chain_served_users_list: List[int] = []
        deployment_list: List[List[int]] = []

        external_traffic_accumulator: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        chain_time_list: List[float] = []

        while True:
            chain_start = time.perf_counter()
            path_res = self._find_best_feasible_chain()
            if path_res is None:
                break

            chain_users = path_res.served_users
            capacity_users = path_res.capacity_users

            if chain_users <= 0:
                break

            per_user_dep = path_res.per_user_deploy_cost
            per_user_comm = path_res.per_user_comm_cost
            per_user_total = path_res.per_user_total_cost
            deployment = path_res.deployment

            chain_deploy_cost = per_user_dep * chain_users
            chain_comm_cost = per_user_comm * chain_users
            chain_cost = per_user_total * chain_users
            chain_profit = self.profit_per_user * chain_users - chain_cost

            total_deploy_cost += chain_deploy_cost
            total_comm_cost += chain_comm_cost
            total_cost += chain_cost
            total_profit += chain_profit
            total_users += chain_users
            chain_count += 1

            used_nodes_count_chain = len(set(deployment))
            avg_mods_chain = (self.module_count / used_nodes_count_chain) if used_nodes_count_chain > 0 else 0.0
            chain_len = self.get_chain_total_hops(deployment)

            chain_len_list.append(int(chain_len))
            chain_avg_modules_list.append(float(avg_mods_chain))
            chain_used_nodes_list.append(int(used_nodes_count_chain))
            chain_capacity_users_list.append(int(capacity_users))
            chain_served_users_list.append(int(chain_users))
            deployment_list.append([int(x) for x in deployment])

            external_edges = self.build_external_traffic_list_for_deployment(
                deployment, chain_users
            )
            self._merge_external_traffic_edges(external_traffic_accumulator, external_edges)

            self._apply_chain_consumption(deployment, chain_users)
            for _, node in enumerate(deployment):
                used_nodes_flags[node] = 1
                modules_assigned_count[node] += 1

            chain_time_list.append(time.perf_counter() - chain_start)
            #  mix_chain_shortest_path
            break

        if chain_count == 0:
            return None

        used_nodes_count = sum(used_nodes_flags)
        if used_nodes_count > 0:
            avg_modules_per_node = sum(modules_assigned_count) / float(used_nodes_count)
        else:
            avg_modules_per_node = 0.0

        edge_traffic_list = self._build_edge_traffic_list()
        external_traffic_list = list(external_traffic_accumulator.values())

        avg_chain_time = sum(chain_time_list) / len(chain_time_list) if chain_time_list else 0.0
        total_deploy_time = sum(chain_time_list)
        return {
            "total_cost": total_cost,
            "total_deploy_cost": total_deploy_cost,
            "total_comm_cost": total_comm_cost,
            "total_profit": total_profit,
            "total_users": total_users,
            "used_nodes": used_nodes_count,
            "avg_modules_per_node": avg_modules_per_node,
            "chain_count": chain_count,
            "chain_len_list": chain_len_list,
            "chain_avg_modules_list": chain_avg_modules_list,
            "chain_used_nodes_list": chain_used_nodes_list,
            "chain_capacity_users_list": chain_capacity_users_list,
            "chain_served_users_list": chain_served_users_list,
            "deployment_list": deployment_list,
            "edge_traffic_list": edge_traffic_list,
            "external_traffic_list": external_traffic_list,
            "chain_time_list": chain_time_list,
            "avg_chain_time": avg_chain_time,
            "total_deploy_time": total_deploy_time,
        }

    # ================================================================
    # ================================================================
    def _find_best_feasible_chain(self) -> Optional[PathResult]:
        m = self.module_count
        if m <= 0:
            return None

        beam_width = max(1, self.beam_width)

        # beam
        # {
        #   "deployment": List[int],
        #   "cost": float,
        #   "access_cost": float,
        # }
        beam_states: List[Dict[str, Any]] = []

        init_candidates = self._global_gpu_rank[: self.global_candidate_limit]

        comp_demand_0, kv_demand_0 = self._get_module_demand(0)
        weight_0 = self.weight_memory_demands[0]

        for node in init_candidates:
            comp_cap, mem_cap = self._get_node_capacity(node)
            if comp_demand_0 > comp_cap:
                continue

            new_weight_0 = weight_0 if 0 not in self.modules_loaded_per_node[node] else 0.0
            if kv_demand_0 + new_weight_0 > mem_cap + 1e-9:
                continue

            dep_cost = self._deploy_cost_per_user(module_idx=0, node_idx=node)
            access_cost = self._get_partial_access_comm_cost_per_user([node])
            if access_cost >= 1e29:
                continue

            beam_states.append({
                "deployment": [int(node)],
                "cost": float(dep_cost + access_cost),
                "access_cost": float(access_cost),
            })

        if not beam_states:
            return None

        beam_states.sort(key=lambda x: x["cost"])
        beam_states = beam_states[:beam_width]

        for layer in range(1, m):
            comp_demand, kv_demand = self._get_module_demand(layer)
            weight_layer = self.weight_memory_demands[layer]
            boundary_data = self._get_data_size(layer - 1)

            next_states: List[Dict[str, Any]] = []

            for state in beam_states:
                deployment_prev = state["deployment"]
                prev_node = int(deployment_prev[-1])

                candidate_nodes = self._build_candidate_nodes_for_prev(prev_node)

                for cur_node in candidate_nodes:
                    comp_cap_cur, mem_cap_cur = self._get_node_capacity(cur_node)
                    if comp_demand > comp_cap_cur:
                        continue

                    new_weight = (
                        weight_layer
                        if layer not in self.modules_loaded_per_node[cur_node]
                        else 0.0
                    )
                    if kv_demand + new_weight > mem_cap_cur + 1e-9:
                        continue

                    dep_cost_cur = self._deploy_cost_per_user(layer, cur_node)

                    comm_cost = 0.0
                    if prev_node != cur_node and boundary_data > 0:
                        bottleneck = self.get_path_bottleneck_bw(prev_node, cur_node)
                        if bottleneck < boundary_data - 1e-12:
                            continue
                        hop = self.get_path_hops(prev_node, cur_node)
                        if hop <= 0 or hop >= 10 ** 9:
                            continue
                        comm_cost = boundary_data * self.bandwidth_cost * hop

                    deployment_cur = deployment_prev + [int(cur_node)]
                    access_cost_cur = self._get_partial_access_comm_cost_per_user(deployment_cur)
                    if access_cost_cur >= 1e29:
                        continue

                    access_delta = access_cost_cur - state["access_cost"]
                    total_cost = state["cost"] + dep_cost_cur + comm_cost + access_delta

                    next_states.append({
                        "deployment": deployment_cur,
                        "cost": float(total_cost),
                        "access_cost": float(access_cost_cur),
                    })

            if not next_states:
                return None

            next_states.sort(key=lambda x: x["cost"])
            beam_states = next_states[:beam_width]

        best_path: Optional[PathResult] = None

        for state in beam_states:
            deployment = state["deployment"]

            capacity_users = self._calculate_max_users_for_deployment(deployment)
            if capacity_users <= 0:
                continue

            served_users = capacity_users
            if served_users <= 0:
                continue

            dep_cost_per_user_excl_weight = self._deploy_cost_for_deployment_per_user(deployment)
            comm_cost_per_user = self._comm_cost_for_deployment_per_user(deployment)

            static_weight_cost = self._weight_cost_for_deployment_static(deployment)
            dep_cost_per_user = dep_cost_per_user_excl_weight + (static_weight_cost / float(served_users))
            total_cost_per_user = dep_cost_per_user + comm_cost_per_user

            candidate = PathResult(
                per_user_deploy_cost=dep_cost_per_user,
                per_user_comm_cost=comm_cost_per_user,
                per_user_total_cost=total_cost_per_user,
                capacity_users=capacity_users,
                served_users=served_users,
                deployment=[int(x) for x in deployment],
            )

            if best_path is None or candidate.per_user_total_cost < best_path.per_user_total_cost:
                best_path = candidate

        return best_path

    def _build_candidate_nodes_for_prev(self, prev_node: int) -> List[int]:
        if not self._fat_tree_selector_enabled():
            return self._global_gpu_rank[: self.global_candidate_limit]

        prev_edge = self._fat_tree_edge_id(prev_node)
        prev_pod = self._fat_tree_pod_id(prev_node)

        out: List[int] = []
        seen = set()

        self._add_candidate_slice(
            self._gpu_nodes_by_edge.get(prev_edge, []),
            self.edge_candidate_limit,
            out,
            seen,
        )
        self._add_candidate_slice(
            self._gpu_nodes_by_pod.get(prev_pod, []),
            self.pod_candidate_limit,
            out,
            seen,
        )
        self._add_candidate_slice(
            self._global_gpu_rank,
            self.global_candidate_limit,
            out,
            seen,
        )
        return out

    # ================================================================
    # ================================================================
    def _apply_chain_consumption(self, deployment: List[int], users: int) -> None:
        """
         /  /
         = KV * users +
        """
        N = self.node_count
        M = self.module_count

        node_comp_usage = [0.0] * N
        node_kv_usage = [0.0] * N
        node_new_weight = [0.0] * N

        for m_idx, node in enumerate(deployment):
            comp_d, kv_d = self._get_module_demand(m_idx)
            node_comp_usage[node] += comp_d * users
            node_kv_usage[node] += kv_d * users

            weight_mem = self.weight_memory_demands[m_idx]
            if m_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem
                self.modules_loaded_per_node[node].add(m_idx)

        for node in range(N):
            comp_cap, mem_cap = self.remaining_computation_capacity[node]
            comp_cap -= node_comp_usage[node]
            mem_cap -= node_kv_usage[node] + node_new_weight[node]
            self.remaining_computation_capacity[node] = [comp_cap, mem_cap]

        #  flow
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
            traffic = flow_per_user * users
            self.topology.decrement_bandwidth(u, v, traffic)

    # ================================================================
    #  /  /
    # ================================================================
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
        if not self._fat_tree_selector_enabled():
            return self._global_gpu_rank[: self.global_candidate_limit]

        if module_idx == 0 or deployment[module_idx - 1] < 0:
            return self._global_gpu_rank[: self.global_candidate_limit]

        prev_node = int(deployment[module_idx - 1])
        prev_edge = self._fat_tree_edge_id(prev_node)
        prev_pod = self._fat_tree_pod_id(prev_node)

        out: List[int] = []
        seen = set()

        self._add_candidate_slice(
            self._gpu_nodes_by_edge.get(prev_edge, []),
            self.edge_candidate_limit,
            out,
            seen,
        )
        self._add_candidate_slice(
            self._gpu_nodes_by_pod.get(prev_pod, []),
            self.pod_candidate_limit,
            out,
            seen,
        )
        self._add_candidate_slice(
            self._global_gpu_rank,
            self.global_candidate_limit,
            out,
            seen,
        )
        return out

    def _get_node_capacity(self, node_idx: int) -> Tuple[float, float]:
        """

        - compute_cap:
        - mem_cap: GB
        """
        comp_cap, mem_cap = self.remaining_computation_capacity[node_idx]
        return float(comp_cap), float(mem_cap)

    def _get_module_demand(self, module_idx: int) -> Tuple[float, float]:
        """
         per-user
        - compute_demand
        - kv_mem_demand
        """
        comp_d, kv_d = self.resource_demands[module_idx]
        return float(comp_d), float(kv_d)

    def _get_data_size(self, boundary_idx: int) -> float:
        if 0 <= boundary_idx < len(self.data_sizes):
            return float(self.data_sizes[boundary_idx])
        return 0.0

    # ================================================================
    # SparseTopology  BFS
    # ================================================================
    def get_path_nodes(self, s: int, t: int) -> Optional[List[int]]:
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

    def _get_gateway_ingress_hop_sum(self, node: int) -> Optional[int]:
        return self._sum_hops_to_node(
            [int(x) for x in self.gateway_uplink_nodes],
            node,
            self._gateway_ingress_hop_sum_cache,
        )

    def _get_gateway_egress_hop_sum(self, node: int) -> Optional[int]:
        return self._sum_hops_to_node(
            [int(x) for x in self.gateway_downlink_nodes],
            node,
            self._gateway_egress_hop_sum_cache,
        )

    def get_chain_total_hops(self, deployment: List[int]) -> int:
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
            hops = self.get_path_hops(a, b)
            if hops >= 10 ** 8:
                continue
            total += int(hops)
        return total

    def get_path_bottleneck_bw(self, s: int, t: int) -> float:
        """
         s->t min edge bw
         0 inf
        """
        return float(self.topology.get_path_bottleneck_bw(s, t))

    def _get_link_bandwidth(self, from_node: int, to_node: int) -> float:
        return float(self.topology.get_link_bandwidth(from_node, to_node))

    def _deploy_cost_per_user(self, module_idx: int, node_idx: int) -> float:
        """
         node_idx  module_idx
        per-user  = compute_demand * gpu_cost + kv_mem_demand * mem_cost
        """
        comp_d, kv_d = self._get_module_demand(module_idx)
        gpu_cost = self.node_gpu_costs[node_idx]
        mem_cost = self.node_mem_costs[node_idx]
        return comp_d * gpu_cost + kv_d * mem_cost

    def _deploy_cost_for_deployment_per_user(self, deployment: List[int]) -> float:
        """
         per-user
         DP
        """
        total = 0.0
        for m_idx, node in enumerate(deployment):
            total += self._deploy_cost_per_user(m_idx, node)
        return total

    def _weight_cost_for_deployment_static(self, deployment: List[int]) -> float:
        """

        -  (module, node)

        """
        static_cost = 0.0
        for m_idx, node in enumerate(deployment):
            weight_mem = self.weight_memory_demands[m_idx]
            if weight_mem <= 0.0:
                continue
            if m_idx in self.modules_loaded_per_node[node]:
                continue
            mem_cost = self.node_mem_costs[node]
            static_cost += weight_mem * mem_cost
        return static_cost

    def _get_fat_tree_gateway_internal_cost_per_user(self, deployment: List[int]) -> float:
        if self.topology_type != "fat_tree":
            return 0.0
        if not self.gateway_uplink_nodes and not self.gateway_downlink_nodes:
            return 0.0
        if not deployment:
            return 0.0

        total = 0.0
        first_node = int(deployment[0])
        last_node = int(deployment[-1])
        ingress_hops = self._get_gateway_ingress_hop_sum(first_node)
        egress_hops = self._get_gateway_egress_hop_sum(last_node)
        if ingress_hops is None or egress_hops is None:
            return 1e30

        if self.gateway_uplink_nodes and self.user_uplink_mb_per_user > 0:
            uplink_per_gw = self.user_uplink_mb_per_user / len(self.gateway_uplink_nodes)
            total += uplink_per_gw * self.bandwidth_cost * ingress_hops

        if self.gateway_downlink_nodes and self.return_to_gateway_mb_per_user > 0:
            downlink_per_gw = self.return_to_gateway_mb_per_user / len(self.gateway_downlink_nodes)
            total += downlink_per_gw * self.bandwidth_cost * egress_hops

        return total

    def _get_clos_gateway_internal_cost_per_user(self, deployment: List[int]) -> float:
        """
        CLOS /per user
          gateway_uplink_nodes  -> first_node  ()
          last_node -> gateway_downlink_nodes  ()
        """
        if self.topology_type != "clos":
            return 0.0
        if not deployment:
            return 0.0

        total = 0.0
        first_node = int(deployment[0])
        last_node = int(deployment[-1])

        if self.gateway_uplink_nodes and self.user_uplink_mb_per_user > 0:
            uplink_per_gw = self.user_uplink_mb_per_user / len(self.gateway_uplink_nodes)
            for gw_node in self.gateway_uplink_nodes:
                gw_node = int(gw_node)
                if gw_node == first_node or uplink_per_gw <= 0:
                    continue
                path = self.get_path_nodes(gw_node, first_node)
                if not path or len(path) < 2:
                    return 1e30
                hop = len(path) - 1
                total += uplink_per_gw * self.bandwidth_cost * hop

        if self.gateway_downlink_nodes and self.user_downlink_mb_per_user > 0:
            downlink_per_gw = self.user_downlink_mb_per_user / len(self.gateway_downlink_nodes)
            for gw_node in self.gateway_downlink_nodes:
                gw_node = int(gw_node)
                if gw_node == last_node or downlink_per_gw <= 0:
                    continue
                path = self.get_path_nodes(last_node, gw_node)
                if not path or len(path) < 2:
                    return 1e30
                hop = len(path) - 1
                total += downlink_per_gw * self.bandwidth_cost * hop

        return total

    def _comm_cost_for_deployment_per_user(self, deployment: List[int]) -> float:
        total = 0.0
        M = self.module_count

        for boundary_idx in range(M - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue

            data_size = self._get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            hop = self.get_path_hops(from_node, to_node)
            if hop <= 0 or hop >= 10 ** 9:
                return 1e30
            total += data_size * self.bandwidth_cost * hop

        # CLOS/ +  gateway /
        if self.topology_type == "clos":
            total += (self.user_uplink_mb_per_user +
                      self.user_downlink_mb_per_user) * self.bandwidth_cost
            total += self._get_clos_gateway_internal_cost_per_user(deployment)

        # fat-tree/ +  gateway /
        if self.topology_type == "fat_tree":
            total += (self.user_uplink_mb_per_user +
                      self.user_downlink_mb_per_user) * self.bandwidth_cost
            total += self._get_fat_tree_gateway_internal_cost_per_user(deployment)

        return total

    def _build_edge_flow_per_user_for_deployment(
            self,
            deployment: List[int]
    ) -> Dict[Tuple[int, int], float]:
        """
         deployment  per-user MB/s per user


        -  (u, v)
        -  (v, u)
        """
        edge_flow_per_user = defaultdict(float)
        m = self.module_count

        # 1)  boundary
        for boundary_idx in range(m - 1):
            from_node = int(deployment[boundary_idx])
            to_node = int(deployment[boundary_idx + 1])

            if from_node == to_node:
                continue

            data_size = self._get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            path = self.get_path_nodes(from_node, to_node)
            if not path or len(path) < 2:
                return {}

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_flow_per_user[(u, v)] += data_size

        # 2) fat-tree: gateway_uplink_nodes -> first_node
        # 3) fat-tree: last_node -> gateway_downlink_nodes
        if self.topology_type == "fat_tree" and deployment:
            first_node = int(deployment[0])
            last_node = int(deployment[-1])

            if self.gateway_uplink_nodes and self.user_uplink_mb_per_user > 0:
                uplink_per_gw = self.user_uplink_mb_per_user / len(self.gateway_uplink_nodes)
                for gw_node in self.gateway_uplink_nodes:
                    if int(gw_node) != first_node and uplink_per_gw > 0:
                        path = self.get_path_nodes(int(gw_node), first_node)
                        if not path or len(path) < 2:
                            return {}
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            edge_flow_per_user[(u, v)] += uplink_per_gw

            if self.gateway_downlink_nodes and self.return_to_gateway_mb_per_user > 0:
                downlink_per_gw = self.return_to_gateway_mb_per_user / len(self.gateway_downlink_nodes)
                for gw_node in self.gateway_downlink_nodes:
                    if int(gw_node) != last_node and downlink_per_gw > 0:
                        path = self.get_path_nodes(last_node, int(gw_node))
                        if not path or len(path) < 2:
                            return {}
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            edge_flow_per_user[(u, v)] += downlink_per_gw

        # 4) & 5) clos
        if self.topology_type == "clos" and deployment:
            first_node = int(deployment[0])
            last_node = int(deployment[-1])

            if self.gateway_uplink_nodes and self.user_uplink_mb_per_user > 0:
                uplink_per_gw = self.user_uplink_mb_per_user / len(self.gateway_uplink_nodes)
                for gw_node in self.gateway_uplink_nodes:
                    if gw_node != first_node and uplink_per_gw > 0:
                        path = self.get_path_nodes(int(gw_node), first_node)
                        if path and len(path) >= 2:
                            for i in range(len(path) - 1):
                                u, v = path[i], path[i + 1]
                                edge_flow_per_user[(u, v)] += uplink_per_gw

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
            node_memory_used_gb = {i: 0.0 for i in range(self.node_count)}
            node_memory_util = {i: 0.0 for i in range(self.node_count)}
            return node_memory_used_gb, node_memory_util

        for deployment, user_count in zip(deployment_list, served_users_list):
            if not deployment or user_count <= 0:
                continue

            for module_idx, node in enumerate(deployment):
                node = int(node)

                if module_idx not in loaded_module_on_node[node]:
                    node_weight_gb[node] += float(self.weight_memory_demands[module_idx])
                    loaded_module_on_node[node].add(module_idx)

                _, kv_per_user_gb = self._get_module_demand(module_idx)
                node_kv_gb[node] += float(kv_per_user_gb) * float(user_count)

        node_memory_used_gb = {}
        node_memory_util = {}

        for node in range(self.node_count):
            used_gb = node_weight_gb[node] + node_kv_gb[node]
            mem_cap_gb = float(self.initial_computation_capacity[node][1])

            node_memory_used_gb[node] = used_gb
            node_memory_util[node] = (used_gb / mem_cap_gb) if mem_cap_gb > 0 else 0.0

        return node_memory_used_gb, node_memory_util

    def calculate_node_memory_stats_for_deployment(
            self,
            deployment: List[int],
            user_count: int
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
         deployment + user_count
        1) node_memory_used_gb
        2) node_memory_util
        """
        node_weight_gb = {i: 0.0 for i in range(self.node_count)}
        node_kv_gb = {i: 0.0 for i in range(self.node_count)}

        for module_idx, node in enumerate(deployment):
            node = int(node)

            weight_gb = float(self.weight_memory_demands[module_idx])
            node_weight_gb[node] += weight_gb

            _, kv_per_user_gb = self._get_module_demand(module_idx)
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

    def _calculate_max_users_for_deployment(self, deployment: List[int]) -> int:
        """
         + KV

        -  /
        -
        """
        N = self.node_count
        M = self.module_count

        if len(deployment) != M:
            return 0

        # per user +
        node_comp_per_user = [0.0] * N
        node_kv_per_user = [0.0] * N
        node_new_weight = [0.0] * N

        for m_idx, node in enumerate(deployment):
            comp_d, kv_d = self._get_module_demand(m_idx)
            node_comp_per_user[node] += comp_d
            node_kv_per_user[node] += kv_d

            weight_mem = self.weight_memory_demands[m_idx]
            if m_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem

        node_limits: List[int] = []

        #  /
        for node in range(N):
            comp_cap, mem_cap = self._get_node_capacity(node)
            comp_use = node_comp_per_user[node]
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

            node_limits.append(int(math.floor(node_limit)))

        #  flow
        edge_flow_per_user = self._build_edge_flow_per_user_for_deployment(deployment)
        if edge_flow_per_user is None:
            return 0

        undirected_edge_flow_per_user: Dict[Tuple[int, int], float] = defaultdict(float)
        for (u, v), flow_per_user in edge_flow_per_user.items():
            if flow_per_user <= 0:
                continue
            key = (u, v) if u < v else (v, u)
            undirected_edge_flow_per_user[key] += flow_per_user

        link_limits: List[int] = []
        for (u, v), flow_per_user in undirected_edge_flow_per_user.items():
            if flow_per_user <= 0:
                continue

            rem_bw = min(
                self._get_link_bandwidth(u, v),
                self._get_link_bandwidth(v, u),
            )
            if rem_bw <= 0:
                return 0

            users_by_edge = rem_bw / flow_per_user
            if users_by_edge <= 0:
                return 0

            link_limits.append(int(math.floor(users_by_edge)))

        all_limits = node_limits + link_limits
        if not all_limits:
            return 0

        max_users = max(0, min(all_limits))
        return max_users
