#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brute Force Single Chain Optimizer

"""

import json
import math
import copy
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from src.tools.sparse_topology import SparseTopology


class BruteForceSingleChainOptimizerSparse:
    """"""

    MAX_NODES = 100
    MAX_MODULES = 100

    def __init__(self, test_data: Dict[str, Any], verbose: bool = False) -> None:
        """
        Args:
            test_data:  sparse_topology: SparseTopology
        """
        self.test_data = test_data
        self.verbose = verbose

        self.test_data_id = test_data.get('test_data_id', 0)
        self.node_count = int(test_data.get('node_count', 0))
        self.module_count = int(test_data.get('module_count', 0))

        st = test_data.get("sparse_topology")
        if not isinstance(st, SparseTopology):
            raise TypeError(
                "BruteForceSingleChainOptimizerSparse  test_data['sparse_topology']  SparseTopology"
            )
        self.topology: SparseTopology = st

        # GPU
        self.gpu_nodes = test_data.get("gpu_nodes", list(range(self.node_count)))
        self.switch_nodes = test_data.get("switch_nodes", [])

        self.gpu_node_count = len(self.gpu_nodes)

        self.computation_capacity = test_data.get('computation_capacity', [])

        self.resource_demands = test_data.get('resource_demands', [])
        self.weight_memory_demands = test_data.get('weight_memory_demands', [])
        self.data_sizes = test_data.get('data_sizes', [])

        # /
        self.gpu_cost = float(test_data.get('gpu_cost', 0.0))
        self.memory_cost = float(test_data.get('memory_cost', 0.0))
        self.bandwidth_cost = float(test_data.get('bandwidth_cost', 0.0))
        self.profit_per_user = float(test_data.get('profit_per_user', 0.0))

        self.node_costs = test_data.get('node_costs', [])
        if self.node_costs is None or len(self.node_costs) == 0:
            self.node_costs = [[self.gpu_cost, self.memory_cost] for _ in range(self.node_count)]

        self.topology_type = str(test_data.get("topology_type", "")).lower().replace("-", "_")

        # gateway_node
        gateway_node_raw = test_data.get("gateway_node", -1)
        if isinstance(gateway_node_raw, str):
            try:
                self.gateway_node = json.loads(gateway_node_raw)
                if isinstance(self.gateway_node, list):
                    self.gateway_node = int(self.gateway_node[0]) if self.gateway_node else -1
                else:
                    self.gateway_node = int(self.gateway_node)
            except:
                self.gateway_node = int(gateway_node_raw) if gateway_node_raw.isdigit() else -1
        else:
            self.gateway_node = int(gateway_node_raw)

        self.user_uplink_mb_per_user = float(test_data.get("user_uplink_mb_per_user", 0.0))
        self.user_downlink_mb_per_user = float(test_data.get("user_downlink_mb_per_user", 0.0))
        self.return_to_gateway_mb_per_user = float(test_data.get("return_to_gateway_mb_per_user", 0.0))

        # GatewayCLOS  fat-tree
        self.gateway_uplink_nodes = test_data.get("gateway_uplink_nodes", [])
        self.gateway_downlink_nodes = test_data.get("gateway_downlink_nodes", [])

        # fat-tree/ gateway_node_list  gateway_node
        if self.topology_type == "fat_tree":
            if not self.gateway_uplink_nodes:
                gateway_node_list_raw = test_data.get("gateway_node_list", [])
                if gateway_node_list_raw:
                    self.gateway_uplink_nodes = [int(x) for x in gateway_node_list_raw]
                elif self.gateway_node >= 0:
                    self.gateway_uplink_nodes = [self.gateway_node]
            if not self.gateway_downlink_nodes:
                self.gateway_downlink_nodes = list(self.gateway_uplink_nodes)
            #  gateway_node
            if self.gateway_node < 0 and self.gateway_uplink_nodes:
                self.gateway_node = int(self.gateway_uplink_nodes[0])

        # GPU
        self.too_large = (
                self.module_count > self.MAX_MODULES
                or self.gpu_node_count > self.MAX_NODES
        )

        if self.too_large:
            self._vprint(
                f"[test_id={self.test_data_id}] "
                f" (N_gpu={self.gpu_node_count}, T={self.module_count})"
                f" ( N_gpu<={self.MAX_NODES}, T<={self.MAX_MODULES})"
            )

        self.evaluated_leaf_count = 0
        self.feasible_leaf_count = 0
        self.infeasible_leaf_count = 0
        self.skipped_prefix_count = 0

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def get_path_nodes(self, s: int, t: int) -> Optional[List[int]]:
        """ s->t """
        return self.topology.get_path_nodes(s, t)

    def get_chain_total_hops(self, deployment: List[int]) -> int:
        """

         hop
         multi
        """
        if not deployment or len(deployment) <= 1:
            return 0

        total_hops = 0
        for i in range(len(deployment) - 1):
            u = int(deployment[i])
            v = int(deployment[i + 1])

            if u == v:
                continue

            path = self.get_path_nodes(u, v)
            if not path or len(path) < 2:
                return 0  #  return -1

            total_hops += len(path) - 1

        return total_hops

    def get_node_capacity(self, node: int) -> Tuple[float, float]:
        """ [, ] """
        if node < 0 or node >= self.node_count:
            return 0.0, 0.0
        return (
            float(self.computation_capacity[node][0]),
            float(self.computation_capacity[node][1]),
        )

    def get_module_demand(self, module: int) -> Tuple[float, float]:
        """ [, KV ]"""
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

    def get_data_size(self, boundary_index: int) -> float:
        """ (MB/s)"""
        if boundary_index < 0 or boundary_index >= len(self.data_sizes):
            return 0.0
        return float(self.data_sizes[boundary_index])

    def _build_edge_flow_per_user_for_deployment(
            self,
            deployment: List[int]
    ) -> Optional[Dict[Tuple[int, int], float]]:
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
           brute
          -
          - edge_traffic_list
        """
        edge_flow_per_user = defaultdict(float)
        m = self.module_count

        # 1)  boundary
        for boundary_idx in range(m - 1):
            from_node = int(deployment[boundary_idx])
            to_node = int(deployment[boundary_idx + 1])

            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            path = self.get_path_nodes(from_node, to_node)
            if not path or len(path) < 2:
                return None

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
                            return None
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            edge_flow_per_user[(u, v)] += uplink_per_gw

            if self.gateway_downlink_nodes and self.return_to_gateway_mb_per_user > 0:
                downlink_per_gw = self.return_to_gateway_mb_per_user / len(self.gateway_downlink_nodes)
                for gw_node in self.gateway_downlink_nodes:
                    if int(gw_node) != last_node and downlink_per_gw > 0:
                        path = self.get_path_nodes(last_node, int(gw_node))
                        if not path or len(path) < 2:
                            return None
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

        return dict(edge_flow_per_user)

    def _build_external_traffic_list(
        self,
        deployment: List[int],
        user_count: int
    ) -> List[Dict[str, Any]]:
        """
         external_traffic_list __user__
         edge_flow_per_user
        """
        external_traffic_list = []
        if not deployment or user_count <= 0:
            return external_traffic_list

        first_node = int(deployment[0])
        last_node = int(deployment[-1])

        if self.topology_type == "clos":
            access_ingress_nodes = self.gateway_uplink_nodes if self.gateway_uplink_nodes else [first_node]
            access_egress_nodes = self.gateway_downlink_nodes if self.gateway_downlink_nodes else [last_node]

            external_uplink = self.user_uplink_mb_per_user * user_count
            external_downlink = self.user_downlink_mb_per_user * user_count

            uplink_bw_per_node = (
                external_uplink / len(access_ingress_nodes) if access_ingress_nodes else 0.0
            )
            downlink_bw_per_node = (
                external_downlink / len(access_egress_nodes) if access_egress_nodes else 0.0
            )

            for gw_node in access_ingress_nodes:
                if uplink_bw_per_node > 1e-12:
                    external_traffic_list.append({
                        "from_node": "__user__",
                        "to_node": int(gw_node),
                        "used_bandwidth": float(uplink_bw_per_node),
                        "initial_bandwidth": None,
                        "traffic_type": "external_uplink",
                    })

            for gw_node in access_egress_nodes:
                if downlink_bw_per_node > 1e-12:
                    external_traffic_list.append({
                        "from_node": int(gw_node),
                        "to_node": "__user__",
                        "used_bandwidth": float(downlink_bw_per_node),
                        "initial_bandwidth": None,
                        "traffic_type": "external_downlink",
                    })

        elif self.topology_type == "fat_tree":
            access_ingress_nodes = self.gateway_uplink_nodes if self.gateway_uplink_nodes else (
                [self.gateway_node] if self.gateway_node >= 0 else []
            )
            access_egress_nodes = self.gateway_downlink_nodes if self.gateway_downlink_nodes else (
                [self.gateway_node] if self.gateway_node >= 0 else []
            )
            external_uplink = self.user_uplink_mb_per_user * user_count
            external_downlink = self.user_downlink_mb_per_user * user_count
            uplink_bw_per_node = external_uplink / len(access_ingress_nodes) if access_ingress_nodes else 0.0
            downlink_bw_per_node = external_downlink / len(access_egress_nodes) if access_egress_nodes else 0.0
            for gw_node in access_ingress_nodes:
                if uplink_bw_per_node > 1e-12:
                    external_traffic_list.append({
                        "from_node": "__user__",
                        "to_node": int(gw_node),
                        "used_bandwidth": float(uplink_bw_per_node),
                        "initial_bandwidth": None,
                        "traffic_type": "external_uplink",
                    })
            for gw_node in access_egress_nodes:
                if downlink_bw_per_node > 1e-12:
                    external_traffic_list.append({
                        "from_node": int(gw_node),
                        "to_node": "__user__",
                        "used_bandwidth": float(downlink_bw_per_node),
                        "initial_bandwidth": None,
                        "traffic_type": "external_downlink",
                    })

        return external_traffic_list

    def _calculate_comm_cost_for_deployment(
        self,
        deployment: List[int],
        user_count: int
    ) -> float:
        """
         multi  deployment
        - boundary
        - clos  +  gateway-> / ->gateway_downlink_nodes
        - fat-tree  +  gateway-> / ->gateway
        """
        if not deployment or user_count <= 0:
            return 0.0

        comm_cost = 0.0
        m = self.module_count
        first_node = int(deployment[0])
        last_node = int(deployment[-1])

        # 1)  boundary
        for boundary_idx in range(m - 1):
            from_node = int(deployment[boundary_idx])
            to_node = int(deployment[boundary_idx + 1])

            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            path = self.get_path_nodes(from_node, to_node)
            if not path or len(path) < 2:
                return float("inf")

            hop = len(path) - 1
            comm_cost += data_size * self.bandwidth_cost * user_count * hop

        # 2) clos
        if self.topology_type == "clos":
            # /
            comm_cost += (
                self.user_uplink_mb_per_user + self.user_downlink_mb_per_user
            ) * self.bandwidth_cost * user_count

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
                    comm_cost += uplink_per_gw * self.bandwidth_cost * user_count * hop

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
                    comm_cost += downlink_per_gw * self.bandwidth_cost * user_count * hop

        # 3) fat-tree
        elif self.topology_type == "fat_tree":
            # __user__ -> gateway_uplink_nodesgateway_downlink_nodes -> __user__
            comm_cost += (
                self.user_uplink_mb_per_user + self.user_downlink_mb_per_user
            ) * self.bandwidth_cost * user_count

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
                    comm_cost += uplink_per_gw * self.bandwidth_cost * user_count * hop

            # last_node -> gateway_downlink_nodes
            if self.gateway_downlink_nodes and self.return_to_gateway_mb_per_user > 0:
                downlink_per_gw = self.return_to_gateway_mb_per_user / len(self.gateway_downlink_nodes)
                for gw_node in self.gateway_downlink_nodes:
                    gw_node = int(gw_node)
                    if gw_node == last_node or downlink_per_gw <= 0:
                        continue
                    path = self.get_path_nodes(last_node, gw_node)
                    if not path or len(path) < 2:
                        return float("inf")
                    hop = len(path) - 1
                    comm_cost += downlink_per_gw * self.bandwidth_cost * user_count * hop

        return comm_cost

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def enumerate_single_chain(
        self,
        objective: str
    ) -> Optional[Tuple]:
        """

         GPU

        Returns:
            16 (MultiFunctionOptimizer):
            (
                total_cost, total_deploy_cost, total_comm_cost, total_profit,
                total_users, used_nodes_count, avg_modules_per_node,
                chain_count, chain_len_list, chain_avg_modules_list,
                chain_used_nodes_list, chain_capacity_users_list,
                chain_served_users_list, chain_deployment_list,
                edge_traffic_list, external_traffic_list
            )
        """
        self.evaluated_leaf_count = 0
        self.feasible_leaf_count = 0
        self.infeasible_leaf_count = 0
        self.skipped_prefix_count = 0

        if self.too_large :
            return None

        m = self.module_count

        if m <= 0:
            return None

        best_solution = None
        eval_count = 0
        feasible_count = 0

        # DFSGPU
        def dfs(module_idx: int, current_deployment: List[int]):
            nonlocal best_solution, eval_count, feasible_count

            if module_idx == m:

                eval_count += 1
                result = self._evaluate_deployment(current_deployment)
                if result is None:
                    return

                (max_users, total_cost, deploy_cost, comm_cost, profit,
                 edge_traffic_list, external_traffic_list) = result

                if max_users <= 0:
                    return

                feasible_count += 1

                # unit_cost = total_cost / max_users
                #  dynamic min_cost
                unit_cost = total_cost / max_users if max_users > 0 else float("inf")
                score = None
                if objective == "min_cost":
                    score = (unit_cost, -profit, -max_users)
                elif objective == "max_profit":
                    score = (-profit, unit_cost, -max_users)
                elif objective == "min_profit":
                    score = (profit, total_cost, max_users)
                elif objective == "max_users":
                    score = (-max_users, unit_cost, -profit)
                else:
                    return

                if best_solution is None or score < best_solution[0]:
                    best_solution = (
                        score, max_users, current_deployment[:],
                        total_cost, deploy_cost, comm_cost, profit,
                        edge_traffic_list, external_traffic_list
                    )
                return

            # GPU
            for gpu_node in self.gpu_nodes:
                current_deployment.append(gpu_node)
                dfs(module_idx + 1, current_deployment)
                current_deployment.pop()

        _t0 = time.perf_counter()
        dfs(0, [])
        _elapsed = time.perf_counter() - _t0
        self._vprint(
            f"[test_id={self.test_data_id}] objective={objective}, "
            f"evaluated={eval_count}, feasible={feasible_count}"
        )
        self.evaluated_leaf_count = eval_count
        self.feasible_leaf_count = feasible_count
        self.infeasible_leaf_count = eval_count - feasible_count

        if best_solution is None:
            return None

        _, max_users, deployment, total_cost, deploy_cost, comm_cost, profit, \
            edge_traffic_list, external_traffic_list = best_solution

        # 19
        used_nodes = list(set(deployment))
        used_nodes_count = len(used_nodes)
        avg_modules_per_node = float(m) / used_nodes_count if used_nodes_count > 0 else 0.0

        chain_len = self.get_chain_total_hops(deployment)
        chain_len_list = [chain_len]
        chain_avg_modules_list = [avg_modules_per_node]
        chain_used_nodes_list = [used_nodes_count]
        chain_capacity_users_list = [max_users]
        chain_served_users_list = [max_users]
        chain_deployment_list = [deployment[:]]

        chain_time_list = [_elapsed]
        avg_chain_time = _elapsed
        total_deploy_time = _elapsed
        return (
            total_cost, deploy_cost, comm_cost, profit,
            max_users, used_nodes_count, avg_modules_per_node,
            1,  # chain_count = 1
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

    def calculate_aggregated_node_memory_stats(
        self,
        deployment_list: List[List[int]],
        served_users_list: List[int],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        node_weight_gb: Dict[int, float] = {}
        node_kv_gb: Dict[int, float] = {}

        for deployment, user_count in zip(deployment_list or [], served_users_list or []):
            node_modules: Dict[int, List[int]] = {}
            for module_idx, node in enumerate(deployment or []):
                node_modules.setdefault(int(node), []).append(module_idx)

            for node, modules in node_modules.items():
                weight_total = sum(
                    self.get_module_weight_memory(module_idx)
                    for module_idx in modules
                    if module_idx < self.module_count
                )
                kv_per_user = sum(
                    self.get_module_demand(module_idx)[1]
                    for module_idx in modules
                    if module_idx < self.module_count
                )
                node_weight_gb[node] = node_weight_gb.get(node, 0.0) + weight_total
                node_kv_gb[node] = node_kv_gb.get(node, 0.0) + kv_per_user * float(user_count)

        used = {}
        util = {}
        for node in set(node_weight_gb) | set(node_kv_gb):
            total = node_weight_gb.get(node, 0.0) + node_kv_gb.get(node, 0.0)
            mem_cap = float(self.computation_capacity[node][1]) if node < len(self.computation_capacity) else 0.0
            used[node] = total
            util[node] = total / mem_cap if mem_cap > 1e-12 else 0.0
        return used, util

    def _evaluate_deployment(
            self,
            deployment: List[int]
    ) -> Optional[Tuple[int, float, float, float, float, List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """

        :
            (
                max_users,
                total_cost,
                deploy_cost,
                comm_cost,
                profit,
                edge_traffic_list,
                external_traffic_list
            )
         None
        """
        if not deployment or len(deployment) != self.module_count:
            return None

        n = self.node_count
        m = self.module_count

        #  / KV /
        temp_comp_on_node = [0.0] * n
        temp_kv_on_node = [0.0] * n
        temp_new_weight_on_node = [0.0] * n

        limits = []

        # 1)
        for module_idx, node in enumerate(deployment):
            node = int(node)

            if node < 0 or node >= n:
                return None

            comp_demand, kv_demand = self.get_module_demand(module_idx)
            weight_mem = self.get_module_weight_memory(module_idx)

            temp_comp_on_node[node] += comp_demand
            temp_kv_on_node[node] += kv_demand

            # brute
            temp_new_weight_on_node[node] += weight_mem

        # 2)
        for node in range(n):
            comp_cap, mem_cap = self.get_node_capacity(node)

            if temp_comp_on_node[node] > 1e-12:
                limits.append(comp_cap / temp_comp_on_node[node])

            #  + KV * users
            if temp_kv_on_node[node] > 1e-12:
                remain_mem_for_kv = mem_cap - temp_new_weight_on_node[node]
                if remain_mem_for_kv < -1e-12:
                    return None
                limits.append(remain_mem_for_kv / temp_kv_on_node[node])
            else:
                if temp_new_weight_on_node[node] - mem_cap > 1e-12:
                    return None

        # 3)  multi  per-user
        edge_flow_per_user_opt = self._build_edge_flow_per_user_for_deployment(deployment)
        if edge_flow_per_user_opt is None:
            return None
        edge_flow_per_user: Dict[Tuple[int, int], float] = edge_flow_per_user_opt
        # multi  edge_flow_per_user
        #  brute  u < v

        undirected_edge_flow_per_user: Dict[Tuple[int, int], float] = defaultdict(float)
        for (u, v), flow in edge_flow_per_user.items():
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            undirected_edge_flow_per_user[(a, b)] += float(flow)

        for (u, v), flow_per_user in undirected_edge_flow_per_user.items():
            if flow_per_user <= 0:
                continue

            bw = float(self.topology.get_initial_bandwidth(u, v))
            if bw <= 0:
                return None

            limits.append(bw / flow_per_user)

        if not limits:
            max_users = 1
        else:
            max_users = int(math.floor(min(limits)))

        if max_users <= 0:
            return None

        # 4)
        deploy_cost = 0.0
        for node in range(n):
            gpu_cost_node, mem_cost_node = self.node_costs[node]
            comp_use = temp_comp_on_node[node] * max_users
            kv_use = temp_kv_on_node[node] * max_users
            weight_use = temp_new_weight_on_node[node]

            deploy_cost += comp_use * float(gpu_cost_node)
            deploy_cost += (kv_use + weight_use) * float(mem_cost_node)

        # 5)
        comm_cost = self._calculate_comm_cost_for_deployment(deployment, max_users)
        if not math.isfinite(comm_cost):
            return None

        # 6) edge_traffic_list
        edge_traffic_list = []
        for (u, v), flow_per_user in undirected_edge_flow_per_user.items():
            used_bw = flow_per_user * max_users
            if used_bw <= 1e-12:
                continue
            edge_traffic_list.append({
                "from_node": int(u),
                "to_node": int(v),
                "used_bandwidth": float(used_bw),
                "initial_bandwidth": float(self.topology.get_initial_bandwidth(u, v)),
                "traffic_type": "internal",
            })
        edge_traffic_list.sort(key=lambda x: (x["from_node"], x["to_node"]))

        # 7) external_traffic_list __user__
        external_traffic_list = self._build_external_traffic_list(deployment, max_users)

        total_cost = deploy_cost + comm_cost
        profit = self.profit_per_user * max_users - total_cost

        return (
            max_users, total_cost, deploy_cost, comm_cost, profit,
            edge_traffic_list, external_traffic_list
        )

    def optimize_for_profit(self) -> Tuple:
        """
        4

        Returns:
            (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)

        plan16MultiFunctionOptimizer
        """
        if self.too_large:
            return None, None, None, None

        self._vprint(f"[test_id={self.test_data_id}] ...")

        min_cost_plan = self.enumerate_single_chain("min_cost")
        max_profit_plan = self.enumerate_single_chain("max_profit")
        min_profit_plan = self.enumerate_single_chain("min_profit")
        max_users_plan = self.enumerate_single_chain("max_users")

        self._vprint(f"[test_id={self.test_data_id}] ")

        return (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)
