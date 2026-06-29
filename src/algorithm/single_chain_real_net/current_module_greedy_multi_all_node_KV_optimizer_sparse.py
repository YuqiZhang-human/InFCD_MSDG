#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Current-module greedy variant of the sparse multi-all-node optimizer.
LMP
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from src.algorithm.single_chain_real_net.multi_all_node_KV_optimizer_sparse import (
    MultiFunctionOptimizerSparse,
)


class CurrentModuleGreedyMultiFunctionOptimizerSparse(MultiFunctionOptimizerSparse):
    """Sparse multi-all-node greedy optimizer using current-module local scores."""

    def _evaluate_current_module_choice(
        self,
        module_idx: int,
        node: int,
        deployment: List[int],
    ) -> Optional[Tuple[int, float, float]]:
        """Return ``(users, local_cost, local_profit)`` for one module choice.

        The local score intentionally uses only:
        - deployment cost caused by placing module ``module_idx`` on ``node``;
        - communication cost between module ``module_idx - 1`` and ``module_idx``;
        - user limits from this node and that one adjacent module boundary.
        """
        node = int(node)
        if node < 0 or node >= self.node_count:
            return None

        comp_demand, kv_demand = self.get_module_demand(module_idx)
        weight_mem = self.get_module_weight_memory(module_idx)
        comp_cap, mem_cap = self.get_node_capacity(node)

        if comp_demand > 0 and comp_cap <= 0:
            return None

        static_weight_add = (
            weight_mem
            if module_idx not in self.modules_loaded_per_node[node]
            else 0.0
        )
        if static_weight_add > mem_cap + 1e-9:
            return None

        limits: List[float] = []
        if comp_demand > 0:
            limits.append(comp_cap / comp_demand)

        if kv_demand > 0:
            avail_mem_for_kv = mem_cap - static_weight_add
            if avail_mem_for_kv <= 0:
                return None
            limits.append(avail_mem_for_kv / kv_demand)

        if module_idx > 0:
            prev_node = int(deployment[module_idx - 1])
            if prev_node < 0:
                return None
            data_size = self.get_data_size(module_idx - 1)
            if prev_node != node and data_size > 0:
                path_bw = self.get_path_bottleneck_bw(prev_node, node)
                if path_bw <= 0:
                    return None
                limits.append(path_bw / data_size)

        if not limits:
            users_local = 1
        else:
            users_local = int(math.floor(min(limits)))
            if users_local <= 0:
                return None

        gpu_cost_node, mem_cost_node = self.node_costs[node]
        deploy_cost_local = comp_demand * users_local * float(gpu_cost_node)
        deploy_cost_local += (
            kv_demand * users_local + static_weight_add
        ) * float(mem_cost_node)

        comm_cost_local = 0.0
        if module_idx > 0:
            prev_node = int(deployment[module_idx - 1])
            if prev_node != node:
                data_size = self.get_data_size(module_idx - 1)
                if data_size > 0:
                    hop = self.get_path_hops(prev_node, node)
                    if hop <= 0 or hop >= 10**9:
                        return None
                    comm_cost_local += (
                        data_size * self.bandwidth_cost * users_local * hop
                    )

        local_cost = deploy_cost_local + comm_cost_local
        local_profit = self.profit_per_user * users_local - local_cost
        return users_local, local_cost, local_profit

    def _deploy_single_chain_greedy(
        self, objective: str
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int], int]]:
        n = self.node_count
        m = self.module_count
        if n <= 0 or m <= 0:
            return None

        deployment = [-1] * m

        for module_idx in range(m):
            best_candidate = None
            candidate_nodes = self._build_candidate_nodes(module_idx, deployment)

            for node in candidate_nodes:
                local_eval = self._evaluate_current_module_choice(
                    module_idx,
                    int(node),
                    deployment,
                )
                if local_eval is None:
                    continue
                users_local, local_cost, local_profit = local_eval

                if objective == "min_cost":
                    score = (
                        float(local_cost),
                        float(-local_profit),
                        float(-users_local),
                    )
                elif objective == "max_profit":
                    score = (
                        float(-local_profit),
                        float(-users_local),
                        float(local_cost),
                    )
                elif objective == "min_profit":
                    score = (
                        float(local_profit),
                        float(local_cost),
                        float(users_local),
                    )
                elif objective == "max_users":
                    score = (
                        float(-users_local),
                        float(local_cost),
                        float(-local_profit),
                    )
                else:
                    continue

                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (score, int(node))

            if best_candidate is None:
                return None

            deployment[module_idx] = best_candidate[1]

        max_users = self.calculate_max_users_for_deployment(deployment)
        if max_users <= 0:
            return None

        total_cost, deploy_cost, comm_cost, profit = self.calculate_costs_for_deployment(
            deployment,
            max_users,
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


CurrentModuleGreedyMultiAllNodeKVOptimizerSparse = (
    CurrentModuleGreedyMultiFunctionOptimizerSparse
)
