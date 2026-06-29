#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10w
-
-
- source-tree BFS  pair-path
- consumed delta  remaining


1.  O(N^2)
2.  (s,t)->
3. reset_bandwidth()
"""

from __future__ import annotations

from array import array
from collections import OrderedDict, deque
from typing import Dict, Iterable, List, Optional, Tuple


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


class SparseTopology:
    """
    - neighbors[u]:  u
    - _initial[(min(u,v), max(u,v))]: MB/s
    - _consumed[(min(u,v), max(u,v))]: MB/s
    - _source_cache[s] = (parent_array, dist_array)
    """

    def __init__(
        self,
        node_count: int,
        neighbors: List[List[int]],
        bandwidth_mb_per_edge: Dict[Tuple[int, int], float],
        source_cache_max: int = 8,
    ) -> None:
        self.node_count = int(node_count)
        self.neighbors: List[Tuple[int, ...]] = [
            tuple(int(v) for v in vs) for vs in neighbors
        ]

        self._initial: Dict[Tuple[int, int], float] = {}
        for (u, v), bw in bandwidth_mb_per_edge.items():
            if u == v:
                continue
            k = edge_key(int(u), int(v))
            self._initial[k] = float(bw)

        self._consumed: Dict[Tuple[int, int], float] = {}
        self._source_cache_max = max(1, int(source_cache_max))
        self._source_cache: "OrderedDict[int, Tuple[array, array]]" = OrderedDict()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def reset_bandwidth(self) -> None:
        """"""
        self._consumed.clear()

    def clear_caches(self) -> None:
        """ BFS  test case """
        self._source_cache.clear()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def get_initial_bandwidth(self, u: int, v: int) -> float:
        if u == v:
            return 0.0
        return float(self._initial.get(edge_key(u, v), 0.0))

    def get_link_bandwidth(self, u: int, v: int) -> float:
        if u == v:
            return 0.0
        k = edge_key(u, v)
        init_bw = float(self._initial.get(k, 0.0))
        used_bw = float(self._consumed.get(k, 0.0))
        rem = init_bw - used_bw
        return rem if rem > 0.0 else 0.0

    def decrement_bandwidth(self, u: int, v: int, consume: float) -> None:
        if u == v or consume <= 0:
            return
        k = edge_key(u, v)
        init_bw = float(self._initial.get(k, 0.0))
        if init_bw <= 0.0:
            return
        used_bw = float(self._consumed.get(k, 0.0))
        new_used = used_bw + float(consume)
        if new_used > init_bw:
            new_used = init_bw
        self._consumed[k] = new_used

    # ------------------------------------------------------------------
    # BFS source-tree cache
    # ------------------------------------------------------------------
    def _get_source_tree(self, s: int) -> Optional[Tuple[array, array]]:
        if s < 0 or s >= self.node_count:
            return None

        if s in self._source_cache:
            self._source_cache.move_to_end(s)
            return self._source_cache[s]

        parent = array("i", [-1]) * self.node_count
        dist = array("i", [-1]) * self.node_count

        q = deque([s])
        parent[s] = s
        dist[s] = 0

        while q:
            u = q.popleft()
            next_dist = dist[u] + 1
            for v in self.neighbors[u]:
                if parent[v] == -1:
                    parent[v] = u
                    dist[v] = next_dist
                    q.append(v)

        self._source_cache[s] = (parent, dist)
        self._source_cache.move_to_end(s)

        while len(self._source_cache) > self._source_cache_max:
            self._source_cache.popitem(last=False)

        return parent, dist

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def get_path_hops(self, s: int, t: int) -> int:
        if s == t:
            return 0
        tree = self._get_source_tree(s)
        if tree is None:
            return 10**9
        _, dist = tree
        d = int(dist[t])
        return d if d >= 0 else 10**9

    def get_path_nodes(self, s: int, t: int) -> Optional[List[int]]:
        if s == t:
            return [s]
        tree = self._get_source_tree(s)
        if tree is None:
            return None

        parent, dist = tree
        if int(dist[t]) < 0:
            return None

        path_rev: List[int] = [int(t)]
        cur = int(t)

        while cur != s:
            cur = int(parent[cur])
            if cur < 0:
                return None
            path_rev.append(cur)

        path_rev.reverse()
        return path_rev

    def get_path_bottleneck_bw(self, s: int, t: int) -> float:
        if s == t:
            return float("inf")
        path = self.get_path_nodes(s, t)
        if not path or len(path) < 2:
            return 0.0
        b = float("inf")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            bw = self.get_link_bandwidth(u, v)
            if bw < b:
                b = bw
        return float(b) if b < float("inf") else 0.0

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def iter_edges_with_usage(self):
        """
         (i, j, initial_bw, used_bw) i < j
        """
        for (i, j), init_bw in sorted(self._initial.items()):
            used_bw = float(self._consumed.get((i, j), 0.0))
            if used_bw > 1e-12 and init_bw > 1e-12:
                yield i, j, float(init_bw), float(used_bw)
