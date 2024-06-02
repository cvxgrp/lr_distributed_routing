import numpy as np
import mlrfit as mf


from lrrouting.utils import *

class DANode:
    def __init__(self, b, c):
        self.b = b
        self.c = c


class DARouting:
    def __init__(self, nodes, adjacency_list):
        self.nodes = nodes
        # for vertex a list of tuples (b, w_ab)
        self.adjacency_list = adjacency_list 


    def dist(self, i, j, ord=2):
        if i == j:
            return 0
        else:
            return np.linalg.norm(self.nodes[i].b - self.nodes[j].c, ord=ord)
    

    def _sort_dist_a_Na_t(self, *, a, t):
        d_at = np.zeros(len(self.adjacency_list[a]))
        for idx, (b, w_b) in enumerate(self.adjacency_list[a]):
            d_at[idx] = w_b + self.dist(b, t)
        idx_sorted = np.argsort(d_at)
        return idx_sorted, d_at
    
    def route_slow(self, s, t):
        # routing using DFS to guarantee path finding
        if s == t:
            return [s], 0
        used = [False] * len(self.nodes)
        stack = [(0, [s])]
        # full_path = []
        while stack != []: 
            wpath, path_a = stack.pop()
            a = path_a[-1]
            if not used[a]:
                if a == t: break
                used[a] = True
                idx_sorted, d_at = self._sort_dist_a_Na_t(a=a, t=t)
                # stack processed nodes in reversed order (stack data structure), 
                # look at nodes in the order of increasing distance from it to t
                for b_idx in idx_sorted[::-1]: 
                    b_next, w_ab = self.adjacency_list[a][b_idx]
                    if not used[b_next]:
                        stack += [(wpath + w_ab, path_a + [b_next])]
        assert path_a[-1] == t
        self.path = path_a
        self.wpath = wpath
        return path_a, wpath

    def add_to_path(self, path, a, parent_a):
        # add node a to path and get valid path by removing a loop 
        # if it exists from parent_a
        if path == [] or path[-1] == parent_a:
            return path + [a]
        p_idx = path.index(parent_a)
        path = path[:p_idx + 1] + [a]
        return path

    def route(self, s, t):
        # routing using DFS to guarantee path finding
        if s == t:
            return [s], 0
        used = [False] * len(self.nodes)
        stack = [(s, None)]
        path = []
        while stack != []: 
            a, parent_a = stack.pop()
            path = self.add_to_path(path, a, parent_a)
            if not used[a]:
                if a == t: break
                used[a] = True
                idx_sorted, d_at = self._sort_dist_a_Na_t(a=a, t=t)
                # stack processed nodes in reversed order (stack data structure), 
                # look at nodes in the order of increasing distance from it to t
                for b_idx in idx_sorted[::-1]: 
                    b_next, w_ab = self.adjacency_list[a][b_idx]
                    if not used[b_next]:
                        stack += [(b_next, a)]
        assert path[-1] == t
        wpath = path_weight(path, self.adjacency_list)
        self.path = path
        self.wpath = wpath
        return path, wpath

    def next_node(self, *, s, t, a):
        d_at = np.zeros(len(self.adjacency_list[a]))
        for idx, (b, w_b) in enumerate(self.adjacency_list[a]):
            d_at[idx] = w_b + self.dist(b, t)
        b_idx = np.argmin(d_at)
        b_next = self.adjacency_list[a][b_idx][0]
        w_b_next = self.adjacency_list[a][b_idx][1]
        return b_next, w_b_next


def construct_nodes(hat_A):
    nodes = []
    for i in range(hat_A.B.shape[0]):
        nodes += [DANode(b = np.split(hat_A.B[hat_A.pi_inv_rows[i]], np.cumsum(hat_A.ranks)[:-1]), \
                        c = np.split(hat_A.C[hat_A.pi_inv_cols[i]], np.cumsum(hat_A.ranks)[:-1]))]
        nodes[i].rblocks = []
        nodes[i].cblocks = []

    for level in range(len(hat_A.ranks)):
        num_blocks = len(hat_A.hpart['rows']['lk'][level])-1
        for block in range(num_blocks):
            r1, r2 = hat_A.hpart['rows']['lk'][level][block], hat_A.hpart['rows']['lk'][level][block+1]
            c1, c2 = hat_A.hpart['cols']['lk'][level][block], hat_A.hpart['cols']['lk'][level][block+1]
            for a in range(r1, r2):
                i = hat_A.pi_rows[a]
                nodes[i].rblocks += [block]
            for b in range(c1, c2):
                j = hat_A.pi_cols[b]
                nodes[j].cblocks += [block]
                    
    return nodes
