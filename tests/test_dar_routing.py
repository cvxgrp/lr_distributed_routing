import numpy as np
import osmnx as ox
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

import random
import pickle

import mlrfit as mf
import lrrouting as ldr


def route_st_dar(s, t, dar, del_cycle=False, max_iters=50):
    route = [s]
    a = s; wpath = 0.
    while len(route) < max_iters and a != t:
        if del_cycle:
            b_next, w_b_next = dar.next_node_del_cycles(s=s, t=t, a=a)
        else:
            b_next, w_b_next = dar.next_node(s=s, t=t, a=a)
        wpath += w_b_next
        a = b_next
        route += [b_next]
        if len(dar.adjacency_list[a]) == 0:
            break
    return route


def route_st_matrix(s, t, func_next_node, adjacency_list, max_iters = 50):
    route = [s]
    a = s; wpath = 0.
    while len(route) < max_iters and a != t:
        b_next = func_next_node(s=s, t=t, a=a)
        for b, w_ab in adjacency_list[a]:
            if b == b_next: 
                w_a_b_next = w_ab 
                break
        wpath += w_a_b_next
        a = b_next
        route += [b_next]
        if len(adjacency_list[a]) == 0:
            break
    return route


rank = 35
for graph in range(5):
    if graph == 0:
        place = "Pacifica, CA, USA"
        G, Adj, Dist, nodes_cc = ldr.dist_matrix_osmnx(place, directed=True, nodes=True)
        m = n = Dist.shape[0]
        diam_G = Dist.max()
        w_min = Dist[Dist>0].min()
        print(m, diam_G, w_min, (Adj>0).sum())
    else:
        n = 500
        G = nx.fast_gnp_random_graph(n, p=0.05, directed=True)
        Adj, Dist, nodes_cc = ldr.nx_graph_to_matrices(G, nodes=True)
        adjacency_list = ldr.adjacency_directed_list(Adj)
        diam_G = Dist.max()
        w_min = Dist[Dist>0].min()
        m = n = len(nodes_cc)
        print(f"{n=}, {G.number_of_nodes()=}, {G.number_of_edges()=}")

    rt_max_iters = min(int(5*Dist.max()/w_min), (10**4) // 2)
    adjacency_list = ldr.adjacency_directed_list(Adj)
    sources, targets = ldr.st_pairs(n, Dist, 5020)
    M = min(5000, sources.size)
    sources = sources[:M]
    targets = targets[:M]

    PSD = False
    symm = np.allclose(Dist, Dist.T) if m==n else False

    hpart = mf.random_hpartition(m, n, num_levels=1, symm=symm, perm=False)
    B1, C1 = mf.single_level_factor_fit(Dist, np.array([rank]), hpart, level=0, symm=symm, PSD=PSD)[:2]

    lr_dar = ldr.construct_lr_graph(B1, C1, adjacency_list)
    lr_A = B1 @ C1.T

    assert np.allclose(ldr.adjacency_list_to_matrix(adjacency_list), Adj)
    print("PASSED adjacency_list_to_matrix")

    total_matches = 0
    dist_func = lambda b,t: lr_A[b,t]
    func_next_node = ldr.next_node(dist_func, adjacency_list, Adj)

    for (s, t) in zip(sources, targets):
        if s == t: continue
        r1 = route_st_dar(s, t, lr_dar, del_cycle=False)
        r2 = route_st_matrix(s, t, func_next_node, adjacency_list)
        total_matches += np.allclose(r1, r2) if len(r1)==len(r2) else 0
        for a in r1[1:]:
            d_at1, d_at2 = np.zeros(len(adjacency_list[a])), np.zeros(len(adjacency_list[a]))
            for idx, (b, w_b) in enumerate(adjacency_list[a]):
                d_at1[idx] = w_b + lr_dar.dist(b, t)
                d_at2[idx] = Adj[a, b] + dist_func(b, t)
            assert np.allclose(d_at2, d_at1)
            b_next1 = lr_dar.next_node(s=s, t=t, a=a)[0]
            b_next2 = func_next_node(s=s, t=t, a=a)
            # print(f"{adjacency_list[a]} \n{d_at1=} {np.argsort(d_at1)} \n{d_at2=} {np.argsort(d_at2)}\n")
            assert b_next1 == adjacency_list[a][np.argmin(d_at1)][0] \
                    and b_next2 == adjacency_list[a][np.argmin(d_at2)][0]
    
    ratios = ldr.subopt_ratios(lr_dar, Dist, sources, targets, rt_max_iters, del_cycles=False)
    ratios_no_cycles = ldr.subopt_ratios(lr_dar, Dist, sources, targets, rt_max_iters, del_cycles=True)
    assert (ratios >= ratios_no_cycles).all()
    print("PASSED, fraction of fully matched paths is %.4f"%( total_matches / len(sources)))
