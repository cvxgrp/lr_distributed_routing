import networkx as nx
import osmnx as ox
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np
from typing import List, Tuple, Callable, TypedDict, List, Set, Optional, Union 
import matplotlib.pyplot as plt
import numba as nb
import sys





class DualLogger:
    def __init__(self, filepath, stdout_original):
        self.terminal = stdout_original
        self.log = open(filepath, "a")

    def write(self, message):
        if message and not message.endswith('\n'):
            message += '\n'
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


@nb.njit()
def is_close(a, b, rtol=1e-05, atol=1e-08):
    return np.sum(np.abs(a - b) > atol + rtol * np.abs(b)) == 0


@nb.njit()
def is_close_scalar(a, b, rtol=1e-05, atol=1e-08):
    return (np.abs(a - b) > atol + rtol * np.abs(b)) == 0


@nb.njit(parallel=True)
def replace_nan_1D(arr, nan=0.0):
    for i in nb.prange(arr.shape[0]):
        if np.isnan(arr[i]):
            arr[i] = nan
    return arr


def eulidean_dist_matrix(X):
    """
    X: n x p 
        an embedding matrix of n items
    """
    n = X.shape[0]
    Gram = X @ X.T 
    d = np.diag(Gram).reshape(-1, 1)
    Dist_sq = np.tile(d.T, (n, 1)) - 2 * Gram + np.tile(d, (1, n))
    assert np.allclose(np.linalg.norm(np.diag(Dist_sq)), 0) and np.min(Dist_sq) > -1e-8, print(np.min(Dist_sq), np.isnan(X).any())
    Dist = np.sqrt(np.maximum(Dist_sq, 0))
    return Dist, Dist_sq


def adjacency_list(A) -> List[List[int]]:
    n = A.shape[0]
    adjacency_list = [[] for _ in range(n)]
    assert np.allclose(A, A.T)
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] != 0:
                adjacency_list[i] += [(j, A[i,j])]
                adjacency_list[j] += [(i, A[i,j])]
    return adjacency_list


def adjacency_list_to_matrix(adjacency_list):
    n = len(adjacency_list)
    A = np.zeros((n, n))
    for i in range(n):
        for j, val_j in adjacency_list[i]:
                A[i, j] = val_j
    return A


def adjacency_directed_list(A) -> List[List[int]]:
    n = A.shape[0]
    adjacency_list = [[] for _ in range(n)]
    if np.allclose(A, A.T): print("graph is not directed")
    for i in range(n):
        for j in range(n):
            if A[i,j] != 0:
                adjacency_list[i] += [(j, A[i,j])]
    return adjacency_list


def create_random_ij_lognormal_graph(n, sigma=0.5):
    # strongly connected
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    while not nx.is_strongly_connected(G):
        for _ in range(100):
            i, j = np.random.choice(n, 2, replace=False)
            weight = np.exp(np.random.randn() * sigma)
            G.add_edge(i, j, weight=weight)
    return G


def dist_matrix_osmnx(place, directed=True, nodes=False):
    G = ox.graph_from_place(place, network_type="drive")
    G.remove_edges_from(nx.selfloop_edges(G))
    Adj_spr = nx.adjacency_matrix(G)
    Adj = np.array(Adj_spr.todense())
    if not directed:
        Adj = (Adj + Adj.T)*0.5
    nodes_cc, Adj, Dist = graph_distance_matrix(Adj, directed=directed, n=G.number_of_nodes())[:3]
    assert (Dist==np.inf).sum() == 0
    # Diam = Dist[Dist != np.inf].max(); Dist[Dist == np.inf] = 2*Diam
    k = {i:val for i, val in enumerate((Adj > 0).sum(axis=1))}
    print("in  degrees:", {n: list(k.values()).count(n) for n in range(max(k.values()) + 1)})
    k = {i:val for i, val in enumerate((Adj > 0).sum(axis=0))}
    print("out degrees:", {n: list(k.values()).count(n) for n in range(max(k.values()) + 1)})
    if nodes:
        return G, Adj, Dist, nodes_cc
    else:
        return G, Adj, Dist
    

def nx_graph_to_matrices(G, nodes=False):
    directed = nx.is_directed(G) 
    G.remove_edges_from(nx.selfloop_edges(G))
    Adj_spr = nx.adjacency_matrix(G)
    Adj = np.array(Adj_spr.todense())
    if not directed:
        Adj = (Adj + Adj.T)*0.5
    nodes_cc, Adj, Dist = graph_distance_matrix(Adj, directed=directed, n=G.number_of_nodes())
    assert (Dist==np.inf).sum() == 0
    # Diam = Dist[Dist != np.inf].max(); Dist[Dist == np.inf] = 2*Diam
    k = {i:val for i, val in enumerate((Adj > 0).sum(axis=1))}
    print("in  degrees:", {n: list(k.values()).count(n) for n in range(max(k.values()) + 1)})
    k = {i:val for i, val in enumerate((Adj > 0).sum(axis=0))}
    print("out degrees:", {n: list(k.values()).count(n) for n in range(max(k.values()) + 1)})
    assert np.allclose(np.diag(Dist), np.zeros(Dist.shape[0]))
    if nodes:
        return Adj, Dist, nodes_cc
    else:
        return Adj, Dist


def graph_distance_matrix(A=None, n=None, sparse=False, printing=False, directed=False) -> np.ndarray:
    """
    Undirected graph is always connected, as A=(A0+A0.T)/2
    For directed graph use the largest strongly connected commponent
    """
    if sparse:
        A_srs = A
    else:
        A_srs = csr_matrix(A)
        # assert directed == (not np.allclose(A, A.T))
    nodes = np.arange(n)
    if directed:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph())
        G_cc = list(sorted(nx.strongly_connected_components(G), key = len, reverse=True))
        print([len(cc) for cc in G_cc])
        nodes = np.array(list(G_cc[0]))
        print(f"n_cc = {nodes.size}, n0 = {A.shape[0]}")
        A = A[nodes, :][:, nodes]
        A_srs = csr_matrix(A)
    Dist = shortest_path(csgraph=A_srs, directed=directed)
    assert (Dist < np.inf).all()
    if printing:
        print(f"|E| = {(A>0).sum()}")
        print(f"deg_max = {A.sum(axis=1).max()}, deg_mean = {A.sum(axis=1).mean()}, deg_min = {A.sum(axis=1).min()}")
        print(f"{Dist.max()=}")
    return nodes, A, Dist


def plot_nx_G(G, pos=None, f_layout=nx.circular_layout, node_color='skyblue', figsize=(4, 4), width=1.0, with_labels=True, node_size=200):
    plt.figure(figsize=figsize, dpi=120)
    if pos == None:
        pos = f_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if with_labels:
        nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    else:
        nx.draw(G, pos, node_color=node_color, with_labels=False, width=width, node_size=node_size)
    plt.show()


def valid_path(path, wpath, adjacency_list, s, t):
    # verify if a path is valid wrt given adjacency list
    wp = 0
    a = s
    assert path[0] == s and path[-1] == t
    for b in path[1:]:
        w_ab = weight_a_b(adjacency_list, a, b)
        if w_ab == np.inf: return False 
        wp += w_ab 
        a = b
    assert np.allclose(wp, wpath)
    return True


def path_weight(path, adjacency_list):
    # get weight of the path given adjacency list
    wp = 0
    a = path[0]
    for b in path[1:]:
        w_ab = weight_a_b(adjacency_list, a, b)
        assert w_ab < np.inf
        wp += w_ab 
        a = b
    return wp


def weight_a_b(adjacency_list, a, b):
    for c in adjacency_list[a]:
        if c[0] == b:
            return c[1]
    return np.inf


def adaptive_row_col(n, Dist, frac=0.5, coeff=1, debug=False, percent=False):
    if percent:
        nsamples = int(coeff * n / 100)
    else:
        nsamples = int(coeff * np.sqrt(n))
    pi_rows, pi_cols = np.zeros(nsamples, dtype=int), np.zeros(nsamples, dtype=int)
    # sample frac * nsamples of rows and cols at uniform
    pi_rows[:int(nsamples * frac)] = np.random.permutation(n)[:int(nsamples * frac)]
    pi_cols[:int(nsamples * frac)] = np.random.permutation(n)[:int(nsamples * frac)]
    print(f"{nsamples=}, random_samples={int(nsamples * frac)}")
    # for the remaining samples adaptively find and add col/row nodes that are the 
    # furthest from the existing nodes in row/col index sets 
    for k in range(int(nsamples * frac), nsamples):
        # add a new node to row index set that is the 
        # furthest from the existing nodes in column index set
        min_dists_all_to_c = np.min(Dist[:, pi_cols[:k]], axis=1)
        row_idx = np.argsort(min_dists_all_to_c)[::-1]
        new_node = row_idx[np.isin(row_idx, pi_rows[:k], invert=True)][0]
        if debug:
            for idx in row_idx:
                if not idx in pi_rows[:k]:
                    new_node2 = idx 
                    break
            assert new_node2 == new_node and (np.diff(min_dists_all_to_c[row_idx]) <= 0).all()
        pi_rows[k] = new_node
        # add a new node to col index set that is the 
        # furthest from the existing nodes in row index set
        min_dists_all_to_r = np.min(Dist[pi_rows[:k+1], :], axis=0)
        col_idx = np.argsort(min_dists_all_to_r)[::-1]
        if debug:
            assert min_dists_all_to_c.size == n and (np.diff(min_dists_all_to_r[col_idx]) <= 0).all()
        new_node = col_idx[np.isin(col_idx, pi_cols[:k], invert=True)][0]
        pi_cols[k] = new_node

    assert np.unique(pi_rows).size == nsamples and np.unique(pi_cols).size == nsamples
    return pi_rows, pi_cols


def furthest_adaptive_row_col(n, Dist, frac=0.5, coeff=1):
    nsamples = int(coeff * np.sqrt(n))
    pi_rows, pi_cols = np.zeros(nsamples, dtype=int), np.zeros(nsamples, dtype=int)
    # sample frac * nsamples of rows and cols at uniform
    pi_rows[:int(nsamples * frac)] = np.random.permutation(n)[:int(nsamples * frac)]
    pi_cols[:int(nsamples * frac)] = np.random.permutation(n)[:int(nsamples * frac)]

    row_idx = np.argsort(Dist.sum(axis=1))[::-1]
    k = int(nsamples * frac)
    pi_rows[k:] = row_idx[np.isin(row_idx, pi_rows[:k], invert=True)][:nsamples-k]
    col_idx = np.argsort(Dist.sum(axis=0))[::-1]
    pi_cols[k:] = col_idx[np.isin(col_idx, pi_cols[:k], invert=True)][:nsamples-k]

    assert np.unique(pi_rows).size == nsamples and np.unique(pi_cols).size == nsamples
    return pi_rows, pi_cols


def sample_dist(n, pi_rows, pi_cols, Dist):
    rDist = Dist[pi_rows]
    cDist = Dist[:, pi_cols].T

    pi_rows_c = np.delete(np.arange(n), pi_rows, axis=0)
    pi_cols_c = np.delete(np.arange(n), pi_cols, axis=0)

    assert np.unique(pi_cols).size + np.unique(pi_cols_c).size == n
    assert np.unique(pi_rows).size + np.unique(pi_rows_c).size == n
    assert (pi_cols).size + (pi_cols_c).size == n
    assert (pi_rows).size + (pi_rows_c).size == n
    return rDist, cDist, pi_rows_c, pi_cols_c


def stats_from_histogram(counts, bins, plot=False):
    bin_centers = (bins[:-1] + bins[1:]) / 2

    hist_mean = np.sum(bin_centers * counts) / np.sum(counts)
    hist_std = np.sqrt(np.sum(((bin_centers - hist_mean) ** 2) * counts) / np.sum(counts))
    hist_min = bins.min()
    hist_max = bins.max()

    cumulative_values = np.cumsum(counts) * (bins[1] - bins[0])
    hist_median = np.interp(0.5, cumulative_values, bin_centers)
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(bins[:-1], bins, weights=counts, color='g', alpha=0.5)
        plt.axvline(hist_mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {hist_mean:.2f}')
        plt.axvline(hist_median, color='b', linestyle='dashed', linewidth=1, label=f'Median: {hist_median:.2f}')
        plt.axvline(hist_mean - hist_std, color='r', linestyle='dotted', linewidth=1, label=f'Std Dev: {hist_std:.2f}')
        plt.axvline(hist_mean + hist_std, color='r', linestyle='dotted', linewidth=1)
        plt.legend()
        plt.show()
    return hist_mean, hist_std, hist_median, hist_min, hist_max
