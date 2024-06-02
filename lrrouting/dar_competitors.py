import numpy as np
import mlrfit as mf
import types
import time, gc
import numba as nb

from astar import AStar

from scipy.sparse.linalg import svds, eigsh

import pymde
from sklearn.manifold import MDS, Isomap, SpectralEmbedding

from lrrouting.dar import *
from lrrouting.utils import *
from lrrouting.routing import *
from lrrouting.bcd_convex_concave import *
from lrrouting.fast_convex_concave import *
from lrrouting.cg import *



def stat_moments(data):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    quartile_25 = np.percentile(data, 25)
    quartile_75 = np.percentile(data, 75)
    maximum = np.max(data)
    minimum = np.min(data)
    return mean, median, std, quartile_25, quartile_75, maximum, minimum


def print_stats_moments(stats):
    # mean, median, std, quartile_25, quartile_75, maximum, minimum
    print(f"mean={stats[0]*100:.2f}, M={stats[1]*100:.2f}, SD={stats[2]*100:.2f}, q_25={stats[3]*100:.2f}, q_75={stats[4]*100:.2f}, max={stats[5]*100:.2f}, min={stats[6]*100:.2f}")


def relative_average_difference(n, Z, pi_rows, pi_cols, rDist, cDist, printing=True, symm=False, bins=30):
    # ignore relative difference on diagonal entries
    if symm:
        assert Z.shape[0] == n
        dists_r, dists_c = pi_asymm_eulidean_dist_matrix(Z, Z, pi_rows, pi_cols)
    else:
        assert Z.shape[0] == 2 * n
        dists_r, dists_c = pi_asymm_eulidean_dist_matrix(Z[:n], Z[n:], pi_rows, pi_cols)

    # remove entries on the diagonal from the distribution
    idx = np.arange(0, pi_rows.size * n, n) + pi_rows 
    mask = ~np.isin(np.arange(n * pi_rows.size), idx)
    r_ratios = np.divide(np.abs(rDist - dists_r), rDist, where=np.abs(rDist)>1e-9).flatten()[mask]
    assert r_ratios.size == (n - 1)* pi_rows.size
    if printing:
        print(f"rows: m={r_ratios.mean()*100:.2f}%, SD={r_ratios.std()*100:.2f}%, M={np.median(r_ratios)*100:.2f}%, max={r_ratios.max()*100:.2f}%")
    r_hist = stat_moments(r_ratios)
    del r_ratios

    idx = np.arange(0, pi_cols.size * n, n) + pi_cols 
    mask = ~np.isin(np.arange(n * pi_cols.size), idx)
    c_ratios = np.divide(np.abs(cDist - dists_c), cDist, where=np.abs(cDist)>1e-9).flatten()[mask]
    assert c_ratios.size == (n - 1) * pi_cols.size 
    if printing:
        print(f"cols: m={c_ratios.mean()*100:.2f}%, SD={c_ratios.std()*100:.2f}%, M={np.median(c_ratios)*100:.2f}%, max={c_ratios.max()*100:.2f}%")
    c_hist = stat_moments(c_ratios)
    del c_ratios
    
    return r_hist, c_hist


class BasicAStar(AStar):
    def __init__(self, info):
        self.nodes = info["nodes"]
        self.adjacency_list = [[] for _ in range(len(self.nodes))]
        self.ws_list = [[] for _ in range(len(self.nodes))]
        for a in range(len(self.nodes)):
            for b, w_ab in info["adj_ws"][a]:
                self.adjacency_list[a] += [b]
                self.ws_list[a] += [w_ab]

    def neighbors(self, n):
        return self.adjacency_list[n]

    def distance_between(self, a, b):
        b_idx = self.adjacency_list[a].index(b)
        return self.ws_list[a][b_idx]
            
    def heuristic_cost_estimate(self, i, j):
        if i == j: 
            return 0
        else:
            return np.linalg.norm(self.nodes[i].b - self.nodes[j].c, ord=2)
    
    def is_goal_reached(self, current, goal):
        return current == goal
    

class IDA_Star():
    def __init__(self, info):
        self.nodes = info["nodes"]
        self.adjacency_list = [[] for _ in range(len(self.nodes))]
        self.ws_list = [[] for _ in range(len(self.nodes))]
        for a in range(len(self.nodes)):
            for b, w_ab in info["adj_ws"][a]:
                self.adjacency_list[a] += [b]
                self.ws_list[a] += [w_ab]

    def neighbors(self, n):
        return self.adjacency_list[n]

    def distance_between(self, a, b):
        b_idx = self.adjacency_list[a].index(b)
        return self.ws_list[a][b_idx]

    def heuristic_cost_estimate(self, i, j):
        if i == j: 
            return 0
        else:
            return np.linalg.norm(self.nodes[i].b - self.nodes[j].c, ord=2)
    
    def is_goal_reached(self, current, goal):
        return current == goal
    

    def astar(self, start, goal):
        def ida_star_recursive(path, g, bound):
            current = path[-1]
            f = g + self.heuristic_cost_estimate(current, goal)
            if f > bound:
                return f, None
            if self.is_goal_reached(current, goal):
                return f, path
            minimum = np.inf
            for neighbor in self.neighbors(current):
                if neighbor not in path:
                    path.append(neighbor)
                    t, result = ida_star_recursive(path, g + self.distance_between(current, neighbor), bound)
                    if result is not None:
                        return t, result
                    if t < minimum:
                        minimum = t
                    path.pop()
            return minimum, None

        bound = self.heuristic_cost_estimate(start, goal)
        path = [start]
        while True:
            t, result = ida_star_recursive(path, 0, bound)
            if result is not None:
                return result
            if t == np.inf:
                return None
            bound = t
    
    
def astar_routing_stats(sources, targets, astar_emb, adjacency_list, Dist):
    ratios = np.zeros(len(sources))
    for i, (s, t) in enumerate(zip(sources, targets)):
            if s == t: ratios[i] = 1
            path = list(astar_emb.astar(s, t))
            wpath = path_weight(path, adjacency_list)
            if path[-1] == t:
                ratios[i] = wpath / Dist[s, t]
            else:
                ratios[i] = np.inf
    median_stretch = np.median(ratios) * 100.
    mean_stretch = ratios.mean() * 100.
    print(f"{median_stretch=:.1f}%, {mean_stretch=:.1f}%")
    fracs = [5, 2, 1.5, 1.2, 1]; f_ratios = {}
    for frac in fracs:
        f_ratios[frac] = 100.*(ratios <= frac + 1e-8).sum() / ratios.size
    print(f"%[ratio<2] = {f_ratios[2]:.2f}%, %[ratio<1.2] = {f_ratios[1.2]:.2f}%, %[ratio=1.] = {f_ratios[1]:.2f}%")
    return ratios


def mds_stress_majorization(rank, deltas_ij, n_init=2, max_iter=500, eps=1e-3, verbose=False):
    best_emb = (None, np.inf)
    n = deltas_ij.shape[0]
    for _ in range(n_init):
        X = np.random.uniform(size=n * rank).reshape(n, rank)
        prev_raw_stress = np.inf
        for t in range(max_iter):
            dists_ij, _ = eulidean_dist_matrix(X)
            raw_stress = 0.5 * np.power(deltas_ij - dists_ij, 2).sum() / (n * (n-1) / 2)
            # reciprocal_dists_ij = np.reciprocal(dists_ij, where=dists_ij!=0)
            # M = np.multiply(deltas_ij, reciprocal_dists_ij)
            mask = dists_ij == 0
            dists_ij[mask] = 1
            M = np.divide(deltas_ij, dists_ij); M[mask] = 0 
            B = -M
            B[np.arange(n), np.arange(n)] += M.sum(axis=1)
            X = (1. / n) * (B @ X)
            assert (prev_raw_stress + 1e-8 >= raw_stress), print(prev_raw_stress + 1e-8 - raw_stress)
            if t >= 1 and np.abs((prev_raw_stress - raw_stress) / prev_raw_stress) < eps:
                if verbose: print(t)
                break
            prev_raw_stress = raw_stress
        if verbose:
            print(f"{raw_stress=}")
        if best_emb[1] > raw_stress:
            best_emb = (X, raw_stress)
    return best_emb


def spectral_embedding(Adj_symm, rank, normalized=False):
    d = Adj_symm.sum(axis=1)
    if normalized:
        d_sqrt = np.power(Adj_symm.sum(axis=1), -0.5)
        L = np.eye(Adj_symm.shape[0]) - ((Adj_symm * d_sqrt).T * d_sqrt).T
    else:
        L = np.diag(d) - Adj_symm   
    lambdas, U = eigsh(L, k=rank + 1, which='SA')
    idx = np.argsort(lambdas)
    lambdas = lambdas[idx] # increasing  order
    U = U[:, idx][:, 1:]
    X_spectral = U
    return X_spectral


def classical_mds(Dist, dim):
    Dist_sq = np.power(Dist, 2)
    n = Dist_sq.shape[0]
    d = Dist_sq.sum(axis=1).reshape(-1, 1)
    # double center Dist_sq
    G_cmds = - 0.5 * (Dist_sq - d @ np.ones((1, n))/n - np.ones((n, 1)) @ d.T/n 
                      + np.ones((n, n)) * d.sum()/(n**2) )

    lambs, V = eigsh(G_cmds, k=dim, which="LA")
    idx = np.argsort(lambs)[::-1]
    lambs = lambs[idx]
    V = V[:, idx]
    X_cmds = V * np.sqrt(np.maximum(lambs, 0))
    return X_cmds, G_cmds


def construct_lr_graph(B, C, adjacency_list):
    nodes = []
    for i in range(B.shape[0]):
        nodes += [DANode(b = B[i], c=C[i])]

    def lr_dist(self, i, j):
        return self.nodes[i].b @ self.nodes[j].c.T

    lr_dar = DARouting(nodes, adjacency_list)
    lr_dar.dist = types.MethodType(lr_dist, lr_dar)
    return lr_dar


def construct_lr_dist2_graph(B, C, adjacency_list):
    nodes = []
    for i in range(B.shape[0]):
        nodes += [DANode(b = B[i], c=C[i])]

    def lr_dist(self, i, j):
        return np.sqrt(np.abs(self.nodes[i].b @ self.nodes[j].c.T))

    lr_dar = DARouting(nodes, adjacency_list)
    lr_dar.dist = types.MethodType(lr_dist, lr_dar)
    return lr_dar


def construct_xy_node_embedding_graph(X, Y, adjacency_list, ord=2):
    nodes = []
    for i in range(X.shape[0]):
        nodes += [DANode(b = X[i], c=Y[i])]

    def node_embedding_dist(self, i, j):
        if i == j: 
            return 0
        else:
            return np.linalg.norm(self.nodes[i].b - self.nodes[j].c, ord=ord)

    nemb_dar = DARouting(nodes, adjacency_list)
    nemb_dar.dist = types.MethodType(node_embedding_dist, nemb_dar)
    return nemb_dar


def construct_node_embedding_graph(node_embeddings, adjacency_list, ord=2):
    nodes = []
    for i in range(node_embeddings.shape[0]):
        nodes += [DANode(b = node_embeddings[i], c=node_embeddings[i])]

    def node_embedding_dist(self, i, j):
        if i == j: 
            return 0
        else:
            return np.linalg.norm(self.nodes[i].b - self.nodes[j].c, ord=ord)

    nemb_dar = DARouting(nodes, adjacency_list)
    nemb_dar.dist = types.MethodType(node_embedding_dist, nemb_dar)
    return nemb_dar


def odlr_approx(A, rank, symm, T=10):
    d = np.zeros(A.shape[0])
    for t in range(T):
        if t%2 == 0:
            B, C = low_rank_approx(A - np.diag(d), dim=rank, symm=symm)
        else:
            d = np.diag(A - B@C.T)
    return B, C
    

def construct_odlr_graph(B, C, adjacency_list):
    nodes = []
    for i in range(B.shape[0]):
        nodes += [DANode(b = B[i], c=C[i])]

    def odlr_dist(self, i, j):
        if i == j: 
            return 0
        else:
            return self.nodes[i].b @ self.nodes[j].c.T

    odlr_dar = DARouting(nodes, adjacency_list)
    odlr_dar.dist = types.MethodType(odlr_dist, odlr_dar)
    return odlr_dar


def low_rank_projection(D, dim, symm):
    U, Vt, sigmas = mf.frob_low_rank(D, dim=dim, symm=symm)
    return U @ np.diag(sigmas) @ Vt


def low_rank_approx(A, dim=None, symm=False, v0=None):
    """
    Return low rank approximation of A \approx B C^T
    """
    M =  min(A.shape[0], A.shape[1])
    if dim is None: dim = M
    dim = min(dim, min(A.shape[0], A.shape[1]))
    if dim < M:
        try:
            U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0)
        except:
            maxiter = min(A.shape) * 100
            try:
                print(f"svds fail: increase {maxiter=}")
                U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0, maxiter=maxiter)
            except:
                print(f"svds fail: decrease tol")
                U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0, tol=1e-2)
    else:
        U, sigmas, Vt = np.linalg.svd(A, full_matrices=False, hermitian=symm)
    # decreasing order of sigmas
    idx = np.argsort(sigmas)[::-1]
    sigmas = sigmas[idx]
    U = U[:, idx]
    Vt = Vt[idx, :]
    sqrt_sigmas = np.sqrt(np.maximum(sigmas, 0))
    B = U * sqrt_sigmas
    C = Vt.T * sqrt_sigmas
    return B, C


def record_subopts_all_methods(rank, Dist, adjacency_list, mde_graph, sources, targets, symm, \
                               methods=["CMDS", "MDS", "xy_pos", "spectral", "isomap", "my MDS"], Adj=None, node_embeddings=None):
    # methods = ["MDS", "MDE", "LR", "LR_sq", "xy_pos", "ODLR"]
    info = {}
    n = Dist.shape[0]
    print(f"{rank=}; {methods}")
    # ['CMDS', 'MDS', 'my MDS', 'spectral', 'isomap']
    if "CMDS" in methods:
        # Classical MDS
        X_cmds, _ = classical_mds((Dist + Dist.T)/2, rank)
        mds_dar = construct_node_embedding_graph(X_cmds, adjacency_list)
        info['C-MDS'] = {'ratios' : subopt_ratios(mds_dar, Dist, sources, targets)}

    if "MDS" in methods:
        # MDS
        mds = MDS(n_components=rank, metric=True, max_iter=1000, dissimilarity='precomputed')
        mds_embedding = mds.fit_transform((Dist + Dist.T)/2)
        mds_dar = construct_node_embedding_graph(mds_embedding, adjacency_list)
        info['MDS'] = {'ratios' : subopt_ratios(mds_dar, Dist, sources, targets)}

    if "my MDS" in methods:
        X_mds, stress = mds_stress_majorization(rank, (Dist + Dist.T)/2, n_init=2, max_iter=500, eps=1e-3, verbose=False)
        mds_dar = construct_node_embedding_graph(X_mds, adjacency_list)
        info['my MDS'] = {'ratios' : subopt_ratios(mds_dar, Dist, sources, targets)}

    if "spectral" in methods:
        # spectral = SpectralEmbedding(n_components=rank, affinity='precomputed') 
        # sigma = Dist.sum() / (n-1)**2 * 0.05
        # spectral_embedding = spectral.fit_transform(np.exp(-np.power(Dist, 2) / (2 * sigma**2)))
        Adj_symm = (Adj + Adj.T)/2
        X_spectral = spectral_embedding(Adj_symm, rank, normalized=False)
        assert X_spectral.shape == (n, rank)
        spectral_dar = construct_node_embedding_graph(X_spectral, adjacency_list)
        info['spectral'] = {'ratios' : subopt_ratios(spectral_dar, Dist, sources, targets)}

    if "isomap" in methods:
        isomap = Isomap(n_components=rank, metric='precomputed') 
        isomap_embedding = isomap.fit_transform((Dist + Dist.T)/2)
        isomap_dar = construct_node_embedding_graph(isomap_embedding, adjacency_list)
        info['isomap'] = {'ratios' : subopt_ratios(isomap_dar, Dist, sources, targets)}

    if "MDE" in methods:
        # MDE
        mde = pymde.preserve_distances(
            data=mde_graph,
            embedding_dim=rank,
            loss=pymde.losses.Quadratic,
            max_distances=1e8,
            verbose=False)
        mde_embedding = mde.embed()
        mde_dar = construct_node_embedding_graph(mde_embedding, adjacency_list)
        info['MDE dist'] = {'ratios' : subopt_ratios(mde_dar, Dist, sources, targets)}
        try:
            mde = pymde.preserve_neighbors(
                data=mde_graph,
                embedding_dim=rank,
                verbose=False)
            mde_embedding = mde.embed()
            mde_dar = construct_node_embedding_graph(mde_embedding, adjacency_list)
            info['MDE nbhd'] = {'ratios' : subopt_ratios(mde_dar, Dist, sources, targets)}
        except: pass

    if "LR" in methods:
        # LR
        B, C = low_rank_approx(Dist, dim=rank, symm=symm)
        lr_dar = construct_lr_graph(B, C, adjacency_list)
        info[r'LR $d(i,j)$'] = {'ratios': subopt_ratios(lr_dar, Dist, sources, targets)}

    if "LR_sq" in methods:
        # LR of $d(i,j)^2$
        B, C = low_rank_approx(np.square(Dist), dim=rank, symm=symm)
        nemb_dar = construct_lr_dist2_graph(B, C, adjacency_list)
        info[r'LR $d(i,j)^2$'] = {'ratios' : subopt_ratios(nemb_dar, Dist, sources, targets)}

    if "xy_pos" in methods:
        # Node embedding in $\mathcal{R}^2$
        if node_embeddings is not None:
            nemb_dar = construct_node_embedding_graph(node_embeddings, adjacency_list)
            info[r'$\mathbf{R}^d$ pos'] = {'ratios' : subopt_ratios(nemb_dar, Dist, sources, targets)}
    
    if "ODLR" in methods:
        # ODLR
        B, C = odlr_approx(Dist, rank, symm, T=100)
        odlr_dar = construct_odlr_graph(B, C, adjacency_list)
        info['ODLR'] = {'ratios' : subopt_ratios(odlr_dar, Dist, sources, targets)}

    return info


def record_subopts_partial(rank, Dist, pi_Dist, dist_row_i, adjacency_list, pi, sources, targets, 
                           percentage_of_nodes, symm=False, methods=[], bcd_iter=200, mds_iter=500,
                           full_iter=200, n_init_mds=3, n_init_cc=2):
    """
    Dist is symmetric
    """
    assert np.allclose(Dist, Dist.T)

    info = {}
    n = Dist.shape[0]

    if "LR" in methods:
        # LR
        B, C = low_rank_approx(Dist, dim=rank, symm=symm)
        lr_dar = construct_lr_graph(B, C, adjacency_list)
        info[r'LR $d(i,j)$'] = {'ratios': subopt_ratios(lr_dar, Dist, sources, targets)}

    if "CMDS" in methods:
        # Classical MDS
        X_cmds, _ = classical_mds((Dist + Dist.T)/2, rank)
        mds_dar = construct_node_embedding_graph(X_cmds, adjacency_list)
        info['C-MDS'] = {'ratios' : subopt_ratios(mds_dar, Dist, sources, targets)}

    if "MDS" in methods:
        X_mds, stress = mds_stress_majorization(rank, (Dist + Dist.T)/2, n_init=n_init_mds, max_iter=mds_iter, eps=1e-3, verbose=False)
        mds_dar = construct_node_embedding_graph(X_mds, adjacency_list)
        info['MDS'] = {'ratios' : subopt_ratios(mds_dar, Dist, sources, targets)}

    if "l2_%d_full"%percentage_of_nodes in methods:
        X_mds, stress = l2_full_convex_concave(rank, pi, pi_Dist, n_init=n_init_cc, max_iter=full_iter, eps=1e-6, verbose=False, freq=200)
        mds_dar = construct_node_embedding_graph(X_mds, adjacency_list)
        info["l2_%d_full"%percentage_of_nodes] = {'ratios' : subopt_ratios(mds_dar, Dist, sources, targets)}

    if "linf_%d_full"%percentage_of_nodes in methods:     
        X_mds, stress = linfty_full_convex_concave(rank, pi, Dist[pi], n_init=n_init_cc, max_iter=full_iter, eps=1e-6, verbose=False, freq=50)
        mds_dar = construct_node_embedding_graph(X_mds, adjacency_list, ord=np.inf)
        info["linf_%d_full"%percentage_of_nodes] = {'ratios' : subopt_ratios(mds_dar, Dist, sources, targets)}

    if "l2_%d"%percentage_of_nodes in methods:    
        ord = 2 
        X_l2, losses_l2 = bcd_convex_concave(ord, dist_row_i, n, rank, pi, n_init=n_init_cc, max_iter=bcd_iter, freq=500, Dist=None, printing=True)
        l2_dar = construct_node_embedding_graph(X_l2, adjacency_list)
        info['l2_%d'%percentage_of_nodes] = {'ratios' : subopt_ratios(l2_dar, Dist, sources, targets)}

    if "linf_%d"%percentage_of_nodes in methods:
        ord = np.inf
        X_linf, losses_linf = bcd_convex_concave(ord, dist_row_i, n, rank, pi, n_init=n_init_cc, max_iter=bcd_iter, freq=500, Dist=None, printing=True)
        linf_dar = construct_node_embedding_graph(X_linf, adjacency_list, ord=np.inf)
        info['linf_%d'%percentage_of_nodes] = {'ratios' : subopt_ratios(linf_dar, Dist, sources, targets)}

    return info


def recompile_jit_functions():
    global asymm_eulidean_dist_matrix, pi_asymm_eulidean_dist_matrix, symm_product_BZ, \
            row_BZ_update_inplace, product_BZ, symm_diag_Laplacian, symm_product_Lb, symm_cg_Laplacian_system,\
            diag_Laplacian, product_Lb, cg_Laplacian_system
    
    asymm_eulidean_dist_matrix = nb.njit()(asymm_eulidean_dist_matrix.py_func)
    pi_asymm_eulidean_dist_matrix = nb.njit()(pi_asymm_eulidean_dist_matrix.py_func)
    symm_product_BZ = nb.njit()(symm_product_BZ.py_func)
    row_BZ_update_inplace = nb.njit(parallel=True)(row_BZ_update_inplace.py_func)
    product_BZ = nb.njit()(product_BZ.py_func)
    symm_diag_Laplacian = nb.njit()(symm_diag_Laplacian.py_func)
    symm_product_Lb = nb.njit()(symm_product_Lb.py_func)
    symm_cg_Laplacian_system = nb.njit()(symm_cg_Laplacian_system.py_func)
    diag_Laplacian = nb.njit()(diag_Laplacian.py_func)
    product_Lb = nb.njit()(product_Lb.py_func)
    cg_Laplacian_system = nb.njit()(cg_Laplacian_system.py_func)


def record_suboptimality(rank, pi_rows, pi_cols, pi_rows_c, pi_cols_c, rDist, cDist, adjacency_list, sources, targets, 
                           Dist, cc_max_iter=1000, cc_eps=1e-6, n_init_cc=3, cg_eps=1e-8, cg_max_iter=100, 
                           symm=True, asymm=True, verbose=False, freq=500, return_emb=False):
    info = {}
    n = rDist.shape[1]
    
    if symm:
        print(f"\n\nSymmetric {rank/2=}")
        start_time = time.time()
        Z_symm, loss, losses = fast_cc(rank//2, pi_rows=pi_rows, pi_rows_c=pi_rows_c, rDist=rDist, symm=True, n_init=n_init_cc,
                                            max_iter=cc_max_iter, eps=cc_eps, verbose=verbose, freq=freq, cg_eps=cg_eps, cg_max_iter=cg_max_iter)
        
        print(f"time={(time.time() - start_time):.2f}, {losses[0]=:.2f}, {losses[-1]=:.2f}, {len(losses)=}")

        collected = gc.collect()
        print("Garbage collector: collected %d objects." % collected)

        print(r"$\|x_i-x_j\|_2$")
        l_dar = construct_node_embedding_graph(Z_symm, adjacency_list)
        r_rel_diff, c_rel_diff = relative_average_difference(n, Z_symm, pi_rows, pi_cols, rDist, cDist, symm=True)
        info[r"symm"] = {'ratios' : subopt_ratios(l_dar, Dist, sources, targets),
                         "r_rel_diff": r_rel_diff, 
                         "c_rel_diff": c_rel_diff}

        recompile_jit_functions()

    if asymm:
        if not symm:
            print(f"\nAsymmetric fit {rank/2=}")
            Z0 = None
        else:
            print(f"\nSplit -> asymmetric fit {rank/2=}")
            Z0 = np.concatenate([Z_symm, Z_symm], axis=0)
            del Z_symm
        start_time = time.time()
        Z, loss, losses = fast_cc(rank, pi_rows, pi_cols, pi_rows_c, pi_cols_c, rDist, cDist, Z0=Z0, n_init=n_init_cc,
                                        max_iter=cc_max_iter, eps=cc_eps, verbose=verbose, freq=freq, cg_eps=cg_eps, cg_max_iter=cg_max_iter)
        print(f"time={(time.time() - start_time):.2f}, {losses[0]=:.2f}, {losses[-1]=:.2f}, {len(losses)=}")

        collected = gc.collect()
        print("Garbage collector: collected %d objects." % collected)

        print(r"$\|x_i-y_j\|_2$")
        l_dar = construct_xy_node_embedding_graph(Z[:n], Z[n:], adjacency_list)
        r_rel_diff, c_rel_diff = relative_average_difference(n, Z, pi_rows, pi_cols, rDist, cDist)

        info[r"symm+asymm"] = {'ratios' : subopt_ratios(l_dar, Dist, sources, targets), 
                               "r_rel_diff": r_rel_diff, 
                                "c_rel_diff": c_rel_diff}
        # cache = nb.caching.NullCache()
        # cache.flush()
        recompile_jit_functions()
        collected = gc.collect()
        print("Garbage collector: collected %d objects." % collected)
    if return_emb:
        return info, Z 
    else:
        return info



def dar_fast_cc(rank, pi_rows, pi_cols, pi_rows_c, pi_cols_c, rDist, cDist, adjacency_list, sources, targets, 
                           Dist, cc_max_iter=1000, cc_eps=1e-6, n_init_cc=3, cg_eps=1e-8, cg_max_iter=100, 
                           verbose=False, freq=500):
    info = {}
    n = rDist.shape[1]
    

    print(f"\n\nSymmetric {rank/2=}")
    start_time = time.time()
    assert np.unique(np.concatenate([pi_rows, pi_rows_c ], axis=0)).size == n
    assert np.unique(np.concatenate([pi_cols, pi_cols_c ], axis=0)).size == n
    Z_symm, loss, losses = fast_cc(rank//2, pi_rows=pi_rows, pi_rows_c=pi_rows_c, rDist=rDist, symm=True, n_init=n_init_cc,
                                        max_iter=cc_max_iter, eps=cc_eps, verbose=verbose, freq=freq, cg_eps=cg_eps, cg_max_iter=cg_max_iter)
    
    print(f"time={(time.time() - start_time):.2f}, {losses[0]=:.2f}, {losses[-1]=:.2f}, {len(losses)=}")
    recompile_jit_functions()
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    print(f"\nSplit -> asymmetric fit {rank=}")
    Z0 = np.concatenate([Z_symm, Z_symm], axis=0)
    del Z_symm
    start_time = time.time()
    Z, loss, losses = fast_cc(rank, pi_rows, pi_cols, pi_rows_c, pi_cols_c, rDist, cDist, Z0=Z0, n_init=n_init_cc,
                                    max_iter=cc_max_iter, eps=cc_eps, verbose=verbose, freq=freq, cg_eps=cg_eps, cg_max_iter=cg_max_iter)
    print(f"time={(time.time() - start_time):.2f}, {losses[0]=:.2f}, {losses[-1]=:.2f}, {len(losses)=}")

    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    print(r"$\|x_i-y_j\|_2$")
    l_dar = construct_xy_node_embedding_graph(Z[:n], Z[n:], adjacency_list)
    info[r"symm+asymm"] = {'ratios' : subopt_ratios(l_dar, Dist, sources, targets)}
    recompile_jit_functions()
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return l_dar, info, Z


def low_dim_embeddings(n, rank, pi_rows, pi_cols, rDist, cDist, 
                       cc_max_iter=1000, cc_eps=1e-6, n_init_cc=3, cg_eps=1e-8, cg_max_iter=100, 
                        verbose=False, freq=500):
    pi_rows_c = np.delete(np.arange(n), pi_rows, axis=0)
    pi_cols_c = np.delete(np.arange(n), pi_cols, axis=0)

    # Symmetric embeddings
    Z_symm, loss, losses = fast_cc(rank//2, pi_rows=pi_rows, pi_rows_c=pi_rows_c, rDist=rDist, symm=True, n_init=n_init_cc,
                                        max_iter=cc_max_iter, eps=cc_eps, verbose=verbose, freq=freq, cg_eps=cg_eps, cg_max_iter=cg_max_iter)
    
    recompile_jit_functions()
    gc.collect()

    # Split -> asymmetric fit
    Z0 = np.concatenate([Z_symm, Z_symm], axis=0)
    del Z_symm
    Z, loss, losses = fast_cc(rank, pi_rows, pi_cols, pi_rows_c, pi_cols_c, rDist, cDist, Z0=Z0, n_init=n_init_cc,
                                    max_iter=cc_max_iter, eps=cc_eps, verbose=verbose, freq=freq, cg_eps=cg_eps, cg_max_iter=cg_max_iter)
    recompile_jit_functions()
    gc.collect()
    print(f"{losses[0]=:.2f}, {losses[-1]=:.2f}, {len(losses)=}")

    return Z
