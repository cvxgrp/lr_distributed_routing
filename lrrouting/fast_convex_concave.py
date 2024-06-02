import numpy as np
import types 
from typing import Tuple
from scipy import sparse
import numba as nb

import tracemalloc

from lrrouting.utils import *
from lrrouting.cg import *
import objgraph



def sparse_sampled_matrix(pi, Dist, rows=True):
    n = Dist.shape[0]
    # create sparse matrix with rows sampled from a distance matrix
    row_indices = np.repeat(pi, n)
    col_indices = np.tile(np.arange(n), pi.size)
    if rows:
        values = Dist[pi].flatten('C')
    else:
        values = Dist[:, pi].T.flatten('C')
    return sparse.csr_matrix((values, (row_indices, col_indices)), shape=(n, n))


@nb.njit()
def asymm_eulidean_dist_matrix(X:"np.ndarray[np.float32]", Y:"np.ndarray[np.float32]")->"np.ndarray[np.float32]":
    """
    X: n1 x p/2;   Y: n2 x p/2 
        embedding matrices of n1 and n2 items
    """
    Gram = X @ Y.T
    d_X = np.multiply(X, X).sum(axis=1).reshape(-1, 1)
    d_Y = np.multiply(Y, Y).sum(axis=1).reshape(-1, 1)
    return np.sqrt(np.maximum(d_Y.T - 2 * Gram + d_X, 0))


@nb.njit()
def pi_asymm_eulidean_dist_matrix(X:"np.ndarray[np.float32]", Y:"np.ndarray[np.float32]", pi_rows:"np.ndarray[np.int32]", pi_cols:"np.ndarray[np.int32]")->Tuple["np.ndarray[np.float32]", "np.ndarray[np.float32]"]:
    """
    X, Y: n x p/2 
        left and right embedding matrices of n items, only for rows in pi_rows and 
        columns in pi_columns
    """
    Dist_r = asymm_eulidean_dist_matrix(X[pi_rows], Y)
    Dist_c = asymm_eulidean_dist_matrix(X, Y[pi_cols])
    return Dist_r, Dist_c.T


def symm_B_matrix(X, n, rDist, pi, dists_ij=None):
    if dists_ij is None:
        dists_ij, _ = eulidean_dist_matrix(X)
    reciprocal_dists_ij = np.zeros((n, n))
    mask = dists_ij > 1e-10
    reciprocal_dists_ij[mask] = np.divide(1, dists_ij[mask])
    B = np.zeros((n, n))
    B[pi] = -np.multiply(rDist, reciprocal_dists_ij[pi]) 
    B[:, pi] = B[pi].T
    B[np.arange(n), np.arange(n)] += -np.array(B.sum(axis=1)).flatten()
    return B


# @profile
@nb.njit()
def symm_product_BZ(X:"np.ndarray[np.float32]", d_X:"np.ndarray[np.float32]", pi:"np.ndarray[np.int32]", pi_c:"np.ndarray[np.int32]", rDist:"np.ndarray[np.float32]")->"np.ndarray[np.float32]":
    """
    Compute (B^k Z^k) with B^k Laplacian matrix for undirected graph with edges connecting
    node i to node j for every (i,j) \in S, with weights w_ij = D[i, j] / \|x_i^k -y_j^k\|_2

    X: n x p 
        embedding matrices of n items
    """
    res = np.zeros(X.shape)
    dists_ij = np.zeros_like(d_X)

    for t in np.arange(pi.size):
        i = pi[t]
        dists_ij *= 0
        dists_ij += np.sqrt(np.maximum(d_X[i] - 2 * np.dot(X[i], X.T) + d_X.T, 0))
        row_BZ_update_inplace(res[i, :], rDist[t], dists_ij, X, X[i])

    dists_ij = np.zeros(pi.size)
    for t in np.arange(pi_c.size):
        i = pi_c[t]
        dists_ij *= 0
        dists_ij += np.sqrt(np.maximum(d_X[i] - 2 * X[i] @ X[pi].T + d_X[pi].T, 0)) 
        row_BZ_update_inplace(res[i, :], rDist[:, i].T , dists_ij, X[pi], X[i])

    return res


@nb.njit(parallel=True)
# @profile
def row_BZ_update_inplace(res:"np.ndarray[np.float32]", Dists_i:"np.ndarray[np.float32]", dists_i:"np.ndarray[np.float32]", 
                         X:"np.ndarray[np.float32]", X_i:"np.ndarray[np.float32]"):
    """
    Return 
       (Dists_i / dists_i).sum() * X_i - (Dists_i / dists_i) @ X, handles division by 0
    """
    val = 0
    res_sum = 0
    for t in nb.prange(X.shape[0]):
        if dists_i[t] > 1e-10:
            val = Dists_i[t] / dists_i[t]
            res -= val * X[t]
            res_sum += val 
    res += res_sum * X_i


@nb.njit()
def product_BZ(X:"np.ndarray[np.float32]", Y:"np.ndarray[np.float32]", d_X:"np.ndarray[np.float32]", d_Y:"np.ndarray[np.float32]", pi_rows:"np.ndarray[np.int32]", pi_cols:"np.ndarray[np.int32]", pi_rows_c:"np.ndarray[np.int32]", pi_cols_c:"np.ndarray[np.int32]", rDist:"np.ndarray[np.float32]", cDist:"np.ndarray[np.float32]")->"np.ndarray[np.float32]":
    """
    Compute (B^k Z^k) with B^k Laplacian matrix for undirected bipartite graph with edges connecting
    node i to node (n+j) for every (i,j) \in S, with weights w_ij = D[i, j] / \|x_i^k -y_j^k\|_2
    B^k is a function of Z^k

    X, Y: n x p/2 
        left and right embedding matrices of n items 
    """ 
    n, half_p = X.shape
    res = np.zeros((2*n, half_p), dtype=np.double)

    dists_ij = np.zeros_like(d_X)
    for t in np.arange(pi_rows.size):
        i = pi_rows[t]
        dists_ij *= 0
        dists_ij += np.sqrt(np.maximum(d_X[i] - 2 * np.dot(X[i], Y.T) + d_Y.T, 0.))  
        row_BZ_update_inplace(res[i, :], rDist[t], dists_ij, Y, X[i])

    for t in np.arange(pi_cols.size):
        j = pi_cols[t]
        dists_ij *= 0
        dists_ij += np.sqrt(np.maximum(d_Y[j] - 2 * Y[j] @ X.T + d_X.T, 0.))
        row_BZ_update_inplace(res[n+j, :], cDist[t], dists_ij, X, Y[j])

    dists_ij = np.zeros(pi_cols.size)
    for t in np.arange(pi_rows_c.size):
        i = pi_rows_c[t]
        dists_ij *= 0
        dists_ij += np.sqrt(np.maximum(d_X[i] - 2 * X[i] @ Y[pi_cols].T + d_Y[pi_cols].T, 0.))
        row_BZ_update_inplace(res[i, :], cDist[:, i].T, dists_ij, Y[pi_cols], X[i])

    dists_ij = np.zeros(pi_rows.size)
    for t in np.arange(pi_cols_c.size):
        j = pi_cols_c[t]
        dists_ij *= 0
        dists_ij += np.sqrt(np.maximum(d_Y[j] - 2 * Y[j] @ X[pi_rows].T + d_X[pi_rows].T, 0.))
        row_BZ_update_inplace(res[n+j, :], rDist[:, j].T, dists_ij, X[pi_rows], Y[j])
    
    return res


def B_matrix(Z0, n, rDist, cDist, pi_rows, pi_cols, dists_ij=None):
    X, Y = Z0[:n], Z0[n:]
    if dists_ij is None:
        dists_ij = asymm_eulidean_dist_matrix(X, Y)
    reciprocal_dists_ij = np.zeros((n, n))
    mask = dists_ij > 1e-10
    reciprocal_dists_ij[mask] = np.divide(1, dists_ij[mask])
    del dists_ij, mask
    M = np.zeros((n, n))
    M[pi_rows] = -np.multiply(rDist, reciprocal_dists_ij[pi_rows]) 
    M[:, pi_cols] = -np.multiply(cDist.T, reciprocal_dists_ij[:, pi_cols]) 
    B = np.block([[np.zeros((n, n)), M], [M.T, np.zeros((n, n))]])
    B[np.arange(2*n), np.arange(2*n)] += -np.array(B.sum(axis=1)).flatten()
    return B


def form_B_compute_BZ(Z0, n, rDist, cDist, pi_rows, pi_cols, dists_ij=None):
    """
    Compute product B(Z) Z, where B is a function of Z
    """
    X, Y = Z0[:n], Z0[n:]
    if dists_ij is None:
        dists_ij = asymm_eulidean_dist_matrix(X, Y)
    reciprocal_dists_ij = np.zeros((n, n))
    mask = dists_ij > 1e-10
    reciprocal_dists_ij[mask] = np.divide(1, dists_ij[mask])
    del dists_ij, mask
    M = np.zeros((n, n))
    M[pi_rows] = -np.multiply(rDist, reciprocal_dists_ij[pi_rows]) 
    M[:, pi_cols] = -np.multiply(cDist.T, reciprocal_dists_ij[:, pi_cols]) 
    D1 = -M.sum(axis=1)
    D2 = -M.sum(axis=0)
    res1 = D1[:, np.newaxis] * X + np.dot(M, Y)
    res2 = D2[:, np.newaxis] * Y + np.dot(M.T, X)
    del M, D1, D2
    return np.concatenate([res1, res2], axis=0)


def symm_fast_cc_single_iteration(X, D_inv, rDist, pi, pi_c, eps=1e-8, max_iter=200, printing=False):
    """
    Modify X in place
    """
    n = X.shape[0]
    diff_dists_r = np.square(rDist - asymm_eulidean_dist_matrix(X[pi], X))
    raw_stress = (diff_dists_r.sum()) / (pi.size * (n-1) / 2)
    del diff_dists_r
    d_X = np.multiply(X, X).sum(axis=1)
    BX0 = symm_product_BZ(X, d_X, pi, pi_c, rDist)
    del d_X
    X *= 0
    for i in range(X.shape[1]):
        X[:, i], _ = symm_cg_Laplacian_system(BX0[:, i], D_inv, pi, pi_c, n, 
                                              eps=eps, max_iter=max_iter, printing=printing)
    del BX0
    return raw_stress


def fast_cc_single_iteration(Z, D_inv, rDist, cDist, pi_rows, pi_cols, pi_rows_c, 
                             pi_cols_c, eps=1e-8, max_iter=200):
    """
    Modify Z in place
    """
    n = Z.shape[0] // 2
    dists_r, dists_c = pi_asymm_eulidean_dist_matrix(Z[:n], Z[n:], pi_rows, pi_cols)
    raw_stress = (np.square(rDist - dists_r).sum() \
                  + np.square(cDist - dists_c).sum()\
                  - np.square(rDist[:, pi_cols] \
                         - dists_r[: , pi_cols]).sum()) \
                    / ((pi_rows.size + pi_cols.size) * (n-1) / 2)
    del dists_r, dists_c
    d_X = np.multiply(Z[:n], Z[:n]).sum(axis=1)
    d_Y = np.multiply(Z[n:], Z[n:]).sum(axis=1)
    BZ0 = product_BZ(Z[:n], Z[n:], d_X, d_Y, pi_rows, pi_cols, pi_rows_c, pi_cols_c, rDist, cDist) 
    del d_X, d_Y
    Z *= 0
    for i in range(Z.shape[1]):
        Z[:, i], _ = cg_Laplacian_system(BZ0[:, i], D_inv, pi_rows, pi_cols, pi_rows_c, pi_cols_c, n, 
                                              eps=eps, max_iter=max_iter, printing=False) 
    del BZ0
    return raw_stress


def fast_cc(rank, pi_rows=None, pi_cols=None, pi_rows_c=None, pi_cols_c=None, rDist=None, 
            cDist=None, symm=False, n_init=2, init=False, Z0=None, max_iter=500, eps=1e-3, verbose=False, 
            freq=50, cg_max_iter=100, cg_eps=1e-7, min_iter=1):
    best_emb = (None, np.inf)
    n = rDist.shape[1]
    # tracemalloc.start()
    if symm:
        D = symm_diag_Laplacian(pi_rows, pi_rows_c, n)
    else:
        D = diag_Laplacian(pi_rows, pi_cols, pi_rows_c, pi_cols_c, n)
    assert (D>0).all(), print(f"sample rows and columns of Dist to guarantee diag(L)>0")
    D_inv = 1 / D 
    del D
    if not Z0 is None: 
        n_init = 1 # single given initialization
        init = True
    for _ in range(n_init):
        if not init and symm:
            Z0 = np.random.randn(n, rank)
        elif not init:
            Z0 = np.random.randn(2*n, rank // 2)
        prev_loss = np.inf
        losses = np.zeros(max_iter)
        for t in range(max_iter):
            # if t % 100 == 0:
            #     snapshot1 = tracemalloc.take_snapshot()
            # print(f"{t=}")
            if symm:
                loss = symm_fast_cc_single_iteration(Z0, D_inv, rDist, pi_rows, pi_rows_c, 
                                                        max_iter=cg_max_iter, eps=cg_eps)
            else:
                loss = fast_cc_single_iteration(Z0, D_inv, rDist, cDist, pi_rows, pi_cols, pi_rows_c, 
                                                   pi_cols_c, max_iter=cg_max_iter, eps=cg_eps)
            if t >= min_iter and np.abs((prev_loss - loss) / prev_loss) < eps:
                if verbose: print(t)
                break
            # if t % 50 == 0:
            #     snapshot2 = tracemalloc.take_snapshot()
            #     top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            #     print("\n*** top 15 stats ***")
            #     for stat in top_stats[:10]:
            #         print(stat)
            losses[t] = loss
            prev_loss = loss
            if verbose and t % freq==0: print(f"{t=}, {loss=}")
        if verbose: print(f"{loss=}, {np.diff(np.array(losses)).max()=}")
        if best_emb[1] > loss:
            best_emb = (Z0+0, loss, losses[:t+1])
    assert not np.isnan(Z0).any()
    return best_emb



def slow_cc_single_iteration(Z0, L, N_inv, rDist, cDist, pi_rows, pi_cols, symm=False, debug=False):
    if symm:
        n = Z0.shape[0]
        dists_ij, _ = eulidean_dist_matrix(Z0)
        raw_stress = (np.square(rDist - dists_ij[pi_rows]).sum()) \
                        / (pi_rows.size * (n-1) / 2)
        B = symm_B_matrix(Z0, n, rDist, pi_rows, dists_ij=dists_ij)
        BZ0 = B @ Z0
    else:
        n = Z0.shape[0] // 2
        dists_ij = asymm_eulidean_dist_matrix(Z0[:n], Z0[n:])
        raw_stress = (np.square(rDist - dists_ij[pi_rows]).sum() \
                    + np.square(cDist.T - dists_ij[:, pi_cols]).sum()\
                    - np.square(rDist[:, pi_cols] \
                            - dists_ij[pi_rows][:, pi_cols]).sum()) \
                        / ((pi_rows.size + pi_cols.size) * (n-1) / 2)
        BZ0 = form_B_compute_BZ(Z0, n, rDist, cDist, pi_rows, pi_cols, dists_ij=dists_ij)
    if debug:
        Z1 = np.linalg.lstsq(L, BZ0, rcond=1e-7)[0] # more precise but slower
    else:
        D_minsqrt = np.power(np.diag(L), -0.5)
        Z1 = D_minsqrt[:, np.newaxis] * (N_inv @ (D_minsqrt[:, np.newaxis] * BZ0))
    return Z1, raw_stress


def slow_cc(rank, pi_rows=None, pi_cols=None, rDist=None, 
            cDist=None, symm=False, n_init=2, Z0=None, max_iter=500, eps=1e-3, verbose=False, freq=50, debug=False):
    best_emb = (None, np.inf)
    n = rDist.shape[1]
    if symm:
        L = laplacian_from_pi(pi_rows, n, debug=debug)
    else:
        L = laplacian_from_row_col_pi(pi_rows, pi_cols, n, debug=debug)
    D_minsqrt = np.power(np.diag(L), -0.5)
    N = (D_minsqrt[:, np.newaxis] * L) * D_minsqrt[np.newaxis, :]
    N_inv = np.linalg.pinv(N, hermitian=True)
    if not Z0 is None: 
        n_init = 1
        Z = Z0 + 0
    for _ in range(n_init):
        if Z0 is None and symm:
            Z = np.random.randn(n, rank)
        elif Z0 is None:
            Z = np.random.randn(2*n, rank // 2)
        prev_loss = np.inf
        losses = []
        for t in range(max_iter):
            Z, loss = slow_cc_single_iteration(Z, L, N_inv, rDist, cDist, 
                                               pi_rows, pi_cols, symm=symm, debug=debug)
            if t >= 1 and np.abs((prev_loss - loss) / prev_loss) < eps:
                if verbose: print(t)
                break
            if debug:
                assert (prev_loss + 1e-5 >= loss), print(t, prev_loss + 1e-6 - loss, "use Z1 = np.linalg.lstsq(L, BZ0)[0]")
            prev_loss = loss
            losses += [loss]
            if verbose and t % freq==0: print(f"{t=}, {loss=}")
        if verbose: print(f"{loss=}")
        if best_emb[1] > loss:
            best_emb = (Z, loss, losses)
    return best_emb
