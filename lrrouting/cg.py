import numpy as np
import numba as nb
from typing import Tuple, List

from lrrouting.utils import *
from memory_profiler import profile



@nb.njit()
def symm_diag_Laplacian(pi, pi_c, n):
    D = np.zeros(n)
    D[pi] = n-1
    D[pi_c] = pi.size
    return D


@nb.njit()
# @profile
def symm_product_Lb(b:"np.ndarray[np.float32]", pi:"np.ndarray[np.int32]", pi_c:"np.ndarray[np.int32]", 
                    n:int)->"np.ndarray[np.float32]":
    """
    Compute (Lb) with L is Laplacian matrix for undirected graph with unit edges connecting
    node i to node j for every (i,j) \in S

    b: n vector 
    """
    res = np.zeros(n)

    res[pi] = n * b[pi] - b.sum()
    res[pi_c] = pi.size * b[pi_c] - b[pi].sum()

    return res


@nb.njit()
# @profile
def symm_cg_Laplacian_system(b:"np.ndarray[np.float32]", D_inv:"np.ndarray[np.float32]", pi:"np.ndarray[np.int32]",
                              pi_c:"np.ndarray[np.int32]", n:int, eps=1e-10, max_iter=100, 
                              printing=False)->Tuple["np.ndarray[np.float32]", "np.ndarray[np.float32]"]:
    """
    Solve Laplacian system using diagonally preconditioned conjugate gradient
        Lx = b
    """ 
    x = np.zeros(n)
    r = b
    z = D_inv * r
    rho0 = np.dot(r, z)
    b_norm = np.linalg.norm(b, ord=2)
    if is_close_scalar(b_norm, 0):
        x = np.ones(n) / np.sqrt(n)
        return x, np.zeros(1)
    p = z + 0
    losses = np.zeros(max_iter)
    Lp = np.zeros(n)
    for k in range(max_iter):
        Lp *= 0
        Lp += symm_product_Lb(p, pi, pi_c, n) 
        Lp += p.sum() / n
        if is_close_scalar(np.linalg.norm(Lp, ord=2), 0):
            x -= x.sum() / n
            return x, losses[:k]
        alpha = np.dot(r, z) / np.dot(p, Lp)
        x += alpha * p 
        r -= alpha * Lp
        z *= 0
        z += D_inv * r
        rho1 = np.dot(r, z)
        p *= (rho1/rho0)
        p += z
        # p = (rho1/rho0) * p + z
        losses[k] = np.sqrt(np.dot(r, r)) / b_norm
        if printing: 
            print(k, losses[k], rho0, rho1)
        if losses[k] < eps: break
        rho0 = rho1
    x -= x.sum() / n
    return x, losses[:k+1]


@nb.njit()
def diag_Laplacian(pi_rows:"np.ndarray[np.int32]", pi_cols:"np.ndarray[np.int32]", pi_rows_c:"np.ndarray[np.int32]",
                    pi_cols_c:"np.ndarray[np.int32]", n:int)->"np.ndarray[np.float32]":
    D = np.zeros(2*n)
    D[pi_rows] = n
    D[n + pi_cols] = n
    if pi_rows_c.size != 0:
        D[pi_rows_c] = pi_cols.size
    if pi_cols_c.size != 0:
        D[n + pi_cols_c] = pi_rows.size
    return D


@nb.njit()
def product_Lb(b:"np.ndarray[np.float32]", pi_rows:"np.ndarray[np.int32]", pi_cols:"np.ndarray[np.int32]", 
               pi_rows_c:"np.ndarray[np.int32]", pi_cols_c:"np.ndarray[np.int32]", n:int)->"np.ndarray[np.int32]":
    """
    Compute (Lb) with L is Laplacian matrix for undirected bipartite graph with unit edges connecting
    node i to node (n+j) for every (i,j) \in S

    b: 2*n vector 
    """
    b1, b2 = b[:n], b[n:]
    res = np.zeros(2*n)

    res[pi_rows] = n * b1[pi_rows] - b2.sum()
    res[n + pi_cols] = n * b2[pi_cols] - b1.sum()
    if pi_rows_c.size != 0:
        res[pi_rows_c] = pi_cols.size * b1[pi_rows_c] - b2[pi_cols].sum()
    if pi_cols_c.size != 0:
        res[n + pi_cols_c] = pi_rows.size * b2[pi_cols_c] -b1[pi_rows].sum()

    return res


@nb.njit()
def cg_Laplacian_system(b:"np.ndarray[np.float32]", D_inv:"np.ndarray[np.float32]", pi_rows:"np.ndarray[np.int32]", 
                        pi_cols:"np.ndarray[np.int32]", pi_rows_c:"np.ndarray[np.int32]", 
                        pi_cols_c:"np.ndarray[np.int32]", n:int, eps=1e-10, max_iter=100, 
                        printing=False)->Tuple["np.ndarray[np.float32]", "np.ndarray[np.float32]"]:
    """
    Solve Laplacian system using diagonally preconditioned conjugate gradient
        Lx = b
    """ 
    x = np.zeros(2*n)
    r = b
    z = D_inv * r
    rho0 = np.dot(r, z)
    b_norm = np.linalg.norm(b, ord=2)
    if is_close_scalar(b_norm, 0):
        x = np.ones(2*n) / np.sqrt(n)
        return x, np.zeros(1)
    p = z + 0
    losses = np.zeros(max_iter)
    Lp = np.zeros(2*n)
    for k in range(max_iter):
        Lp *= 0
        Lp += product_Lb(p, pi_rows, pi_cols, pi_rows_c, pi_cols_c, n)
        Lp += p.sum() / (2*n)
        if is_close_scalar(np.linalg.norm(Lp, ord=2), 0):
            x -= x.sum() / (2*n)
            return x, losses[:k]
        alpha = np.dot(r, z) / np.dot(p, Lp)
        x += alpha * p 
        r -= alpha * Lp
        z *= 0
        z += D_inv * r
        rho1 = np.dot(r, z)
        p *= (rho1/rho0)
        p += z
        # p = (rho1/rho0) * p + z
        losses[k] = np.sqrt(np.dot(r, r)) / b_norm
        if printing: 
            print(k, losses[k], rho0, rho1)
        if losses[k] < eps: break
        rho0 = rho1
    x -= x.sum() / (2*n)
    return x, losses[:k+1]


def cg_Ax_b(A, b, diag_precond=True, D=None, eps=1e-10, max_iter=100, printing=False):
    if diag_precond:
        if D is None: D = np.diag(A)
    else:
        D = np.ones(b.shape[0])
    D_inv = 1 / D
    x = np.zeros(b.shape[0])
    r = b 
    z = D_inv * r
    rho0 = np.dot(r, z)
    b_norm = np.linalg.norm(b, ord=2)
    p = z + 0
    losses = [np.sqrt(np.dot(r, r)) / b_norm]
    for k in range(max_iter):
        Ap = np.dot(A, p)
        if is_close_scalar(np.linalg.norm(Ap, ord=2), 0):
            return x, losses
        alpha = np.dot(r, z) / np.dot(p, Ap)
        x += alpha * p 
        r -= alpha * Ap
        z = D_inv * r
        rho1 = np.dot(r, z)
        p = (rho1/rho0) * p + z
        losses += [np.sqrt(np.dot(r, r)) / b_norm]
        if printing: print(k, losses[-1], np.linalg.norm(A @ x - b, ord=2), rho0, rho1)
        if losses[-1] < eps: break
        rho0 = rho1
    return x, losses


def laplacian_from_row_col_pi(pi_rows, pi_cols, n, debug=False):
    # create Laplacian matrix from the row and col indices in pi
    # we assume the nodes in pi_rows are connected to everyone
    # and nodes in pi_cols are connected to everyone
    A = np.zeros((2*n, 2*n))
    A[pi_rows, n:] = -1
    A[n+pi_cols, :n] = -1
    A[n:, pi_rows] = -1
    A[:n, n+pi_cols] = -1
    L = A 
    L[np.arange(2*n), np.arange(2*n)] += -L.sum(axis=1)
    # L_inv = np.linalg.pinv(L, hermitian=True)
    if debug:
        assert np.allclose(np.linalg.norm(L.sum(axis=0)), 0) and np.allclose(np.linalg.norm(L.sum(axis=1)), 0)
        assert ((np.linalg.eigvalsh(L) >= -1e-10).all()), print("L is not a PSD matrix")
    return L


def laplacian_from_pi(pi, n, debug=False):
    # Laplacian for distance matrix coming from undirected graph G
    # create Laplacian matrix from the node indices in pi
    # we assume the nodes in pi are connected to everyone else
    A = np.zeros((n, n))
    A[pi] = -1
    A[:, pi] = -1
    L = A 
    L[np.arange(n), np.arange(n)] += -L.sum(axis=1)
    if debug:
        assert np.allclose(np.linalg.norm(L.sum(axis=0)), 0) and np.allclose(np.linalg.norm(L.sum(axis=1)), 0)
        assert ((np.linalg.eigvalsh(L) >= -1e-10).all()), print("L is not a PSD matrix")
    return L
