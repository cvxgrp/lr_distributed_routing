import numpy as np
import cvxpy as cp
import osqp
from scipy import sparse

from lrrouting.utils import *


def norm_subgradient(a, ord=np.inf, debug=False):
    if ord in [np.inf, 'inf']:
        ord = np.inf
        max_idx = np.argmax(np.abs(a))
        g = np.zeros(a.shape)
        g[max_idx] = np.sign(a[max_idx])
    elif ord == 2:
        a_norm = np.linalg.norm(a, ord=ord)
        g = a
        if a_norm > 1e-8:
            g /= a_norm
    if debug:
        assert np.allclose(np.linalg.norm(g, ord=1/(1-1/ord)), 1)
        assert np.allclose(np.linalg.norm(a, ord=ord), g.T @ (a)), print(np.linalg.norm(a, ord=ord), g.T @ (a))
    return g


def norm_all_subgradients(X, ord):
    n, p = X.shape
    G = np.zeros((n, n, p))
    for i in range(n):
        for j in range(i+1, n):
            G[i, j] = norm_subgradient(X[i] - X[j], ord=ord)
            G[j, i] = -G[i, j]
    return G


def compute_lq_norm_matrix(X, ord):
    n, p = X.shape
    diff = X.reshape(n, 1, p) - X.reshape(1, n, p)
    lp_norms = np.linalg.norm(diff, ord=ord, axis=-1)
    return lp_norms


def convex_concave_forloop(Dist_i, i, X, ord, solver=cp.CLARABEL, verbose=False, g_i=None):
    n, p = X.shape
    x_i = cp.Variable(p)
    obj = 0
    for j in range(n):
        if i == j: continue
        x_j = X[j]
        if g_i is None:
            g_ij = norm_subgradient(X[i] - X[j], ord=ord)
        else:
            g_ij = g_i[j]
        obj += cp.square(cp.norm(x_i - x_j, ord)) - 2 * Dist_i[j] * g_ij @ (x_i - x_j) 
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=solver, verbose=verbose)
    return x_i.value, obj.value + np.square(Dist_i).sum()


def convex_concave(Dist_i, i, X, ord, solver=cp.CLARABEL, verbose=False, g_i=None):
    # g_i: n x p
    n, p = X.shape
    x_i = cp.Variable((1, p))
    weights = 2 * (Dist_i[:, np.newaxis] * g_i)
    obj = 0
    if i > 0:
        obj += cp.sum_squares(cp.norm(x_i - X[:i], p=ord, axis=1)) - cp.sum(cp.multiply(weights[:i], x_i - X[:i])) 
    if i < n-1: 
        obj += cp.sum_squares(cp.norm(x_i - X[i+1:], p=ord, axis=1)) - cp.sum(cp.multiply(weights[i+1:], x_i - X[i+1:]))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=solver, verbose=verbose)
    return x_i.value.reshape(-1), obj.value + np.square(Dist_i).sum()


def bcd_convex_concave(ord, dist_row_i, n, rank, pi, n_init=1, max_iter=200, freq=50, Dist=None, 
                       printing=False, solver=cp.CLARABEL, verbose=False):
    norm_ord = lambda a: norm_subgradient(a, ord=ord)
    best_emb = (None, np.inf)
    for _ in range(n_init):
        X_emb = np.random.randn(n, rank)
        losses = []
        if Dist is not None: losses = [obj_distortion_function(Dist, X_emb, ord)]  
        for t in range(max_iter):
            loss = 0
            for i in pi:
                Dist_i = dist_row_i(i)
                g_i = np.apply_along_axis(norm_ord, 1, np.repeat(X_emb[i:i+1], n, axis=0) - X_emb)
                if ord == 2:
                    x_i, hat_ell_i = bcd_l2_single(n, rank, i, Dist_i, X_emb, g_i)
                else:
                    x_i, hat_ell_i = convex_concave(Dist_i, i, X_emb, ord, solver=solver, g_i=g_i)
                X_emb[i] = x_i
                loss += hat_ell_i
            if Dist is not None:
                losses += [obj_distortion_function(Dist, X_emb, ord)]
            else:
                losses += [loss / (n * (n-1) / 2) ] 
            if (t % freq == 0 or t == max_iter-1) and printing:
                    print(f"{t=}, distortion={losses[-1]}") 
        mean_loss = sum(losses[-20:])/20
        if verbose: print(f"{mean_loss=}")
        if best_emb[1] > mean_loss:
            best_emb = (X_emb, mean_loss)  
    return best_emb[0], best_emb[1]


def obj_distortion_function(Dist, X, ord, pi=None):
    n = X.shape[0]
    if pi is None: pi = np.arange(n)
    # average distortion
    hat_D = compute_lq_norm_matrix(X, ord)
    return np.square(Dist - hat_D[pi]).sum() / (pi.size * (n-1) / 2)


def simple_laplacian_from_pi(pi, n, debug=False):
    # Laplacian for distance matrix coming from undirected graph G
    # create Laplacian matrix from the node indices in pi
    # we assume the nodes in pi are connected to everyone else
    A = np.zeros((n, n))
    A[pi] = -1
    A[:, pi] = -1
    L = A 
    L[np.arange(n), np.arange(n)] += -L.sum(axis=1)
    L_inv = np.linalg.pinv(L, hermitian=True)
    if debug:
        assert np.allclose(np.linalg.norm(L.sum(axis=0)), 0) and np.allclose(np.linalg.norm(L.sum(axis=1)), 0)
        assert ((np.linalg.eigvalsh(L) >= -1e-10).all()), print("L is not a PSD matrix")
    return L, L_inv


def l2_full_convex_concave(rank, pi, pi_Dist, n_init=2, max_iter=500, eps=1e-3, verbose=False, freq=50, debug=False):
    best_emb = (None, np.inf)
    n = pi_Dist.shape[0]
    L, L_inv = simple_laplacian_from_pi(pi, n, debug=debug)
    pi_Disth =  pi_Dist.transpose()
    for _ in range(n_init):
        X = np.random.randn(n, rank)
        prev_loss = np.inf
        for t in range(max_iter):
            X, loss = l2_full_single(X, L, L_inv, pi_Dist, pi_Disth, pi, debug=debug)
            if t >= 1 and np.abs((prev_loss - loss) / prev_loss) < eps:
                if verbose: print(t)
                break
            if debug:
                assert (prev_loss + 1e-7 * np.abs(prev_loss) >= loss), print(t, prev_loss + 1e-7 * np.abs(prev_loss) - loss)
            prev_loss = loss
            if verbose and t % freq==0: print(f"{t=}, {loss=}")
        if verbose: print(f"{loss=}")
        if best_emb[1] > loss:
            best_emb = (X, loss)
    return best_emb


def l2_full_single(X0, L, L_inv, pi_Dist, pi_Disth, pi, debug=False):
    n = X0.shape[0]
    dists_ij, _ = eulidean_dist_matrix(X0)
    raw_stress = np.square(pi_Dist[pi] - dists_ij[pi]).sum() / (pi.size * (n-1) / 2)
    # assert (np.allclose(raw_stress, ldr.obj_distortion_function(Dist[pi], X0, ord, pi)))
    reciprocal_dists_ij = np.zeros((n, n))
    mask = dists_ij > 1e-10
    reciprocal_dists_ij[mask] = np.divide(1, dists_ij[mask])
    M = -(pi_Dist.multiply(reciprocal_dists_ij) + pi_Disth.multiply(reciprocal_dists_ij))
    B = M.toarray()
    B[np.ix_(pi, pi)] /= 2
    B[np.arange(n), np.arange(n)] += -np.array(B.sum(axis=1)).flatten()
    if debug:
        assert np.allclose(np.linalg.norm(B.sum(axis=0)), 0) and np.allclose(np.linalg.norm(B.sum(axis=1)), 0)
        assert ((np.linalg.eigvalsh(B) >= -1e-10).all()), print("B is not a PSD matrix")
    X1 = L_inv @ (B @ X0)
    # X1 = np.linalg.lstsq(L, B @ X0)[0] # more precise but slower
    # upper_bound = (np.square(pi_Dist[pi].toarray()).sum() - 2 * np.trace(X1.T @ B @ X0) + np.trace(X1.T @ L @ X1)) / (pi.size * (n-1) / 2)
    return X1, raw_stress


def bcd_l2_single(n, rank, i, Dist_i, X_emb, g_i):
    # X: n x p;  g_i: n x p
    # P = (n-1) * np.eye(rank)
    # minimize_{x_i} \sum_{j != i} (-2D_ij (g_ij^k)^T(x_i - x_j^k) + \|x_i - x_j^T\|_2^2)
    q = - (X_emb.sum(axis=0) - X_emb[i] + (Dist_i[:, np.newaxis] * g_i).sum(axis=0))
    x_i = - (1./(n-1)) * q 
    obj_i = (n-1) * np.square(x_i).sum() + 2 * q.T @ x_i + np.square(Dist_i).sum() \
                        + 2 * np.trace((Dist_i[:, np.newaxis] * g_i) @ X_emb.T) + np.square(X_emb).sum() - np.square(X_emb[i]).sum()
    return x_i, obj_i


def matrices_bcd_linf_single(n, rank, format='lil'):
    P = sparse.block_diag([sparse.eye(n), sparse.csc_matrix((rank, rank))], format='csc')
    M = []
    for j in range(n):
        ones_e_j = np.zeros((rank, n))
        ones_e_j[:, j] = 1
        block = np.block([[-ones_e_j, np.eye(rank)],
                            [ones_e_j, np.eye(rank)]],)
        M += [block]
    M = np.concatenate(M, axis=0)
    M = sparse.csc_matrix(M)
    return P, M


def bcd_linf_single(n, rank, i, Dist_i, X_emb, g_i, M0, eps_rel=1e-3, eps_abs=1e-3, max_iter=4000):
    # variable: y = [s_i, x_i] \in R^{n + p}
    # X: n x p;  g_i: n x p
    diag = np.ones(n)
    diag[i] = 0
    diag_no_i = sparse.diags(diag, 0)
    P = sparse.block_diag([diag_no_i, sparse.csc_matrix((rank, rank))], format='csc')
    u = np.concatenate([X_emb, np.full((n,rank), np.inf)], axis=1).flatten(order='C')
    l = np.concatenate([np.full((n,rank), -np.inf), X_emb], axis=1).flatten(order='C')
    # same as zeroing out entries of M corresponding to x_i bounds
    # -\infty <= x_i - s_ii <= \infy
    u[2*rank*i : 2*rank*(i+1)] = np.inf
    l[2*rank*i : 2*rank*(i+1)] = -np.inf
    q = np.concatenate([np.zeros(n), -(Dist_i[:, np.newaxis] * g_i).sum(axis=0)], axis=0)
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=M0, l=l, u=u, verbose=False, eps_rel=eps_rel, eps_abs=eps_abs, max_iter=max_iter)
    res = prob.solve()

    x_i = res.x[-rank:]
    y = res.x
    obj_i = (y.T @ P) @ y + 2 * q.T @ y + np.square(Dist_i).sum() \
            + 2 * np.trace((Dist_i[:, np.newaxis] * g_i) @ X_emb.T)  
    return x_i, obj_i


def convex_concave_full_pi(pi_Dist_compressed, pi, ord, solver=cp.CLARABEL, verbose=False, g_ij=None):
    # X0: n x p; g_ij: g_i: n x n x p; pi_Dist: |pi| x n
    n, p = g_ij.shape[1:]
    X = cp.Variable((n, p))
    obj = 0
    for di_idx, i in enumerate(pi):
        weights = 2 * (pi_Dist_compressed[di_idx][:, np.newaxis] * g_ij[i])
        if i > 0:
            obj += cp.sum_squares(cp.norm(X[i:i+1] - X[:i], p=ord, axis=1)) - cp.sum(cp.multiply(weights[:i], X[i:i+1] - X[:i])) 
        if i < n-1: 
            obj += cp.sum_squares(cp.norm(X[i:i+1] - X[i+1:], p=ord, axis=1)) - cp.sum(cp.multiply(weights[i+1:], X[i:i+1] - X[i+1:]))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=solver, verbose=verbose)
    return X.value, (obj.value + np.square(pi_Dist_compressed).sum()) / (pi.size * (n-1) / 2)


def linfty_full_convex_concave(rank, pi, pi_Dist_compressed, n_init=2, eps=1e-3, max_iter=500, debug=False, verbose=False, freq=50):
    best_emb = (None, np.inf)
    n = pi_Dist_compressed.shape[1]
    for _ in range(n_init):
        # X = np.random.uniform(size=n * rank).reshape(n, rank)
        X = np.random.randn(n, rank)
        prev_loss = np.inf
        for t in range(max_iter):
            g_ij = norm_all_subgradients(X, ord=np.inf)
            X, raw_stress = convex_concave_full_pi(pi_Dist_compressed, pi, ord=np.inf, g_ij=g_ij, verbose=verbose)
            if verbose and t % freq==0: print(f"{t=}, {raw_stress=}")
            loss = obj_distortion_function(pi_Dist_compressed, X, np.inf, pi)
            if t >= 1 and np.abs((prev_loss - loss) / prev_loss) < eps:
                if verbose: print(t)
                break
            if debug:
                assert prev_loss + 1e-6 > loss, print(prev_loss + 1e-6 - loss) 
                prev_loss = loss
        if verbose: print(f"{raw_stress=}")
        if best_emb[1] > raw_stress:
            best_emb = (X, raw_stress)
    return best_emb


def full_convex_concave(rank, pi, pi_Dist_compressed, n_init=2, ord=np.inf, max_iter=500, debug=False, 
                        verbose=False, solver=cp.CLARABEL, freq=50):
    best_emb = (None, np.inf)
    n = pi_Dist_compressed.shape[1]
    for _ in range(n_init):
        # X = np.random.uniform(size=n * rank).reshape(n, rank)
        X = np.random.randn(n, rank)
        prev_loss = np.inf
        for t in range(max_iter):
            g_ij = norm_all_subgradients(X, ord=ord)
            X, raw_stress = convex_concave_full_pi(pi_Dist_compressed, pi, ord=ord, g_ij=g_ij, verbose=verbose, solver=solver)
            if verbose and t % freq==0: print(f"{t=}, {raw_stress=}")
            loss = obj_distortion_function(pi_Dist_compressed, X, ord, pi)
            if debug:
                assert prev_loss + 1e-6 > loss, print(prev_loss + 1e-6 - loss) 
                prev_loss = loss
        if verbose: print(f"{raw_stress=}")
        if best_emb[1] > raw_stress:
            best_emb = (X, raw_stress)
    return best_emb
