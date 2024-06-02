import numpy as np
import networkx as nx


import random
import pickle, sys, os

import lrrouting as ldr

import warnings
import numba as nb
import pickle
import argparse



np.random.seed(1001)
random.seed(1001)


# Ignore specific warning
warnings.filterwarnings("ignore", message="omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.")

# Ignore NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=nb.NumbaPerformanceWarning)


parser = argparse.ArgumentParser(description='Solve a problem based on rank and n.')
parser.add_argument('--n', type=int)
parser.add_argument('--nsamples', type=int, default=1) # percent of total number of rows
parser.add_argument('--storage', type=float)
parser.add_argument('--slurm', type=int, default=-1)
args = parser.parse_args()

n = args.n
if args.slurm == -1:
    storage = args.storage
else:
    storage = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2][args.slurm - 1]


mtype = "manhattan"


if args.slurm == -1:
    # Open a file to write
    original_stdout = sys.stdout
    sys.stdout = ldr.DualLogger('../outputs/%s.txt'%mtype, sys.stdout)


place = "Manhattan, NY, USA"
G, Adj, Dist, nodes_cc = ldr.dist_matrix_osmnx(place, directed=True, nodes=True)

m = n = Dist.shape[0]

rank = int(np.ceil(n * storage / 100))

diam_G = Dist.max()
w_min = Dist[Dist>0].min()
print(m, diam_G, w_min, (Adj>0).sum())
print(f"{n=}, {rank=}")
print(np.histogram(Dist.flatten(), bins=5, density=True))

adjacency_list = ldr.adjacency_directed_list(Adj)
sources, targets = ldr.st_pairs(n, Dist, 1020)
M = min(1000, sources.size)
sources = sources[:M]
targets = targets[:M]

PSD = False
w_min = Dist[Dist>0].min()
rt_max_iters = min(int(5*Dist.max()/w_min), (10**4) // 2)
symm = np.allclose(Dist, Dist.T)
print(f"{symm=}")
filename = "%s_r%d_%d"%(mtype, rank, n)
print(f"{filename=}")

print(f"{np.histogram(Adj[Adj>0], bins=5, density=True)=}")

del G, Adj

info = {} 

# \sqrt{n} rows and columns sampled

# sampling is adaptive
rand_frac = 0.975
print(f"{rand_frac=}")
pi_rows, pi_cols = ldr.adaptive_row_col(n, Dist, frac=rand_frac, coeff=args.nsamples, percent=True)
rDist, cDist, pi_rows_c, pi_cols_c = ldr.sample_dist(n, pi_rows, pi_cols, Dist)

nsamples = pi_rows.size

info_ranks = {}
 

print(f"\n{rank=}")
info_ranks[rank] = ldr.record_suboptimality(rank, pi_rows, pi_cols, pi_rows_c, pi_cols_c, rDist, cDist, adjacency_list, sources, targets, 
                           Dist, asymm=True, cc_max_iter=1000, n_init_cc=2, cg_eps=1e-8, cg_max_iter=200, verbose=True, freq=500)


script_dir = os.getcwd()
parent_dir = os.path.dirname(script_dir)
output_path = os.path.join(parent_dir, 'outputs/%s_nsp%d_rank%d_n%d_rf%d.pickle' % (mtype, args.nsamples, rank, n, int(rand_frac*100)))

with open(output_path, 'wb') as handle:
    pickle.dump(info_ranks, handle, protocol=pickle.HIGHEST_PROTOCOL)



if args.slurm == -1:
    sys.stdout.close()
    sys.stdout = original_stdout
