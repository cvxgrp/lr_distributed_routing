# Approximate Distributed Routing via Low Dimensional Embedding

joint work with T. Marcucci and S. Boyd.


We address the problem of approximate distributed routing
by focusing on two key objectives:  
1) finding a route between any two pairs
of nodes in linear time (with respect to the number of edges), 
and 
2) constraining each node to a small storage capacity.

To meet the first objective,
we propose a routing policy based on approximate dynamic programming 
combined with depth-first search. 
For the second objective,
we introduce an efficient method for node embeddings based on
multidimensional scaling.



In this repository, we provide `lrrouting` package implementing proposed methods.


## Installation
To install `lrrouting` 1) activate virtual environment, 2) clone the repo, 3) from inside the directory run 
```python3
pip install -e .
```
Requirements
* python == 3.9
* [mlrfit](https://github.com/cvxgrp/mlr_fitting) == 0.0.1
* numpy >= 1.21
* scipy >= 1.10
* scikit-learn == 1.1.3
* cvxpy == 1.4.2
* matplotlib == 3.7.1
* osmnx == 1.2.1
* numba == 0.55.0
* networkx == 3.0



## `hello_world`
See the [`examples/hello_world.ipynb`](https://github.com/cvxgrp/lr_distributed_routing/blob/main/examples/hello_world.ipynb) notebook or explanation below.


**Step 1.** Load `lrrouting` and graph given by the weighted adjacency list `adjacency_list`. 
```python3
import lrrouting as ldr 
```

**Step 2.** 
Sample rows and columns of the distance matrix using `pi_rows` for row indices with `rDist`, and `pi_cols` for column indices with `cDist`.

**Step 3.** Generate low dimensional embeddings for nodes.
```python3
Z = ldr.low_dim_embeddings(n, rank, pi_rows, pi_cols, rDist, cDist)
```

**Step 4.** Create a `DARouting` object instance using node embeddings and weighted adjacency list.
```python3
lr_dar = ldr.construct_xy_node_embedding_graph(Z[:n], Z[n:], adjacency_list)
```

**Step 5.** Use `DARouting` object to find a route between nodes `s` and `t`.
```python3
route_lr, w_lr = lr_dar.route(s, t)
```
 
## Example notebooks
See the notebooks in [`examples/`](https://github.com/cvxgrp/lr_distributed_routing/tree/main/examples) folder
that show how to use `lrrouting` to route between nodes,
and how to find good trade-offs
between routing suboptimality and storage.

