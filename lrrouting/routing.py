import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dask






def route_st(s, t, dar, max_iters=10**4):
    route = [s]
    if s == t: return route
    a = s; wpath = 0.
    while len(route) < max_iters and a != t:
        b_next, w_b_next = dar.next_node(s=s, t=t, a=a)
        wpath += w_b_next
        a = b_next
        route += [b_next]
        if len(dar.adjacency_list[a]) == 0:
            break
    return route, wpath


def map_to_node_ids(route, node_ids, nodes_cc):
    return [node_ids[nodes_cc[a]] for a in route]


def st_pairs(n, Dist, M):
    sources = np.random.uniform(0, n, M).astype(int)
    targets = np.random.uniform(0, n, M).astype(int)
    ss, tt = [], []
    for (s, t) in zip(sources, targets):
        if Dist[s,t] == np.inf or s == t:
            continue
        ss += [s]
        tt += [t]
    sources = np.array(ss)
    targets = np.array(tt)
    return sources, targets


def calculate_ratio(s, t, dar, Dist):
    if s == t:
        return 1 
    path, wpath = dar.route(s=s, t=t)
    a = path[-1]
    if a == t:
        return wpath / Dist[s, t]
    else:
        return np.inf
    

def dp_routing_stats(sources, targets, dar, Dist):
    ratios = np.zeros(len(sources))
    for i, (s, t) in enumerate(zip(sources, targets)):
        if s == t: ratios[i] = 1
        ratios[i] = calculate_ratio(s, t, dar, Dist)
    
    fracs = [5, 2, 1.5, 1.2, 1]; f_ratios = {}
    for frac in fracs:
        f_ratios[frac] = 100.*(ratios <= frac + 1e-8).sum() / ratios.size
    median_stretch = np.median(ratios) * 100.
    mean_stretch = ratios.mean() * 100.
    print(f"{median_stretch=:.1f}%, {mean_stretch=:.1f}%")
    print(f"%[ratio<2] = {f_ratios[2]:.2f}%, %[ratio<1.2] = {f_ratios[1.2]:.2f}%, %[ratio=1.] = {f_ratios[1]:.2f}%")
    return ratios


def dp_routing_stats_parallel(sources, targets, dar, Dist):
    delayed_ratios = [dask.delayed(calculate_ratio)(s, t, dar, Dist) for s, t in zip(sources, targets)]
    
    ratios = np.array(dask.compute(*delayed_ratios))
    
    fracs = [5, 2, 1.5, 1.2, 1]; f_ratios = {}
    for frac in fracs:
        f_ratios[frac] = 100. * (ratios <= frac + 1e-8).sum() / ratios.size
    median_stretch = np.median(ratios) * 100.
    mean_stretch = ratios.mean() * 100.
    print(f"{median_stretch=:.1f}%, {mean_stretch=:.1f}%")
    print(f"%[ratio<2] = {f_ratios[2]:.2f}%, %[ratio<1.2] = {f_ratios[1.2]:.2f}%, %[ratio=1.] = {f_ratios[1]:.2f}%")
    return ratios


def dp_routing_stats_base(sources, targets, func_next_node, Dist, Adj, adjacency_list, max_iters=10**4, tau=1):
    ratios = []
    total_pairs = 0
    for (s, t) in zip(sources, targets):
        if s == t: continue
        total_pairs += 1
        a = s; k = 0; 
        wpath = 0.
        while k < max_iters and a != t:
            b_next = func_next_node(s=s, t=t, a=a)
            wpath += Adj[a, b_next]
            a = b_next
            k += 1
            if len(adjacency_list[a]) == 0:
                break
        if a == t:
            ratios += [wpath / Dist[s,t]]
        else:
            ratios += [np.inf]
    ratios = np.array(ratios)
    fracs = [5, 2, 1.5, 1.2, 1]; f_ratios = {}
    for frac in fracs:
        f_ratios[frac] = 100.*(ratios <= frac + 1e-8).sum() / total_pairs
    median_stretch = np.median(ratios) * 100.
    mean_stretch = ratios.mean() * 100.
    print(f"{median_stretch=:.1f}%, {mean_stretch=:.1f}%")
    print(f"%[ratio<2] = {f_ratios[2]:.2f}%, %[ratio<1.2] = {f_ratios[1.2]:.2f}%, %[ratio=1.] = {f_ratios[1]:.2f}%")
    return ratios


def next_node(dist_func, adjacency_list, Adj):
    def get_b(*, s, t, a):
        d_at = np.zeros(len(adjacency_list[a]))
        for idx, (b, w_b) in enumerate(adjacency_list[a]):
            d_at[idx] = Adj[a, b] + dist_func(b, t)
        b_idx = np.argmin(d_at)
        b_next = adjacency_list[a][b_idx][0]
        return b_next
    return get_b


def subopt_ratios_mtrx(mtrx_A, adjacency_list, Adj, Dist, sources, targets, rt_max_iters):
    dist_func = lambda b,t: mtrx_A[b,t]
    func_next_node = next_node(dist_func, adjacency_list, Adj)
    ratios = dp_routing_stats_base(sources, targets, func_next_node, Dist, Adj, adjacency_list, max_iters=rt_max_iters)
    return ratios


def subopt_ratios_base(graph, adjacency_list, Adj, Dist, sources, targets, rt_max_iters):
    dist_func = lambda b, t: graph.dist(b, t)
    func_next_node = next_node(dist_func, adjacency_list, Adj)
    ratios = dp_routing_stats_base(sources, targets, func_next_node, Dist, Adj, adjacency_list, max_iters=rt_max_iters)
    return ratios


def subopt_ratios(dar, Dist, sources, targets):
    ratios = dp_routing_stats(sources, targets, dar, Dist)
    return ratios


def plot_cdf_algo_subopt_ratio(stats, fracs=None, title="", width=3, log_scale=False, figsize=(6, 4), dpi=150):
    if fracs is None:
        fracs = np.linspace(1, 4, 100)
    plt.rcParams['text.usetex'] = True
    algos = sorted(stats.keys())
    ratios = {algo:[] for algo in algos} 
    
    for j, (algo, rats) in enumerate(stats.items()):
        rats = rats["ratios"]
        for frac in fracs:
            ratios[algo] += [(rats <= frac + 1e-8).sum() / rats.size]
    
    num_algo = len(algos)
    num_rows = 1
    fig, ax = plt.subplots(num_rows, \
                           figsize=figsize, \
                           dpi=dpi, sharey=True)
    cmp = sns.color_palette("hls", num_algo+1)
    for j, algo in enumerate(algos): 
        ax.plot(fracs, ratios[algo], label=algo, linewidth=1, color=cmp[j])
        ax.grid(True)
        if log_scale:
            ax.set_yscale('log')
    ax.set_ylabel('fraction')
    ax.set_xlabel(r'$\alpha$ suboptimality')
    size = (stats[algos[0]]["ratios"]).size
    fig.suptitle("Subopt ratio cdf: %s, %d samples"%(title, size), \
                            fontsize="x-large")
    fig.tight_layout()
    fig.legend(algos, fancybox=True, framealpha=0.5, loc='center left', bbox_to_anchor=(1, 0.5))
    if title:
        plt.savefig("plots/"+title+".pdf", format="pdf", bbox_inches="tight")
    plt.show()
    

def plot_fracs_subopt(info_ranks, rank_max, fracs = [1.05, 1.2, 1.5, 2.0], mtype="", marker=None, figsize=(10, 3), 
                      notitle=False, dpi=130, ylim=None):
    plt.rcParams['text.usetex'] = True
    r = list(info_ranks.keys())[0]
    algos = sorted(info_ranks[r].keys())
    ratios = {frac:{algo:[] for algo in algos} for frac in fracs}

    if isinstance(rank_max, int): ranks = list(range(1, rank_max+1))
    else: ranks = sorted(info_ranks.keys())

    for frac in fracs:
        for rank in ranks:
            for (algo, rats) in info_ranks[rank].items():
                ratios[frac][algo] += [(rats["ratios"] <= frac).sum() / rats["ratios"].size]

    cmp = sns.color_palette("hls", len(algos) + 1)
    fig, axs = plt.subplots(1, len(fracs), figsize=figsize, dpi=dpi, sharey=True, gridspec_kw={'hspace': 0.5})
    plt.subplots_adjust(top=0.8)
    axs[0].set_ylabel('fraction')
    for i, frac in enumerate(fracs):
        for j, algo in enumerate(algos): 
            axs[i].plot(ranks, ratios[frac][algo], label=algo, linewidth=0.8, color=cmp[j], marker=marker)
            axs[i].grid(True)
            axs[i].set_title(r'$\alpha=%d$'%int(np.round(((frac-1)*100)))+r"$\%$")
            axs[i].set_xlabel('storage')
            
            if ylim is not None:
                axs[i].set_ylim(ylim[0], ylim[1])
    if not notitle:
        fig.suptitle(r"Fraction of paths $\alpha\%$ suboptimal, "+ mtype, fontsize="x-large", y=0.95)
    fig.legend(algos, fancybox=True, framealpha=0.5, loc='center left', bbox_to_anchor=(0.9, 0.5))
    if mtype:
        fname = "plots/subopt_ranks_"+mtype+".pdf"
        fname = fname.replace('$', '')
        plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.show()


def plot_median_storage(n, info_ranks, rank_max, mtype="", marker=None, figsize=(10, 3), 
                      notitle=False, dpi=130, yticks=[1e-1, 1, 10], 
                      ylim=None, symm=False, yscale='log'):
    plt.rcParams['text.usetex'] = True
    r = list(info_ranks.keys())[0]
    algos = sorted(info_ranks[r].keys())
    if yticks is not None:
        ylabels = [f'$10^{{{int(np.log10(tick))}}}$' if tick!=0 else f'$0$' for tick in yticks]

    if isinstance(rank_max, int): ranks = list(range(1, rank_max+1))
    else: ranks = np.array(sorted(info_ranks.keys()))

    quantiles = {algo:[] for algo in algos}
    for rank in ranks:
        for (algo, rats) in info_ranks[rank].items():
                quantiles[algo] += [np.median(rats["ratios"]) - 1]

    cmp = sns.color_palette("hls", len(algos) + 1)
    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi, sharey=True, gridspec_kw={'hspace': 0.5})
    plt.subplots_adjust(top=0.8)
    for j, algo in enumerate(algos): 
        axs.plot(ranks / n * 100, quantiles[algo], label=algo, linewidth=0.8, color=cmp[j], marker=marker)
    axs.grid(True)

    # axs.set_xlabel(r'storage for $n=%d$'%n)
    if symm:
        axs.set_xlabel(r'$p/n \times 100\%$')
    else:
        axs.set_xlabel(r'$2p/n \times 100\%$')
    axs.set_ylabel('Median routing suboptimality')
    axs.set_yscale(yscale)
    if yticks is not None:
        axs.set_yticks(yticks)
        axs.set_yticklabels(ylabels)
        
    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[1])
    fig.legend(algos, fancybox=True, framealpha=0.5, loc='center left', bbox_to_anchor=(0.9, 0.5))
    if mtype:
        fname = "plots/median_storage_"+mtype+".pdf"
        fname = fname.replace('$', '')
        plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.show()


def plot_emb_vs_routing_subopt_lines(graphs, names, coeff = 100, dpi=120, figsize=(6, 4), msize=1, marker='.', savefig=True):
    cmp = sns.color_palette("hls", len(names) + 1)
    plt.rcParams['text.usetex'] = True

    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi, sharey=True, gridspec_kw={'hspace': 0.5})
    plt.subplots_adjust(top=0.8)

    for j, mtype in enumerate(graphs[coeff]): 
        axs.plot(graphs[coeff][mtype]["delta_r"], np.array(graphs[coeff][mtype]["subopt"])-1, 
                 label=mtype, linewidth=0.8, color=cmp[j], marker=marker, ms=msize)
        
    axs.grid(True)
    fig.legend(names, fancybox=True, framealpha=0.5, loc='center left', bbox_to_anchor=(0.9, 0.5))
    axs.set_ylabel('Median routing suboptimality')
    axs.set_yscale('symlog')
    axs.set_xlabel('Median embedding suboptimality')
    if savefig:
        fname = "plots/emb_vs_routing_coef%d.pdf"%coeff
        fname = fname.replace('$', '')
        plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.show()


def plot_emb_vs_routing_subopt(graphs, names, coeff = 100, dpi=120, figsize=(6, 4), msize=50, marker='.', savefig=True):
    cmp = sns.color_palette("hls", len(names) + 1)
    plt.rcParams['text.usetex'] = True

    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi, sharey=True, gridspec_kw={'hspace': 0.5})
    plt.subplots_adjust(top=0.8)

    for j, mtype in enumerate(graphs[coeff]): 
        axs.scatter(graphs[coeff][mtype]["delta_r"], np.array(graphs[coeff][mtype]["subopt"])-1, 
                 label=mtype, linewidth=0.8, color=cmp[j], marker=marker, s=msize)
        
    axs.grid(True)
    fig.legend(names, fancybox=True, framealpha=0.5, loc='center left', bbox_to_anchor=(0.9, 0.5))
    axs.set_ylabel('Median routing suboptimality')
    axs.set_yscale('symlog')
    axs.set_xlabel('Median embedding suboptimality')
    if savefig:
        fname = "plots/emb_vs_routing_coef%d.pdf"%coeff
        fname = fname.replace('$', '')
        plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.show()
