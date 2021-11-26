from matplotlib import pyplot as plt
from matplotlib import animation

import networkx as nx
import numpy as np
import pandas as pd


def run_simulation(X, A, n_iter, eps, M, minfriends, minbad):
    Xs = [X]

    for _ in range(n_iter):
        if X.sum() < M:
            # go if enough of your friends went last week
            # whether or not you went
            X = (A.dot(X) > minfriends)
        else:
            X = (A.dot(X) > minbad)

        X += rng.uniform(size=(len(A),1)) < eps
        Xs.append(X)

    return np.hstack(Xs)


def maximal_fixed_point(G):
    g = G.copy()
    while len(g.nodes) > 0:
        k = g.degree()
        to_keep = [node[0] for node in k if node[1] > minfriends]
        if len(to_keep) == len(g.nodes):
            return g
        g = g.subgraph(to_keep)

    return g


def run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng):
    Xnps = []

    for i in range(batch_size):
        X = rng.uniform(size=(len(A),1)) < 0.3
        Xnps.append(
            run_simulation(X, A, n_iter, eps, M, minfriends, minbad))
        print(f"Ran simulation {i}")

    # count the number of times each node goes to the bar
    # in each batch
    # skip first 50 rounds to give time to settle
    attendance = [np.where(Xnp[:,50:]) for Xnp in Xnps]
    ppl = [pd.Series(a[0]) for a in attendance]

    cnts = pd.DataFrame(index=range(len(A)))
    for i in range(batch_size):
        cnts[i] = ppl[i].value_counts()

    always = (cnts==n_iter - 49).mean(axis=1)
    sometimes = (cnts < n_iter - 49).mean(axis=1)
    never = cnts.isna().mean(axis=1)

    cnts["Always"] = always
    cnts["Sometimes"] = sometimes
    cnts["Never"] = never

    g = maximal_fixed_point(
        nx.convert_matrix.from_numpy_array(A)
    )

    cnts["maximal_fixed_point"] = pd.Series(range(len(A))).apply(
        lambda x: x in g.nodes)

    return cnts


def truncated_power_law(gamma, k_0, n, rng, size=None):
    """
    Generate a sample of size *size* from a power law distribution
    with mininmum *k_0*, maximum *n*, and power *gamma*
    """
    k_0=np.float64(k_0)
    gamma=np.float64(gamma)
    n=np.float64(n)
    
    if size:
        U = rng.uniform(size=size)
    else:
        U = rng.uniform()
        
    return (
        (k_0**(1-gamma) - 
             ( k_0**(1-gamma) - n**(1-gamma) ) * U 
        )**(1/(1-gamma))
    )


def generate_chung_lu(k, rng):
    n = len(k)

    k_mean = k.mean()
    # k_mean times is twice the expected number of edges
    # this will be the denominator of all edge probabilities
    m2 = k_mean * n

    # initialize adjacency matrix
    A = np.zeros((n,n),dtype=int)
    choices = rng.random(n*(n-1) // 2)
    choice_i = 0
    
    for i in range(n):
        for j in range(i+1,n):
            # no self loops
            if i == j:
                continue
                
            # compute probability of edge
            p = min(
                (k[i] * k[j] / m2,
                1))
            
            # generate edge
            if choices[choice_i] < p:
                A[i,j] = 1
                A[j,i] = 1
            
            choice_i += 1
                
    return A

def not_sometimes(cnts):
    maska = cnts.Always > 0
    maskn = cnts.Never > 0
    masks = cnts.Sometimes == 0
    
    return (maska & maskn & masks).sum()


if __name__ == "__main__":

    rng = np.random.default_rng(2357111)

    n = 100
    size=300
    n_iter = 1000
    eps = 0
    M = 600
    minfriends = 4
    minbad = 4
    batch_size = 50
    
    # generate configuration model
    print("Running simulation on configuration model")
    gamma = 2
    k_0 = 3
    k = truncated_power_law(gamma, k_0, n, rng, size=size).astype(int)
    k.sort()
    A = generate_chung_lu(k, rng)
    print(f"Mean degree: {A.sum(axis=0).mean()}")

    cnts_cm = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
    print(f"Not sometimes: {not_sometimes(cnts_cm)}")
    print(f"Size of maximal fixed point: {cnts_cm.maximal_fixed_point.sum()}")
    
    plt.xlabel("Proportion of population attending")
    plt.ylabel("Frequency")
    plt.title("Configuration Model")
    plt.hist((cnts_cm.sum(axis=0)[:50]/(951*300)),bins=20,range=(0,1))
    plt.savefig("images/cm_attendance.png")
    
    # generate Erdos-Renyi random graph
    print("\nRunning simulation on Erdos-Renyi random graph")
    k_bar = A.sum() / len(A)
    N = len(A)
    p = k_bar / N

    # G = nx.gnp_random_graph(N, 0.03268, seed=8888777)
    G = nx.gnp_random_graph(N, p, seed=8888777)
    A = nx.convert_matrix.to_numpy_array(G)
    print(f"Mean degree: {A.sum(axis=0).mean()}")

    cnts_gnp = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
    print(f"Not sometimes: {not_sometimes(cnts_gnp)}")
    print(f"Size of maximal fixed point: {cnts_gnp.maximal_fixed_point.sum()}")

    plt.figure()
    plt.hist((cnts_gnp.sum(axis=0)[:50]/(951*300)),bins=20,range=(0,1))
    plt.xlabel("Proportion of population attending")
    plt.ylabel("Frequency")
    plt.title("Erdos Renyi")
    plt.savefig("images/gnp_attendance.png")
    

    print("\nRunning simulation on stochastic block model.")
    sizes = [100, 100, 100]
    probs = [[0.09, 0.01, 0.01], [0.01, 0.09, 0.01], [0.01, 0.01, 0.09]]
    G = nx.stochastic_block_model(sizes, probs, seed=3089277)
    A = nx.convert_matrix.to_numpy_matrix(G)
    print(f"Mean degree: {A.sum(axis=0).mean()}")

    cnts_sbm = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
    print(f"Not sometimes: {not_sometimes(cnts_sbm)}")
    print(f"Size of maximal fixed point: {cnts_sbm.maximal_fixed_point.sum()}")

    # plot weekly attendance
    plt.figure()
    plt.hist((cnts_sbm.sum(axis=0)[:50]/(951*300)),bins=20,range=(0,1))
    plt.xlabel("Proportion of population attending")
    plt.ylabel("Frequency")
    plt.title("Stochastic Block Model")
    plt.savefig("images/sbm_attendance.png")

    # generate plot
    always = ((cnts_gnp.Always > 0).sum(), (cnts_cm.Always > 0).sum(), (cnts_sbm.Always > 0).sum())
    sometimes = ((cnts_gnp.Sometimes > 0).sum(), (cnts_cm.Sometimes > 0).sum(), (cnts_sbm.Sometimes > 0).sum())
    never = ((cnts_gnp.Never > 0).sum(), (cnts_cm.Never > 0).sum(), (cnts_sbm.Never > 0).sum())
    
    # create plot
    # fig, ax = plt.subplots()
    index = np.arange(3)
    bar_width = 0.3
    opacity = 0.8

    rects1 = plt.bar(index, always, bar_width,
    alpha=opacity,
    color='tab:green',
    label='Always')

    rects2 = plt.bar(index + bar_width, sometimes, bar_width,
    alpha=opacity,
    color='tab:orange',
    label='Sometimes')

    rects2 = plt.bar(index + 2*bar_width, never, bar_width,
    alpha=opacity,
    color='tab:red',
    label='Never')

    plt.xlabel('Random Graph')
    plt.ylabel('Node Count')
    plt.title('Attendance Patterns on Random Graph Models')
    plt.xticks(index + 2*bar_width, ('Erdos\nRenyi', 'Configuration\nModel', 'SBM'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/always_sometimes_never.png")


    # plot members of the maximal fixed point only
    cnts_gnp = cnts_gnp[cnts_gnp.maximal_fixed_point]
    cnts_cm = cnts_cm[cnts_cm.maximal_fixed_point]
    cnts_sbm = cnts_sbm[cnts_sbm.maximal_fixed_point]
    
    # generate plot
    always = ((cnts_gnp.Always > 0).mean(), (cnts_cm.Always > 0).mean(), (cnts_sbm.Always > 0).mean())
    sometimes = ((cnts_gnp.Sometimes > 0).mean(), (cnts_cm.Sometimes > 0).mean(), (cnts_sbm.Sometimes > 0).mean())
    never = ((cnts_gnp.Never > 0).mean(), (cnts_cm.Never > 0).mean(), (cnts_sbm.Never > 0).mean())
    
    # create plot
    # fig, ax = plt.subplots()
    plt.figure()

    index = np.arange(3)
    bar_width = 0.3
    opacity = 0.8

    rects1 = plt.bar(index, always, bar_width,
    alpha=opacity,
    color='tab:green',
    label='Always')

    rects2 = plt.bar(index + bar_width, sometimes, bar_width,
    alpha=opacity,
    color='tab:orange',
    label='Sometimes')

    rects2 = plt.bar(index + 2*bar_width, never, bar_width,
    alpha=opacity,
    color='tab:red',
    label='Never')

    plt.xlabel('Random Graph')
    plt.ylabel('Node Count')
    plt.title('Attendance Patterns on Random Graph Models')
    plt.xticks(index + 2*bar_width, ('Erdos\nRenyi', 'Configuration\nModel', 'SBM'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/always_sometimes_never_mfp.png")
