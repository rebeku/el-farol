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
        # print(f"Ran simulation {i}")

    # count the number of times each node goes to the bar
    # in each batch
    # skip first 50 rounds to give time to settle
    attendance = [np.where(Xnp[:,51:]) for Xnp in Xnps]
    ppl = [pd.Series(a[0]) for a in attendance]

    cnts = pd.DataFrame(index=range(len(A)))
    for i in range(batch_size):
        cnts[i] = ppl[i].value_counts()

    always = (cnts==n_iter - 50).mean(axis=1)
    sometimes = (cnts < n_iter - 50).mean(axis=1)
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


def simulate_with_noise(A, title, fname):
    n_trials = 5
    epsilons = np.arange(0, 0.2, 0.01)
    epsilons = np.hstack([epsilons] * n_trials)
    epsilons.sort()

    cnts_mean = []
    cnts_std = []
    batch_size = 20

    for eps in epsilons:
        cnts = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
        mfp = cnts.maximal_fixed_point.sum()

        weekly = (cnts.sum(axis=0)[:batch_size]/(950*mfp))
        cnts_std.append(weekly.std())
        cnts_mean.append(weekly.mean())
            
    plt.figure()
    plt.scatter(epsilons, cnts_mean)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Proportion of MFP Attending")
    plt.title(title.format("Mean"))
    plt.savefig(fname.format("mean"))
    
    plt.figure()
    plt.scatter(epsilons, cnts_std)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Proportion of MFP Attending")
    plt.title(title.format("STD"))
    plt.savefig(fname.format("std"))

if __name__ == "__main__":
    rng = np.random.default_rng(2357111)

    n = 100
    size=300
    n_iter = 1000
    eps = 0
    M = 600
    minfriends = 4
    minbad = 4

    k_std = []
    cnts_std = []
    n_trials =  5
    batch_size = 30
    
    gammas = np.arange(1.3, 4.1, 0.05)
    gammas = np.hstack([gammas] * n_trials)
    gammas.sort()

    k_0 = 10

    """
    
    for gamma in gammas:
        k = truncated_power_law(gamma, k_0, n, rng, size=size).astype(int)
        k.sort()
        
        # find k so that the mean degree is between
        # 10 and 11
        for _ in range(10):
            if k.mean() < 10:
                k_0 = k_0 + 1
                k = truncated_power_law(gamma, k_0, n, rng, size=size).astype(int)
                
            elif k.mean() > 11:
                k_0 = k_0 - 1
                k = truncated_power_law(gamma, k_0, n, rng, size=size).astype(int)
            else:
                break
                    
        A = generate_chung_lu(k, rng)
        print(f"K_0:{k_0}")
        print(f"Mean degree: {A.sum(axis=0).mean()}")
        k_std.append(A.sum(axis=0).std())
        
        cnts = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
        cnts_std.append((cnts.sum(axis=0)[:batch_size]/(950)).std())
        
        print(f"K std: {k_std[-1]}\tcnts_std: {cnts_std[-1]}")
    
    k_std = np.array(k_std)
    cnts_std = np.array(cnts_std)

    mask = np.where(k_std > 0)[0]

    plt.figure()
    plt.scatter(k_std[mask], cnts_std[mask])
    plt.xlabel("Std of node degrees")
    plt.ylabel("Std of limiting attendance")
    plt.title("Chung-Lu Model: Impact of degree distribution")
    plt.savefig("images/chung_lu_std.png")
    """

    print("Simulating attendance vs. degree with noise")
    eps = 0.2
    N = size

    probs = np.arange(0.02, 0.031, 0.0005)
    probs = np.hstack([probs] * n_trials)
    probs.sort()

    attendance = []
    mean_deg = []

    for p in probs:
        G = nx.gnp_random_graph(N, p, seed=rng.choice(1000))
        A = nx.convert_matrix.to_numpy_array(G)
        mean_deg.append(A.sum(axis=0).mean())
        cnts = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
        
        mfp = cnts.maximal_fixed_point.sum()

        weekly = (cnts.sum(axis=0)[:batch_size]/(950 * mfp))
        attendance.append(weekly.mean())

    attendance = np.array(attendance)
    mean_deg = np.array(mean_deg)

    mask = np.where(attendance < 2)[0]

    plt.scatter(mean_deg[mask], attendance[mask])
    plt.xlabel("Mean Degree")
    plt.ylabel("Proportion of MFP Attending")
    plt.title("Attendance on a Noisy Erdos-Renyi Network")
    plt.savefig("images/mean_degree_v_noise.png")

    print("Now simulating with noise")
    gamma = 2.3
    k_0 = 4
    k = truncated_power_law(gamma, k_0, n, rng, size=size).astype(int)
    k.sort()
    A = generate_chung_lu(k, rng)
    print(f"Mean degree: {A.sum(axis=0).mean()}")

    simulate_with_noise(A, "Chung-Lu Model: {} Weekly Attendance", "images/noisy_chung_lu_{}.png")


    # generate Erdos-Renyi random graph
    print("\nRunning simulation on Erdos-Renyi random graph")
    k_bar = A.sum() / len(A)
    N = len(A)
    p = k_bar / N

    G = nx.gnp_random_graph(N, p, seed=8888777)
    A = nx.convert_matrix.to_numpy_array(G)
    print(f"Mean degree: {A.sum(axis=0).mean()}")
    
    simulate_with_noise(A, "Erods-Renyi: {} Weekly Attendance", "images/noisy_erdos_renyi_{}.png")
    
    print("\nRunning simulation on stochastic block model.")
    sizes = [100, 100, 100]
    probs = [[0.09, 0.01, 0.01], [0.01, 0.09, 0.01], [0.01, 0.01, 0.09]]
    G = nx.stochastic_block_model(sizes, probs, seed=3089277)
    A = nx.convert_matrix.to_numpy_matrix(G)
    print(f"Mean degree: {A.sum(axis=0).mean()}")

    simulate_with_noise(A, "SBM: {} Weekly Attendance", "images/noisy_sbm_{}.png")