from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from simulation import run_simulation_batch
from chung_lu import generate_chung_lu, truncated_power_law


if __name__=="__main__":
    rng = np.random.default_rng(2357111)

    n = 100
    size=300
    n_iter = 1000
    eps = 0
    M = 600
    minfriends = 4
    minbad = 4
    batch_size = 10 

    # plot impact of mean degree on Chung Lu Model
    weekly = []
    mean_deg = []

    k_0s = [2, 3, 4]
    gammas = np.hstack(
        [np.arange(2.0, 2.4, 0.025), 
        np.arange(2.4,2.7,0.1)
        ])

    for gamma in gammas:
        for k_0 in k_0s:
            k = truncated_power_law(gamma, k_0, n, rng, size=size).astype(int)
            k.sort()
            A = generate_chung_lu(k, rng)
            mean_deg.append([A.sum(axis=0).mean()])
            
            cnts = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
            weekly.append(cnts.sum(axis=0)[:batch_size]/950)

    weekly = np.vstack(weekly)
    mean_deg = np.array(mean_deg)

    mask = np.where(weekly.mean(axis=1) < np.inf)[0]

    plt.figure()
    plt.errorbar(mean_deg[mask], weekly.mean(axis=1)[mask], weekly.std(axis=1)[mask], fmt="P")
    plt.xlabel("Mean Degree")
    plt.ylabel("Steady State Attendance")
    plt.title("Chung Lu Model")
    plt.savefig("images/chung_lu_degree.png")

    # Erdos-Renyi Model
    probs = np.arange(0.026, 0.05, 0.0005)

    weekly = []
    mean_deg = []

    for p in probs:
        G = nx.gnp_random_graph(size, p, seed=rng.choice(1000))
        A = nx.convert_matrix.to_numpy_array(G)
        mean_deg.append([A.sum(axis=0).mean()])
        
        cnts = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
        weekly.append(cnts.sum(axis=0)[:batch_size]/950)

    weekly = np.vstack(weekly)
    mean_deg = np.array(mean_deg)

    mask = np.where(weekly.mean(axis=1) < np.inf)[0]

    plt.figure()
    plt.errorbar(mean_deg[mask], weekly.mean(axis=1)[mask], weekly.std(axis=1)[mask], fmt="P")
    plt.xlabel("Mean Degree")
    plt.ylabel("Steady State Attendance")
    plt.title("Erdos Renyi Model")
    plt.savefig("images/erdos_renyi_degree.png")

    # SBM
    # baseline range used to construct the mixing matrix
    probs = np.arange(0.042, 0.085, 0.001)

    weekly = []
    mean_deg = []

    sizes=[100,100,100]

    for p in probs:
        p_in = p*1.5
        p_out = p_in / 9
        p = [[p_in, p_out, p_out],[p_out, p_in, p_out], [p_out, p_out, p_in]]
        G = nx.stochastic_block_model(sizes, p, seed=rng.choice(1000))
        A = nx.convert_matrix.to_numpy_array(G)
        mean_deg.append([A.sum(axis=0).mean()])
        
        cnts = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
        weekly.append(cnts.sum(axis=0)[:batch_size]/950)

    weekly = np.vstack(weekly)
    mean_deg = np.array(mean_deg)

    mask = np.where(weekly.mean(axis=1) < np.inf)[0]

    plt.figure()
    plt.errorbar(mean_deg[mask], weekly.mean(axis=1)[mask], weekly.std(axis=1)[mask], fmt="P")
    plt.xlabel("Mean Degree")
    plt.ylabel("Steady State Attendance")
    plt.title("SBM")
    plt.savefig("images/sbm_degree.png")