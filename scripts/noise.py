from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from simulation import run_simulation_batch
from chung_lu import generate_chung_lu, truncated_power_law


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
    n_trials = 5
    batch_size = 30

    print("Simulating attendance vs. degree with noise")
    eps = 0.15
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

    print("Now varying epsilon")
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