from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from simulation import run_simulation_batch
from chung_lu import generate_chung_lu, truncated_power_law


def simulate_with_noise(A, title, fname):
    epsilons = np.arange(0, 0.25, 0.01)
    weekly = []

    batch_size = 20

    for eps in epsilons:
        cnts = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
        mfp = cnts.maximal_fixed_point.sum()

        weekly.append((cnts.sum(axis=0)[:batch_size]/(950*mfp)))

    w = np.vstack(weekly)
    plt.figure()
    plt.errorbar(epsilons, w.mean(axis=1), w.std(axis=1), fmt="P")
    plt.xlabel("$\epsilon$")
    plt.ylabel("Proportion of MFP Attending")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)

    
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
    batch_size = 3

    print("Simulating attendance vs. degree with noise")
    eps = 0.15
    N = size

    probs = np.arange(0.02, 0.05, 0.001)
    probs = np.hstack([probs] * n_trials)
    probs.sort()

    attendance = []
    mean_deg = []
    nonoise = []

    for p in probs:
        G = nx.gnp_random_graph(N, p, seed=rng.choice(1000))
        A = nx.convert_matrix.to_numpy_array(G)
        mean_deg.extend([A.sum(axis=0).mean()] * batch_size)
        
        cnts = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
        mfp = cnts.maximal_fixed_point.sum()
        weekly = (cnts.sum(axis=0)[:batch_size]/(950 * mfp))
        attendance.extend(weekly.tolist())
        
        cnts = run_simulation_batch(A, n_iter, 0, M, minfriends, minbad, batch_size, rng)
        mfp = cnts.maximal_fixed_point.sum()
        weekly = (cnts.sum(axis=0)[:batch_size]/(950 * mfp))
        nonoise.extend(weekly.tolist())

    attendance = np.array(attendance)
    mean_deg = np.array(mean_deg)
    nonoise = np.array(nonoise)

    plt.scatter(mean_deg, attendance)
    plt.scatter(mean_deg, nonoise)
    plt.legend(["$\epsilon=0.15$", "$\epsilon=0$"])
    plt.xlabel("Mean Degree")
    plt.ylabel("Proportion of MFP Attending")
    plt.title("Noisy vs. Determininistic Erdos-Renyi")
    plt.savefig("images/mean_degree_v_noise.png")

    print("Now varying epsilon")
    gamma = 2.3
    k_0 = 4
    k = truncated_power_law(gamma, k_0, n, rng, size=size).astype(int)
    k.sort()
    A = generate_chung_lu(k, rng)
    print(f"Mean degree: {A.sum(axis=0).mean()}")

    simulate_with_noise(A, "Noisy Chung-Lu Model", "images/noisy_chung_lu.png")


    # generate Erdos-Renyi random graph
    print("\nRunning simulation on Erdos-Renyi random graph")
    k_bar = A.sum() / len(A)
    N = len(A)
    p = k_bar / N

    G = nx.gnp_random_graph(N, p, seed=8888777)
    A = nx.convert_matrix.to_numpy_array(G)
    print(f"Mean degree: {A.sum(axis=0).mean()}")
    
    simulate_with_noise(A, "Noisy Erods-Renyi", "images/noisy_erdos_renyi.png")
    
    print("\nRunning simulation on stochastic block model.")
    sizes = [100, 100, 100]
    probs = [[0.09, 0.01, 0.01], [0.01, 0.09, 0.01], [0.01, 0.01, 0.09]]
    G = nx.stochastic_block_model(sizes, probs, seed=3089277)
    A = nx.convert_matrix.to_numpy_matrix(G)
    print(f"Mean degree: {A.sum(axis=0).mean()}")

    simulate_with_noise(A, "Noisy SBM", "images/noisy_sbm.png")