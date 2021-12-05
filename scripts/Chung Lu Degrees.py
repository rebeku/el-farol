from matplotlib import pyplot as plt
from matplotlib import animation

import networkx as nx
import numpy as np
import pandas as pd

from simulation import run_simulation_batch
from chung_lu import generate_chung_lu, truncated_power_law


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