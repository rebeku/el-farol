from matplotlib import pyplot as plt
from matplotlib import animation

import networkx as nx
import numpy as np

from simulation import run_simulation_batch
from chung_lu import generate_chung_lu, truncated_power_law


def not_sometimes(cnts):
    maska = cnts.Always > 0
    maskn = cnts.Never > 0
    masks = cnts.Sometimes == 0
    
    return (maska & maskn & masks).sum()


def plot_freq(cnts, title, fn):
    plt.figure()
    mfp = cnts.maximal_fixed_point.sum()
    plt.hist((cnts.sum(axis=0)[:50]/(950*mfp)),bins=20,range=(0,1))
    plt.xlabel("Proportion of MFP attending",fontsize=14)
    plt.ylabel("Frequency",fontsize=14)
    plt.title(title,fontsize=16)
    plt.savefig(fn)

def mfp_freq(cnts):
    mfp = cnts.maximal_fixed_point.sum()
    return (cnts.sum(axis=0)[:50] / 950 == mfp).sum()


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
    print(f"Frequency of MFP: {mfp_freq(cnts_cm)}")
    
    plot_freq(cnts_cm, "Chung-Lu Model", "images/cm_attendance.png")
    
    # save table of attendance
    print(cnts_cm.max().max())
    (cnts_cm.sum(axis=0)[:50]/950).value_counts().sort_index().to_csv("data/cm.csv",header=False)
    
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
    print(f"Frequency of MFP: {mfp_freq(cnts_gnp)}")

    plot_freq(cnts_gnp, "Erdos-Renyi", "images/gnp_attendance.png")
    
    # save table of attendance
    (cnts_gnp.sum(axis=0)[:50]/950).value_counts().sort_index().to_csv("data/gnp.csv",header=False)

    print("\nRunning simulation on stochastic block model.")
    sizes = [100, 100, 100]
    probs = [[0.09, 0.01, 0.01], [0.01, 0.09, 0.01], [0.01, 0.01, 0.09]]
    G = nx.stochastic_block_model(sizes, probs, seed=3089277)
    A = nx.convert_matrix.to_numpy_matrix(G)
    print(f"Mean degree: {A.sum(axis=0).mean()}")

    cnts_sbm = run_simulation_batch(A, n_iter, eps, M, minfriends, minbad, batch_size, rng)
    print(f"Not sometimes: {not_sometimes(cnts_sbm)}")
    print(f"Size of maximal fixed point: {cnts_sbm.maximal_fixed_point.sum()}")
    print(f"Frequency of MFP: {mfp_freq(cnts_sbm)}")

    # plot weekly attendance
    plot_freq(cnts_sbm, "Stochastic Block Model", "images/sbm_attendance.png")

    # save table of attendance
    (cnts_sbm.sum(axis=0)[:50]/950).value_counts().sort_index().to_csv("data/sbm.csv",header=False)

    # generate plot
    always = (
        (cnts_gnp.Always > 0).sum()/N, 
        (cnts_cm.Always > 0).sum()/N, 
        (cnts_sbm.Always > 0).sum()/N)
    sometimes = (
        (cnts_gnp.Sometimes > 0).sum()/N, 
        (cnts_cm.Sometimes > 0).sum()/N, 
        (cnts_sbm.Sometimes > 0).sum()/N)
    never = (
        (cnts_gnp.Never > 0).sum()/N, 
        (cnts_cm.Never > 0).sum()/N, 
        (cnts_sbm.Never > 0).sum()/N)
    
    # create plot
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
    plt.ylabel('Fraction of Distinct Agents')
    plt.title('Proportion of Population')
    plt.xticks(index + 2*bar_width, ('Erdos\nRenyi', 'Chung\nLu', 'SBM'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/always_sometimes_never.png")


    # plot members of the maximal fixed point only
    cnts_gnp = cnts_gnp[cnts_gnp.maximal_fixed_point]
    cnts_cm = cnts_cm[cnts_cm.maximal_fixed_point]
    cnts_sbm = cnts_sbm[cnts_sbm.maximal_fixed_point]
    
    # generate plot
    always = (
        (cnts_gnp.Always > 0).mean(), 
        (cnts_cm.Always > 0).mean(), 
        (cnts_sbm.Always > 0).mean())
    sometimes = (
        (cnts_gnp.Sometimes > 0).mean(), 
        (cnts_cm.Sometimes > 0).mean(), 
        (cnts_sbm.Sometimes > 0).mean())
    never = (
        (cnts_gnp.Never > 0).mean(), 
        (cnts_cm.Never > 0).mean(), 
        (cnts_sbm.Never > 0).mean())
    
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
    plt.ylabel('Fraction of Distinct Agents')
    plt.title('Proportion of MFP')
    plt.xticks(index + 2*bar_width, ('Erdos\nRenyi', 'Configuration\nModel', 'SBM'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/always_sometimes_never_mfp.png")