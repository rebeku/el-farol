import networkx as nx
import numpy as np
import pandas as pd


def run_simulation(X, A, n_iter, eps, M, minfriends, minbad, rng):
    Xs = [X]

    for _ in range(n_iter):
        if X.sum() < M:
            # go if enough of your friends went last week
            # whether or not you went
            X = (A.dot(X) > minfriends)
        else:
            X = (A.dot(X) > minbad)

        # a few extras attend randomly
        X += rng.uniform(size=(len(A),1)) < eps
        
        # a few do not attend randomly
        X = np.bitwise_and(X, rng.uniform(size=(len(A),1)) > eps)
        
        Xs.append(X)

    return np.hstack(Xs)


def maximal_fixed_point(G, minfriends):
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
            run_simulation(X, A, n_iter, eps, M, minfriends, minbad, rng))

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
        nx.convert_matrix.from_numpy_array(A), minfriends
    )

    cnts["maximal_fixed_point"] = pd.Series(range(len(A))).apply(
        lambda x: x in g.nodes)

    return cnts