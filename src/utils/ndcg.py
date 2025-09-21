import numpy as np

def ndcg(true_prefs, est_prefs):
    """Calculates the normalized discounted cumulative gain (nDCG) for a given set of true and estimated preferences."""
    true_prefs = np.asarray(true_prefs)
    est_prefs = np.asarray(est_prefs)

    length = true_prefs.shape[1]

    DCG_opt = np.sum( np.arange(length)[::-1]  / np.log2(2 + np.arange(length)) )

    DCG = []
    for i in range(est_prefs.shape[0]):
        DCG.append( np.sum( np.asarray([length - 1 - list(true_prefs[i,:]).index(j) for j in est_prefs[i,:]])  / np.log2(2 + np.arange(length)) ) )

    nDCG = np.asarray(DCG) / DCG_opt

    return nDCG