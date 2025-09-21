import numpy as np

from src.utils.estimate_choice_probs import estimate_choice_probs
from src.utils.get_cdf import get_cdf

def approx_choice_probs_true_prefs(true_utils, target_cdf_name, cdf_loc=0, cdf_scale=1):
    #  Compute choice probs in user batches to avoid memory issues
    choice_probs_true_prefs = []

    print("Approx choice probs - getting cdf...", flush=True)

    cdf = get_cdf(
        cdf_name=target_cdf_name,
        loc=cdf_loc,
        scale=cdf_scale,
    )

    batch_size = 25
    for i in range(0, true_utils.shape[0], batch_size): 
        print(f"Approx choice probs - batch {i}...", flush=True)
        choice_probs_true_prefs.append(
            estimate_choice_probs(
                cdf=cdf,
                predictions=true_utils[i:i+batch_size],
                n_samples_per_pred=10_000_000,
            )
        )
        
    # Concatenate along the user axis
    choice_probs_true_prefs = np.concatenate(choice_probs_true_prefs, axis=0)

    return choice_probs_true_prefs

def compute_choice_probs_est_prefs(recommender, user_ids, items_set_B):
    # Get the estimated utilities
    est_utils = []
    for user in user_ids:
        utilities = recommender.model(np.repeat(user, len(items_set_B)), np.asarray(items_set_B))
        utilities = np.asarray(utilities).reshape(-1)
        est_utils.append(utilities)

    est_utils = np.asarray(est_utils)

    #  Compute choice probs in user batches to avoid memory issues
    choice_probs_est_prefs = []

    # Compute estimated choice probabilities
    batch_size = 25
    for i in range(0, len(user_ids), batch_size): 
        choice_probs_est_prefs.append(recommender.model.probs(est_utils[i:i+batch_size]))
        
    # Concate along the user axis
    choice_probs_est_prefs = np.concatenate(choice_probs_est_prefs, axis=0)

    return choice_probs_est_prefs

def jenson_shannon_divergence(choice_probs_true_prefs, choice_probs_est_prefs):
    """Calculates the Kullback-Leibler divergence between the true and estimated preference distributions for a set of users."""
    # Smooth the choice probabilities
    choice_probs_true_prefs = choice_probs_true_prefs * (1 - 1e-10) + 1e-10
    choice_probs_est_prefs = choice_probs_est_prefs * (1 - 1e-10) + 1e-10

    # Calculate the KL divergence
    kl = np.mean(
        np.sum(
            choice_probs_true_prefs * np.log(
                (choice_probs_true_prefs) / (choice_probs_est_prefs)
            ), 
            axis=-1
        )
    )

    reverse_kl = np.mean(
        np.sum(
            choice_probs_est_prefs * np.log(
                (choice_probs_est_prefs) / (choice_probs_true_prefs)
            ),
            axis=-1
        )
    )

    print(f"KL: {kl}")
    print(f"Reverse KL: {reverse_kl}")
    
    jensen_shannon_divergence = 0.5 * kl + 0.5 * reverse_kl

    print(f"Jensen-Shannon divergence: {jensen_shannon_divergence}")

    return {
        "kl": kl,
        "reverse_kl": reverse_kl,
        "jensen_shannon_divergence": jensen_shannon_divergence,
    }
