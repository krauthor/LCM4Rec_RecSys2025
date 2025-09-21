import numpy as np

def estimate_choice_probs(cdf, predictions, n_samples_per_pred = 1000):
    """Estimate choice probabilities with monte-carlo based on a matrix of (predicted) utilities and a cdf"""

    n_preds = predictions.shape[0]
    n_items = predictions.shape[1]
    assert (n_samples_per_pred % n_items == 0), "Number of samples per pred are not a multiple of the number of items"

    # Sample from the cdf
    samples = []
    for i in range(n_preds): # For each prediction because the cluster keeps getting stuck at this line for the exponomial and bimix gaussian cdfs
        if n_samples_per_pred > 1000000:
            for _ in range(n_samples_per_pred//1000000):
                samples.append(cdf.sample(1000000))
        else:
            samples.append(cdf.sample(n_samples_per_pred))
    samples = np.concatenate(samples)

    samples = np.reshape(
        samples,
        (n_preds, int(n_samples_per_pred/n_items), n_items)
    )

    predictions = np.expand_dims(predictions, 1)

    u = predictions + samples

    choices = np.asarray([np.argmax(u_i, axis=-1) for u_i in u])

    # Calculate the choice probabilities
    choice_probs = np.zeros((n_preds, n_items))
    for i in range(n_preds):
        item_ids, counts_per_user = np.unique(choices[i], return_counts=True)

        choice_probs[i][item_ids] = counts_per_user / counts_per_user.sum()

    return choice_probs