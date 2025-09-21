import numpy as np

# https://github.com/krauthorDFKI/DiscreteChoiceForBiasMitigation/blob/main/src/evaluation/process_results.py

def bootstrapped_two_sample_t_test(x, y, B=100000, twosided=False):
    """Performs a two-sample t-test on the difference of the means of two samples."""
    if np.mean(x) < np.mean(y):  # Mean value of x should be greater
        a = x
        x = y
        y = a

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    n = len(x)
    m = len(y)
    x_var = np.var(x)
    y_var = np.var(y)

    t = (x_bar - y_bar) / np.sqrt(x_var / n + y_var / m)

    z_bar = np.mean(np.concatenate((x, y)))
    x_dash = x - x_bar + z_bar
    y_dash = y - y_bar + z_bar

    t_stars = []
    x_star = np.random.choice(x_dash, size=(B, n), replace=True)
    y_star = np.random.choice(y_dash, size=(B, m), replace=True)

    x_star_bar = np.mean(x_star, axis=1)
    y_star_bar = np.mean(y_star, axis=1)
    x_star_var = np.var(x_star, axis=1)
    y_star_var = np.var(y_star, axis=1)

    t_stars = (x_star_bar - y_star_bar) / np.sqrt(x_star_var / n + y_star_var / m)

    if twosided:
        p = np.mean(np.abs(t_stars) >= np.abs(t))  # Two-sided test
    else:
        p = np.mean(np.asarray(t_stars) >= t)  # One-sided test
    return p

def bootstrapped_one_sample_t_test(x, B=200000):
    """Performs a one-sample t-test on the difference of the means of two samples."""
    p = bootstrapped_two_sample_t_test(x, np.repeat(0, len(x)), B=B)
    return p