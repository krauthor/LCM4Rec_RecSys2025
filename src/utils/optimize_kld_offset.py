import numpy as np

from src.Models.cdf_estimator import CDFEstimator

def optimize_KLD_offset(cdf, target_cdf):
    rough_offset = 0
    i = 0
    while np.abs(rough_offset) == i:
        _, rough_offset = scale_dist(cdf, target_cdf, rough_offset, -1, 1, 0.1, 10000) # Keep shifting the cdf until the offset is inbetween -1 and 1
        i += 1
    kl_error_dist, best_offset = scale_dist(cdf, target_cdf, rough_offset, -0.1, 0.1, 0.01, 1000000)

    return kl_error_dist, best_offset


def scale_dist(est_cdf, true_cdf, prior_offset, offset_min, offset_max, offset_interval, n_samples):
    if est_cdf.__class__ == CDFEstimator:
        est_cdf.center_mean()
        est_cdf.set_variance(true_cdf.variance()) # target_cdf.variance() differs from cdf_scale
    
    best_offset = None
    best_kld = 1e10
    for offset in np.arange(offset_min + prior_offset, offset_max + prior_offset, offset_interval):
        kld = KLD_offset(est_cdf, true_cdf, offset, n_samples)

        if kld < best_kld:
            best_kld = kld
            best_offset = offset

    print(f"Best kld score: {best_kld}, best offset: {best_offset}")
    
    if est_cdf.__class__ == CDFEstimator:
        est_cdf.set_mean(best_offset.astype(np.float32))

    return best_kld, best_offset


def KLD_offset(est_cdf, true_cdf, offset, n_samples=100000):
    x_samples = true_cdf.sample(n_samples) 

    y_true = (1 - 1e-10) * true_cdf.prob(x_samples).numpy() + 1e-10
    y_est = (1 - 1e-10) * est_cdf.prob(x_samples - offset).numpy() + 1e-10
    
    kld = np.mean(
        np.log(y_true / y_est)
    )

    return kld