import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions

def plot_goodnessOfFit(min_val, max_val, estimated_cdf=None, target_cdf=None, initial_cdf=None, plot_pdf=True, title=None):
    """
    Plot the estimated cdf and pdf, the target cdf 
    and pdf and the initial cdf and pdf if applicable.
    Print the Kullback-Leibler divergence of the estimated cdf and the true cdf."""
    
    _, ax = plt.subplots(figsize=(12, 4))

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if estimated_cdf is not None:
        min_val = np.min([min_val, estimated_cdf.get_support_min()])
        max_val = np.max([max_val, estimated_cdf.get_support_max()])

    if target_cdf is not None:
        try:
            min_val = np.min([min_val, target_cdf.quantile(0.001).numpy()])
            max_val = np.max([max_val, target_cdf.quantile(0.999).numpy()])
        except NotImplementedError:
            print("Target cdf does not have a quantile method.")

    plot_x_samples = np.arange(min_val, max_val, 0.01)

    # Plot the estimated cdf
    if estimated_cdf:
        ax.plot(plot_x_samples, estimated_cdf.cdf(plot_x_samples), color="blue", label='Estimated cdf')
    # Plot the target cdf
    if target_cdf:
        ax.plot(plot_x_samples, target_cdf.cdf(plot_x_samples), color="red", label='Target cdf')
    # Plot the initial cdf
    if initial_cdf:
        ax.plot(plot_x_samples, initial_cdf.cdf(plot_x_samples), color="grey", label='Initial cdf')

    # Plot the pdf
    if plot_pdf:
        # Plot the estimated pdf
        if estimated_cdf:
            ax.plot(plot_x_samples, estimated_cdf.prob(plot_x_samples), color="blue", linestyle="--", label='Estimated pdf')
        # Plot the target pdf
        if target_cdf:
            ax.plot(plot_x_samples, target_cdf.prob(plot_x_samples), color="red", linestyle="--", label='Target pdf')
        # Plot the initial pdf
        if initial_cdf:
            ax.plot(plot_x_samples, initial_cdf.prob(plot_x_samples), color="grey", linestyle="--", label='Initial pdf')

    # Plot the weights of the estimated cdf if applicable
    if hasattr(estimated_cdf, "des_points"):
        ax.bar(estimated_cdf.scaled_des_points(), estimated_cdf.weights(), color="blue", width=estimated_cdf.bandwidth())

    # Output the Kullback-Leibler divergence of the estimated cdf and the true cdf
    if estimated_cdf is not None:
        print(f"Kullback-Leibler divergence of the estimated cdf and the true cdf: {estimated_cdf.kld(target_cdf)}")

    ax.legend()

    # add title if applicable
    if title is not None:
        plt.title(title)

    plt.show()