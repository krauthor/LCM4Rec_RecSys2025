import os

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display

from src.utils.inverse_functions import inverse_sigmoid, inverse_softplus

class CDFEstimator(Model):
    def __init__(
            self, 
            min_init = -1., 
            max_init = 1., 
            n_kernels = 20, 
            learn_bandwidth = True,
        ):
        """Initializes the CDF Estimator.
        
        Args:
            min_init (float): The lower bound of the initial support of the distribution.
            max_init (float): The upper bound of the initial support of the distribution.
            n_kernels (int): The number of design points.
            learn_bandwidth (bool): Whether or not to lern the bandwidth.
        """

        print("Initializing CDF Estimator")

        super(CDFEstimator, self).__init__()

        self.gif_plot_queue = []

        self.n_kernels = n_kernels

        self.des_points =  tf.Variable(np.arange(-1, 1+(1/n_kernels), 2/(n_kernels-1)), dtype=tf.float32, trainable=False, name="des_points") # Keep trainable false as setting it to true creates too many local minima

        self.offset = tf.Variable(0., dtype=tf.float32, trainable=False, name="offset")

        self.alphas = tf.Variable(np.zeros(n_kernels), dtype=tf.float32, trainable=True, name="alphas") # Für Gewichte der Design Punkte (Stufenhöhen)

        self.learn_bandwidth = learn_bandwidth
        if learn_bandwidth:            
            self.beta = tf.Variable(tf.zeros(n_kernels), dtype=tf.float32, trainable=True, name="beta") # Für Bandbreite
        else:
            self.beta = tf.Variable(tf.zeros(n_kernels), dtype=tf.float32, trainable=False, name="beta") # Für Bandbreite
        
        self.beta_lower_bound = tf.constant(-3., dtype=tf.float32, name="beta_lower_bound")
        self.beta_upper_bound = tf.constant(5., dtype=tf.float32, name="beta_upper_bound")

        self.gamma = tf.Variable(inverse_softplus((max_init-min_init) / 2), dtype=tf.float32, trainable=True, name="gamma")    

        self.gamma_lower_bound = tf.constant(-1., dtype=tf.float32, name="gamma_lower_bound")
        self.gamma_upper_bound = tf.constant(10., dtype=tf.float32, name="gamma_upper_bound")

        self.compile()
    

    @tf.function
    def weights(self):
        """Returns the weights of the design points."""

        weights = tf.exp(self.alphas)
        weights /= tf.reduce_sum(weights) # Weights should add up to 1

        return weights
    

    @tf.function
    def width(self):
        """Returns the width of the cdf."""
        return tf.math.softplus(self.gamma)
    
    
    @tf.function
    def scaled_des_points(self):
        """Returns the design points scaled by the width and the offset."""
        return self.des_points * self.width() + self.offset
    

    @tf.function
    def bandwidth(self):
        """Returns the bandwidths of the design points."""
        if self.learn_bandwidth:
            return tf.math.softplus(self.beta) * self.width() / self.n_kernels
        else:
            return tf.ones(tf.shape(self.beta)) * self.width() / self.n_kernels
    

    @tf.function()
    def apply_variable_bounds(self):
        """Clips the variables inside the bounds."""
        self.beta.assign(
            tf.clip_by_value(
                self.beta, 
                self.beta_lower_bound + 1e-3, 
                self.beta_upper_bound - 1e-3
            )
        )

        self.gamma.assign(
            tf.clip_by_value(
                self.gamma, 
                self.gamma_lower_bound + 1e-3, 
                self.gamma_upper_bound - 1e-3
            )
        )


    @tf.function
    def cdf(self, x):
        """Returns the cdf of the distribution at x."""

        x = tf.cast(tf.expand_dims(x, axis=1), tf.float32)

        cdf = tf.reduce_sum(self.weights() * tf.sigmoid((x - self.scaled_des_points()) / self.bandwidth()), axis=1)

        return cdf


    @tf.function
    def prob(self, x):
        """Returns the pdf of the distribution at x."""

        x = tf.expand_dims(tf.cast(x, tf.float32), axis=1)

        input = (x - self.scaled_des_points()) / self.bandwidth()

        pdf = tf.reduce_sum(self.weights() * tf.sigmoid(input) * (1 - tf.sigmoid(input)) / self.bandwidth(), axis=1) # https://stackoverflow.com/questions/10626134/derivative-of-sigmoid und das 1/h von https://www.wolframalpha.com/input?i=derive+1%2F%281%2Bexp%28-%28x-u%29%2Fh%29%29
        
        return pdf


    def sample(self, n_samples):
        """Returns n_samples samples from the distribution."""

        # Sample des_point first
        des_point_samples = np.random.choice(
            a=self.n_kernels, 
            p=self.weights().numpy() / np.sum(self.weights().numpy()), 
            size=n_samples, 
            replace=True,
        )
        
        epsilon_samples = (
            tf.gather(self.scaled_des_points(), des_point_samples)
            + inverse_sigmoid(tf.random.uniform((n_samples,), 0, 1))
            * tf.gather(self.bandwidth(), des_point_samples)
        )
    
        return epsilon_samples
    

    @tf.function()
    def mean(self):
        """Returns the distribution's mean."""

        n_samples = 1000

        min_val = self.get_support_min()
        max_val = self.get_support_max()

        x = tf.cast(tf.linspace(min_val, max_val, n_samples), tf.float32)

        pdf = self.prob(x)

        measure = tf.cast((max_val - min_val) / n_samples, tf.float32)

        mean = tf.reduce_sum(x * pdf * measure)

        return mean
    

    @tf.function()
    def set_mean(self, val):
        """Sets the distribution's mean to val."""

        self.offset.assign(
            tf.cast(
                self.offset + val- self.mean(), 
                tf.float32,
            )
        )
    

    @tf.function()
    def center_mean(self):
        """Sets the distribution's mean to 0."""
        self.set_mean(0)
    

    @tf.function()
    def variance(self):
        """Returns the distribution's variance."""

        n_samples = 1000

        min_val = self.get_support_min()
        max_val = self.get_support_max()

        x = tf.cast(tf.linspace(min_val, max_val, n_samples), tf.float32)

        pdf = self.prob(x)

        measure = tf.cast((max_val - min_val) / n_samples, tf.float32)
        
        first_momentum_squared = tf.pow( tf.reduce_sum((x * pdf * measure)), 2)

        second_momentum = tf.reduce_sum(
            tf.pow(x, 2) * pdf * measure
        )

        return second_momentum - first_momentum_squared
    

    @tf.function()
    def set_variance(self, val):
        """Sets the distribution's variance to val."""
        var = self.variance()

        rel_sqrt = tf.sqrt(tf.cast(val, tf.float32)) / tf.sqrt(var)
        
        self.gamma.assign(tf.cast(inverse_softplus(self.width() * rel_sqrt), tf.float32)) # Since the bandwidth is proportional to the width, and the width is adjusted by gamma, we do not need to adjust beta here


    @tf.function()
    def variance_to_one(self):
        """Sets the distribution's variance to 1."""
        self.set_variance(1)


    @tf.function()
    def standardize(self):
        """Sets the distribution's variance to 1 and the mean to 0."""

        self.variance_to_one()

        self.center_mean()

    
    @tf.function(reduce_retracing=True)
    def mse(self, samples, target_cdf_scores):
        """Returns the mean squared error between the cdf of the distribution and the target_cdf at the samples."""

        samples = tf.cast(samples, tf.float32)

        cdf_scores = tf.cast(self.cdf(samples), tf.float32)

        return tf.reduce_mean(tf.math.pow(cdf_scores - target_cdf_scores, 2))
    

    def kld(self, target_cdf, n_samples=10000):
        """Returns the Kullback-Leibler-Divergence between the cdf of the distribution and the target_cdf."""
        
        samples = self.sample(n_samples)

        p = self.prob(samples) * (1 - 1e-10) + 1e-10 # Smooth to avoid p=0

        q = target_cdf.prob(samples) * (1 - 1e-10) + 1e-10 # Smooth to avoid q=0

        return tf.reduce_mean(tf.math.log(p / q))
    

    @tf.function()
    def get_support_min(self):
        """Returns the approximate minimum of the support of the distribution."""
        return tf.cast(tf.reduce_min(inverse_sigmoid(1e-03) * self.bandwidth() + self.scaled_des_points()), tf.float32)


    @tf.function()
    def get_support_max(self):
        """Returns the approximate maximum of the support of the distribution."""
        return tf.cast(tf.reduce_max(inverse_sigmoid(1 - 1e-03) * self.bandwidth() + self.scaled_des_points()), tf.float32) #  lassen hier / self.weights() weg, weil sonst error (macht mathematisch sinn, ist alles ok)


    def plot(self, n_samples=1000, show=False, supplemental_info=None):
        """Plots the estimated cdf and pdf."""

        # Get plot's boundaries
        min_val = self.get_support_min()
        max_val = self.get_support_max()

        x = tf.linspace(min_val, max_val, n_samples)

        fig, ax = plt.subplots()

        # Plot cdf and pdf of the distribution
        plt.plot(x, self.cdf(x), color="blue", linestyle="")
        plt.plot(x, self.prob(x), color="blue", linestyle="--")
        
        # Plot weights and bandwidths
        ax.bar(self.scaled_des_points(), self.weights(), color="blue", width=self.bandwidth())

        if supplemental_info is not None:
            plt.title(supplemental_info)

        if show:
            plt.show()
            return None
        else:
            fig.canvas.draw()
            im = PIL.Image.frombytes(
                'RGB', 
                fig.canvas.get_width_height(), 
                fig.canvas.tostring_rgb()
            )
            plt.close("all")
            return im


    def add_plot_to_gif_queue(self, n_samples=1000, supplemental_info=None):
        """Adds a plot of the distribution to the gif queue."""

        self.gif_plot_queue.append(
            self.plot(
                n_samples=n_samples, 
                show=False, 
                supplemental_info=supplemental_info
            )
        )


    def clear_gif_plot_queue(self):
        """Clears the gif plot queue."""
        self.gif_plot_queue = []


    def save_gif(self, path):
        """Saves the gif of the distribution to path."""

        # Check if path exists
        if not os.path.exists(os.path.dirname(path)):
            raise FileNotFoundError(f"Path {path} does not exist. Current working directory: {os.getcwd()}")

        self.gif_plot_queue[0].save(
            path, 
            save_all=True, 
            append_images=self.gif_plot_queue[1:], 
            duration=250, 
            loop=0,
        )


    def show_gif(self, filename="data/gifs/tmp.gif"):
        """Shows the gif of the distribution."""    
        # Temporarily save cdf gif
        self.save_gif(filename)

        # Load and display cdf gif
        display(Image(filename=filename))