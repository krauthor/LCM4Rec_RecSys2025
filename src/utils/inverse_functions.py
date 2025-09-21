import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
    
@tf.function()
def inverse_softplus(y):
    """Returns the inverse of the softplus function at x"""
    return tf.math.log(tf.exp(y) - 1.)

        
@tf.function()
def inverse_sigmoid(y):
    """Returns the inverse of the softplus function at y"""
    y = tf.clip_by_value(y, 1e-10, 1 - 1e-7)
    
    return tf.math.log(y / (1. - y))