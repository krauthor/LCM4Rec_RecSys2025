import tensorflow as tf

from src.Models.binary_logit_negative_sampling import Recommender_Network as Parent

class Recommender_Network(Parent):
    # Based on the original implementation at https://dl.acm.org/doi/10.1145/3604915.3608783, https://github.com/asash/gSASRec-pytorch (Scroll down in the ReadMe file)
    
    def __init__(self, n_users, n_items, batch_size, embedding_size, l2_embs=0, n_early_stop=5, optimizer_class_name="SGD", n_negative_samples=3, t=1):

        self.alpha = n_negative_samples / ((n_items/2) - 1) # n_items/2 because we always sample from either set A or set B
        self.beta = tf.constant(
            self.alpha * ((1 - 1/self.alpha)*t + 1/self.alpha), 
            dtype=tf.float64,
        )
        self.t = t

        super(Recommender_Network, self).__init__(
            n_users=n_users, 
            n_items=n_items, 
            batch_size=batch_size, 
            embedding_size=embedding_size,
            l2_embs= l2_embs, 
            n_early_stop=n_early_stop, 
            n_negative_samples=n_negative_samples, 
            optimizer_class_name=optimizer_class_name,
        )


    # We use the softmax function to calculate the probabilities from the parent class because the sigmoid probabilities do not sum to 1
    def probs(self, predictions):
        """Compute choice probabilities based on a matrix of utilities"""
        # Choice probs
        probs = tf.sigmoid(predictions)
        return (probs / tf.reduce_sum(probs, axis=1, keepdims=True)).numpy()


    @tf.function()
    def mf_loss(self, ratings, predictions):
        ratings = tf.cast(ratings, tf.float64)

        positive_logits = tf.cast(predictions, 'float64') #use float64 to increase numerical stability
        negative_logits = predictions
        eps = 1e-10
        positive_probs = tf.clip_by_value(tf.sigmoid(positive_logits), eps, 1-eps)
        positive_probs_adjusted = tf.clip_by_value(tf.math.pow(positive_probs, -self.beta), 1+eps, tf.float64.max)
        to_log = tf.clip_by_value(tf.math.divide(1.0, (positive_probs_adjusted  - 1)), eps, tf.float64.max)
        positive_logits_transformed = tf.math.log(to_log)
        negative_logits = tf.cast(negative_logits, 'float64')
        logits = ratings * positive_logits_transformed + (1 - ratings) * negative_logits # Edited because it works better with our framework

        return tf.cast(tf.keras.losses.BinaryCrossentropy(from_logits=True)(ratings, logits), tf.float32)
        