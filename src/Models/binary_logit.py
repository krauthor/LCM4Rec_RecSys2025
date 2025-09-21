import tensorflow as tf
import numpy as np

from src.Models.multinomial_logit import Recommender_Network as Parent

class Recommender_Network(Parent):

    def generate_dataset(self, data):

        data = data.astype('int32')
            
        dataset = tf.data.Dataset.from_tensor_slices((list(data[:,0]), list(data[:,1]), list(data[:,2]))).cache().shuffle(buffer_size=1000000, reshuffle_each_iteration=True).batch(self.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset


    @tf.function()
    def mf_loss(self, ratings, predictions):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(ratings, predictions)
    
    def probs(self, predictions):
        """Compute the probability that exactly that item gets chosen."""

        # Apply sigmoid
        choice_probs = tf.sigmoid(predictions).numpy()

        # Compute column-wise choice probs
        def get_exactly_one_choice_prob(choice_probs, choices_pos):
            choice_probs_choice = np.take_along_axis(choice_probs, choices_pos, axis=1) # choice_probs[:, np.reshape(choices_pos_test, -1)]

            final_choice_probs = 1 - choice_probs
            np.put_along_axis(final_choice_probs, choices_pos, choice_probs_choice, axis=1)

            return np.prod(final_choice_probs, axis=1)
        
        choice_probs_by_pos = []
        for pos in range(predictions.shape[1]):
            choice_probs_by_pos.append(get_exactly_one_choice_prob(choice_probs, pos * np.ones((predictions.shape[0], 1)).astype(np.int32)))

        # Concatenate
        choice_probs_by_pos = np.asarray(choice_probs_by_pos).T

        # Condition on the event that exactly one item gets chosen
        choice_probs_by_pos = choice_probs_by_pos / np.expand_dims(choice_probs_by_pos.sum(axis=1), 1)

        return choice_probs_by_pos
        

    @tf.function()
    def gradient_step(self, users, items, ratings, optimizer):
        
        with tf.GradientTape() as tape:
            predictions = self(users, items)
            mf_loss = self.mf_loss(ratings, predictions)
            reg_loss = tf.math.reduce_mean(self.losses)
            loss = mf_loss + reg_loss
            
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss


    @tf.function()
    def val_step(self, users, items, ratings):
        
        predictions = self(users, items)
        loss = self.mf_loss(ratings, predictions)
        
        probs = 1 / (1 + tf.exp(-predictions))
        accuracy = tf.reduce_mean(tf.cast(tf.cast(probs  > 0.5, tf.bool) == tf.cast(ratings, tf.bool), tf.float32))
        
        return loss, accuracy