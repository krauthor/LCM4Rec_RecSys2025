import tensorflow as tf
import numpy as np

from src.Models.multinomial_logit import Recommender_Network as Parent

class Recommender_Network(Parent):
        
    @tf.function()
    def G_i(self, u, i, lmbda, m):
        u_i = u[:,i]
        return tf.math.exp(-lmbda*tf.reduce_sum(u[:,i:m]-tf.reshape(u_i, (tf.size(u_i,),1)), axis=1)) / (tf.cast(m, dtype=tf.float32)-(tf.cast(i, dtype=tf.float32)+1)+1)

    @tf.function()
    def invert_permutation(self, indices, n_alternatives):
        row_ids = tf.repeat(tf.range(tf.shape(indices)[0]), n_alternatives)
        row_ids = tf.reshape(row_ids, [tf.shape(indices)[0],n_alternatives,1])
        idx = tf.stack([row_ids, tf.reshape(indices, [tf.shape(indices)[0],n_alternatives,1])], axis=-1)
        idx = tf.squeeze(idx)
        return tf.scatter_nd(updates = tf.tile(tf.reshape(tf.range(n_alternatives),(1,n_alternatives)), [tf.shape(indices)[0],1]), 
                                indices=idx, 
                                shape=tf.shape(indices))
    
    def probs(self, predictions, lmbda=1):
        """Compute choice probabilities based on a matrix of utilities"""
        g_discount = self.get_g_discount(options=predictions)
        m = predictions.shape[1]
        u = predictions

        u, indices = tf.nn.top_k(u, k=m, sorted=True)
        u = tf.reverse(u, axis=[1])
        indices = tf.reverse(indices, axis=[1])

        G = tf.exp( lmbda * (tf.tile(tf.expand_dims(tf.reverse(tf.range(m, dtype=tf.float32)+1, axis=[0]), 0), multiples=[tf.shape(u)[0], 1]) * u - tf.reverse(tf.cumsum(tf.reverse(u, axis=[1]), axis=1), axis=[1]) ) )
        G = G / tf.reverse(tf.range(m, dtype=tf.float32)+1, axis=[0])

        G_discounted = G[:,0:-1] * g_discount # list comp auslgelagert, geht aktuell nur mit konstanter choice set size
        G_discounted = tf.concat((tf.zeros((tf.shape(G)[0],1)), G_discounted), axis=1)

        p_sorted = G - tf.math.cumsum(G_discounted, axis=1)

        p = np.asarray([p[np.argsort(indices.numpy()[idx])] for idx, p in enumerate(p_sorted.numpy())])

        return p

    @tf.function()
    def en_loss(self, u, m, choice, lmbda, g_discount):
        """Compute the loss (cannot use self.probs because we need to track the permutation order)"""
        u, indices = tf.nn.top_k(u, k=m, sorted=True)
        u = tf.reverse(u, axis=[1])
        indices = tf.reverse(indices, axis=[1])
        
        choice_ranks = tf.gather(self.invert_permutation(indices, m), choice, axis=1, batch_dims=1)

        G = tf.exp( lmbda * (tf.tile(tf.expand_dims(tf.reverse(tf.range(m, dtype=tf.float32)+1, axis=[0]), 0), multiples=[tf.shape(u)[0], 1]) * u - tf.reverse(tf.cumsum(tf.reverse(u, axis=[1]), axis=1), axis=[1]) ) )
        G = G / tf.reverse(tf.range(m, dtype=tf.float32)+1, axis=[0])

        G_discounted = G[:,0:-1] * g_discount # list comp auslgelagert, geht aktuell nur mit konstanter choice set size
        G_discounted = tf.concat((tf.zeros((tf.shape(G)[0],1)), G_discounted), axis=1)

        Q = tf.gather(G - tf.math.cumsum(G_discounted, axis=1), choice_ranks, axis=1, batch_dims=1)

        return - tf.reduce_mean(tf.math.log(Q))
    
    @tf.function()
    def mf_loss(self, predictions, choices_pos, lmbda, g_discount):
        return self.en_loss(u=predictions, m=tf.shape(predictions)[1], choice=choices_pos, lmbda=lmbda, g_discount=g_discount)
            

    @tf.function()
    def gradient_step_tf_function(self, users, options, choices_pos, optimizer, lmbda, g_discount):
        
        with tf.GradientTape() as tape:
            predictions = self(tf.repeat(users, tf.size(options[0])), tf.reshape(options, [-1]))
            predictions = tf.reshape(predictions, tf.shape(options))

            mf_loss = self.mf_loss(predictions, choices_pos, lmbda, g_discount)
            reg_loss = tf.math.reduce_mean(self.losses)
            loss = mf_loss + reg_loss
            
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    def gradient_step(self, users, options, choices_pos, optimizer):
        g_discount = [1/(tf.shape(options)[1]-i-1) for i in range(tf.shape(options)[1]-1)]
        lmbda = 1
        return self.gradient_step_tf_function(users, options, choices_pos, optimizer, lmbda, g_discount)

    @tf.function()
    def val_step_tf_function(self, users, options, choices_pos, lmbda, g_discount):
        
        predictions = self(tf.repeat(users, tf.size(options[0])), tf.reshape(options, [-1]))
        predictions = tf.reshape(predictions, tf.shape(options))

        loss = self.mf_loss(predictions, choices_pos, lmbda, g_discount)
        accuracy = tf.reduce_mean(tf.cast(tf.squeeze(choices_pos)==tf.math.argmax(predictions, axis=1, output_type=tf.int32), dtype=tf.float32))
        
        return loss, accuracy
    
    def get_g_discount(self, options):
        return [1/(tf.shape(options)[1]-i-1) for i in range(tf.shape(options)[1]-1)]
    
    def val_step(self, users, options, choices_pos):
        g_discount = self.get_g_discount(options)
        lmbda = 1
        return self.val_step_tf_function(users, options, choices_pos, lmbda, g_discount)