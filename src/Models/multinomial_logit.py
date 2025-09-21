import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, regularizers
tfkl = tf.keras.layers

from src.utils.is_running_on_cluster import is_running_on_cluster

class Recommender_Network(Model):
    
    def __init__(self, n_users, n_items, batch_size, embedding_size, l2_embs=0, n_early_stop=5, optimizer_class_name="SGD"):
        """Initialize the Recommender Network"""

        super(Recommender_Network, self).__init__()
        
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.l2_embs   = l2_embs
        self.n_early_stop = n_early_stop

        if optimizer_class_name=="SGD":
            self.optimizer_class = tf.keras.optimizers.SGD
        elif optimizer_class_name=="Adam":
            self.optimizer_class = tf.keras.optimizers.Adam
        else:
            raise ValueError(f"{optimizer_class_name} is not a valid value for optimizer_class_name")
        
        self.user_emb = tfkl.Embedding(input_dim=n_users, output_dim=embedding_size, trainable=True, embeddings_regularizer=regularizers.l2(self.l2_embs))
        self.item_emb = tfkl.Embedding(input_dim=n_items, output_dim=embedding_size, trainable=True, embeddings_regularizer=regularizers.l2(self.l2_embs))
        self.bias = tfkl.Embedding(input_dim=n_items, output_dim=1, trainable=True)

        self.out = tfkl.Dense(units=1)
        
    @tf.function()
    def call(self, users, items):
        """Feed input through the network layer by layer"""   

        # Set items with index -1 (padding) to zero, as those will be overwritten below anyway
        items_with_padding = tf.cast(items, tf.int32)

        items = tf.where(items_with_padding==tf.cast(-1, tf.int32), 0, items_with_padding)
        
        # Get embeddings
        user_embedding = self.user_emb(users)
        item_embedding = self.item_emb(items)

        # Add item-specific constants
        bias = self.bias(items)

        # Apply scalar product
        out = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), axis=1)
        
        out = out + tf.reshape(bias, out.shape)

        # Consider padding
        out = tf.where(items_with_padding==tf.cast(-1, tf.int32), -100., out)

        return out
    
    def probs(self, predictions):
        """Compute choice probabilities based on a matrix of utilities"""
        return tf.nn.softmax(predictions).numpy()
    
    @tf.function()
    def mf_loss(self, predictions, choices_pos):
        return - tf.math.reduce_mean(tf.math.log(tf.gather(tf.nn.softmax(predictions), choices_pos, batch_dims=1)))

    def generate_dataset(self, users, options, choices):
            
        user = list(np.asarray(users).astype('int32'))
        options_list = list(np.array(options.tolist()).astype('int32'))
        choice_pos = [[np.where(options[i]==choices[i])[0][0].astype('int32')] for i in range(len(users))]

        dataset = tf.data.Dataset.from_tensor_slices((user, options_list, choice_pos)).cache().shuffle(buffer_size=1000000, reshuffle_each_iteration=True).batch(self.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset

    @tf.function()
    def gradient_step(self, users, options, choices_pos, optimizer):
        """Perform a training step for the given model"""
        
        with tf.GradientTape() as tape:

            predictions = self(tf.repeat(users, tf.size(options[0])), tf.reshape(options, [-1]))
            predictions = tf.reshape(predictions, tf.shape(options))

            mf_loss = self.mf_loss(predictions, choices_pos)
            reg_loss = tf.math.reduce_mean(self.losses)
            loss = mf_loss + reg_loss
            
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    @tf.function()
    def val_step(self, users, options, choices_pos):
        """Tests the models loss over the given data set"""
        
        predictions = self(tf.repeat(users, tf.size(options[0])), tf.reshape(options, [-1]))
        predictions = tf.reshape(predictions, tf.shape(options))
        
        loss = self.mf_loss(predictions, choices_pos)
        accuracy = tf.reduce_mean(tf.cast(tf.squeeze(choices_pos)==tf.math.argmax(predictions, axis=1, output_type=tf.int32), dtype=tf.float32))
        
        return loss, accuracy

    def train_model(self, train_dataset, val_dataset, learning_rate, n_epochs):
        """Trains the given model"""  
        
        tf.keras.backend.clear_session()
        
        if self.optimizer_class == tf.keras.optimizers.Adam:
            learning_rate /= 100
            
        optimizer = self.optimizer_class(learning_rate=learning_rate)
        
        train_losses = []
        val_losses = []
        val_acc = []
        
        users_val = []
        options_val = []
        choices_val = []

        for users, options, choices in val_dataset:
            users_val.extend(list(users.numpy()))
            options_val.extend(list(options.numpy()))
            choices_val.extend(list(choices.numpy()))

        min_val_loss = 0
        steps_since_improved = 0
        best_weights = None
        
        for epoch in range(n_epochs):
            
            # Training 
            running_average_train = 0.0
            for users, options, choices in train_dataset:
                train_loss = self.gradient_step(users, options, choices, optimizer)

                if running_average_train == 0:
                    running_average_train = train_loss
                else:
                    running_average_train = 0.95 * running_average_train + (1 - 0.95) * train_loss
            
            train_losses.append(running_average_train)
        
            # Testing
            if val_dataset is not None:

                val_loss, val_accuracy= self.val_step(np.asarray(users_val), np.asarray(options_val), np.asarray(choices_val))

                val_losses.append(val_loss)
                val_acc.append(val_accuracy)

                steps_since_improved += 1

                if val_losses[-1] < min_val_loss or best_weights is None:
                    steps_since_improved = 0
                    min_val_loss = val_losses[-1]
                    best_epoch = epoch
                    best_weights = [w.numpy() for w in self.trainable_weights]

                # print train and val loss
                if not is_running_on_cluster():
                    print(f"Epoch: {epoch}, train_loss: {train_losses[-1]:.5f}, val_loss: {val_losses[-1]:.5f}, best val_loss: {min_val_loss:.5f}, steps_since_improved: {steps_since_improved}")
                
                if steps_since_improved >= self.n_early_stop:
                    print(f"early stopped at epoch {epoch}")
                    break
            # print(time()-s)

        if val_dataset is None:  
            val_losses = None
        else:
            self.set_weights(best_weights)

            print(f"Took best model from epoch {best_epoch} at train loss {train_losses[best_epoch]} and validation loss {val_losses[best_epoch]}.")

        return np.asarray(train_losses), np.asarray(val_losses)