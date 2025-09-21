import tensorflow as tf
import numpy as np

from src.Models.binary_logit import Recommender_Network as Parent
from src.utils.is_running_on_cluster import is_running_on_cluster

class Recommender_Network(Parent):
    
    def __init__(self, n_users, n_items, batch_size, embedding_size, l2_embs=0, n_early_stop=5, n_negative_samples=3, optimizer_class_name="SGD"):
        
        self.n_negative_samples = n_negative_samples

        self.n_items = n_items
        self.set_A = np.arange(int(n_items/2))
        self.size_set_A = len(self.set_A)
        self.set_B = np.arange(int(n_items/2), n_items)
        self.size_set_B = len(self.set_B)

        super(Recommender_Network, self).__init__(
            n_users=n_users,
            n_items=n_items,
            batch_size=batch_size,
            embedding_size=embedding_size,
            l2_embs=l2_embs,
            n_early_stop=n_early_stop,
            optimizer_class_name=optimizer_class_name,
        )

    @tf.function()
    def generate_dataset(self, users, items, ratings):
        users = tf.cast(users, tf.int32)
        items = tf.cast(items, tf.int32)
        ratings = tf.cast(ratings, tf.int32)
            
        dataset = tf.data.Dataset.from_tensor_slices((users, items, ratings)).cache().shuffle(buffer_size=1000000, reshuffle_each_iteration=True).batch(self.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset


    def sample_negatives(self, users, validation):

        # import time
        # s = time.time()

        if validation:
            items_consumed_by_user = self.items_consumed_by_user_val
        else:
            items_consumed_by_user = self.items_consumed_by_user_train

        unique_users = np.unique(users)

        pos_users = [[user_idx] * len(items_consumed_by_user[user_idx]) for user_idx in unique_users]
        pos_users = np.concatenate(pos_users)

        pos_items = [items_consumed_by_user[user_idx] for user_idx in unique_users]
        pos_items = np.concatenate(pos_items)

        neg_users = [[user_idx] * len(items_consumed_by_user[user_idx]) * self.n_negative_samples for user_idx in unique_users]
        neg_users = np.concatenate(neg_users)

        # Instead of the above, sample from self.set_size_set_A - 1, add self.set_size_A if the pos item is from set B and then add 1 to the result if the sampled item is greater or equal to the positive item
        assert self.size_set_A == self.size_set_B, "This implementation only works for equal set A and set B sizes"
        neg_items = np.random.choice(self.size_set_A - 1, len(neg_users), replace=True)
        neg_items = np.where(np.repeat(pos_items, 3) >= self.size_set_A, neg_items + self.size_set_A, neg_items)
        neg_items = np.where(neg_items >= np.repeat(pos_items, 3), neg_items + 1, neg_items)

        data_pos = np.stack([pos_users, pos_items, np.ones(len(pos_users))], axis=1)
        data_neg = np.stack([neg_users, neg_items, np.zeros(len(neg_users))], axis=1)

        data = np.concatenate([data_pos, data_neg]).astype(np.int32)

        np.random.shuffle(data)

        return data


    def train_model(self, data_train, data_val, learning_rate, n_epochs):
        
        tf.keras.backend.clear_session()
        
        if self.optimizer_class == tf.keras.optimizers.Adam:
            learning_rate /= 100
        
        optimizer = self.optimizer_class(learning_rate=learning_rate)
        
        train_losses = []
        val_losses = []
        val_acc = []

        min_val_loss = 0
        steps_since_improved = 0
        best_weights = None
        
        from collections import defaultdict
        self.items_consumed_by_user_train = defaultdict(list)
        for k, v in data_train[:,[0,2]]:
            self.items_consumed_by_user_train[k].append(v)
        for key in self.items_consumed_by_user_train.keys():
            self.items_consumed_by_user_train[key] = list(set(self.items_consumed_by_user_train[key])) # Remove duplicates
        
        if data_val is not None:
            self.items_consumed_by_user_val = defaultdict(list)
            for k, v in data_val[:,[0,2]]:
                self.items_consumed_by_user_val[k].append(v)
            for key in self.items_consumed_by_user_val.keys():
                self.items_consumed_by_user_val[key] = list(set(self.items_consumed_by_user_val[key])) # Remove duplicates

        for epoch in range(n_epochs):  
            train_dataset = self.generate_dataset(*(self.sample_negatives(users=data_train[:,0], validation=False)).T)
            if data_val is not None:
                val_dataset = self.sample_negatives(users=data_val[:,0], validation=True)
            else: 
                val_dataset = None
        
            users_val = list(val_dataset[:, 0])
            options_val = list(val_dataset[:, 1])
            choices_val = list(val_dataset[:, 2])

            running_average_train = 0.0
            for users, items, choices in train_dataset:
                train_loss = self.gradient_step(users, items, choices, optimizer)
                
                if running_average_train == 0:
                    running_average_train = train_loss
                else:
                    running_average_train = 0.95 * running_average_train + (1 - 0.95) * train_loss
            
            train_losses.append(running_average_train)
            
            if val_dataset is not None:

                val_loss, val_accuracy= self.val_step(np.asarray(users_val), np.asarray(options_val), np.asarray(choices_val))
                
                val_losses.append(val_loss)
                val_acc.append(val_accuracy)

                steps_since_improved += 1

                if val_losses[-1] < min_val_loss or best_weights is None: 
                    steps_since_improved = 0
                    min_val_loss = val_losses[-1]
                    best_epoch = epoch
                    best_weights = [w.numpy() for w in self.trainable_weights] # get_weights() wirft Fehler aus. Mit .numpy() erhält man eine echte Kopie
                # print train and val loss
                if not is_running_on_cluster():
                    print(f"Epoch: {epoch}, train_loss: {train_losses[-1]:.5f}, val_loss: {val_losses[-1]:.5f}, best val_loss: {min_val_loss:.5f}, steps_since_improved: {steps_since_improved}")
                
                if steps_since_improved >= self.n_early_stop:
                    print(f"early stopped at epoch {epoch}")
                    break

        if val_dataset is None:  
            val_losses = None
        else:
            self.set_weights(best_weights)

            print(f"Took best model from epoch {best_epoch} at train loss {train_losses[best_epoch]} and validation loss {val_losses[best_epoch]}.")

        return np.asarray(train_losses), np.asarray(val_losses)