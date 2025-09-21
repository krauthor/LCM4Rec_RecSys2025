import sys
import warnings

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tensorflow_probability as tfp
tfd = tfp.distributions

from src.Models.cdf_estimator import CDFEstimator
from src.Models.nll import nll
from src.Models.approx_gradient import approx_gradient
from src.Models.multinomial_logit import Recommender_Network as Parent
from src.utils.estimate_choice_probs import estimate_choice_probs

from src.utils.is_running_on_cluster import is_running_on_cluster

class Recommender_Network(Parent):
    
    def __init__(
            self, 
            n_users, 
            n_items, 
            batch_size, 
            embedding_size, 
            n_kernels=20, 
            l2_embs=0, 
            n_early_stop=3, 
            optimizer_class_name="SGD",
            normalize_bias=True, 
            learn_bandwidth=True,
            plot_cdf_gif=True,
        ):
        
        """Initialize the Model"""

        warnings.warn("L2-regularisierung mit neuer Berechnung der Gradienten noch nicht implementiert.")

        super(Recommender_Network, self).__init__(
            n_users=n_users,
            n_items=n_items,
            batch_size=batch_size,
            embedding_size=embedding_size,
            l2_embs=l2_embs,
            n_early_stop=n_early_stop,
            optimizer_class_name=optimizer_class_name,
        )

        self.n_kernels = n_kernels
        
        self.normalize_bias = normalize_bias

        self.plot_cdf_gif = plot_cdf_gif

        if not normalize_bias and l2_embs > 0:
            print("Warning: When setting l2_embs > 0, normalize_bias=True is required for ensuring stability.")
            
        self.cdf = CDFEstimator(min_init=-1., max_init=1., n_kernels=n_kernels, learn_bandwidth=learn_bandwidth)

        
    def apply_bias_normalization(self):
        """Scale the bias vector to the interval [0,1]"""
        self.bias.weights[0].assign(
            (self.bias.weights[0] - tf.reduce_min(self.bias.weights[0])) 
            / (tf.reduce_max(self.bias.weights[0]) - tf.reduce_min(self.bias.weights[0]) + 1e-10)
        )

    
    def probs(self, predictions, n_samples_per_pred = 10_000_000):
        """Compute choice probabilities based on a matrix of utilities"""

        choice_probs = estimate_choice_probs(
            cdf=self.cdf,
            predictions=predictions,
            n_samples_per_pred=n_samples_per_pred
        )

        return choice_probs


    @tf.function()
    def mf_loss(self, predictions, choices_pos):
        """Return the estimated negative log-likelihood"""

        return nll(
            cdf=self.cdf, 
            users=None,
            choices=None,
            options=None, 
            choices_pos=choices_pos, 
            preds=predictions, 
            use_variational_inference=True,
            use_unbiased_variational_inference=True,
        )
    

    @tf.function()
    def gradient_step(self, users, options, choices, optimizer_model, optimizer_cdf):
        """Perform a training step for the given model"""
        
        gradients, loss = approx_gradient(
            cdf=self.cdf, 
            model=self,
            users=users,
            options=options, 
            choices_pos=choices, 
        )

        # Distribute the learnable variables on the two optimizers
        # Yes, there are prettier ways of doing is and I do not care
        var_names = [v.name for v in self.trainable_variables]

        for var_name, grad in zip(var_names, gradients):
            tf.debugging.check_numerics(grad, f"Nan in gradients for variable {var_name}. CDF width: {self.cdf.width()}")

        model_names = [name for name in var_names if "_emb" in name or "bias" in name]
        model_grads = [gradients[i] for i in range(len(var_names)) if var_names[i] in model_names] 
        model_vars = [self.trainable_variables[i] for i in range(len(var_names)) if var_names[i] in model_names] 
        optimizer_model.apply_gradients(zip(model_grads, model_vars))

        cdf_names = [name for name in var_names if not name in model_names]
        cdf_grads = [gradients[i] for i in range(len(var_names)) if var_names[i] in cdf_names] 
        cdf_vars = [self.trainable_variables[i] for i in range(len(var_names)) if var_names[i] in cdf_names] 
        optimizer_cdf.apply_gradients(zip(cdf_grads, cdf_vars))

        self.cdf.apply_variable_bounds()

        if self.normalize_bias:
            self.apply_bias_normalization()

        return loss


    @tf.function()
    def val_step(self, users, items, choices):
        """Compute the validation loss"""

        predictions = self(tf.repeat(users, tf.size(items[0])), tf.reshape(items, [-1]))
        predictions = tf.reshape(predictions, tf.shape(items))

        loss = self.mf_loss(predictions, choices)

        accuracy = tf.reduce_mean(tf.cast(tf.squeeze(choices)==tf.math.argmax(predictions, axis=1, output_type=tf.int32), dtype=tf.float32))
        
        return loss, accuracy
    

    def train_model(self, train_dataset, val_dataset, learning_rate_model, learning_rate_cdf, n_epochs):
        """Train the model"""  
        
        tf.keras.backend.clear_session()
        
        if self.optimizer_class == tf.keras.optimizers.Adam:
            learning_rate_model /= 100
            learning_rate_cdf /= 100
        
        optimizer_model = self.optimizer_class(learning_rate=learning_rate_model)
        optimizer_cdf = self.optimizer_class(learning_rate=learning_rate_cdf)
        
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

        steps_since_improved = 0
        min_val_loss = 10000

        for epoch in range(n_epochs):
            import time
            s = time.time()

            # Training
            running_average_train = 0.
            pbar = tqdm(train_dataset, ascii=True, leave=False, file=sys.stdout, disable=is_running_on_cluster())
            for users, options, choices in pbar:

                train_loss = self.gradient_step(users, options, choices, optimizer_model, optimizer_cdf)
                
                if running_average_train == 0:
                    running_average_train = train_loss
                else:
                    running_average_train = 0.95 * running_average_train + (1 - 0.95) * train_loss

                tf.debugging.check_numerics(self.cdf.alphas, "Nan in alphas")

                pbar.set_description(f"Epoch: {epoch}, Model Loss: {running_average_train:.4f}")

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
                    best_weights = [w.numpy() for w in self.trainable_weights] # get_weights() return error. .numpy() returns a true copy

                if not is_running_on_cluster():
                    print(f"Epoch: {epoch}, train_loss: {train_losses[-1]:.4f}, val_loss: {val_losses[-1]:.4f}, best val_loss: {min_val_loss:.4f}, steps_since_improved: {steps_since_improved}, cdf width: {self.cdf.width():.2f}")
                
                # Plot cdf
                if self.plot_cdf_gif:
                    self.cdf.add_plot_to_gif_queue(supplemental_info=f"Full Training - Epoch: {epoch} - NLL Validation: {val_losses[-1]:.2f}")

                if steps_since_improved >= self.n_early_stop:
                    print(f"early stopped at epoch {epoch}")
                    break
            else:
                if not is_running_on_cluster():
                    print(f"Epoch: {epoch}, train_loss: {train_losses[-1]:.5f}")

                # Plot cdf
                if self.plot_cdf_gif:
                    self.cdf.add_plot_to_gif_queue(supplemental_info=f"Full Training - Epoch: {epoch}")
        
            if not is_running_on_cluster():
                    print(f"Duration past epoch: {time.time() - s:.2f}")
                    
        if val_dataset is None:
            val_losses = None
            best_weights = [w.numpy() for w in self.trainable_weights] # get_weights() return error. .numpy() returns a true copy
            best_epoch=epoch
            min_val_loss=train_losses[-1]

        for idx, var in enumerate(self.trainable_variables):
            var.assign(best_weights[idx])

        print(f"Took best model from epoch {best_epoch} at train loss {train_losses[best_epoch]} and validation loss {val_losses[best_epoch]}.")

        return np.asarray(train_losses), np.asarray(val_losses)