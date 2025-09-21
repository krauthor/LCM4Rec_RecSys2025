import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from src.settings import MONTE_CARLO_N_SAMPLES_PER_KERNEL
from src.utils.inverse_functions import inverse_sigmoid

@tf.function()
def cartesian_product(a, b):
    return tf.reshape(tf.transpose(tf.reshape(tf.tile(tf.expand_dims(a, -1), [1, tf.shape(b)[0]]), [-1])), [-1]), tf.reshape(tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1]), [-1])


@tf.function()
def var(sample, num_samples_per_kernel):
    num_samples_per_kernel = tf.cast(num_samples_per_kernel, tf.float32)
    return tf.reduce_sum(tf.square(sample - tf.reduce_mean(sample, axis=-1, keepdims=True)), axis=-1) / num_samples_per_kernel / (num_samples_per_kernel - 1)


@tf.function()
def var3(sample, num_samples_per_kernel):
    num_samples_per_kernel = tf.cast(num_samples_per_kernel, tf.float32)
    return tf.reduce_sum(tf.pow(sample - tf.reduce_mean(sample, axis=-1, keepdims=True), 3), axis=-1) / num_samples_per_kernel / (num_samples_per_kernel - 1) / (num_samples_per_kernel - 2)


@tf.function()
def nll(cdf, users, choices, options, choices_pos, utils = None, preds = None, use_variational_inference=False, min_val = None, max_val = None, n_samples=100, num_samples_per_kernel=None, use_unbiased_variational_inference=False):
    """Calculate the negative log-likelihood (nll) for a given choice model"""

    # Determine the differences between the utilities of the choices and the utilities of the other options
    if preds is None: # If not used within the model
        # Use true or provided utilities
        utils_options = utils[tf.expand_dims(users,-1), options]

        utils_choices = utils[users, choices]
        utils_choices = tf.expand_dims(utils_choices, -1)

    else: # Wenn innerhalb des Modells verwendet
        tf.debugging.check_numerics(preds, "Nan in preds")

        utils_options = preds

        utils_choices = tf.gather(preds, choices_pos, batch_dims=1)

    utils_diffs = utils_choices - utils_options

    # Calculate the nll
    if (not use_variational_inference and not use_unbiased_variational_inference) or not hasattr(cdf, "des_points"):

        # Select values from the integrated-over space
        n_samples = tf.constant(n_samples)

        epsilon_samples = tf.cast(tf.linspace(min_val, max_val, n_samples), tf.float32)
        epsilon_samples = tf.cast(
            tf.tile(
                tf.expand_dims(tf.expand_dims(epsilon_samples, axis=0), axis=0), 
                [utils_diffs.shape[0], utils_diffs.shape[1], 1]
            ), 
            tf.float32,
        )

        cdf_inputs = epsilon_samples + tf.cast(tf.tile(tf.expand_dims(utils_diffs, axis=-1), multiples=[1, 1, n_samples]), tf.float32) # Values of the cdfs
        flat_cdf_inputs = tf.reshape(cdf_inputs, [-1]) # flatten to 1D

        # Cdf values (probability that the other options are worse )
        flat_cdf_values = cdf.cdf(flat_cdf_inputs) # Compute CDF for flattened samples
        cdf_values = tf.reshape(flat_cdf_values, [cdf_inputs.shape[0], cdf_inputs.shape[1], cdf_inputs.shape[2]]) # Reshape CDF to the original shape of diffs_samples

        # Exclude the cdf value that corresponds to the choice. Update these with the pdf values for multiplication
        # Create an index tensor for tf.tensor_scatter_nd_update
        choice_indices = tf.cast(
            tf.stack(
                [
                    tf.repeat(tf.range(cdf_values.shape[0]), cdf_values.shape[2]), 
                    tf.repeat(choices_pos, cdf_values.shape[2]), 
                    tf.tile(tf.range(cdf_values.shape[2]), [cdf_values.shape[0]]),
                ], 
                axis=1,
            ),
            tf.int64,
        )

        # Replace cdf value of chosen item with pdf value
        pdf_values = cdf.prob(tf.cast(tf.linspace(min_val, max_val, n_samples), tf.float32))
        pdf_values = tf.tile(pdf_values, [tf.cast(choice_indices.shape[0]/pdf_values.shape[0],tf.int32)])
        factors_in_integral = tf.tensor_scatter_nd_update(cdf_values, choice_indices, pdf_values) # Update the selected elements to pdf values

        # Calculate probabilities
        # Multiply factors inside integral
        product_inside_integral = tf.reduce_prod(factors_in_integral, axis=1) # Multiply pdf and cdf values
        
        # Integrate over epsilon_1
        integral_values = tf.reduce_sum(product_inside_integral, axis=1) * tf.cast((max_val-min_val)/tf.cast(n_samples, tf.float32), tf.float32) # Calculate integral; (max-min)/n_samples wegen des Lebesques-Maßes

        # Consider probability of hittng values outside integrated-over space
        outside_probability = 1 - tf.reduce_sum(cdf.prob(tf.cast(tf.linspace(min_val, max_val, n_samples), tf.float32))) * tf.cast((max_val-min_val)/tf.cast(n_samples, tf.float32), tf.float32) # Wahrscheinlichkeit außerhalb der Grenzen  abziehen?
        integral_values *= 1 - outside_probability

        # Negative log-likelihood
        nll = - tf.reduce_mean(tf.math.log(integral_values + 1e-10)) # Mean over all observations
        
    else: 
        # Sample epsilon, omit pdf from integral

        # Set constants
        num_obs = tf.shape(utils_diffs)[0]

        n_kernels = tf.constant(cdf.n_kernels)

        if num_samples_per_kernel is None:
            num_samples_per_kernel = tf.constant(MONTE_CARLO_N_SAMPLES_PER_KERNEL) # Number of samples per kernel

        num_alternatives = tf.shape(utils_diffs)[1]

        # Samples epsilon values

        des_point_mat = tf.repeat(tf.expand_dims(tf.transpose(cdf.scaled_des_points()), 1), [num_samples_per_kernel], axis=1) # Contains position of the des_points of each kernel, shape [n_kernels, num_samples_per_kernel]
        des_point_mat = tf.tile(
            tf.expand_dims(
                des_point_mat,
                axis=0
            ),
            [
                num_obs,
                1,
                1,
            ]
        )
        
        uniform_samples = tf.random.uniform(tf.shape(des_point_mat), 0., 1.) # Uniform samples. IMPORTANT: Added small values (1e-03, 1 - 1e-03) to avoid NaNs in the gradients of the variables that affect the bandwidth
        # epsilon_samples = des_point_mat + inverse_sigmoid(uniform_samples) * cdf.bandwidth() # Add to des_points
        epsilon_samples = des_point_mat + inverse_sigmoid(uniform_samples) * tf.expand_dims(tf.expand_dims(cdf.bandwidth(), -1), 0) # Add to des_points
        
        # Create inputs to the cdfs
        epsilon_samples = tf.cast( # Potential values of epsilon
            tf.tile(
                tf.expand_dims(
                    epsilon_samples, 
                    axis=-1
                ), 
                [
                        1, 
                        1,
                        1,
                        num_alternatives, 
                ]
            ), 
            tf.float32
        )
        
        cdf_inputs = epsilon_samples + tf.cast(
            tf.tile(
                tf.expand_dims(
                    tf.expand_dims(utils_diffs, axis=1), 
                    axis=1
                ), 
                multiples=[1, n_kernels, num_samples_per_kernel, 1]
            ), 
            tf.float32
        ) # Values that epsilon would need to at least take given the samples

        # Compute probability for the observations (that the other options are worse) according to the cdf
        flat_cdf_inputs = tf.reshape(cdf_inputs, [-1]) # flatten to 1D
        flat_cdf_values = cdf.cdf(flat_cdf_inputs) # Compute CDF for flattened samples
        cdf_values = tf.reshape(flat_cdf_values, cdf_inputs.shape) # Reshape CDF to the original shape of diffs_samples

        tf.debugging.check_numerics(cdf_values, "Nan in cdf_values")
        
        # Exclude the cdf value for the chosen alternative
        cart_prod_indices = cartesian_product(tf.range(n_kernels), tf.range(num_samples_per_kernel))
        choice_indices = tf.cast(
            tf.stack(
                [
                    tf.repeat(tf.range(num_obs), n_kernels*num_samples_per_kernel), 
                    tf.tile(cart_prod_indices[0], [num_obs]),
                    tf.tile(cart_prod_indices[1], [num_obs]),
                    tf.repeat(choices_pos, n_kernels*num_samples_per_kernel), 
                ], 
                axis=1
            ),
            tf.int64
        ) # Create an index tensor for tf.tensor_scatter_nd_update
        ones = tf.ones((num_obs*n_kernels*num_samples_per_kernel,)) # Set to 1, because we sampled the values according to the pdf (effectively remove pdf as a factor in the integral)
        factors_in_integral = tf.tensor_scatter_nd_update(cdf_values, choice_indices, ones) # Update the selected elements to pdf values

        # Calculate integrals per kernel
        product_inside_integral = tf.reduce_prod(factors_in_integral, axis=-1) # Multiply pdf and cdf values
        tf.debugging.check_numerics(product_inside_integral, "Nan in product_inside_integral")

        if not use_unbiased_variational_inference:
            integral_values = tf.reduce_mean(product_inside_integral, axis=-1) # Calculate integral
            tf.debugging.check_numerics(integral_values, "Nan in integral_values")

            # average over kernels w.r.t. weights
            average_over_kernels = tf.reduce_sum(cdf.weights() * integral_values, axis=1)
            tf.debugging.check_numerics(average_over_kernels, "Nan in average_over_kernels")

            # clip
            average_over_kernels = tf.clip_by_value(average_over_kernels, 1e-10, 1)

            log = tf.math.log(average_over_kernels)
            tf.debugging.check_numerics(log, "Nan in log")

            # average over observations
            nll = - tf.reduce_mean(log) # Mean over all observations
            tf.debugging.check_numerics(nll, "Nan in nll")

        elif use_unbiased_variational_inference:
            # average over kernels w.r.t. weights
            average_over_kernels = tf.reduce_sum(tf.expand_dims(tf.expand_dims(cdf.weights(), 0), -1) * product_inside_integral, axis=1)
            tf.debugging.check_numerics(average_over_kernels, "Nan in average_over_kernels")

            # Increase marginally to avoid log of zero and division by zero later (could do this later but is more efficient here)
            average_over_kernels += 1e-10

            # Average samples
            integral_values = tf.reduce_mean(average_over_kernels, axis=-1) # Calculate integral
            tf.debugging.check_numerics(integral_values, "Nan in integral_values")

            log = tf.math.log(integral_values)
            tf.debugging.check_numerics(log, "Nan in log")

            # Correct for bias https://stats.stackexchange.com/questions/384446/unbiased-estimator-for-log-left-int-px-mid-zpz-dz-right
            unbiased_log = log + 1 / 2 * var(average_over_kernels, num_samples_per_kernel) / tf.square(integral_values) - 1 / 3 * var3(average_over_kernels, num_samples_per_kernel) / tf.pow(integral_values, 3) 
            tf.debugging.check_numerics(unbiased_log, "Nan in unbiased log")

            # average over observations
            nll = - tf.reduce_mean(unbiased_log) # Mean over all observations
            tf.debugging.check_numerics(nll, "Nan in nll")

    return nll


import numpy as np
from tqdm import tqdm
import warnings
def calc_nll(cdf, utils, users, choices, options, choices_pos, min_val, max_val, 
             target_cdf=None, cdf_mse=None, preds = None, batch_size=250):
    """
    Calculate the negative log-likelihood (nll) for a given choice model.
    """
    try:

        losses_est = []
        losses_target = []
        losses_mse = []

        # users, choices, options, choices_pos = shuffle_training_data(users, choices, options, choices_pos)
        for i in tqdm(range(int(np.floor(len(users) / batch_size))), ascii=True, total=int(np.floor(len(users) / batch_size))):
            users_batch = list(users[i*batch_size:(i+1)*batch_size])
            choices_batch = list(choices[i*batch_size:(i+1)*batch_size])
            options_batch = options[i*batch_size:(i+1)*batch_size]
            choices_pos_batch = tf.expand_dims(list(choices_pos[i*batch_size:(i+1)*batch_size]), -1)
            
            if preds is None:
                preds_batch = utils[tf.expand_dims(users_batch,-1), options_batch]
            else:
                 preds_batch = preds[i*batch_size:(i+1)*batch_size]

            losses_est.append(
                 nll(
                    cdf=cdf, 
                    utils=utils, 
                    users=users_batch, 
                    choices=choices_batch, 
                    options=options_batch, 
                    choices_pos=choices_pos_batch, 
                    min_val=min_val, 
                    max_val=max_val, 
                    n_samples=2000, 
                    preds=preds_batch, 
                    use_variational_inference=False,
                )
            )
            if target_cdf is not None:
                losses_target.append(
                    nll(
                        cdf=target_cdf, 
                        utils=utils, 
                        users=users_batch, 
                        choices=choices_batch, 
                        options=options_batch, 
                        choices_pos=choices_pos_batch, 
                        min_val=min_val, 
                        max_val=max_val, 
                        n_samples=2000, 
                        preds=preds_batch, 
                        use_variational_inference=False,
                    )
                )
            if cdf_mse is not None:
                losses_mse.append(
                    nll(
                        cdf=cdf_mse, 
                        utils=utils, 
                        users=users_batch, 
                        choices=choices_batch, 
                        options=options_batch, 
                        choices_pos=choices_pos_batch, 
                        min_val=min_val, 
                        max_val=max_val, 
                        n_samples=2000, 
                        preds=preds_batch, 
                        use_variational_inference=False,
                    )
                )
                
        if target_cdf is not None:
                print(f"Negative log-likelihood of the target cdf: {np.mean(losses_target):.4f}")

        print(f"Negative log-likelihood of the estimated cdf: {np.mean(losses_est):.4f}")
        
        if cdf_mse is not None:
                print(f"Negative log-likelihood of the mse-optimized cdf: {np.mean(losses_mse):.4f}")
                
        return np.mean(losses_est)
    
    except tf.errors.ResourceExhaustedError as e:
        print("ResourceExhaustedError in calc_nll. Re-try with smaller batch size.")