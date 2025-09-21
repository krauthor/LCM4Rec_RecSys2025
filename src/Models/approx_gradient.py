import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from src.settings import MONTE_CARLO_N_SAMPLES_PER_KERNEL
from src.utils.inverse_functions import inverse_sigmoid

@tf.function()
def cartesian_product(a, b):
    return tf.reshape(tf.transpose(tf.reshape(tf.tile(tf.expand_dims(a, -1), [1, tf.shape(b)[0]]), [-1])), [-1]), tf.reshape(tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1]), [-1])


@tf.function()
def compute_g_k(cdf, utils_diffs, epsilon_samples, choices_pos, num_samples_per_kernel, n_kernels, num_obs, drop_denstity_from_product):
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
    
    if drop_denstity_from_product:
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
            tf.int64,
        ) # Create an index tensor for tf.tensor_scatter_nd_update
        ones = tf.ones((num_obs*n_kernels*num_samples_per_kernel,)) # Set to 1, because we sampled the values according to the pdf (effectively remove pdf as a factor in the integral)
        factors_in_integral = tf.tensor_scatter_nd_update(cdf_values, choice_indices, ones) # Update the selected elements to pdf values

    else: 
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
            tf.int64,
        ) # Create an index tensor for tf.tensor_scatter_nd_update
        ones = tf.reshape(cdf.prob(flat_cdf_inputs), cdf_inputs.shape) # Set to 1, because we sampled the values according to the pdf (effectively remove pdf as a factor in the integral)
        ones = tf.gather_nd(ones, choice_indices) # TODO: Check if this is correct
        factors_in_integral = tf.tensor_scatter_nd_update(cdf_values, choice_indices, ones) # Update the selected elements to pdf values


    # Calculate integrals per kernel
    product_inside_integral = tf.reduce_prod(factors_in_integral, axis=-1) # Multiply pdf and cdf values
    tf.debugging.check_numerics(product_inside_integral, "Nan in product_inside_integral")

    return product_inside_integral


@tf.function()
def approx_grad_log_likelihood(
        model,
        cdf, 
        users,
        options,
        choices_pos,
        num_samples_per_kernel=None,
    ):

    # Estimate utilities
    preds = model(
        users=tf.repeat(users, tf.size(options[0])), 
        items=tf.reshape(options, [-1]),
    )
    
    preds = tf.reshape(preds, tf.shape(options))

    tf.debugging.check_numerics(preds, "Nan in preds")

    utils_options = preds

    utils_choices = tf.gather(preds, choices_pos, batch_dims=1)

    utils_diffs = utils_choices - utils_options

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

    g_k_no_pdf = compute_g_k(
        cdf=cdf,
        utils_diffs=utils_diffs,
        epsilon_samples=epsilon_samples,
        choices_pos=choices_pos,
        num_samples_per_kernel=num_samples_per_kernel,
        n_kernels=n_kernels,
        num_obs=num_obs,
        drop_denstity_from_product=True,
    )

    aggregation_factor = model.cdf.weights() / (model.cdf.bandwidth() * tf.cast(num_samples_per_kernel, tf.float32))

    # do not clip because this is not a likelihood
    likelihoods = tf.reduce_sum(
        tf.reduce_sum(
            g_k_no_pdf*model.cdf.bandwidth(),
            axis=-1
        ),
        axis=1,
    )

    
    with tf.GradientTape() as tape:
        # Estimate utilities
        preds = model(
            users=tf.repeat(users, tf.size(options[0])), 
            items=tf.reshape(options, [-1]),
        )
        
        preds = tf.reshape(preds, tf.shape(options))

        tf.debugging.check_numerics(preds, "Nan in preds")

        utils_options = preds

        utils_choices = tf.gather(preds, choices_pos, batch_dims=1)

        utils_diffs = utils_choices - utils_options

        # Set constants
        num_obs = tf.shape(utils_diffs)[0]

        n_kernels = tf.constant(cdf.n_kernels)

        if num_samples_per_kernel is None:
            num_samples_per_kernel = tf.constant(MONTE_CARLO_N_SAMPLES_PER_KERNEL) # Number of samples per kernel

        num_alternatives = tf.shape(utils_diffs)[1]

        # Samples epsilon values

        # des_point_mat = tf.repeat(tf.expand_dims(tf.transpose(cdf.scaled_des_points()), 1), [num_samples_per_kernel], axis=1) # Contains position of the des_points of each kernel, shape [n_kernels, num_samples_per_kernel]
        # des_point_mat = tf.tile(
        #     tf.expand_dims(
        #         des_point_mat,
        #         axis=0
        #     ),
        #     [
        #         num_obs,
        #         1,
        #         1,
        #     ]
        # )

        # uniform_samples = tf.random.uniform(tf.shape(des_point_mat), 0., 1.) # Uniform samples. IMPORTANT: Added small values (1e-03, 1 - 1e-03) to avoid NaNs in the gradients of the variables that affect the bandwidth
        # # epsilon_samples = des_point_mat + inverse_sigmoid(uniform_samples) * cdf.bandwidth() # Add to des_points
        # epsilon_samples = des_point_mat + inverse_sigmoid(uniform_samples) * tf.expand_dims(tf.expand_dims(cdf.bandwidth(), -1), 0) # Add to des_points

        # # Create inputs to the cdfs
        # epsilon_samples = tf.cast( # Potential values of epsilon
        #     tf.tile(
        #         tf.expand_dims(
        #             epsilon_samples, 
        #             axis=-1
        #         ), 
        #         [
        #                 1, 
        #                 1,
        #                 1,
        #                 num_alternatives, 
        #         ]
        #     ), 
        #     tf.float32
        # )

        g_k_pdf = compute_g_k(
            cdf=cdf,
            utils_diffs=utils_diffs,
            epsilon_samples=epsilon_samples,
            choices_pos=choices_pos,
            num_samples_per_kernel=num_samples_per_kernel,
            n_kernels=n_kernels,
            num_obs=num_obs,
            drop_denstity_from_product=False,
        )
    
        log_g_k = tf.math.log(g_k_pdf + 1e-10) # Add small value to avoid NaNs in the gradients
        
        value_inside_integral = g_k_no_pdf * log_g_k

        value_inside_integral *= model.cdf.weights() / (model.cdf.bandwidth() * tf.cast(num_samples_per_kernel, tf.float32)) # aggregation_factor

        # raise ValueError("Check if the aggregation factor is correct and if it should really not affect the gradient")

        integral_values = tf.reduce_sum(value_inside_integral, axis=-1) # Calculate integral, sum because the number of samples per kernel is already included in aggregation_factor
        tf.debugging.check_numerics(integral_values, "Nan in integral_values")

        # average over kernels w.r.t. weights
        average_over_kernels = tf.reduce_sum(integral_values, axis=1) # weights() is already included in aggregation_factor
        tf.debugging.check_numerics(average_over_kernels, "Nan in average_over_kernels")
        total_average = tf.reduce_mean(average_over_kernels / likelihoods)
        
    grad_likelihood = tape.gradient(total_average, model.trainable_variables) # Aggregates

    grad_likelihood = [-grad for grad in grad_likelihood] # "-"" because we want to minimize the negative log-likelihood

    return grad_likelihood, utils_diffs

@tf.function()
def approx_likelihood(
        cdf, 
        utils_diffs,
        choices_pos,
        num_samples_per_kernel=None,
    ):

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

    integral_values = tf.reduce_mean(product_inside_integral, axis=-1) # Calculate integral
    tf.debugging.check_numerics(integral_values, "Nan in integral_values")

    # average over kernels w.r.t. weights
    average_over_kernels = tf.reduce_sum(cdf.weights() * integral_values, axis=1)
    tf.debugging.check_numerics(average_over_kernels, "Nan in average_over_kernels")

    # clip
    average_over_kernels = tf.clip_by_value(average_over_kernels, 1e-10, 1)

    return tf.reduce_mean(average_over_kernels)

@tf.function()
def approx_gradient(
        cdf, 
        model,
        users, 
        options, 
        choices_pos, 
    ):
    """Compute the gradient of the negative log-likelihood (nll) via Monte-Carlo approximation for a given choice model"""


    # Approximate gradient of the log-likelihood

    grad_log_likelihood, utils_diffs = approx_grad_log_likelihood(
        cdf=cdf, 
        model=model,
        users=users,
        options=options,
        choices_pos=choices_pos,
    )

    likelihood = approx_likelihood(
        cdf=cdf, 
        utils_diffs=utils_diffs,
        choices_pos=choices_pos,
    )

    # grad_log_likelihood = [-grad / likelihood for grad in grad_likelihood] # "-"" because we want to minimize the negative log-likelihood

    nll = - tf.math.log(likelihood + 1e-10) # Add small value to avoid NaNs in the gradients

    return grad_log_likelihood, nll

