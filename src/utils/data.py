# %%
import sys
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import tqdm

from src.utils.is_running_on_cluster import is_running_on_cluster

def getRandomSamplesOnNSphere(k, numberOfSamples, R=1): 
    '''Uniformly draws random samples from the k-dimensional unit sphere.'''
    # https://mathworld.wolfram.com/HyperspherePointPicking.html
    X = np.random.normal(size=(numberOfSamples , k))
    return R / np.sqrt(np.sum(X**2, 1, keepdims=True)) * X

def generate_embeddings(n_users, n_items, k, item_specific_constant=False):
    '''Generates random embeddings for users and items.'''
    # Introducing users and items
    U = getRandomSamplesOnNSphere(k, n_users)
    V = getRandomSamplesOnNSphere(k, n_items)
        
    if item_specific_constant:
        U = np.c_[ U, np.ones((n_users, 1)) ]
        V = np.c_[ V, np.random.uniform(0, 1, n_items) ]
        
    # introducing picking prob matrix
    utils = np.matmul(U, V.transpose())

    T = np.exp(utils)

    return U, V, utils, T

def generate_data(cdf, n_users, n_items, k, n_choices_per_user, choice_set_size=3, item_specific_constant=False, seed=42):
    import numpy as np
    from sklearn.model_selection import train_test_split

    if type(choice_set_size) == list:
        dynamic_choice_set_size = True
        max_choice_set_size = np.max(choice_set_size)
    else:
        dynamic_choice_set_size = False
        max_choice_set_size = choice_set_size

    U, V, _, T = generate_embeddings(n_users, n_items, k, item_specific_constant)

    indices_train, indices_validation, indices_test = None, None, None
    if n_choices_per_user > 0:

        U_perChoice = tf.repeat(tf.cast(U, tf.float32), n_choices_per_user, axis=0)
        U_perChoice = tf.reshape(U_perChoice, shape=(U_perChoice.shape[0], 1, U_perChoice.shape[1]))
        U_perChoice = tf.tile(U_perChoice, multiples=(1, max_choice_set_size, 1))

        # assume we have a numpy 2D matrix V with m columns and n rows
        m = V.shape[1]
        n = V.shape[0]
        V_ph = tf.cast(V, tf.float32)
        indices = tf.random.uniform(shape=(n_users*n_choices_per_user*max_choice_set_size,), maxval=n, dtype=tf.int32)
        rows = tf.gather(V_ph, indices)
        V_perChoice = tf.reshape(rows, shape=(n_users*n_choices_per_user, max_choice_set_size, m))

        utils_choices = tf.multiply(U_perChoice, V_perChoice)
        utils_choices = tf.reduce_sum(utils_choices, axis=-1)
        utils_choices += cdf.sample(utils_choices.shape)
        
        users = np.repeat(range(n_users), n_choices_per_user)
        options = tf.reshape(indices, (n_users*n_choices_per_user, max_choice_set_size)).numpy()

        if dynamic_choice_set_size:
            choice_set_sizes = np.random.choice(a=choice_set_size, replace=True, size=utils_choices.shape[0])

            utils_choices = np.asarray(
                [
                    list(u[:choice_set_sizes[idx]]) + [-100]*(max_choice_set_size-choice_set_sizes[idx])
                    for idx, u in enumerate(utils_choices.numpy())
                ]
            ).astype(np.float32)

            options = np.asarray(
                [
                    list(o[:choice_set_sizes[idx]]) + [-1]*(max_choice_set_size-choice_set_sizes[idx])
                    for idx, o in enumerate(options)
                ]
            ).astype(np.int32)

        choices = tf.gather(options, tf.argmax(utils_choices, axis=-1).numpy(), batch_dims=1)
                    
        # Splitting data into train/test sets
        import numpy as np
        from sklearn.model_selection import train_test_split
        indices = [i for i in range(len(choices))]
        indices_train, indices_validation = train_test_split(indices, test_size=0.4, random_state=42)
        indices_validation, indices_test = train_test_split(indices_validation, test_size=0.5, random_state=42)

    return U, V, T, np.array(users), np.array(choices), np.array(options), indices_train, indices_validation, indices_test


def generate_choices_for_subsets_bias(
        user_subset, 
        item_subset, 
        U, 
        V, 
        cdf, 
        choice_set_size, 
        n_choices_per_user, 
        bias_mode_set_B, 
        biased_items=None, 
        top_20_percent_unbiased_item_ids_set_B=None, 
        bottom_20_percent_unbiased_item_ids_set_B=None,
    ):
    """Generates biased or unbiased  choices for a subset of users and items."""
    print("Generating choices...")

    assert(bias_mode_set_B in ["unbiased", "overexposure", "unpopular_alternatives", "popular_alternatives"]),  f"Invalid bias mode {bias_mode_set_B}"

    user_subset = np.asarray(user_subset)
    item_subset = np.asarray(item_subset)

    n_users = len(user_subset)
    n_items = len(item_subset)
    
    embedding_size = V.shape[1]

    unbiased_items = [i for i in item_subset if i not in biased_items]

    # Get user embeddings
    users = np.repeat(user_subset, n_choices_per_user)

    U_perChoice = tf.gather(U, tf.convert_to_tensor(users, dtype=tf.int32)) # Not U_subset!
    U_perChoice = tf.reshape(U_perChoice, shape=(U_perChoice.shape[0], 1, U_perChoice.shape[1]))
    U_perChoice = tf.tile(U_perChoice, multiples=(1, choice_set_size, 1))

    # Get item embeddings

    # Get choice sets
    if bias_mode_set_B == "unbiased": # Fast path for unbiased choices
        print("Generating unbiased choices...")

        # Assume we have a numpy 2D matrix V with m columns and n rows
        indices = tf.random.uniform(shape=(n_users*n_choices_per_user*choice_set_size,), minval=0, maxval=n_items, dtype=tf.int32) # Sample items

        options = tf.reshape(item_subset[indices], (n_users*n_choices_per_user, choice_set_size)).numpy()

    else: # Slow path for biased choices
        print("Generating biased choices for training users...")

        options = []
        for _ in tqdm(user_subset, ascii=True, file=sys.stdout, disable=is_running_on_cluster()):
            for _ in range(n_choices_per_user):
                # Get bias mode
                if bias_mode_set_B == "popular_alternatives":
                    # Generate a choice set uniformly from unbiased items at 50% chance or from one biased item and three top 20% popular items at 50% chance
                    if random.random() < 0.5:
                        new_options = np.random.choice(unbiased_items, size=choice_set_size, replace=False)
                    else:
                        new_options = np.asarray(
                            list(np.random.choice(biased_items, size=1, replace=False)) 
                            + list(np.random.choice(top_20_percent_unbiased_item_ids_set_B, size=choice_set_size-1, replace=False).tolist())
                        )
                # Get bias mode
                elif bias_mode_set_B == "unpopular_alternatives":
                    # Generate a choice set uniformly from unbiased items at 50% chance or from one biased item and three bottom 20% popular items at 50% chance
                    if random.random() < 0.5:
                        new_options = np.random.choice(unbiased_items, size=choice_set_size, replace=False)
                    else:
                        new_options = np.asarray(
                            list(np.random.choice(biased_items, size=1, replace=False)) 
                            + list(np.random.choice(bottom_20_percent_unbiased_item_ids_set_B, size=choice_set_size-1, replace=False).tolist())
                        )
                elif bias_mode_set_B == "overexposure":
                    # Generate a choice set uniformly from all items at 50% chance or from one biased item and any other item at 50% chance
                    if random.random() < 0.5:
                        new_options = np.random.choice(item_subset, size=choice_set_size, replace=False)
                    else:
                        new_options = np.asarray(
                            list(np.random.choice(biased_items, size=1, replace=False)) 
                            + list(np.random.choice(unbiased_items, size=choice_set_size-1, replace=False).tolist())
                        )
                else:
                    raise NotImplementedError(f"Bias mode {bias_mode_set_B} is not implemented")
                
                options.append(new_options)
        
        options = np.array(options)

    indices = tf.convert_to_tensor(options.flatten(), dtype=tf.int32)
    rows = tf.gather(V, indices) # Not V_subset!
    V_perChoice = tf.reshape(rows, shape=(n_users*n_choices_per_user, choice_set_size, embedding_size))

    # Get deterministic utilities for each choice set
    utils_choices = (
        tf.cast(
            tf.reduce_sum(
                tf.multiply(U_perChoice, V_perChoice),
                axis=-1
            ),
            tf.float32
        )
        + cdf.sample(options.shape)
    )

    # Get choices
    choices = tf.gather(options, tf.argmax(utils_choices, axis=-1).numpy(), batch_dims=1)

    return users, options, choices


def generate_choices_for_subsets_performance(
        user_subset, 
        item_subset, 
        U, 
        V, 
        cdf, 
        choice_set_size, 
        n_choices_per_user,
        bias_mode_set_B="unbiased",
    ):
    """Generates biased or unbiased  choices for a subset of users and items."""
    print("Generating choices for a user and/or item subset for the performance experiment...", flush=True)

    assert bias_mode_set_B in ["unbiased", "proportional_to_item_popularity"],  f"Invalid bias mode {bias_mode_set_B}"

    user_subset = np.asarray(user_subset)
    item_subset = np.asarray(item_subset)

    n_users = len(user_subset)
    
    embedding_size = V.shape[1]

    # Get user embeddings
    users = np.repeat(user_subset, n_choices_per_user)

    U_perChoice = tf.gather(U, tf.convert_to_tensor(users, dtype=tf.int32)) # Not U_subset!
    U_perChoice = tf.reshape(U_perChoice, shape=(U_perChoice.shape[0], 1, U_perChoice.shape[1]))
    U_perChoice = tf.tile(U_perChoice, multiples=(1, choice_set_size, 1))


    user_subset_embs = tf.gather(U, tf.convert_to_tensor(user_subset, dtype=tf.int32)).numpy()
    item_subset_embs = tf.gather(V, tf.convert_to_tensor(item_subset, dtype=tf.int32)).numpy()

    utils_subset = np.matmul(user_subset_embs, item_subset_embs.transpose())

    softmax_choice_probs = tf.nn.softmax(utils_subset, axis=1).numpy()
    average_choice_probs = np.mean(softmax_choice_probs, axis=0)
    
    # Get choice sets

    options = []
    for user_idx in tqdm(np.repeat(np.arange(n_users), n_choices_per_user), ascii=True, file=sys.stdout, disable=is_running_on_cluster()):
        # Generate a choice set of items proportional to their popularity for this user when using a softmax
        if bias_mode_set_B == "unbiased":
            new_options = np.random.choice(item_subset, size=choice_set_size, replace=False)
        else:
            new_options = np.random.choice(item_subset, p=average_choice_probs, size=choice_set_size, replace=False)
        
        options.append(new_options)
        
    options = np.array(options)

    indices = tf.convert_to_tensor(options.flatten(), dtype=tf.int32)
    rows = tf.gather(V, indices) # Not V_subset!
    V_perChoice = tf.reshape(rows, shape=(n_users*n_choices_per_user, choice_set_size, embedding_size))

    # Get deterministic utilities for each choice set
    utils_choices = (
        tf.cast(
            tf.reduce_sum(
                tf.multiply(U_perChoice, V_perChoice),
                axis=-1
            ),
            tf.float32
        )
        + cdf.sample(options.shape)
    )

    # Get choices
    choices = tf.gather(options, tf.argmax(utils_choices, axis=-1).numpy(), batch_dims=1)

    return users, options, choices


def generate_biased_data(
        cdf, 
        n_users, 
        n_items, 
        k, 
        n_choices_per_user, 
        choice_set_size=3, 
        item_specific_constant=False, 
        seed=42, 
        bias_mode="competition",
        ratio_training_users=0.5,
        n_biased_items=10,
):
    '''
    Generates embeddings and choices for a biased choice task.
    
    The output dataset has the format
    Users / Items      Set A   Set B

    Training           Unbiased  Biased

    Validation         Unbiased  Unbiased
    '''
    import numpy as np
    from sklearn.model_selection import train_test_split

    U, V, utils, T = generate_embeddings(n_users, n_items, k, item_specific_constant)

    indices_train, indices_validation = None, None

    if n_choices_per_user > 0: # Sometimes, we only want to generate the embeddings

        # Split users into training and validation users
        n_users_train = int(n_users * ratio_training_users)

        users_train = np.arange(n_users_train)
        users_validation = np.arange(n_users_train, n_users)

        # Split items into set A and set B
        item_ids_set_A = [i for i in range(int(n_items / 2))]
        item_ids_set_B = [i for i in range(int(n_items / 2), n_items)]

        # Set biased items
        biased_item_ids = np.random.choice(item_ids_set_B, size=int(len(item_ids_set_B) / n_biased_items), replace=False)
        unbiased_item_ids_set_B = [i for i in item_ids_set_B if i not in biased_item_ids]

        # Get set B item popularities
        item_popularities = utils.mean(axis=0)

        # Sort items by popularity
        sorted_item_ids = [
            (item_id, popularity) 
            for item_id, popularity in sorted(zip(range(n_items), item_popularities), key=lambda x: x[1], reverse=True)
        ]

        # Select five biased items from set B at each quintile of the popularity distribution
        sorted_item_ids_set_B = [item for item in sorted_item_ids if item[0] in item_ids_set_B]
        
        sorted_unbiased_item_ids_set_B = [item for item in sorted_item_ids_set_B if item[0] in unbiased_item_ids_set_B]

        # Get top 20% most popular items from set B that are not biased
        top_20_percent_unbiased_item_ids_set_B = [
            item[0] 
            for item in sorted_unbiased_item_ids_set_B[:int(len(sorted_item_ids_set_B) * 0.2)]
        ]
        bottom_20_percent_unbiased_item_ids_set_B = [
            item[0] 
            for item in sorted_unbiased_item_ids_set_B[int(len(sorted_item_ids_set_B) * 0.2):]
        ]

        # Create unbiased choices on set A for all users
        users_set_A, options_set_A, choices_set_A = generate_choices_for_subsets_bias(
            user_subset=[i for i in range(n_users)],
            item_subset=item_ids_set_A,
            U=U,
            V=V,
            cdf=cdf,
            choice_set_size=choice_set_size,
            n_choices_per_user=n_choices_per_user,
            bias_mode_set_B="unbiased",
            biased_items=biased_item_ids,
            top_20_percent_unbiased_item_ids_set_B = top_20_percent_unbiased_item_ids_set_B,
            bottom_20_percent_unbiased_item_ids_set_B=bottom_20_percent_unbiased_item_ids_set_B,
        )

        # Create first set of choices on set B for the training users 
        # ("unbiased" for bias modes "unbiased" and "overexposure", and "unpopular_alternatives" for "competition")
        if bias_mode in ["unbiased", "overexposure"]:
            bias_mode_set_B = "unbiased"
        else:
            bias_mode_set_B = "unpopular_alternatives"
        
        users_set_B_unbiased, options_set_B_unbiased, choices_set_B_unbiased = generate_choices_for_subsets_bias(
            user_subset=users_train,
            item_subset=item_ids_set_B,
            U=U,
            V=V,
            cdf=cdf,
            choice_set_size=choice_set_size,
            n_choices_per_user=n_choices_per_user,
            bias_mode_set_B=bias_mode_set_B,
            biased_items=biased_item_ids,
            top_20_percent_unbiased_item_ids_set_B = top_20_percent_unbiased_item_ids_set_B,
            bottom_20_percent_unbiased_item_ids_set_B=bottom_20_percent_unbiased_item_ids_set_B,
        )

        # Create second set of choices on set B for the training users 
        # (unbiased for bias mode "unbiased", overexposure for "overexposure", and "popular_alternatives" for "competition")
        if bias_mode == "unbiased":
            bias_mode_set_B = "unbiased"
        elif bias_mode == "overexposure":
            bias_mode_set_B = "overexposure"
        else:
            bias_mode_set_B = "popular_alternatives"
        
        train_users_set_B_biased, train_options_set_B_biased, train_choices_set_B_biased = generate_choices_for_subsets_bias(
            user_subset=users_train,
            item_subset=item_ids_set_B,
            U=U,
            V=V,
            cdf=cdf,
            choice_set_size=choice_set_size,
            n_choices_per_user=n_choices_per_user,
            bias_mode_set_B=bias_mode_set_B,
            biased_items=biased_item_ids,
            top_20_percent_unbiased_item_ids_set_B = top_20_percent_unbiased_item_ids_set_B,
            bottom_20_percent_unbiased_item_ids_set_B=bottom_20_percent_unbiased_item_ids_set_B,
        )

        # Combine choices from set A and set B

        # Unbiased (control) data
        users_unbiased = np.concatenate([users_set_A, users_set_B_unbiased])
        choices_unbiased = np.concatenate([choices_set_A, choices_set_B_unbiased])
        options_unbiased = np.concatenate([options_set_A, options_set_B_unbiased], axis=0)
        choices_pos_unbiased = np.asarray(
            [
                np.where(options_unbiased[i]==choices_unbiased[i])[0][0]
                .astype('int32') for i in range(len(choices_unbiased))
            ]
        )

        # Biased (treatment) data
        users_biased = np.concatenate([users_set_A, train_users_set_B_biased])
        choices_biased = np.concatenate([choices_set_A, train_choices_set_B_biased])
        options_biased = np.concatenate([options_set_A, train_options_set_B_biased], axis=0)
        choices_pos_biased = np.asarray(
            [
                np.where(options_biased[i]==choices_biased[i])[0][0]
                .astype('int32') for i in range(len(choices_biased))
            ]
        )
                    
        # Split data into train/test sets
        indices_train, indices_validation = train_test_split([i for i in range(len(choices_biased))], test_size=0.2, random_state=seed)

        # Generate test data (unbiased choices on set B for the validation users)
        validation_users_set_B_biased, validation_options_set_B_biased, validation_choices_set_B_biased = generate_choices_for_subsets_bias(
            user_subset=users_validation,
            item_subset=item_ids_set_B,
            U=U,
            V=V,
            cdf=cdf,
            choice_set_size=choice_set_size,
            n_choices_per_user=n_choices_per_user,
            bias_mode_set_B="unbiased",
            biased_items=biased_item_ids,
            top_20_percent_unbiased_item_ids_set_B = top_20_percent_unbiased_item_ids_set_B,
            bottom_20_percent_unbiased_item_ids_set_B=bottom_20_percent_unbiased_item_ids_set_B,
        )

    # Compute biased item popularity ranks
    popularity_dict_set_B = {item_id[0]: rank for rank, item_id in enumerate(sorted_item_ids_set_B)}
    biased_items_popularity_ranks = [popularity_dict_set_B[item_id] for item_id in biased_item_ids]

    return {
        "U": U,
        "V": V,
        "utils": utils,
        "T": T,
        #
        "users_unbiased" : users_unbiased,
        "choices_unbiased" : choices_unbiased,
        "options_unbiased" : options_unbiased,
        "choices_pos_unbiased" : choices_pos_unbiased,
        "users_biased" : users_biased,
        "choices_biased" : choices_biased,
        "options_biased" : options_biased,
        "choices_pos_biased" : choices_pos_biased,
        "users_train" : users_train,
        "users_validation" : users_validation,
        "indices_train" : indices_train,
        "indices_validation" : indices_validation,
        "test_users" : validation_users_set_B_biased,
        "test_choices" : validation_choices_set_B_biased,
        "test_options" : validation_options_set_B_biased,
        #
        "items_set_A" : item_ids_set_A,
        "items_set_B" : item_ids_set_B,
        "biased_item_ids" : biased_item_ids,
        "unbiased_item_ids_set_B" : unbiased_item_ids_set_B,
        "biased_item_ranks": biased_items_popularity_ranks,
        # "top_20_percent_item_ids_set_B" : top_20_percent_unbiased_item_ids_set_B,
    }


def generate_performance_data(
        cdf, 
        n_users, 
        n_items, 
        k, 
        n_choices_per_user, 
        choice_set_size=3, 
        item_specific_constant=True, 
        seed=42,
        ratio_training_users=0.5,
    ):
    import numpy as np # We have to reimport numpy here for whatever reason
    from sklearn.model_selection import train_test_split

    print("Generating data for performance experiment...", flush=True)

    print("Generating embeddings...", flush=True)
    U, V, utils, T = generate_embeddings(n_users, n_items, k, item_specific_constant)

    indices_train, indices_validation = None, None

    if n_choices_per_user > 0: # Sometimes, we only want to generate the embeddings

        # Split users into trainig and validation users
        n_users_train = int(n_users * ratio_training_users)

        users_train = np.arange(n_users_train)
        users_validation = np.arange(n_users_train, n_users)

        # Split items into set A and set B
        item_ids_set_A = [i for i in range(int(n_items / 2))]
        item_ids_set_B = [i for i in range(int(n_items / 2), n_items)]

        # Create choices on set A for all users and sample proportional to item popularity
        users_set_A, options_set_A, choices_set_A = generate_choices_for_subsets_performance(
            user_subset=[i for i in range(n_users)],
            item_subset=item_ids_set_A,
            U=U,
            V=V,
            cdf=cdf,
            choice_set_size=choice_set_size,
            n_choices_per_user=n_choices_per_user,
            bias_mode_set_B="unbiased",
        )

        # Create second set of choices on set B for the training users and sample proportional to item popularity
        train_users_set_B, train_options_set_B, train_choices_set_B = generate_choices_for_subsets_performance(
            user_subset=users_train,
            item_subset=item_ids_set_B,
            U=U,
            V=V,
            cdf=cdf,
            choice_set_size=choice_set_size,
            n_choices_per_user=n_choices_per_user,
            bias_mode_set_B="unbiased",
        )

        # Combine choices from set A and set B
        train_users = np.concatenate([users_set_A, train_users_set_B])
        train_choices = np.concatenate([choices_set_A, train_choices_set_B])
        train_options = np.concatenate([options_set_A, train_options_set_B], axis=0)
        train_choices_pos = np.asarray([np.where(train_options[i]==train_choices[i])[0][0].astype('int32') for i in range(len(train_choices))])

        # Split data into train/test sets
        indices_train, indices_validation = train_test_split([i for i in range(len(train_choices))], test_size=0.2, random_state=seed)

        # Generate test data (unbiased (uniformly sampled) choice sets on set B for the validation users)
        validation_users_set_B_biased, validation_options_set_B_biased, validation_choices_set_B_biased = generate_choices_for_subsets_performance(
            user_subset=users_validation,
            item_subset=item_ids_set_B,
            U=U,
            V=V,
            cdf=cdf,
            choice_set_size=choice_set_size,
            n_choices_per_user=n_choices_per_user,
            bias_mode_set_B="unbiased",
        )

    return {
        "U": U,
        "V": V,
        "utils": utils,
        "T": T,
        #
        "train_users" : train_users,
        "train_choices" : train_choices,
        "train_options" : train_options,
        "train_choices_pos" : train_choices_pos,
        "users_train" : users_train,
        "users_validation" : users_validation,
        "indices_train" : indices_train,
        "indices_validation" : indices_validation,
        "test_users" : validation_users_set_B_biased,
        "test_choices" : validation_choices_set_B_biased,
        "test_options" : validation_options_set_B_biased,
        #
        "items_set_A" : item_ids_set_A,
        "items_set_B" : item_ids_set_B,
    }


def shuffle_training_data(users, choices, options, choices_pos):
    """
    Shuffles the training data in the four input arrays.
    
    Args:
        users: A numpy array of shape (n_samples,) containing the user IDs.
        choices: A numpy array of shape (n_samples,) containing the IDs of the chosen options.
        options: A numpy array of shape (n_samples, n_options) containing the option features.
        choices_pos: A numpy array of shape (n_samples, n_options) containing the position of the chosen options.
    
    Returns:
        A tuple of four numpy arrays of the same shape as the inputs, containing the shuffled data.
    """
    # Get the number of samples
    n_samples = len(users)
    
    # Generate a random permutation of the sample indices
    perm = np.random.permutation(n_samples)
    
    # Shuffle the data arrays using the permutation
    users_shuffled = users[perm]
    choices_shuffled = choices[perm]
    options_shuffled = options[perm]
    choices_pos_shuffled = choices_pos[perm]
    
    # Return the shuffled data arrays
    return users_shuffled, choices_shuffled, options_shuffled, choices_pos_shuffled

    
def prepare_data(users, alternatives, choices):
    '''Converts the input data into a numpy array of shape (n_samples, 3) where each row contains the user ID, the alternative IDs, and the choice.'''
    return(np.asarray([[users[i], alternatives[i], choices[i]] for i in range(len(users))], dtype=object)) # Geht das noch schneller?


def train_test_val_split(users, options, choices, choices_pos, n_choices_per_user, train_size=0.6, val_size=0.2):
    """Splits the data into training, validation and test set (only used for testing)"""

    assert(train_size + val_size < 1.0), "The sum of train_size and val_size must be less than 1.0"

    inds_train_singleuser = range(0, int(train_size * n_choices_per_user))
    inds_validation_singleuser = range(int(train_size * n_choices_per_user), int((train_size + val_size) * n_choices_per_user))
    inds_test_singleuser = range(int((train_size + val_size) * n_choices_per_user), n_choices_per_user)

    inds_train = [i for i in range(len(users)) if i%n_choices_per_user in inds_train_singleuser]
    inds_validation = [i for i in range(len(users)) if i%n_choices_per_user in inds_validation_singleuser]
    inds_test = [i for i in range(len(users)) if i%n_choices_per_user in inds_test_singleuser]

    users_train      = users[inds_train] 
    options_train    = options[inds_train]
    choices_train    = choices[inds_train]

    users_validation      = users[inds_validation] 
    options_validation    = options[inds_validation]
    choices_validation    = choices[inds_validation]

    users_test      = users[inds_test] 
    options_test    = options[inds_test]
    choices_test    = choices[inds_test]

    data_train = prepare_data(users_train, options_train, choices_train)
    data_validation = prepare_data(users_validation, options_validation, choices_validation)
    data_test = prepare_data(users_test, options_test, choices_test)
    
    return data_train, data_validation, data_test