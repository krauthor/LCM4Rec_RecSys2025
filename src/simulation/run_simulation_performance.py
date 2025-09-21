# Imports
import os
from importlib import reload
import datetime
import time
import random
import sys
import pathlib
import json

# Set the maximum number of cpu threads to 1
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Suppress TensorFlow warnings so that stdout only contains error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
import tensorflow as tf

# Add parent directory to path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
if os.getcwd()=="/vol/das-nobackup/users/tkrause":
    os.chdir("/vol/das-nobackup/users/tkrause/repo")
    sys.path.append("/vol/das-nobackup/users/tkrause/repo")

import src.Models.recommenders
import src.Models
from src.utils.data import generate_performance_data
from src.Models.recommenders import Recommender_NPMLE, Recommender_multinomial_logit, Recommender_exponomial, Recommender_binary_logit, Recommender_binary_logit_negative_sampling, Recommender_gBCE
from src.utils.get_cdf import get_cdf
from src.utils.load_hyperparameters import load_hyperparameters
from src.utils.parsers import create_simulation_parser
from src.utils.store_results import store_results
from src.utils.get_ranking import get_ranking
from src.utils.ndcg import ndcg
from src.utils.optimize_kld_offset import optimize_KLD_offset
from src.utils.jenson_shannon_divergence import jenson_shannon_divergence, approx_choice_probs_true_prefs, compute_choice_probs_est_prefs
from src.utils.get_nll_and_accuracy import get_nll_and_accuracy
from src.utils.data import prepare_data
from src.utils.store_results import store_results
from src.utils.is_running_on_cluster import is_running_on_cluster


def run_simulation(
        cdf_name: str, 
        cdf_loc: float, 
        cdf_scale:float, 
        n_users: int, 
        n_items: int, 
        k: int, 
        n_choices_per_user: int, 
        choiceSet_size: int, 
        n_kernels: int,
        model_name: str = None, # Name of the model to be trained, only used for hyperparameter optimization
        optimize_hyperparameters_fg: bool = False, # Whether this simulation is part of the hyperparameter optimization (different output directory, skips computationally expensive evaluation steps)
        hyperparameters: dict = None, # Hyperparameters for the models, to be used instead of loading from the hyperparameter file, only used for hyperparameter optimization
        debug_fg: bool = False, # Whether to run in debug mode
    ):
    reload(src.Models.recommenders)

    print("Starting simulation...", flush=True)

    print("Parameters:", flush=True)
    print(f"cdf_name: {cdf_name}", flush=True)
    print(f"cdf_loc: {cdf_loc}", flush=True)
    print(f"cdf_scale: {cdf_scale}", flush=True)
    print(f"n_users: {n_users}", flush=True)
    print(f"n_items: {n_items}", flush=True)
    print(f"k: {k}", flush=True)
    print(f"n_choices_per_user: {n_choices_per_user}", flush=True)
    print(f"choiceSet_size: {choiceSet_size}", flush=True)
    print(f"n_kernels: {n_kernels}", flush=True)
    print(f"model_name: {model_name}", flush=True)
    print(f"optimize_hyperparameters_fg: {optimize_hyperparameters_fg}", flush=True)
    print(f"debug_fg: {debug_fg}", flush=True)

    s = time.time()

    # Generate data from the target distribution
    print("Generating data...", flush=True)

    target_cdf = get_cdf(cdf_name, loc=cdf_loc, scale=cdf_scale)

    data = generate_performance_data(
        target_cdf, 
        n_users, 
        n_items, 
        k, 
        n_choices_per_user, 
        choiceSet_size, 
        item_specific_constant=True,
    )

    print("Pre-processing data...", flush=True)

    # Unbiased data
    data_train = prepare_data(
        data["train_users"][data["indices_train"]], 
        data["train_options"][data["indices_train"]], 
        data["train_choices"][data["indices_train"]],
    )

    data_validation = prepare_data(
        data["train_users"][data["indices_validation"]], 
        data["train_options"][data["indices_validation"]], 
        data["train_choices"][data["indices_validation"]],
    )

    users_test = data["test_users"]
    options_test = data["test_options"]
    choices_test = data["test_choices"]
    choices_pos_test = np.asarray(
        [[np.where(options_test[i]==choices_test[i])[0][0].astype('int32')] for i in range(len(choices_test))]
    )
    
    data_test = prepare_data(users_test, options_test, choices_test)

    metrics = {
        "nDCG": {},
        "nll": {},
        "acc": {},
        "kl": {},
        "reverse_kl": {},
        "js": {},
        "kl_error_dist": {},
    }

    est_pdf = None
    best_offset = None 

    items_set_B = data["items_set_B"]

    validation_users = data["users_validation"]

    if not optimize_hyperparameters_fg:
        print("Approximating the true choice probabilities...", flush=True)

        # Wait for a brief random period of time. Maybe, this fixes slurm's issues
        time.sleep(random.uniform(0, 2*60))
        
        # Calculate the choice probabilities for the true and estimated preferences for the validation users on set B
        choice_probs_true_prefs = approx_choice_probs_true_prefs(
            true_utils=data["utils"][validation_users][:, items_set_B], 
            target_cdf_name=cdf_name,
            cdf_loc=cdf_loc,
            cdf_scale=cdf_scale,
        )

    ## Train the models

    # Load hyperparameter settings
    if hyperparameters is None:
        hyperparameters = load_hyperparameters()["performance"]

    if model_name is None:
        model_list = [Recommender_multinomial_logit, Recommender_exponomial, Recommender_NPMLE, Recommender_binary_logit, Recommender_binary_logit_negative_sampling, Recommender_gBCE]
    else:
        if model_name == "Recommender_multinomial_logit":
            model_class = Recommender_multinomial_logit
        elif model_name == "Recommender_exponomial":
            model_class = Recommender_exponomial
        elif model_name == "Recommender_NPMLE":
            model_class = Recommender_NPMLE
        elif model_name == "Recommender_binary_logit":
            model_class = Recommender_binary_logit
        elif model_name == "Recommender_binary_logit_negative_sampling":
            model_class = Recommender_binary_logit_negative_sampling
        elif model_name == "Recommender_gBCE":
            model_class = Recommender_gBCE

        # model_class = getattr(Models.recommenders, model_name)

        model_list = [model_class]

    for model_class in model_list:
        print(f"Training model {model_class.__name__}...", flush=True)

        # Add n_users etc. to hyperparameters (to be able to store them in the results file)
        hyperparameters[model_class.__name__]["n_users"] = n_users
        hyperparameters[model_class.__name__]["n_alternatives"] = n_items
        hyperparameters[model_class.__name__]["k"] = k
        hyperparameters[model_class.__name__]["n_kernels"] = n_kernels

        if debug_fg:
            hyperparameters[model_class.__name__]["n_epochs"] = 2
        
        hp_this_model = hyperparameters[model_class.__name__]

        print(f"Hyperparameters: {hp_this_model}")

        model = model_class(hp_this_model)

        model.train(data_train, data_validation, hp_this_model)

        ##  Compute results
        print("Computing results...", flush=True)

        ## Nll and accuracy
        nll, _ = get_nll_and_accuracy(model, model_class, data_test, users_test, options_test, choices_pos_test)

        metrics["nll"][model_class.__name__] = nll.numpy()
        # metrics["acc"][model_class.__name__] = acc.numpy()

        # Compute the nll as the probability to choose only the chosen item among the options for the univariate models
        if model_class in [Recommender_binary_logit, Recommender_binary_logit_negative_sampling, Recommender_gBCE]:
            scores = model.model(np.repeat(users_test, choiceSet_size), np.reshape(options_test, -1)).numpy()
            scores = np.reshape(scores, options_test.shape)

            choice_probs = model.model.probs(scores)

            nll = - np.mean(
                np.log(
                    np.take_along_axis(choice_probs, choices_pos_test, axis=1)
                )
            )

            metrics["nll"][model_class.__name__] = nll
            

        scores = model.model(np.repeat(users_test, choiceSet_size), np.reshape(options_test, -1))
        scores = np.reshape(scores, options_test.shape)
        choices = np.argmax(scores, axis=1)
        acc = np.mean(choices == np.reshape(choices_pos_test, -1))
        print("Accuracy:", acc)
        
        metrics["acc"][model_class.__name__] = acc

        if not optimize_hyperparameters_fg:

            ## nDCG and rank diffs

            # For each validaton user, get the preference ranking

            nDCG = []
            for user_id in validation_users:
                # Get estimated ranking
                ranking = list(get_ranking(model, user_id, items_set_B))
                
                # Get true ranking
                utils_this_user = data["utils"][:, items_set_B]
                
                ranking_true = list(np.argsort(utils_this_user[user_id])[::-1] + np.min(items_set_B)) # + np.min(items_set_B) to get the original item ids (Set A starts at 0, Set B starts at min(Set B))

                # Calculate the nDCG scores
                nDCG.append(ndcg([ranking_true], [ranking]))
                
            metrics["nDCG"][model_class.__name__] = np.mean(nDCG)

            # Calculate the choice probabilities for the estimated preferences for the validation users on set B
            choice_probs_est_prefs = compute_choice_probs_est_prefs(
                recommender=model, 
                user_ids=validation_users, 
                items_set_B=items_set_B,
            )

            # Calculate the Kullback-Leibler divergence between the true and estimated preference distributions
            divergence = jenson_shannon_divergence(
                choice_probs_true_prefs=choice_probs_true_prefs,
                choice_probs_est_prefs=choice_probs_est_prefs
            )

            metrics["js"][model_class.__name__] = divergence["jensen_shannon_divergence"]
            metrics["kl"][model_class.__name__] = divergence["kl"]
            metrics["reverse_kl"][model_class.__name__] = divergence["reverse_kl"]

            metrics["kl_error_dist"][model_class.__name__], best_offset = None, None
            if model_class == Recommender_NPMLE:
                metrics["kl_error_dist"][model_class.__name__], best_offset = optimize_KLD_offset(model.model.cdf, target_cdf)

                x = np.arange(
                    model.model.cdf.get_support_min(),
                    model.model.cdf.get_support_max(),
                    0.05,
                )

                est_pdf = np.asarray(
                    [
                        x,
                        model.model.cdf.prob(x),
                    ]
                ).T
            elif model_class in [Recommender_multinomial_logit, Recommender_exponomial]:            
                if model_class == Recommender_multinomial_logit:
                    assumed_cdf = get_cdf("gumbel", 0, np.sqrt(6 * target_cdf.variance() / (np.math.pi**2)))
                elif model_class == Recommender_exponomial:
                    assumed_cdf = get_cdf("exponomial", 0, np.sqrt(1 / target_cdf.variance()))

                metrics["kl_error_dist"][model_class.__name__], best_offset = optimize_KLD_offset(assumed_cdf, target_cdf)

        # if model_class == Recommender_NPMLE:
        #     model.model.cdf.plot(show=True)
            # model.model.cdf.save_gif("tmp.gif")

        # Clear and delete the models just in case to reset the graph or whatever
        model.clear()
        
        print()

        del model

    # Print the results
    if not is_running_on_cluster():
        print(
            pd.DataFrame.from_dict(
                metrics,
                orient='index'
            ).round(3)
        )

    results = {
        "metrics": metrics,
        "other":{
            "est_pdf" : est_pdf,
            "best_offset": best_offset,
            "runtime": time.time() - s,
        },
        "parameters": {
            "cdf_name" : cdf_name,
            "cdf_loc" : cdf_loc,
            "cdf_scale" : cdf_scale,
            "n_users": n_users,
            "n_items": n_items,
            "k": k,
            "n_choices_per_user": n_choices_per_user,
            "choiceSet_size": choiceSet_size,
            "n_kernels": n_kernels,
            "hyperparameters": hyperparameters,
        },
    }

    # Wait for a random amount of time to avoid storing all files with the same name when running multiple simulations in parallel
    time.sleep(random.uniform(0, 0.2))

    # Get datetime as string with maximum precision
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    
    if optimize_hyperparameters_fg:
        sub_dir = "hyperparameter_tuning/"
    else:
        sub_dir = ""

    result_file_path = "data/simulation_results/" + sub_dir + f"performance/{cdf_name}/{now_str}.json"

    store_results(results, result_file_path)

if __name__ == "__main__":
    print("run_simulation_performance.py", flush=True)

    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress warnings so that stdout only contains error messages

    # Print the numpy ressources
    print("Numpy resources:", flush=True)
    np.show_config()

    # Print the tensorflow ressources
    print("Tensorflow resources:", flush=True)
    print(tf.config.list_physical_devices('GPU'), flush=True)
    print(tf.config.list_physical_devices('CPU'), flush=True)

    # Set the maximum number of threads to 1
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    print("Parsing args...", flush=True)
    
    # Get knwargs from command line
    parser = create_simulation_parser(parser_type="performance")
    args = parser.parse_args()

    print("Running simulation...", flush=True)
    run_simulation(
        cdf_name=args.cdf_name,
        cdf_loc=args.cdf_loc,
        cdf_scale=args.cdf_scale,
        n_users=args.n_users,
        n_items=args.n_items,
        k=args.k,
        n_choices_per_user=args.n_choices_per_user,
        choiceSet_size=args.choiceSet_size,
        n_kernels=args.n_kernels,
        model_name=args.model_name,
        optimize_hyperparameters_fg=args.optimize_hyperparameters_fg,
        hyperparameters=json.loads(args.hyperparameters) if args.hyperparameters is not None else None,
        debug_fg=args.debug_fg,
    )