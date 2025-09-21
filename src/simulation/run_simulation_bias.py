# Imports
import json
import os
from importlib import reload
import datetime
import time
import random
import sys
import pathlib

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
if os.getcwd()=="/vol/das-nobackup/users/tkrause":
    os.chdir("/vol/das-nobackup/users/tkrause/repo")
    sys.path.append("/vol/das-nobackup/users/tkrause/repo")

import src.Models
from src.utils.data import generate_biased_data, prepare_data
from src.Models.recommenders import Recommender_NPMLE, Recommender_multinomial_logit, Recommender_exponomial, Recommender_binary_logit, Recommender_binary_logit_negative_sampling, Recommender_gBCE
from src.utils.get_cdf import get_cdf
from src.utils.load_hyperparameters import load_hyperparameters
from src.utils.parsers import create_simulation_parser
from src.utils.store_results import store_results
from src.utils.get_ranking import get_ranking
from src.utils.ndcg import ndcg
from src.utils.optimize_kld_offset import optimize_KLD_offset
from src.utils.jenson_shannon_divergence import jenson_shannon_divergence
from src.utils.get_nll_and_accuracy import get_nll_and_accuracy
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
        bias_mode: str,
        model_name: str = None, # Name of the model to be trained, only used for hyperparameter optimization
        optimize_hyperparameters_fg: bool = False, # Whether this simulation is part of the hyperparameter optimization (different output directory, skips computationally expensive evaluation steps)
        hyperparameters: dict = None, # Hyperparameters for the models, to be used instead of loading from the hyperparameter file, only used for hyperparameter optimization
        debug_fg: bool = False, # Whether to run in debug mode
    ):
    # reload(Models.npmle)
    reload(src.Models.recommenders)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress warnings so that stdout only contains error messages

    print("Starting simulation...")

    print("Parameters:")
    print(f"cdf_name: {cdf_name}")
    print(f"cdf_loc: {cdf_loc}")
    print(f"cdf_scale: {cdf_scale}")
    print(f"n_users: {n_users}")
    print(f"n_items: {n_items}")
    print(f"k: {k}")
    print(f"n_choices_per_user: {n_choices_per_user}")
    print(f"choiceSet_size: {choiceSet_size}")
    print(f"n_kernels: {n_kernels}")
    print(f"bias_mode: {bias_mode}")
    print(f"model_name: {model_name}")
    print(f"optimize_hyperparameters_fg: {optimize_hyperparameters_fg}")
    print(f"debug_fg: {debug_fg}")

    s = time.time()

    # Generate data from the target distribution
    print("Generating biased data...")

    target_cdf = get_cdf(cdf_name, loc=cdf_loc, scale=cdf_scale)

    data = generate_biased_data(
        target_cdf, 
        n_users, 
        n_items, 
        k, 
        n_choices_per_user, 
        choiceSet_size, 
        item_specific_constant=True,
        bias_mode=bias_mode,
    )

    size_set_A = int(n_items/2)

    # Train - validation split (Same ids for unbiased and biased data)
    indices_train = data["indices_train"]
    indices_validation = data["indices_validation"]

    # Unbiased data
    users_unbiased = data["users_unbiased"]
    options_unbiased = data["options_unbiased"]
    choices_unbiased = data["choices_unbiased"]

    data_train_unbiased = prepare_data(
        users_unbiased[indices_train], 
        options_unbiased[indices_train], 
        choices_unbiased[indices_train],
    )
    data_validation_unbiased = prepare_data(
        users_unbiased[indices_validation], 
        options_unbiased[indices_validation], 
        choices_unbiased[indices_validation],
    )

    # Biased data
    users_biased = data["users_biased"]
    options_biased = data["options_biased"]
    choices_biased = data["choices_biased"]

    data_train_biased = prepare_data(
        users_biased[indices_train], 
        options_biased[indices_train], 
        choices_biased[indices_train],
    )
    data_validation_biased = prepare_data(
        users_biased[indices_validation],  
        options_biased[indices_validation], 
        choices_biased[indices_validation],
    )

    users_test = data["test_users"]
    options_test = data["test_options"]
    choices_test = data["test_choices"]
    choices_pos_test = np.asarray(
        [[np.where(options_test[i]==choices_test[i])[0][0].astype('int32')] for i in range(len(choices_test))]
    )
    
    data_test = prepare_data(users_test, options_test, choices_test)

    metrics = {
        "nDCG_unbiased": {},
        "nDCG_biased": {},
        "mean_bias_per_item": {},
        "nll_unbiased": {},
        "nll_biased": {},
        "acc_unbiased": {},
        "acc_biased": {},
        "kl_unbiased": {},
        "kl_biased": {},
        "reverse_kl_unbiased": {},
        "reverse_kl_biased": {},
        "js_unbiased": {},
        "js_biased": {},
        "kl_error_dist_unbiased": {},
        "kl_error_dist_biased": {},
    }
    
    est_pdf_unbiased = None
    best_offset_unbiased = None 
    est_pdf_biased = None
    best_offset_biased = None 

    ## Train the models
    
    # Load hyperparameter settings
    if hyperparameters is None:
        hyperparameters = load_hyperparameters()["performance"] # We use the performance hyperparameters for the bias simulations

    if model_name is None:
        model_list = [Recommender_multinomial_logit, Recommender_exponomial, Recommender_NPMLE]
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

        model_list = [model_class]

    for model_class in model_list:
        print(f"Training model {model_class.__name__}...")

        # Add n_users etc. to hyperparameters (to be able to store them in the results file)
        hyperparameters[model_class.__name__]["n_users"] = n_users
        hyperparameters[model_class.__name__]["n_alternatives"] = n_items
        hyperparameters[model_class.__name__]["k"] = k
        hyperparameters[model_class.__name__]["n_kernels"] = n_kernels

        if debug_fg:
            hyperparameters[model_class.__name__]["n_epochs"] = 2

        # Add n_users etc. to hyperparameters
        hp_this_model = hyperparameters[model_class.__name__]

        print(f"Hyperparameters: {hp_this_model}")

        # Train model on unbiased data
        print("Training unbiased model...")

        model_unbiased = model_class(hp_this_model)

        model_unbiased.train(data_train_unbiased, data_validation_unbiased, hp_this_model)

        model_unbiased.clear()
        
        # Train model on biased data
        print("Training biased model...")

        model_biased = model_class(hp_this_model)

        model_biased.train(data_train_biased, data_validation_biased, hp_this_model)

        model_biased.clear()

        ##  Compute results
        print("Computing results...")

        ## Nll and accuracy
        nll_unbiased, acc_unbiased = get_nll_and_accuracy(model_unbiased, model_class, data_test, users_test, options_test, choices_pos_test)
        nll_biased, acc_biased = get_nll_and_accuracy(model_biased, model_class, data_test, users_test, options_test, choices_pos_test)

        metrics["nll_unbiased"][model_class.__name__] = nll_unbiased.numpy()
        # metrics["acc_unbiased"][model_class.__name__] = acc_unbiased.numpy()

        metrics["nll_biased"][model_class.__name__] = nll_biased.numpy()
        # metrics["acc_biased"][model_class.__name__] = acc_biased.numpy()

        # if model_name in ["Recommender_binary_logit", "Recommender_binary_logit_negative_sampling", "Recommender_gBCE"]:
        def get_acc(model, data, users_test, options_test, choices_pos_test):
            scores = model.model(np.repeat(users_test, choiceSet_size), np.reshape(options_test, -1))
            scores = np.reshape(scores, options_test.shape)
            choices = np.argmax(scores, axis=1)
            acc = np.mean(choices == np.reshape(choices_pos_test, -1))
            return acc
        
        acc_unbiased = get_acc(model_unbiased, data_test, users_test, options_test, choices_pos_test)
        acc_biased = get_acc(model_biased, data_test, users_test, options_test, choices_pos_test)
        
        metrics["acc_unbiased"][model_class.__name__] = acc_unbiased
        metrics["acc_biased"][model_class.__name__] = acc_biased

        if not optimize_hyperparameters_fg:
            ## nDCG and rank diffs

            # For each validaton user, get the preference ranking
            items_set_B = data["items_set_B"]

            validation_users = data["users_validation"]

            nDCG_unbiased = []
            nDCG_biased = []
            rank_diffs = []
            for user_id in validation_users:
                # Get estimated ranking
                ranking_unbiased = list(get_ranking(model_unbiased, user_id, items_set_B))
                ranking_biased = list(get_ranking(model_biased, user_id, items_set_B))

                # Compare the rankings
                rank_diffs.append([ranking_unbiased.index(item_id) - ranking_biased.index(item_id) for item_id in items_set_B])
                
                # Get true ranking
                utils_this_user = data["utils"][:, items_set_B]
                ranking_true = list(np.argsort(utils_this_user[user_id])[::-1] + size_set_A)

                # Calculate the nDCG scores
                nDCG_unbiased.append(ndcg([ranking_true], [ranking_unbiased]))
                nDCG_biased.append(ndcg([ranking_true], [ranking_biased]))
                
            metrics["nDCG_unbiased"][model_class.__name__] = np.mean(nDCG_unbiased)
            metrics["nDCG_biased"][model_class.__name__] = np.mean(nDCG_biased)

            # Calculate rank diffs
            rank_diffs = np.array(rank_diffs)

            # Get the rank diffs of the biased items
            biased_item_ids = np.asarray(data["biased_item_ids"])

            # metrics["mean_rank_diff"][model_class.__name__] = rank_diffs.mean(axis=0)
            metrics["mean_bias_per_item"][model_class.__name__] = rank_diffs[:,biased_item_ids - size_set_A].mean(axis=0)

            # Skip the KLD calculation for the biased models due to the high computational cost and because we do not plan to use it in the paper
            # # Calculate the Kullback-Leibler divergence
            # divergence_unbiased = jenson_shannon_divergence(
            #     true_utils=data["utils"][validation_users][:, items_set_B], 
            #     recommender=model_unbiased, 
            #     user_ids=validation_users, 
            #     items_set_B=items_set_B, 
            #     target_cdf_name=cdf_name,
            #     cdf_loc=cdf_loc,
            #     cdf_scale=cdf_scale,
            # )
            # divergence_biased = jenson_shannon_divergence(
            #     true_utils=data["utils"][validation_users][:, items_set_B], 
            #     recommender=model_biased, 
            #     user_ids=validation_users, 
            #     items_set_B=items_set_B, 
            #     target_cdf_name=cdf_name,
            #     cdf_loc=cdf_loc,
            #     cdf_scale=cdf_scale,
            # )

            # metrics["js_unbiased"][model_class.__name__] = divergence_unbiased["jensen_shannon_divergence"]
            # metrics["kl_unbiased"][model_class.__name__] = divergence_unbiased["kl"]
            # metrics["reverse_kl_unbiased"][model_class.__name__] = divergence_unbiased["reverse_kl"]

            # metrics["js_biased"][model_class.__name__] = divergence_biased["jensen_shannon_divergence"]
            # metrics["kl_biased"][model_class.__name__] = divergence_biased["kl"]
            # metrics["reverse_kl_biased"][model_class.__name__] = divergence_biased["reverse_kl"]

            metrics["kl_error_dist_unbiased"][model_class.__name__] = 0
            metrics["kl_error_dist_biased"][model_class.__name__] = 0

            # Store estimated cdf values for our model
            metrics["kl_error_dist_unbiased"][model_class.__name__], best_offset_unbiased = None, None
            metrics["kl_error_dist_biased"][model_class.__name__], best_offset_biased = None, None
            if model_class == Recommender_NPMLE:
                # Offset the cdf to minimize the KLD to the true distritbution
                metrics["kl_error_dist_unbiased"][model_class.__name__], best_offset_unbiased = optimize_KLD_offset(model_unbiased.model.cdf, target_cdf)
                metrics["kl_error_dist_biased"][model_class.__name__], best_offset_biased = optimize_KLD_offset(model_biased.model.cdf, target_cdf)

                x_unbiased = np.arange(
                    model_unbiased.model.cdf.get_support_min(),
                    model_unbiased.model.cdf.get_support_max(),
                    0.05,
                )
                est_pdf_unbiased = np.asarray(
                    [
                        x_unbiased,
                        model_unbiased.model.cdf.prob(x_unbiased),
                    ]
                ).T
                
                x_biased = np.arange(
                    model_biased.model.cdf.get_support_min(),
                    model_biased.model.cdf.get_support_max(),
                    0.05,
                )
                est_pdf_biased = np.asarray(
                    [
                        x_biased,
                        model_biased.model.cdf.prob(x_biased),
                    ]
                ).T
            elif model_class in [Recommender_multinomial_logit, Recommender_exponomial]:            
                if model_class == Recommender_multinomial_logit:
                    assumed_cdf = get_cdf("gumbel", 0, np.sqrt(6 * target_cdf.variance() / (np.math.pi**2)))
                elif model_class == Recommender_exponomial:
                    assumed_cdf = get_cdf("exponomial", 0, np.sqrt(1 / target_cdf.variance()))

                metrics["kl_error_dist_unbiased"][model_class.__name__], best_offset_unbiased = optimize_KLD_offset(assumed_cdf, target_cdf)
                metrics["kl_error_dist_biased"][model_class.__name__], best_offset_biased = optimize_KLD_offset(assumed_cdf, target_cdf)

                print("KLD between assumed and true distribution: ", metrics["kl_error_dist_unbiased"][model_class.__name__])
                print("KLD between assumed and true distribution: ", metrics["kl_error_dist_biased"][model_class.__name__])

            # Clear and delete the models just in case to reset the graph or whatever
            model_unbiased.clear()
            model_biased.clear()

            del model_unbiased
            del model_biased

    # Print the results
    if not is_running_on_cluster():
        if not optimize_hyperparameters_fg:
            print(
                pd.DataFrame.from_dict(
                    metrics,
                    orient='index'
                ).drop("mean_bias_per_item", axis=0)
            )
        else:
            print(
                pd.DataFrame.from_dict(
                    metrics,
                    orient='index'
                )
            )


    results = {
        "metrics": metrics,
        "other":{
            "est_pdf_unbiased" : est_pdf_unbiased,
            "est_pdf_biased" : est_pdf_biased,
            "best_offset_unbiased": best_offset_unbiased,
            "best_offset_biased": best_offset_biased,
            "biased_item_ranks": data["biased_item_ranks"],
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

    result_file_path = f"data/simulation_results/" + sub_dir + f"bias/{bias_mode}/{cdf_name}/{now_str}.json"

    store_results(results, result_file_path)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress warnings so that stdout only contains error messages

    # Get knwargs from command line
    parser = create_simulation_parser(parser_type="bias")
    args = parser.parse_args()

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
        bias_mode=args.bias_mode,
        model_name=args.model_name,
        optimize_hyperparameters_fg=args.optimize_hyperparameters_fg,
        hyperparameters=json.loads(args.hyperparameters) if args.hyperparameters is not None else None,
        debug_fg=args.debug_fg,
    )