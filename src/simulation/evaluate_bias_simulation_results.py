import json
import os
import sys
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.utils.get_cdf import get_cdf
from src.utils.parsers import create_simulation_parser

def evaluate_simulation_results(args):
        
    # Load data
    result_table_df = pd.DataFrame()
    pd.options.display.float_format = "{:,.3f}".format
    for cdf_name in args.cdf_names:
        for bias_mode in args.bias_modes:
            
            dire_path = f"data/simulation_results/bias/{bias_mode}/{cdf_name}"

            # If path does not exist, print an error message and return
            if not os.path.exists(dire_path):
                print(f"No results found for {dire_path}")
                continue

            # Load all JSON files in the directory
            loaded_results = []
            for file in os.listdir(dire_path):
                if file.endswith(".json"):
                    try:
                        with open(f"{dire_path}/{file}", "r") as f:
                            loaded_results.append(json.load(f))
                    except Exception as e:
                        print(f"Could not load {file}: {e}")

            n_runs = len(loaded_results)

            print(f"Found {n_runs} results")

            # tmp
            loaded_results = [
                {key: val for key, val in r.items() if val != {}}
                for r in loaded_results
            ]

            # Add an absolute bias columns
            def add_abs_bias(r):
                r["metrics"]["abs_bias_per_item"] = {}
                for key in r["metrics"]["mean_bias_per_item"].keys():
                    r["metrics"]["abs_bias_per_item"][key] = np.abs(r["metrics"]["mean_bias_per_item"][key])
                return r
            loaded_results = [add_abs_bias(r) for r in loaded_results]

            def confidence_interval(val, n_runs):
                # Get length of val if val is a list or array
                if isinstance(val, (list, np.ndarray)) and len(val.shape) > 1:
                    l = val.shape[1]
                else:
                    l = 1

                # # Use scipy's' bootstrap instead
                # from scipy.stats import bootstrap
                # ci_l, ci_u = bootstrap(
                #     (val.flatten(), ),
                #     np.mean,
                #     method='percentile',
                #     confidence_level=0.95,
                # ).confidence_interval

                # # return ci_l, ci_u
                # print(f"Mean: {np.mean(val):.2f}, Center: {(ci_l + ci_u) / 2:.2f}, CI: {ci_l:.2f}, {ci_u:.2f}")
            
                return 1.96 * np.std(val) / np.sqrt(n_runs * l)

            
            aggregated_results_means_stds  = {
                metric_name : {
                    "_".join(model_name.split("_")[1:]) : (
                        f'{np.asarray([r["metrics"][metric_name][model_name] for r in loaded_results]).mean():.3f}'
                        + f' ± '
                        + f'{confidence_interval(np.asarray([r["metrics"][metric_name][model_name] for r in loaded_results]), n_runs=n_runs):.3f}'
                    )
                    for model_name in loaded_results[0]["metrics"][metric_name].keys()
                }
                for metric_name in loaded_results[0]["metrics"].keys()
            } 
            
            aggregated_results_means_stds["cdf_name"] = cdf_name
            aggregated_results_means_stds["bias_mode"] = bias_mode

            # Convert to pandas dataframe
            results_df = pd.DataFrame.from_dict(aggregated_results_means_stds)

            # Move cdf_name and bias_mode to the front
            results_df = results_df[["cdf_name", "bias_mode"] + [col for col in results_df.columns if col not in ["cdf_name", "bias_mode"]]]

            # Append to bias_result_table_df
            result_table_df = result_table_df._append(results_df, ignore_index=False)
        
    result_table_df["model_name"] = result_table_df.index

    # Rename columns to avoid underscores
    result_table_df.rename(
        columns={
            "cdf_name": "$\\epsilon$",
            "bias_mode": "Bias source",
            "model_name": "Model",
            "nDCG_unbiased": "$O'$ nDCG ↑",
            "nDCG_biased": "$O''$ nDCG ↑",
            "mean_bias_per_item": "Bias |·|↓",
            "abs_bias_per_item": "|Bias| ↓",
            "nll_unbiased": "$O'$ NLL ↓",
            "nll_biased": "$O''$ NLL ↓",
            "acc_unbiased": "$O'$ Acc ↑",
            "acc_biased": "$O''$ Acc ↑",
            "kl_unbiased": "$O'$ KL ↓",
            "kl_biased": "$O''$ KL ↓",
            "reverse_kl_unbiased": "$O'$ Reverse KL ↓",
            "reverse_kl_biased": "$O''$ Reverse KL ↓",
            "js_unbiased": "$O'$ JS ↓",
            "js_biased": "$O''$ JS ↓",
            "kl_error_dist_unbiased": "$O'$ KL $\\epsilon$ ↓",
            "kl_error_dist_biased": "$O''$ KL $\\epsilon$ ↓",
        },
        inplace=True
    )

    # Drop the kl, reverse kl and js columns because we did not evaluate them for the bias experiment
    result_table_df.drop(columns=["$O'$ KL ↓", "$O''$ KL ↓", "$O'$ Reverse KL ↓", "$O''$ Reverse KL ↓", "$O'$ JS ↓", "$O''$ JS ↓"], inplace=True)

    # Move columns "$\\epsilon$", "Bias source" and "Model" to the front
    result_table_df.set_index(["$\\epsilon$", "Bias source", "Model"], inplace=True)

    index = np.asarray(list(result_table_df.index), dtype="object")

    # For now, remove anything after the plus-minus symbol in all columns except for the bias one
    # for col in result_table_df.columns:
    #     if col != "Bias |·|↓":
    #         result_table_df[col] = result_table_df[col].apply(lambda x: x.split(" ± ")[0]).values

    # Sort according to "↑", "↓" and "|·|↓"
    # For each set of entries per error distribution and bias source, wrap the best value of every column into a \textbf{} command and the second best into a \underline{} command
    for cdf_name in np.unique(index[:, 0]):
        for bias_mode in np.unique(index[:, 1]):
            for col in result_table_df.columns:
                if col not in ["cdf_name", "bias_mode", "model_name"]:
                    # Get the best value (highest if column name contains "↑", lowest if column name contains "↓" and lowest absolute value if column name contains "|·|↓")
                    # Get the second best value (second highest if column name contains "↑", second lowest if column name contains "↓" and second lowest absolute value if column name contains "|·|↓")
                    if "|·|↓" in col:
                        # Remember to keep the values sign
                        best_value_arg = result_table_df.loc[(cdf_name, bias_mode)][col].apply(lambda x: x.split(" ± ")[0]).astype(float).abs().argmin()
                        best_value = result_table_df.loc[(cdf_name, bias_mode)][col].apply(lambda x: x.split(" ± ")[0]).astype(float).iloc[best_value_arg]
                        second_best_value_arg = result_table_df.loc[(cdf_name, bias_mode)][col].apply(lambda x: x.split(" ± ")[0]).astype(float).abs().argsort().iloc[1]
                        second_best_value = result_table_df.loc[(cdf_name, bias_mode)][col].apply(lambda x: x.split(" ± ")[0]).astype(float).iloc[second_best_value_arg]
                    elif "↑" in col:
                        best_value = result_table_df.loc[(cdf_name, bias_mode)][col].apply(lambda x: x.split(" ± ")[0]).astype(float).max()
                        second_best_value = result_table_df.loc[(cdf_name, bias_mode)][col].apply(lambda x: x.split(" ± ")[0]).astype(float).sort_values(ascending=False).iloc[1]
                    elif "↓" in col:
                        best_value = result_table_df.loc[(cdf_name, bias_mode)][col].apply(lambda x: x.split(" ± ")[0]).astype(float).min()
                        second_best_value = result_table_df.loc[(cdf_name, bias_mode)][col].apply(lambda x: x.split(" ± ")[0]).astype(float).sort_values().iloc[1]

                    # Wrap the best value into a \textbf{} command and the second best into a \underline{} command
                    result_table_df.loc[(cdf_name, bias_mode), col] = result_table_df.loc[(cdf_name, bias_mode)][col].apply(
                        lambda x: f"\\textbf{{{x}}}" if float(x.split(" ± ")[0]) == best_value else f"\\underline{{{x}}}" if float(x.split(" ± ")[0]) == second_best_value else x
                    ).values

    # Rename the index columns
    # For renaming the index columns we need to reset the index
    result_table_df.reset_index(inplace=True)

    result_table_df.replace(
        {
            "$\\epsilon$": {
                "gumbel": "Gumbel",
                "exponomial": "Sign. Exponential",
                "bimix_gaussian": "Gaussian Mix.",
            },
            "Bias source": {
                "unbiased": "Unbiased",
                "overexposure": "Overexposure",
                "competition": "Competition",
            },
            "Model": {
                "multinomial_logit": "MNL",
                "exponomial": "ENL",
                "NPMLE": "Ours",
            }
        },
        inplace=True
    )

    result_table_df.set_index(["$\\epsilon$", "Bias source", "Model"], inplace=True)

    # Replace " ± 0." with " ± ." in all cells to make the table more compact
    result_table_df = result_table_df.applymap(lambda x: x.replace(" ± 0.", " ± .") if " ± 0." in x else x)

    # Split the table into two tables, one for the bias and the other for all other metrics
    result_table_df_bias = result_table_df[["Bias |·|↓", "|Bias| ↓"]]
    result_table_df_performance = result_table_df[[col for col in result_table_df.columns if col not in result_table_df_bias.columns]]

    # In result_table_df_bias, remove the rows with Bias Mode "Unbiased" if they are even in there
    if "Unbiased" in result_table_df_bias.index.get_level_values("Bias source"):
        result_table_df_bias = result_table_df_bias.drop("Unbiased", level="Bias source")

    # For the columns of result_table_df_other create a multiindex so that the columns are grouped by the metric
    result_table_df_performance.columns = pd.MultiIndex.from_tuples(
        [(" ".join(col.split(" ")[1:]), " ".join(col.split(" ")[:1])) for col in result_table_df_performance.columns],
    ) 

    # # Split the performance table into two tables where the second one contains the kl, reverse kl and js metrics
    # result_table_df_divergence = result_table_df_performance[["KL ↓", "Reverse KL ↓", "JS ↓", "KL $\\epsilon$ ↓"]]
    # result_table_df_performance = result_table_df_performance[[col for col in result_table_df_performance.columns if col[0] not in ["KL ↓", "Reverse KL ↓", "JS ↓", "KL $\\epsilon$ ↓"]]]

    # Print the tables
    print(result_table_df_bias)
    # print(result_table_df_performance)
    # print(result_table_df_divergence)

    # Define file paths
    results_paths = {
        "bias": 
            {
                "table": result_table_df_bias,
                "path": "data/out/tables and figures/simulation_bias_results.tex",
            },
        # "performance":
        #     {
        #         "table": result_table_df_performance,
        #         "path": "data/out/tables/simulation_performance_results.tex",
        #     },
        # "divergence":
        #     {
        #         "table": result_table_df_divergence,
        #         "path": "data/out/tables/simulation_divergence_results.tex",
        #     },
    }

    for key, val in results_paths.items():
        table = val["table"]
        path = val["path"]

        # Check if the directories exist, if not create them
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Write the tables in LaTeX format
        table.to_latex(path, escape=False)

        # Open the .tex files, left-align the index columns, center align the other columns and replace {r} with {c}
        with open(path, "r") as f:
            lines = f.readlines()

        if key == "bias":
            with open(path, "w") as f:
                for line in lines:
                    f.write(
                        line
                        .replace("{llll}", "{lllc}")
                        .replace("{r}", "{c}")                        
                        .replace("Gumbel", r"{\rotatebox{90}{\hspace{-1.3cm}Gumbel}}")
                        .replace("Sign. Exponential", r"{\rotatebox[origin=br]{90}{\parbox{1.4cm}{\centering Sign.\\ Exponential}}}")
                        .replace("Gaussian Mix.", r"{\rotatebox[origin=br]{90}{\parbox{1.3cm}{\centering Gaussian\\ Mixture}}}")
                    )
        elif key == "performance": 
            with open(path, "w") as f:
                for line in lines:
                    f.write(
                        line
                        .replace("{lllllllll}", "{lllcccccc}")
                        .replace("{r}", "{c}")                    
                        .replace("Gumbel", r"{\rotatebox{90}{\hspace{-1.3cm}Gumbel}}")
                        .replace("Sign. Exponential", r"{\rotatebox{90}{\hspace{-1.6cm}Sign. Exponential}}")
                        .replace("Gaussian Mix.", r"{\rotatebox{90}{\hspace{-1.4cm}Gaussian Mixture}}")
                    )
        elif key == "divergence":
            with open(path, "w") as f:
                for line in lines:
                    f.write(line
                        .replace("{lllllllll}", "{lllcccccc}")
                        .replace("{r}", "{c}")                    
                        .replace("Gumbel", r"{\rotatebox{90}{\hspace{-1.3cm}Gumbel}}")
                        .replace("Sign. Exponential", r"{\rotatebox{90}{\hspace{-1.6cm}Sign. Exponential}}")
                        .replace("Gaussian Mixture", r"{\rotatebox{90}{\hspace{-1.4cm}Gaussian Mixture}}")
                    )

    # Compute the final bias values by a difference-in-differences approach
    # For each model and each error distribution, compute the difference between the models bias on its assumed distribution and the error distribution
    difference_in_differences = pd.DataFrame(columns=["$\\epsilon$", "Bias source", "MNL", "ENL"])

    difference_in_differences.set_index(["$\\epsilon$", "Bias source"], inplace=True)

    for cdf_name in result_table_df_bias.index.get_level_values("$\\epsilon$").unique():
        for bias_mode in result_table_df_bias.index.get_level_values("Bias source").unique():
            if bias_mode == "Unbiased":
                continue

            for model_name in result_table_df_bias.index.get_level_values("Model").unique():
                if model_name == "Ours":
                    continue

                # Get the bias of the model on its assumed distribution
                assumed_distribution = (
                    "Gumbel" if model_name == "MNL" else
                    "Sign. Exponential"
                )
                bias_assumed = result_table_df_bias.loc[(assumed_distribution, bias_mode, model_name), "Bias |·|↓"]
                if "{" in bias_assumed:
                    bias_assumed = bias_assumed[bias_assumed.find("{")+1:bias_assumed.find("}")] 
                bias_assumed = float(bias_assumed.split(" ± ")[0])
                
                # Get the bias of the model on the error distribution
                bias_error = result_table_df_bias.loc[(cdf_name, bias_mode, model_name), "Bias |·|↓"]
                if "{" in bias_error:
                    bias_error = bias_error[bias_error.find("{")+1:bias_error.find("}")]
                bias_error = float(bias_error.split(" ± ")[0])

                # Compute the difference in differences
                difference_in_differences.loc[(cdf_name, bias_mode), model_name] = bias_error - bias_assumed
                
    # Swap the index levels
    difference_in_differences = difference_in_differences.swaplevel(0, 1)

    # Sort the index
    difference_in_differences.sort_index(inplace=True)

    print(difference_in_differences)

    ## Plot estimated pdfs
    
    # Change axis font to libertine
    import matplotlib
    font = {'family' : 'serif'}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'font.family': 'serif'})

    # Create a 1x3 canvas
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for cdf_idx, cdf_name in enumerate(args.cdf_names):
        ax = axs[cdf_idx]
        for bias_idx, bias_mode in enumerate(args.bias_modes):
            print("Results for bias mode {} and CDF {}:".format(bias_mode, cdf_name))

            pd.options.display.float_format = "{:,.3f}".format
            
            dire_path = f"data/simulation_results/bias/{bias_mode}/{cdf_name}"

            # If path does not exist, print an error message and return
            if not os.path.exists(dire_path):
                print(f"No results found for {dire_path}")
                continue

            # Load all JSON files in the directory
            loaded_results = []
            for file in os.listdir(dire_path):
                if file.endswith(".json"):
                    try:
                        with open(f"{dire_path}/{file}", "r") as f:
                            loaded_results.append(json.load(f))
                    except Exception as e:
                        print(f"Could not load {file}: {e}")

            n_runs = len(loaded_results)

            print(f"Found {n_runs} results")

            x = (
                [np.asarray(r["other"]["est_pdf_unbiased"])[:,0] for r in loaded_results]
                + [np.asarray(r["other"]["est_pdf_biased"])[:,0] for r in loaded_results]
            )

            y = (
                [np.asarray(r["other"]["est_pdf_unbiased"])[:,1] for r in loaded_results]
                + [np.asarray(r["other"]["est_pdf_biased"])[:,1] for r in loaded_results]
            )
            
            # # Shift distributions to the sides to minimize the KLD to the true distribution
            # def KLD(x, y, cdf_name, offset):
            #     cdf = get_cdf(cdf_name, loc=args.cdf_loc, scale=args.cdf_scale)

            #     # Get interval scale
            #     x_steps = (x.max(axis=1) - x.min(axis=1)) / x.shape[1]

            #     y = np.tile(y, (x.shape[0], 1))

            #     klds = []

            #     for x_arr, y_arr, x_s, off in zip(x, y, x_steps, offset):

            #         # Expand y
            #         y_arr = np.concatenate(
            #             [
            #                 np.repeat(0, len(np.arange(np.min([-4, x_arr.min() + off]), x_arr.min() + off, x_s))),
            #                 y_arr,
            #                 np.repeat(0, len(np.arange(np.max([4, x_arr.max() + off]), x_arr.max() + off, x_s))),
            #             ]
            #         )#

            #         # Expand x
            #         x_arr = np.concatenate(
            #             [
            #                 np.arange(np.min([-4, x_arr.min() + off]), x_arr.min() + off, x_s),
            #                 x_arr,
            #                 np.arange(np.max([4, x_arr.max() + off]), x_arr.max() + off, x_s),
            #             ]
            #         )

            #         pdf = cdf.prob(x_arr + cdf.mean() + off) # Get pdf of centered distribution

            #         y_arr = y_arr * (1 - 1e-10) + 1e-10
            #         pdf = pdf * (1 - 1e-10) + 1e-10

            #         klds.append(
            #             np.sum(
            #                 pdf * np.log(pdf / y_arr),
            #             )
            #             * (x_arr.max()- x_arr.min()) / x_arr.shape[0]
            #         )

            #     return np.asarray(klds)
            
            # def get_best_offset(x_arr, y_arr, cdf_name, offset_min, offset_max, offset_interval):
            
            #     x_offset_arr = np.arange(offset_min, offset_max, offset_interval)

            #     kld_arr = KLD(
            #         x=np.tile(x_arr, (len(x_offset_arr), 1)), # + np.expand_dims(x_offset_arr, 1),
            #         y=y_arr,
            #         cdf_name=cdf_name,
            #         offset=x_offset_arr,
            #     )

            #     if cdf_name == "exponomial":
            #         print(x_offset_arr[np.argmin(kld_arr)])

            #         if np.abs(x_offset_arr[np.argmin(kld_arr)]) > 2:
            #             pass

            #     return x_arr + x_offset_arr[np.argmin(kld_arr)]

            # for arr_idx, (x_arr, y_arr) in enumerate(zip(x, y)):
            #     x_arr = get_best_offset(
            #         x_arr=x_arr,
            #         y_arr=y_arr,
            #         cdf_name=cdf_name,
            #         offset_min=-3,
            #         offset_max=3,
            #         offset_interval=0.1,
            #     )
                
            #     x_arr = get_best_offset(
            #         x_arr=x_arr,
            #         y_arr=y_arr,
            #         cdf_name=cdf_name,
            #         offset_min=-0.2,
            #         offset_max=0.2,
            #         offset_interval=0.01,
            #     )

            #     x[arr_idx] = x_arr

            # ax = axs[cdf_idx]

            for x_arr, y_arr in zip(x, y):
                ax.plot(
                    x_arr,
                    y_arr,
                    alpha=0.05,
                    solid_capstyle='butt',
                    c="blue"
                )

            # Plot cdf
            cdf = get_cdf(cdf_name, loc=args.cdf_locs[cdf_idx], scale=args.cdf_scales[cdf_idx])

            plot_x = np.linspace(
                np.min([ll for l in x for ll in l] + [cdf.quantile(0.0001).numpy() if type(cdf).__name__!='Mixture' else -3]), 
                np.max([ll for l in x for ll in l] + [cdf.quantile(0.9999).numpy() if type(cdf).__name__!='Mixture' else 3]), 
                1000,
            )

            plot_y = cdf.prob(plot_x).numpy()
            
            ax.plot(
                plot_x,
                plot_y,
                c="red",
            )

        ax.set_xlim(-3.5, 2.5)
        ax.set_ylim(0, 0.8)

        # Add a subtitle including the cdf name
        ax.set_title(f"{cdf_name.capitalize()}")

    # # Add a legend
    # axs[0].legend(["Estimated PDF", "True PDF"], loc='upper right')

    # # Set the legend colors correctly
    # axs[0].get_legend().legendHandles[0].set_color('blue')
    # axs[0].get_legend().legendHandles[1].set_color('red')
    
    # axs[0].get_legend().legendHandles[0].set_alpha(1)
    # axs[0].get_legend().legendHandles[1].set_alpha(1)

    # # Set the legend alpha level to 1
    # axs[0].get_legend().set_alpha(1)

    plt.show()

    # Save the figure
    fig.savefig("data/out/simulation_estimated_pdfs.svg")

            # # pdf data
            # pdf_data_unbiased = pd.DataFrame(
            #     np.concatenate([r["other"]["est_pdf_unbiased"] for r in loaded_results], axis=0),
            #     columns=["x", "pdf"]
            # )
            # pdf_data_biased = pd.DataFrame(
            #     np.concatenate([r["other"]["est_pdf_biased"] for r in loaded_results], axis=0),
            #     columns=["x", "pdf"]
            # )

            # plot_lowess(pdf_data_unbiased["x"], pdf_data_unbiased["pdf"], nonnegative=True)
            # plot_lowess(pdf_data_biased["x"], pdf_data_biased["pdf"], nonnegative=True)

            # Plot item poularity versus measured bias
            # from utils.lowess import plot_lowess
            # for model_name in loaded_results[0]["metrics"]["mean_bias_per_item"].keys():
            #     item_popularities = np.asarray([r["meta"]["biased_item_ranks"] for r in loaded_results]).flatten()
            #     item_biases = np.asarray([r["metrics"]["mean_bias_per_item"][model_name] for r in loaded_results]).flatten()
            #     plot_lowess(item_popularities, item_biases)
            
            # # Compare mean rank differences
            # rank_diffs_softmax = [r["mean_rank_diff_softmax"] for r in loaded_results]
            # rank_diffs_nonparametric = [r["mean_rank_diff_nonparametric"] for r in loaded_results]

            # mean_rank_diff_softmax = np.mean([r["mean_rank_diff_softmax"] for r in loaded_results])
            # mean_rank_diff_nonparametric = np.mean([r["mean_rank_diff_nonparametric"] for r in loaded_results])

            # mean_bias_per_item_softmax = np.mean([r["mean_bias_per_item_softmax"] for r in loaded_results], axis=0)
            # mean_bias_per_item_nonparametric = np.mean([r["mean_bias_per_item_nonparametric"] for r in loaded_results], axis=0)

            # # plot the mean rank differences for all items in two overlapping histograms
            # ranks_diffs_softmax_flattened = np.asarray(rank_diffs_softmax).flatten()
            # ranks_diffs_nonparametric_flattened = np.asarray(rank_diffs_nonparametric).flatten()
            # bins = np.linspace(
            #     np.min([ranks_diffs_softmax_flattened, ranks_diffs_nonparametric_flattened]), 
            #     np.max([ranks_diffs_softmax_flattened, ranks_diffs_nonparametric_flattened]), 
            #     100
            # )
            # # plt.hist(ranks_diffs_softmax_flattened, bins=bins, alpha=0.5, label='softmax', density=True)
            # # plt.hist(ranks_diffs_nonparametric_flattened, bins=bins, alpha=0.5, label='nonparametric', density=True)
            # # plt.legend(loc='upper right')
            # # plt.show()
            
            # # Create confidence intervals for the mean rank differences of the biased items
            # std_rank_diff_biased_items_softmax = np.std([r["mean_bias_per_item_softmax"] for r in loaded_results], axis=0)
            # std_rank_diff_biased_items_nonparametric = np.std([r["mean_bias_per_item_nonparametric"] for r in loaded_results], axis=0)

            # confidence_interval_rank_diff_biased_items_softmax = 1.96 * std_rank_diff_biased_items_softmax / np.sqrt(n_runs)
            # confidence_interval_rank_diff_biased_items_nonparametric = 1.96 * std_rank_diff_biased_items_nonparametric / np.sqrt(n_runs)

            # bias_results_df = pd.DataFrame(
            #     {
            #         "biased item popularity": [f"Top {i * 20}%" for i in range(5)],
            #         "mean rank diff softmax": [
            #             f"{m:.2f} +- {ci:.2f}"
            #             for m, ci in zip(mean_bias_per_item_softmax, confidence_interval_rank_diff_biased_items_softmax)
            #         ],
            #         "mean rank diff nonparametric": [
            #             f"{m:.2f} +- {ci:.2f}"
            #             for m, ci in zip(mean_bias_per_item_nonparametric, confidence_interval_rank_diff_biased_items_nonparametric)
            #         ]
            #     }
            # )
            # with pd.option_context('display.precision', 3):
            #     print(bias_results_df.head())

            # # Create the above plot in a 5x1 grid for each biased item
            # fig, axs = plt.subplots(5, 1, figsize=(4, 8), constrained_layout=True)
            # n_bins = 20
            # # Get minimum and maximum values for the x-axis
            # x_min = np.min(
            #     [
            #         np.min([r["mean_bias_per_item_softmax"] for r in loaded_results]), 
            #         np.min([r["mean_bias_per_item_nonparametric"] for r in loaded_results])
            #     ]
            # )
            # x_max = np.max(
            #     [
            #         np.max([r["mean_bias_per_item_softmax"] for r in loaded_results]), 
            #         np.max([r["mean_bias_per_item_nonparametric"] for r in loaded_results])
            #     ]
            # )

            # for i in range(5):
            #     # Apply min and max values to the x-axis
            #     axs[i].set_xlim(x_min, x_max)
            #     axs[i].hist([r["mean_bias_per_item_softmax"][i] for r in loaded_results], bins=n_bins, alpha=0.5, label='softmax', density=True)
            #     axs[i].hist([r["mean_bias_per_item_nonparametric"][i] for r in loaded_results], bins=n_bins, alpha=0.5, label='nonparametric', density=True)
            #     axs[i].legend(loc='upper right')

            #     # Set the title
            #     axs[i].set_title(f"Top {i * 20}% most popular item from set B")
            # plt.show()



if __name__ == "__main__":
    parser = create_simulation_parser("multi")
    args = parser.parse_args()

    evaluate_simulation_results(args)