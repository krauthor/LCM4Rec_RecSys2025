import json
import os
import sys
import pathlib        
import re     

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.utils.get_cdf import get_cdf
from src.utils.parsers import create_simulation_parser
from src.utils.hypothesis_tests import bootstrapped_two_sample_t_test

def evaluate_simulation_results(args):

    correct_param_model_by_dist = {
        "gumbel": "multinomial_logit",
        "exponomial": "exponomial",
        "bimix_gaussian": None,
    }
        
    # Load data
    result_table_df = pd.DataFrame()
    pd.options.display.float_format = "{:,.3f}".format
    for cdf_name in args.cdf_names:
        dire_path = f"data/simulation_results/performance/{cdf_name}"

        # If path does not exist, print an error message and continue
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

        # One result had an nan for the exponomial distribution, bl model and kl metric. Remove that one
        if cdf_name=="exponomial":
            for idx, r in enumerate(loaded_results):
                if np.isnan(r["metrics"]["kl"]["Recommender_binary_logit"]):
                    loaded_results.remove(r)
                    break

        def confidence_interval(val, n_runs):
            # Get length of val if val is a list or array
            if isinstance(val, (list, np.ndarray)) and len(val.shape) > 1:
                l = val.shape[1]
            else:
                l = 1

            return 1.96 * np.std(val) / np.sqrt(n_runs * l)
        
        def repl_None(list):
            return [0 if x is None else x for x in list]
        
        aggregated_results_means_stds  = {
            metric_name : {
                "_".join(model_name.split("_")[1:]) : (
                    f'{np.asarray(repl_None([r["metrics"][metric_name][model_name] for r in loaded_results])).mean():.3f}'
                    + f' ± '
                    + f'{confidence_interval(np.asarray(repl_None([r["metrics"][metric_name][model_name] for r in loaded_results])), n_runs=n_runs):.3f}'
                )
                for model_name in loaded_results[0]["metrics"][metric_name].keys()
            }
            for metric_name in loaded_results[0]["metrics"].keys()
        } 
        
        aggregated_results_means_stds["cdf_name"] = cdf_name

        # Convert to pandas dataframe
        results_df = pd.DataFrame.from_dict(aggregated_results_means_stds)

        # Move cdf_name and bias_mode to the front
        results_df = results_df[["cdf_name"] + [col for col in results_df.columns if col not in ["cdf_name"]]]

        # Append to bias_result_table_df
        result_table_df = result_table_df._append(results_df, ignore_index=False)
    
    result_table_df["model_name"] = result_table_df.index

    # # Rename columns to avoid underscores
    # col_name_to_fancy_dict = {
    #     "cdf_name": "$\\epsilon$",
    #     "model_name": "Model",
    #     "nDCG": "nDCG ↑",
    #     "nll": "NLL ↓",
    #     "acc": "Acc ↑",
    #     "kl": "KLD ↓",
    #     "reverse_kl": "Reverse KLD ↓",
    #     "js": "JS ↓",
    #     "kl_error_dist": "KLD $\\epsilon$ ↓",
    # }
    # result_table_df.rename(
    #     columns=col_name_to_fancy_dict,
    #     inplace=True
    # )

    # # Move columns "$\\epsilon$" and "Model" to the front
    # result_table_df.set_index(["$\\epsilon$", "Model"], inplace=True)

    # # Sort according to "↑", "↓" and "|·|↓"
    # # For each set of entries per $\\epsilon$ribution and bias source, wrap the best value of every column into a \textbf{} command and the second best into a \underline{} command
    # for cdf_name in set(args.cdf_names).intersection(np.asarray(list(result_table_df.index))[:,0]):
        
    #     dire_path = f"data/simulation_results/performance/{cdf_name}"

    #     # If path does not exist, print an error message and continue
    #     if not os.path.exists(dire_path):
    #         print(f"No results found for {dire_path}")
    #         continue

    #     # Load all JSON files in the directory
    #     loaded_results = []
    #     for file in os.listdir(dire_path):
    #         if file.endswith(".json"):
    #             try:
    #                 with open(f"{dire_path}/{file}", "r") as f:
    #                     loaded_results.append(json.load(f))
    #             except Exception as e:
    #                 print(f"Could not load {file}: {e}")

    #     for col_name in result_table_df.columns:
    #         if col_name not in ["cdf_name", "model_name"]:
    #             # Get the best value (highest if column name contains "↑", lowest if column name contains "↓" and lowest absolute value if column name contains "|·|↓")
    #             # Get the second best value (second highest if column name contains "↑", second lowest if column name contains "↓" and second lowest absolute value if column name contains "|·|↓")
    #             cond = (
    #                 ~result_table_df.index.isin(
    #                     [("exponomial", "exponomial"),
    #                     ("gumbel", "multinomial_logit"),]
    #                 )
    #                 &
    #                 (result_table_df.index.get_level_values(0) == cdf_name)
    #             )
                
    #             rows = result_table_df[cond][col_name]
    #             vals = rows.apply(lambda x: x.split(" ± ")[0]).astype(float)
                
    #             if "|·|↓" in col_name:
    #                 # Remember to keep the values sign
    #                 best_value_arg = vals.abs().argmin()
    #                 best_value = vals.iloc[best_value_arg]
    #                 second_best_value_arg = vals.abs().argsort().iloc[1]
    #                 second_best_value = vals.iloc[second_best_value_arg]
    #             elif "↑" in col_name:
    #                 best_value = vals.max()
    #                 second_best_value = vals.sort_values(ascending=False).iloc[1]
    #             elif "↓" in col_name:
    #                 best_value = vals.min()
    #                 second_best_value = vals.sort_values().iloc[1]

    #             # Wrap the best value into a \textbf{} command and the second best into a \underline{} command
    #             new_col = rows.apply(
    #                 lambda x: f"\\textbf{{{x}}}" if float(x.split(" ± ")[0]) == best_value else f"\\underline{{{x}}}" if float(x.split(" ± ")[0]) == second_best_value else x
    #             )

    #             result_table_df.update(new_col)

    #             # Hypothesis tests   
    #             cond = (result_table_df.index.get_level_values(0) == cdf_name)
    #             rows = result_table_df[cond][col_name]
    #             vals = rows.apply(lambda x: re.findall('{(.+?)}',x)[0].split(" ± ")[0] if "{" in x else x.split(" ± ")[0]).astype(float)

    #             # For each value, get the next worst value
    #             # Then, get the corresponding values from loaded_results and the corresponding models and test if the difference is significant
    #             for model_idx, value in vals.items():
    #                 if "↓" in col_name:
    #                     if value == vals.max():
    #                         continue
    #                     # next_worst_value = vals[vals > value].min()
    #                     next_worst_value_idx = vals[vals > value].idxmin()
    #                 elif "↑" in col_name:
    #                     if value == vals.min():
    #                         continue
    #                     # next_worst_value = vals[vals < value].max()
    #                     next_worst_value_idx = vals[vals < value].idxmax()
    #                 elif "|·|↓" in col_name:
    #                     if value == vals.abs().max():
    #                         continue
    #                     # next_worst_value = vals.abs()[vals.abs() > value].min()
    #                     next_worst_value_idx = vals.abs()[vals.abs() > value].idxmin()
                    
    #                 # Get the corresponding models
    #                 model = model_idx[1]
    #                 next_worst_model = next_worst_value_idx[1]

    #                 # Get the original name of the metric by reversing the dictionary
    #                 orig_col_name = {v: k for k, v in col_name_to_fancy_dict.items()}[col_name]

    #                 # Get the corresponding values from loaded_results
    #                 results_this_model = [r["metrics"][orig_col_name]["Recommender_" + model] for r in loaded_results if r["parameters"]["cdf_name"] == cdf_name]
    #                 results_next_worst_model = [r["metrics"][orig_col_name]["Recommender_" + next_worst_model] for r in loaded_results if r["parameters"]["cdf_name"] == cdf_name]

    #                 # Perform a two-sample t-test
    #                 # Some metrics are only defined for multivariate models, so we need to check if the results are not all None
    #                 if not results_this_model==[None]*len(results_this_model) and not results_next_worst_model==[None]*len(results_next_worst_model):
    #                     p_value = bootstrapped_two_sample_t_test(
    #                         np.asarray(results_this_model),
    #                         np.asarray(results_next_worst_model),
    #                         B=100000,
    #                         twosided=False,
    #                     )

    #                 # If the p-value is smaller than 0.1, add a star, if it is smaller than 0.05, add two stars, if it is smaller than 0.01, add three stars
    #                 # If the cell value contains curly braces, add the stars before the closing curly brace
    #                 if "}" in result_table_df.loc[model_idx, col_name]:
    #                     if p_value < 0.1:
    #                         result_table_df.loc[model_idx, col_name] = result_table_df.loc[model_idx, col_name].replace("}", "*}")
    #                     if p_value < 0.05:
    #                         result_table_df.loc[model_idx, col_name] = result_table_df.loc[model_idx, col_name].replace("}", "*}")
    #                     if p_value < 0.01:
    #                         result_table_df.loc[model_idx, col_name] = result_table_df.loc[model_idx, col_name].replace("}", "*}")
    #                 else:
    #                     if p_value < 0.1:
    #                         result_table_df.loc[model_idx, col_name] += "*"
    #                     if p_value < 0.05:
    #                         result_table_df.loc[model_idx, col_name] += "*"
    #                     if p_value < 0.01:
    #                         result_table_df.loc[model_idx, col_name] += "*"

    # # Rename the index columns
    # # For renaming the index columns we need to reset the index
    # result_table_df.reset_index(inplace=True)

    # result_table_df.replace(
    #     {
    #         "$\\epsilon$": {
    #             "gumbel": "Gumbel",
    #             "exponomial": "Sign. Exponential",
    #             "bimix_gaussian": "Gaussian Mix.",
    #         },
    #         "Model": {
    #             "multinomial_logit": "MNL",
    #             "exponomial": "ENL",
    #             "NPMLE": "LCM4Rec",
    #             "binary_logit": "BL",
    #             "binary_logit_negative_sampling": "BCE",
    #             "gBCE": "gBCE",
    #         }
    #     },
    #     inplace=True
    # )

    # result_table_df.set_index(["$\\epsilon$", "Model"], inplace=True)

    # # Replace " ± 0." with " ± ." in all cells to make the table more compact
    # result_table_df = result_table_df.applymap(lambda x: x.replace(" ± 0.", " ± .") if " ± 0." in x else x)

    # # Actually, just remove the confidence intervals for now
    # result_table_df = result_table_df.applymap(lambda x: x.split(" ±")[0] + "*"*x.count("*") if " ±" in x else x)
    # result_table_df = result_table_df.applymap(lambda x: x+"}" if "{" in x else x)

    # # # Memorize KLDs to the true distribution for later
    # # klds = result_table_df["KLD $\\epsilon$ ↓"].copy() # THESE ARE OUTDATED DUE TO INACCURATE SHIFTING

    # result_table_df = result_table_df[[
    #     "KLD ↓",
    #     "NLL ↓",
    #     "nDCG ↑",
    #     "Acc ↑",
    # ]]

    # # Append a star to the Model name when its assumption matches the true error distribution
    # as_list = result_table_df.index.tolist()
    # idx = as_list.index(('Gumbel', 'MNL'))
    # as_list[idx] = ('Gumbel', 'MNL$^*$')
    # idx = as_list.index(('Sign. Exponential', 'ENL'))
    # as_list[idx] = ('Sign. Exponential', 'ENL$^*$')
    # result_table_df.index = as_list


    # # Re-order the Model column to "BCE", "gBCE", "BL", "MNL", "ENL", "LCM4Rec"
    # result_table_df = result_table_df.reindex(
    #     pd.MultiIndex.from_product(
    #         [
    #             ["Gumbel", "Sign. Exponential", "Gaussian Mix."],
    #             [
    #                 "BCE",
    #                 "gBCE",
    #                 "BL",
    #                 "MNL",
    #                 "MNL$^*$",
    #                 "ENL",
    #                 "ENL$^*$",
    #                 "LCM4Rec",
    #             ],
    #         ]
    #     ),
    # ).dropna()

    # # Print the table
    # print(result_table_df)

    # # print(klds)

    # path = "data/out/tables and figures/simulation_performance_results.tex"

    # # Check if the directories exist, if not create them
    # if not os.path.exists(os.path.dirname(path)):
    #     os.makedirs(os.path.dirname(path))

    # # Write the tables in LaTeX format
    # result_table_df.to_latex(path, escape=False)

    # # Open the .tex files, left-align the index columns, center align the other columns and replace {r} with {c}
    # with open(path, "r") as f:
    #     lines = f.readlines()

    # with open(path, "w") as f:
    #     for line in lines:
    #         f.write(
    #             line
    #             .replace("{llll}", "{lllc}")
    #             .replace("{r}", "{c}")
    #             .replace("Gumbel", r"{\rotatebox{90}{\hspace{-1.3cm}Gumbel}}")
    #             .replace("Sign. Exponential", r"{\rotatebox[origin=br]{90}{\parbox{1.4cm}{\centering Sign.\\ Exponential}}}")
    #             .replace("Gaussian Mix.", r"{\rotatebox[origin=br]{90}{\parbox{1.3cm}{\centering Gaussian\\ Mixture}}}")
    #         )

    #         # Add a horizontal line after the BL model row
    #         if "BL" in line:
    #             f.write(r"\cline{2-6}" + "\n")

    # Plot estimated pdfs

    def pdf_ys_to_cdf_ys(pdf_ys):
        """Convert pdf values to cdf values"""
        return np.cumsum(pdf_ys) / np.sum(pdf_ys)
    
    # Change axis font to libertine
    import matplotlib

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14 # 18
    BIGGER_SIZE = 16 # 20

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    font = {'family' : 'serif'}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'font.family': 'serif'})

    # Create a 2x3 canvas
    fig, axs = plt.subplots(
        2, 
        3, 
        # figsize=(18, 5.16), 
        figsize=(18, 4.2),
        constrained_layout=True, 
        sharex=True, 
        # sharey=True, 
        # gridspec_kw={'height_ratios': [1.1, 1.65]},
    )

    for plot_type in["cdf", "pdf"]:
        for cdf_idx, (cdf_name, cdf_loc, cdf_scale) in enumerate(zip(args.cdf_names, args.cdf_locs, args.cdf_scales)):

            print(f"Results for CDF {cdf_name}:")

            pd.options.display.float_format = "{:,.3f}".format
            
            dire_path = f"data/simulation_results/performance/{cdf_name}"

            # If path does not exist, print an error message and continue with the next cdf
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


            # loaded_results = loaded_results[:10]

            n_runs = len(loaded_results)

            print(f"Found {n_runs} results")

            if n_runs == 0:
                print(f"No results found for {dire_path}")
                continue

            x = [np.asarray(r["other"]["est_pdf"])[:,0] for r in loaded_results]
            y = [np.asarray(r["other"]["est_pdf"])[:,1] for r in loaded_results]
            
            # # Shift distributions to the sides to minimize the KLD to the true distribution
            def KLD(x, y, cdf_name, cdf_scale, cdf_loc, offset):
                cdf = get_cdf(cdf_name, loc=cdf_loc, scale=cdf_scale)

                lower_bound = cdf.quantile(0.001).numpy() if type(cdf).__name__ != 'Mixture' else -7
                upper_bound = cdf.quantile(0.999).numpy() if type(cdf).__name__ != 'Mixture' else 7

                # Get interval scale
                x_steps = (x.max(axis=1) - x.min(axis=1)) / x.shape[1]

                y = np.tile(y, (x.shape[0], 1))

                klds = []

                for x_arr, y_arr, x_s, off in zip(x, y, x_steps, offset):

                    if 1.15<off<1.25:
                        pass
                    if -1.15>off>-1.25:
                        pass
                    if -0.1<off<0.1:
                        pass

                    lower_bound_offset = lower_bound - off
                    upper_bound_offset = upper_bound - off

                    # Expand y
                    y_arr = np.concatenate(
                        [
                            np.repeat(0, len(np.arange(np.min([lower_bound_offset, x_arr.min()]), x_arr.min(), x_s))),
                            y_arr,
                            np.repeat(0, len(np.arange(x_arr.max(), np.max([upper_bound_offset, x_arr.max()]), x_s))),
                        ]
                    )#

                    # Expand x
                    x_arr = np.concatenate(
                        [
                            np.arange(np.min([lower_bound_offset, x_arr.min()]), x_arr.min(), x_s),
                            x_arr,
                            np.arange(x_arr.max(), np.max([upper_bound_offset, x_arr.max()]), x_s),
                        ]
                    )

                    pdf = cdf.prob(x_arr + off) # Get pdf of negatively shifted distribution

                    y_arr = y_arr * (1 - 1e-10) + 1e-10
                    pdf = pdf * (1 - 1e-10) + 1e-10

                    klds.append(
                        np.sum(
                            pdf * np.log(pdf / y_arr),
                        )
                        * (x_arr.max()- x_arr.min()) / x_arr.shape[0]
                    )

                return np.asarray(klds)
            
            def get_offset_mean_diff(x_arr, y_arr, cdf_name, cdf_loc, cdf_scale, offset):
                """Compute the difference in means of the offset cdf from x_arr and y_arr and the true cdf"""

                offset_mean = np.random.choice(
                    a=x_arr + offset, 
                    p=y_arr/np.sum(y_arr), 
                    replace=True, 
                    size=10000,
                ).mean()

                cdf_mean = get_cdf(cdf_name, loc=cdf_loc, scale=cdf_scale).mean().numpy()

                return offset_mean - cdf_mean
            
            def get_best_offset(x_arr, y_arr, cdf_name, cdf_loc, cdf_scale, offset_min, offset_max, offset_interval):
            
                x_offset_arr = np.arange(offset_min, offset_max, offset_interval)

                kld_arr = KLD(
                    x=np.tile(x_arr, (len(x_offset_arr), 1)), # + np.expand_dims(x_offset_arr, 1),
                    y=y_arr,
                    cdf_name=cdf_name,#
                    cdf_loc=cdf_loc,
                    cdf_scale=cdf_scale,
                    offset=x_offset_arr,
                )

                # if cdf_name == "exponomial":
                #     print(x_offset_arr[np.argmin(kld_arr)])
                #     print(kld_arr)

                #     if np.abs(x_offset_arr[np.argmin(kld_arr)]) > 2:
                #         pass

                return x_arr + x_offset_arr[np.argmin(kld_arr)], x_offset_arr[np.argmin(kld_arr)], np.min(kld_arr)

            best_klds = []

            for arr_idx, (x_arr, y_arr) in enumerate(zip(x, y)):
                x_arr, _, _ = get_best_offset(
                    x_arr=x_arr,
                    y_arr=y_arr,
                    cdf_name=cdf_name,
                    cdf_loc=cdf_loc,
                    cdf_scale=cdf_scale,
                    offset_min=-3,
                    offset_max=3,
                    offset_interval=0.1,
                )
                
                x_arr, best_offset, best_kld = get_best_offset(
                    x_arr=x_arr,
                    y_arr=y_arr,
                    cdf_name=cdf_name,
                    cdf_loc=cdf_loc,
                    cdf_scale=cdf_scale,
                    offset_min=-0.2,
                    offset_max=0.2,
                    offset_interval=0.01,
                )

                x[arr_idx] = x_arr

                best_klds.append(best_kld)

            print("Mean best KLD for cdf", cdf_name, ":", np.mean(best_klds), "±", np.std(best_klds))

            plot_x_lims = {
                "gumbel": (-3, 4.5),
                "exponomial": (-5.5, 2),
                "bimix_gaussian": (-3, 3),
            }
            if plot_type == "pdf":
                plot_y_lims = {
                    "gumbel": (-0.05, 1.05),
                    "exponomial": (-0.05, 1.05),
                    "bimix_gaussian": (-0.05, 1.6),
                }
            else:
                plot_y_lims = {
                    "gumbel": (-0.05, 1.05),
                    "exponomial": (-0.05, 1.05),
                    "bimix_gaussian": (-0.05, 1.05),
                }

            plot_x_lims_all = (np.min([v for v in plot_x_lims.values()]), np.max([v for v in plot_x_lims.values()]))
            plot_y_lims_all = (np.min([v for v in plot_y_lims.values()]), np.max([v for v in plot_y_lims.values()]))

            plot_colors = { # Estimated CDF is already blue
                "gumbel": "red",
                "exponomial": "purple",
                "bimix_gaussian": "orange",
            }

            ax = axs[0 if plot_type=="cdf" else 1][cdf_idx]
            
            ax.set_xlim(plot_x_lims_all)
            # ax.set_ylim(plot_y_lims_all)

            for x_arr, y_arr in zip(x, y):
                # Plot within the plot limits of the cdf
                x_arr = np.asarray(x_arr)
                y_arr = np.asarray(y_arr)
                
                # y_arr = y_arr[(x_arr >= plot_x_lims[cdf_name][0]) & (x_arr <= plot_x_lims[cdf_name][1])]
                # x_arr = x_arr[(x_arr >= plot_x_lims[cdf_name][0]) & (x_arr <= plot_x_lims[cdf_name][1])]
                
                y_arr = y_arr[(x_arr >= plot_x_lims_all[0]) & (x_arr <= plot_x_lims_all[1])]
                x_arr = x_arr[(x_arr >= plot_x_lims_all[0]) & (x_arr <= plot_x_lims_all[1])]

                ax.plot(
                    x_arr,
                    y_arr if plot_type=="pdf" else pdf_ys_to_cdf_ys(y_arr),
                    alpha=0.2,
                    linewidth=0.5,
                    solid_capstyle='butt',
                    c="lightblue"
                )

            # Plot the mean line of the estimated pdfs
            x_mean = np.arange(plot_x_lims_all[0], plot_x_lims_all[1], 0.01)
            y_mean = [
                np.mean(
                    np.reshape(
                        [
                            l
                            for x_arr, y_arr in zip(x, y)
                            if np.sum(np.abs(x_arr-x_pos) <= 0.025) > 0
                            for l in y_arr[np.abs(x_arr-x_pos) <= 0.025]
                        ],
                        -1
                    )
                )
                for x_pos in x_mean
            ]

            # Replace all NaNs in y_mean with 0
            y_mean = np.nan_to_num(y_mean)

            ax.plot(
                x_mean,
                y_mean if plot_type=="pdf" else pdf_ys_to_cdf_ys(y_mean),
                c="blue",
                linewidth=2,
            )

            # Plot cdf
            cdf = get_cdf(cdf_name, loc=cdf_loc, scale=cdf_scale)

            plot_x = np.arange(
                plot_x_lims_all[0],
                plot_x_lims_all[1],
                0.01,
            )
            
            # Plot other cdfs dashed
            for alt_cdf_name in args.cdf_names:
                if alt_cdf_name != cdf_name:
                    alt_cdf = get_cdf(alt_cdf_name, loc=args.cdf_locs[args.cdf_names.index(alt_cdf_name)], scale=args.cdf_scales[args.cdf_names.index(alt_cdf_name)])
                    plot_y = alt_cdf.prob(plot_x).numpy() if plot_type=="pdf" else alt_cdf.cdf(plot_x).numpy()
                    ax.plot(
                        plot_x,
                        plot_y,
                        linewidth=0.75,
                        c=plot_colors[alt_cdf_name],
                        linestyle="--",
                    )

            plot_y = cdf.prob(plot_x).numpy() if plot_type=="pdf" else cdf.cdf(plot_x).numpy()
            ax.plot(
                plot_x,
                plot_y,
                c=plot_colors[cdf_name],
            )

            # Add a subtitle including the cdf name
            subtitle_plot_name = {
                "gumbel": "Gumbel",
                "exponomial": "Sign. Exponential",
                "bimix_gaussian": "Gaussian Mixture",
            }

            if plot_type == "cdf":
                ax.set_title(subtitle_plot_name[cdf_name])
                if cdf_idx == 0:
                    ax.set_ylabel("CDF")
            else:
                if cdf_idx == 0:
                    ax.set_ylabel("PDF")

            if plot_type == "cdf":
                # Compute KLD score between the mean estimated cdf and the true cdf
                smoothed_pdf_ys = cdf.prob(x_mean).numpy() * (1 - 1e-10) + 1e-10
                smoothed_estimated_cdf_ys = np.asarray(y_mean) * (1 - 1e-10) + 1e-10
                KLD_of_mean_estimate = (
                    (smoothed_pdf_ys * np.log(smoothed_pdf_ys / smoothed_estimated_cdf_ys)).sum() 
                    / len(x_mean) * (x_mean.max()-x_mean.min())
                )
                print(f"KLD of mean estimate for {cdf_name}: {KLD_of_mean_estimate:.2f}")
                # Annotate at the top left of the plot
                ax.annotate(
                    xy=(ax.get_xlim()[0] + 0.3, ax.get_ylim()[1] - 0.14), 
                    text=f"KLD: {KLD_of_mean_estimate:.2f}", 
                    fontsize=10,
                    horizontalalignment='left',
                )

            # Include a legend where the lines of the other cdfs are dashed and without a border frame
            if plot_type == "pdf":
                ax.legend(
                    [
                        "LCM4Rec\n(Mean estimate)",
                        "LCM4Rec\n(Individual estimates)",
                        "Gumbel",
                        "Sign. Exponential",
                        "Gaussian Mixture",
                    ],
                    loc="upper left",
                    frameon=False,
                    handlelength=0.75,
                )

                legend_linestyles = {
                    "LCM4Rec\n(Mean estimate)": "-",
                    "Gumbel": "-" if cdf_name == "gumbel" else "--",
                    "Sign. Exponential": "-" if cdf_name == "exponomial" else "--",
                    "Gaussian Mixture": "-" if cdf_name == "bimix_gaussian" else "--",
                    "LCM4Rec\n(Individual estimates)": "-",
                }

                legend_linewidths = {
                    "LCM4Rec\n(Mean estimate)": 2,
                    "Gumbel": 1 if cdf_name == "gumbel" else 0.75,
                    "Sign. Exponential": 1 if cdf_name == "exponomial" else 0.75,
                    "Gaussian Mixture": 1 if cdf_name == "bimix_gaussian" else 0.75,
                    "LCM4Rec\n(Individual estimates)": 0.5,
                }

                legend_colors = {
                    "LCM4Rec\n(Mean estimate)": "blue",
                    "Gumbel": "red",
                    "Sign. Exponential": "purple",
                    "Gaussian Mixture": "orange",
                    "LCM4Rec\n(Individual estimates)": "lightblue",
                }

                # Set the color, linestyle and linewidth inside the legend
                for line, text in zip(ax.get_legend().get_lines(), ax.get_legend().get_texts()):
                    legend_cdf_name = text.get_text()
                    line.set_color(legend_colors[legend_cdf_name])
                    line.set_linestyle(legend_linestyles[legend_cdf_name])
                    line.set_linewidth(legend_linewidths[legend_cdf_name])
                    line.set_alpha(1)

    plt.show()

    # # Save the figure
    fig.savefig("data/out/tables and figures/simulation_estimated_pdfs.svg")

    # plt.tight_layout()

    # Save the figure
    # plt.savefig("data/out/tables and figures/simulation_estimated_pdfs.svg", bbox_inches='tight')


if __name__ == "__main__":
    parser = create_simulation_parser(parser_type="multi")
    args = parser.parse_args()

    evaluate_simulation_results(args)