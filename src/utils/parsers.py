import argparse

def create_simulation_parser(parser_type):
    """Creates the parser for the simulation settings."""

    assert parser_type in ["performance", "bias", "multi"], f"Invalid parser type: {parser_type}"

    parser = argparse.ArgumentParser(description='Simulation settings')
    parser.add_argument('--debug_fg', type=bool, default=False)

    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--n_items', type=int, default=500)
    parser.add_argument('--n_choices_per_user', type=int, default=250)
    parser.add_argument('--choiceSet_size', type=int, default=4) # Increasing this value will hinder the model from learning the left side of the target pdf
    
    parser.add_argument('--k', type=int, default=3)

    parser.add_argument('--n_kernels', type=int, default=5)

    if parser_type in ["performance", "bias"]:
        parser.add_argument('--cdf_loc', type=float, default=0.0)
        parser.add_argument('--cdf_scale', type=float, default=0.75)
        parser.add_argument("--cdf_name", type=str, default="gumbel", choices=["gumbel", "exponomial", "bimix_gaussian"])
        
        parser.add_argument("--model_name", type=str, default=None)
        parser.add_argument("--optimize_hyperparameters_fg", type=bool, default=False)
        parser.add_argument("--hyperparameters", type=str, default=None)
    
    if parser_type == "bias":
        parser.add_argument("--bias_mode", type=str, default="unbiased", choices=["unbiased", "overexposure", "competition"])

    if parser_type == "multi": # Evaluation
        parser.add_argument('--simulation_type', type=str, default="performance", choices=["performance", "bias"])
        parser.add_argument('--bias_modes', nargs='+', type=str, default=["unbiased", "overexposure", "competition"], choices=["unbiased", "overexposure", "competition"])
        parser.add_argument('--cdf_names', nargs='+', type=str, default=["gumbel", "exponomial", "bimix_gaussian"], choices=["gumbel", "exponomial", "bimix_gaussian"])
        parser.add_argument('--cdf_locs', nargs='+', type=float, default=[0., 0., 0.75])
        parser.add_argument('--cdf_scales', nargs='+', type=float, default=[0.75, 0.75, 0.25])
        parser.add_argument('--n_runs', type=int, default=4)
        parser.add_argument(
            '--model_names', 
            type=str,
            nargs='+', 
            default=["Recommender_multinomial_logit", "Recommender_exponomial", "Recommender_NPMLE", "Recommender_binary_logit", "Recommender_binary_logit_negative_sampling", "Recommender_gBCE"], 
            choices=["Recommender_multinomial_logit", "Recommender_exponomial", "Recommender_NPMLE", "Recommender_binary_logit", "Recommender_binary_logit_negative_sampling", "Recommender_gBCE"],
        )

    return parser