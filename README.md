# LCM4Rec: Learned Choice Model for Recommendation

This repository contains the implementation of [_A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options_](https://dl.acm.org/doi/full/10.1145/3705328.3748090), presented at RecSys'25.

The creation of this ReadMe was supported by GitHub Copilot.

## Overview

LCM4Rec is a research project that implements Non-Parametric Maximum Likelihood Estimation methods for recommendation systems. The project focuses on modeling user preferences and item utilities using general distributional assumptions and comparing their performance against traditional modeling approaches.

### Key Features

- **Multiple Model Implementations**: Including Multinomial Logit (MNL), Exponomial (ENL), Binary Logit, [gBCE](https://github.com/asash/gSASRec-pytorch), BCE, and the proposed LCM4Rec approach
- **Performance Evaluation**: Evaluation of LCM4Rec's ability to recover ground-truth choice models
- **Bias Analysis**: Evaluation of the effect of mis-specifying the choice model on exposure bias robustness

## Repository Structure

```
├── src/
│   ├── Models/              # Model implementations
│   │   ├── npmle.py         # Main NPMLE model
│   │   ├── multinomial_logit.py
│   │   ├── exponomial.py
│   │   ├── binary_logit.py
│   │   └── recommenders.py  # Model wrappers
│   ├── simulation/          # Simulation scripts
│   │   ├── run_simulation_performance.py
│   │   ├── run_simulation_bias.py
│   │   ├── evaluate_performance_simulation_results.py
│   │   └── evaluate_bias_simulation_results.py
│   └── utils/               # Utility functions
├── data/
│   ├── parameters/          # Hyperparameter configurations
│   └── simulation_results/  # Output directory for results
├── notebooks/               # Jupyter notebooks for analysis
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.13.0

### Setup

1. Clone the repository:
```bash
git clone https://github.com/krauthor/LCM4Rec_RecSys2025
cd LCM4Rec_RecSys2025
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Performance Simulations

To evaluate model performance across different error distributions:

```bash
python src/simulation/run_simulation_performance.py \
    --cdf_name gumbel \
    --n_users 500 \
    --n_items 500 \
    --n_choices_per_user 250 \
    --choiceSet_size 4 \
    --k 3 \
    --n_kernels 5
```

**Parameters:**
- `--cdf_name`: Error distribution type (`gumbel`, `exponomial`, `bimix_gaussian`)
- `--n_users`: Number of simulated users
- `--n_items`: Number of items in the catalog
- `--n_choices_per_user`: Number of choice observations per user
- `--choiceSet_size`: Size of choice sets presented to users
- `--k`: Embedding dimension
- `--n_kernels`: Number of kernels for NPMLE estimation

Please see `src/utils/parsers.py` for additional configuration options.

### Running Bias Simulations

To evaluate model behavior under different bias conditions:

```bash
python src/simulation/run_simulation_bias.py \
    --cdf_name gumbel \
    --bias_mode overexposure \
    --n_users 500 \
    --n_items 500 \
    --n_choices_per_user 250 \
    --choiceSet_size 4
```

Please see `src/utils/parsers.py` for additional configuration options.

**Bias Modes:**
- `unbiased`: No bias in data generation
- `overexposure`: Models exposure bias effects
- `competition`: Models competition bias effects

### Available Models
- `Recommender_NPMLE`: Our proposed method LCM4Rec
- `Recommender_multinomial_logit`: Multinomial Logit (MNL)
- `Recommender_exponomial`: Exponomial (ENL)
- `Recommender_binary_logit`: Binary Logit
- `Recommender_binary_logit_negative_sampling`: Binary Logit with Negative Sampling
- `Recommender_gBCE`: Generalized Binary Cross Entropy

## Evaluation

### Performance Evaluation

After running simulations, evaluate the results:

```bash
python src/simulation/evaluate_performance_simulation_results.py \
    --cdf_names gumbel exponomial bimix_gaussian
```

This generates performance tables comparing models across different metrics:
- **nDCG**: Normalized Discounted Cumulative Gain
- **NLL**: Negative Log-Likelihood
- **Accuracy**: Prediction accuracy
- **KL Divergence**: Kullback-Leibler divergence measures

### Bias Evaluation

Evaluate bias simulation results:

```bash
python src/simulation/evaluate_bias_simulation_results.py \
    --cdf_names gumbel exponomial bimix_gaussian
```

This produces bias analysis including:
- Mean bias per item
- Absolute bias measurements
- Difference-in-differences analysis
- Estimated PDF visualizations

### Results Location

Results are automatically saved to:
- `data/simulation_results/performance/[cdf_name]/` - Performance simulation outputs
- `data/simulation_results/bias/[bias_mode]/[cdf_name]/` - Bias simulation outputs
- `data/out/tables and figures/` - Generated tables and figures

## Hyperparameters

Hyperparameters are configured in:
- `data/parameters/simulation/hyperparameter_grid.json` - Search grid
- `data/parameters/simulation/tuned_hyperparameters.json` - Optimized values

## Notebooks

Interactive analysis notebooks used for generating tables and plots in the paper are available in the `notebooks/` directory:

- `cannibalisation_example.ipynb`: Demonstrates cannibalization effects
- `Exposure_bias_plot.ipynb`: Visualizes exposure bias analysis

## Configuration

Key configuration options in `src/settings.py`:

```python
USE_MONTE_CARLO = True                    # Enable Monte Carlo sampling, default True. False disables Monte Carlo sampling, deprecated.
MONTE_CARLO_N_SAMPLES_PER_KERNEL = 5     # Number of samples per kernel
HYPERPARAMETER_PATH = "data/parameters/simulation/tuned_hyperparameters.json"
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{10.1145/3705328.3748090,
author = {Krause, Thorsten and Oosterhuis, Harrie},
title = {A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options},
year = {2025},
isbn = {9798400713644},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3705328.3748090},
doi = {10.1145/3705328.3748090},
booktitle = {Proceedings of the Nineteenth ACM Conference on Recommender Systems},
pages = {21–30},
numpages = {10},
keywords = {Recommender Systems, User Choice Modeling, Exposure Bias},
location = {
},
series = {RecSys '25}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please contact <a href="mailto:thorsten.krause@ru.nl">thorsten.krause@ru.nl</a> or open an issue on GitHub.