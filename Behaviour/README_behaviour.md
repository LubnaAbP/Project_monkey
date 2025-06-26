# Behavioral Analysis – README

## Introduction

This folder contains all analyses related to modeling latent cognitive states from behavioral data using a **Generalized Linear Model Hidden Markov Model (GLM-HMM)**. This approach combines the flexibility of GLMs with the ability of HMMs to capture **discrete internal states** that evolve over time. In our task, monkeys flexibly switch between categorizing stimuli by color or motion. The GLM-HMM helps us uncover state-dependent strategies that may not be directly observable from behavior alone.

The model used in these analyses is configured with the following characteristics:

- `observations = 'input_driven_obs'`: Each latent state defines its own GLM that maps input features to choice probabilities.
- `transitions = 'standard'`: Transition probabilities between states are modeled independently of the inputs.

This configuration allows us to identify shifts in behavioral strategies while keeping the state transitions unconstrained by trial features.

Each latent state is characterized by its own set of GLM weights that determine how current and past trial features influence choices. The output includes posterior state probabilities, GLM weights per state, and transition dynamics between internal states.

---

## Features and Data Structure

The GLM-HMM is fitted using the following default input features:

```python
feature_names = [
    'color', 'direction', 'p_color', 'dist_prev_current_color',
    'p_direction', 'dist_prev_current_motion', 'prev_response', 'intercept'
]
```
These regressors capture both current stimuli and history-dependent factors, such as the previous trial’s stimulus or response. The dependent variable choices contains binary responses from the monkeys (e.g., left = 1, right = 0). Session-level structure is preserved throughout the analysis, with one input/response pair per session.

To customize or inspect how these features are constructed, refer to:
scripts_beh/preprocessing_hmm_glm.py

## Notebooks Overview

### `Optimal_number_of_hidden_states.ipynb`
Runs cross-validated k-fold fitting of the GLM-HMM to determine the optimal number of latent states based on log-likelihood performance.

### `main_HMM_GLM_fitting.ipynb`
Fits a Generalized Linear Model Hidden Markov Model (GLM-HMM) to behavioral data using the SSM framework.

- Supports flexible configuration for any number of latent states.
- Uses `observations='input_driven_obs'` and `transitions='standard'`.
- Plots EM log-likelihood over iterations.
- Visualizes GLM weights for each state.
- Displays the learned transition matrix.
- Plots posterior state probabilities for selected sessions.

### `Bootstrapping_3_states.ipynb`
Performs non-parametric bootstrapping(for 3 states) to estimate confidence intervals for GLM weights and transition matrices.

- Selects the best model from multiple random initializations.
- Aligns states across bootstrap iterations.
- Computes mean weights and 95% confidence intervals.
- Saves aligned results and visualizes transition structures.

### `State_based_analysis.ipynb`
Performs post-hoc analysis on identified cognitive states.

- Compares accuracy and reaction times across states.
- Analyzes context distributions per state.
- Plots 2D and 1D psychometric curves.
- Evaluates serial dependence effects on previous responses.
- Visualizes state probabilities as a function of stimulus features.
