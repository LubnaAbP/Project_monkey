import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import ssm
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

# Make sure we're adding the root of Project_monkey
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))  # Two levels up from notebooks
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Behaviour.scripts_beh.preprocessing_hmm_glm import format_glmhmm_inputs

#### processed_data = [preprocess_behavioral_session(df) for df in data]

##data = processed_data
def find_best_parameters(
    data,
    num_states,
    input_dim,
    N_iters,
    tolerance,
    num_initializations,
    feature_names,
    filename="best_parameters.pkl"
):
    """
    Run multiple random initializations of GLM-HMM and select the best based on log-likelihood.

    Args:
        data (List[pd.DataFrame]): Preprocessed session data.
        num_states (int): Number of HMM states.
        input_dim (int): Number of input features.
        N_iters (int): Number of EM iterations per fit.
        tolerance (float): EM convergence tolerance.
        num_initializations (int): Number of random restarts.
        filename (str): Name of file to save best parameters (stored in results/).

    Returns:
        Tuple: (best_obs_params, best_trans_mat)
    """
    inpts, choices = format_glmhmm_inputs(data, feature_names)

    def fit_initialization(init_num):
        print(f"Running initialization {init_num + 1} of {num_initializations}...")
        hmm = ssm.HMM(num_states, 1, input_dim, observations="input_driven_obs", transitions="standard")
        hmm.fit(choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=tolerance)
        log_likelihood = hmm.log_probability(choices, inputs=inpts)
        return log_likelihood, hmm

    results = Parallel()(delayed(fit_initialization)(i) for i in range(num_initializations))
    best_log_likelihood, best_hmm = max(results, key=lambda x: x[0])

    obs_params = best_hmm.observations.params
    trans_mat = np.exp(best_hmm.transitions.log_Ps)

    # Save to results/ folder
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    save_path = os.path.join(project_root, "results", filename)

    with open(save_path, "wb") as f:
        pickle.dump({"obs_params": obs_params, "trans_mat": trans_mat}, f)

    print(f"[INFO] Best Log-Likelihood: {best_log_likelihood:.2f}")
    print(f"[INFO] Best parameters saved to results")

    return obs_params, trans_mat

def single_bootstrap_parametric(
    processed_data,
    num_states,
    input_dim,
    N_iters,
    tolerance,
    best_obs_params,
    best_trans_mat,
    inpts,
    num_categories=2
):
    """
    Perform one bootstrap replicate by simulating data from the best model and re-fitting.

    Returns:
        Tuple: (fit_obs_params, fit_trans_mat, log_likelihood)
    """
    num_sess = len(processed_data)
    simulated_choices = []

    sim_glmhmm = ssm.HMM(
        num_states, 1, input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories),
        transitions="standard"
    )
    sim_glmhmm.observations.params = best_obs_params
    sim_glmhmm.transitions.log_Ps = np.log(best_trans_mat)

    for sess_idx in range(num_sess):
        num_trials = len(processed_data[sess_idx])
        inpt = inpts[sess_idx]
        _, y = sim_glmhmm.sample(num_trials, input=inpt)
        simulated_choices.append(y)

    fit_model = ssm.HMM(
        num_states, 1, input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories),
        transitions="standard"
    )
    fit_model.fit(simulated_choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=tolerance)
    loglik = fit_model.log_probability(simulated_choices, inputs=inpts)

    return fit_model.observations.params, np.exp(fit_model.transitions.log_Ps), loglik

def bootstrap_glmhmm_safe_exit_parametric(
    data,
    num_bootstrap,
    num_states,
    input_dim,
    N_iters,
    tolerance,
    feature_names,
    results_filename="glm_hmm_parametric_bootstrap.pkl",
    best_params_filename="best_parameters.pkl"

):
    """
    Run parametric bootstrapping to assess variability in GLM-HMM parameters.

    Args:
        data (List[pd.DataFrame]): Raw session data.
        num_bootstrap (int): Total number of bootstrap replicates.
        num_states (int): Number of latent HMM states.
        input_dim (int): Number of input features.
        N_iters (int): Number of EM iterations per model.
        tolerance (float): EM convergence tolerance.
        results_filename (str): Filename to save bootstrapped weights and transitions.
        best_params_filename (str): Filename to load pre-fitted best model parameters.
        save_every (int): Save intermediate results every N replicates.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of bootstrapped weights, transitions, and log-likelihoods.
    """

    # Resolve file paths relative to project root
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    save_path = os.path.join(project_root, "results", results_filename)
    best_params_path = os.path.join(project_root, "results", best_params_filename)

    # Load best parameters
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Best parameters not found at: {best_params_path}")

    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)
        best_obs_params = best_params["obs_params"]
        best_trans_mat = best_params["trans_mat"]

    inpts, choices = format_glmhmm_inputs(data,  feature_names)

    # Resume or start fresh
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            saved = pickle.load(f)
            boot_weights = saved.get("weights", [])
            boot_transitions = saved.get("transitions", [])
            boot_loglikes = saved.get("log_likelihoods", [])
        start_idx = len(boot_weights)
        print(f"[INFO] Resuming from {start_idx}/{num_bootstrap} bootstraps.")
    else:
        boot_weights, boot_transitions, boot_loglikes = [], [], []
        start_idx = 0
        print("[INFO] Starting new bootstrap run.")

    try:
        with tqdm(total=num_bootstrap, initial=start_idx) as pbar:
            results = Parallel()(
                delayed(single_bootstrap_parametric)(
                    data, num_states, input_dim, N_iters, tolerance,
                    best_obs_params, best_trans_mat, inpts
                )
                for _ in range(start_idx, num_bootstrap)
            )

            for obs_params, trans_mat, loglike in results:
                boot_weights.append(obs_params)
                boot_transitions.append(trans_mat)
                boot_loglikes.append(loglike)
                pbar.update(1)

            with open(save_path, "wb") as f:
                pickle.dump({
                    "weights": boot_weights,
                    "transitions": boot_transitions,
                    "log_likelihoods": boot_loglikes
                }, f)
            print(f"[INFO] Bootstrapped results saved to results")

    except KeyboardInterrupt:
        print("[WARNING] Interrupted — saving progress.")
        with open(save_path, "wb") as f:
            pickle.dump({
                "weights": boot_weights,
                "transitions": boot_transitions,
                "log_likelihoods": boot_loglikes
            }, f)
        print(f"[INFO] Partial results saved to: {save_path}")

    return np.array(boot_weights), np.array(boot_transitions), np.array(boot_loglikes)


def print_best_glmhmm_parameters(filename="glm_best_params.pkl", regressor_labels=None):
    """
    Load and print best GLM-HMM parameters from a file in the results/ folder.

    Args:
        filename (str): Name of the .pkl file in the results/ folder.
        regressor_labels (List[str], optional): Names of the input regressors.
    """
    # Resolve project root and full path to results/
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    result_path = os.path.join(project_root, "results", filename)


    if not os.path.exists(result_path):
        raise FileNotFoundError(f"[ERROR] File not found at: {result_path}")

    with open(result_path, "rb") as f:
        params = pickle.load(f)

    obs_params = params["obs_params"]  # shape: (num_states, 1, input_dim)
    trans_mat = params["trans_mat"]

    num_states = obs_params.shape[0]
    input_dim = obs_params.shape[2]

    # Transition matrix
    print("\n=== Best Transition Matrix ===")
    trans_df = pd.DataFrame(
        trans_mat,
        index=[f"State {i}" for i in range(num_states)],
        columns=[f"State {i}" for i in range(num_states)]
    )
    print(trans_df.round(4))

    # GLM weights
    print("\n=== GLM Observation Weights ===")
    for state in range(num_states):
        print(f"\nState {state}:")
        weights = obs_params[state][0]
        if regressor_labels and len(regressor_labels) == input_dim:
            df = pd.DataFrame({
                "Regressor": regressor_labels,
                "Weight": weights
            })
            print(df.round(4))
        else:
            for i, w in enumerate(weights):
                print(f"  Input {i}: {w:.4f}")



def load_glmhmm_bootstrap_results(filename="glm_hmm_parametric_bootstrap.pkl", verbose=True):
    """
    Load bootstrapped GLM-HMM results from the results/ directory.

    Args:
        filename (str): Name of the bootstrap .pkl file in the results/ folder.
        verbose (bool): Whether to print shapes of loaded arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: weights, transitions, log-likelihoods
    """
    # Resolve full path to results/ folder
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    filepath = os.path.join(project_root, "results", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Bootstrap results not found at: {filepath}")

    with open(filepath, "rb") as f:
        saved = pickle.load(f)

    boot_weights = np.array(saved["weights"])
    boot_transitions = np.array(saved["transitions"])
    boot_loglikes = np.array(saved["log_likelihoods"])

    if verbose:
        print(f"[INFO] Loaded {len(boot_weights)} bootstrap iterations from {filename}")
        print(f"  Weights shape:     {boot_weights.shape}")
        print(f"  Transitions shape: {boot_transitions.shape}")
        print(f"  Log-likelihoods:   {boot_loglikes.shape}")

    return boot_weights, boot_transitions, boot_loglikes



def reorder_states_by_weights(weight_array):
    """
    Reorder HMM states in bootstrapped weights using top weight features.
    Assumes:
        - color regressor is at index 0
        - motion regressor is at index 1
    Returns:
        np.ndarray: reordered_weight_array [num_bootstrap, 3, 1, input_dim]
    """
    reordered = []

    for weights in weight_array:
        color_weights = weights[:, 0, 0]
        motion_weights = weights[:, 0, 1]

        color_state = np.argmax(color_weights)
        motion_state = np.argmax(motion_weights)

        if color_state == motion_state:
            sorted_motion_states = np.argsort(motion_weights)[::-1]
            for alt_motion_state in sorted_motion_states:
                if alt_motion_state != color_state:
                    motion_state = alt_motion_state
                    break

        all_states = set(range(weights.shape[0]))
        disengaged_state = list(all_states - {color_state, motion_state})[0]

        reordered_weights = weights[[motion_state, color_state, disengaged_state], :, :]
        reordered.append(reordered_weights)

    return np.array(reordered)




def save_bootstrap_summary(weights, transitions, filename="glm_bootstrap_summary.pkl"):
    """
    Computes and saves mean bootstrapped GLM-HMM parameters to file.

    Args:
        weights (np.ndarray): Bootstrapped weights of shape [n_bootstraps, n_states, 1, input_dim].
        transitions (np.ndarray): Bootstrapped transitions of shape [n_bootstraps, n_states, n_states].
        filename (str): Name of the .pkl file to save the summary (under results/).

    Returns:
        summary (dict): Dictionary with keys:
            - 'glm_weights': mean GLM weights [n_states, 1, input_dim]
            - 'transition_matrix': mean transition matrix [n_states, n_states]
            - 'num_states': number of latent states
            - 'input_dim': number of input regressors
            - 'obs_dim': output dimensionality (always 1)
    """
    # Reorder weights
    reordered_weights = reorder_states_by_weights(weights)

    # Reorder transitions
    reordered_transitions = []
    for idx in range(len(transitions)):
        color_weights = weights[idx, :, 0, 0]  # color regressor
        motion_weights = weights[idx, :, 0, 1]  # motion regressor

        color_state = np.argmax(color_weights)
        motion_state = np.argmax(motion_weights)

        if color_state == motion_state:
            sorted_motion_states = np.argsort(motion_weights)[::-1]
            for alt_motion_state in sorted_motion_states:
                if alt_motion_state != color_state:
                    motion_state = alt_motion_state
                    break

        all_states = set(range(weights.shape[1]))
        disengaged_state = list(all_states - {color_state, motion_state})[0]
        new_order = [motion_state, color_state, disengaged_state]

        reordered_transitions.append(transitions[idx][np.ix_(new_order, new_order)])

    reordered_transitions = np.stack(reordered_transitions)

    # Compute means
    glm_weights = np.mean(reordered_weights, axis=0)  # [n_states, 1, input_dim]
    transition_matrix = np.mean(reordered_transitions, axis=0)  # [n_states, n_states]

    # Compose summary
    summary = {
        "glm_weights": glm_weights,
        "transition_matrix": transition_matrix,
        "num_states": glm_weights.shape[0],
        "input_dim": glm_weights.shape[2],
    }

    # Save to file
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    save_path = os.path.join(project_root, "results", filename)
    with open(save_path, "wb") as f:
        pickle.dump(summary, f)

    print(f"[INFO] Bootstrap summary saved to results")
    return summary




def load_bootstrap_summary(filename="glm_bootstrap_summary.pkl"):
    """
    Load mean weights and transition matrix from bootstrap summary file in results/.

    Args:
        filename (str): Name of the .pkl file in the results/ folder.

    Returns:
        dict: Dictionary with keys 'mean_weights', 'mean_transition_matrix', etc.
    """
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    summary_path = os.path.join(project_root, "results", filename)

    with open(summary_path, "rb") as f:
        summary = pickle.load(f)

    return summary
