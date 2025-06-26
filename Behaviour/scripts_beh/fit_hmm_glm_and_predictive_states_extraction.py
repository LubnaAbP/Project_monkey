import ssm
import os
import pickle
import numpy as np


def define_glmhmm_model(
    inpts,
    num_states,
    obs_dim = 1,
    observations="input_driven_obs",
    transitions="standard"
):
    """
    Initialize a GLM-HMM model without fitting.

    Args:
        inpts (List[np.ndarray]): List of input arrays (to infer input_dim from the first session).
        num_states (int): Number of latent HMM states.
        obs_dim (int): Output dimensionality (1 for binary responses).
        observations (str): Observation model type (default: 'input_driven_obs').
        transitions (str): Transition model type (default: 'standard').

    Returns:
        ssm.HMM: Initialized GLM-HMM model (untrained).
    """
    if not inpts:
        raise ValueError("`inpts` must be a non-empty list of input arrays.")

    input_dim = inpts[0].shape[1]

    model = ssm.HMM(
        K=num_states,
        D=obs_dim,
        M=input_dim,
        observations=observations,
        transitions=transitions
    )

    return model

def fit_glmhmm(
        inpts,
        choices,
        num_states,
        obs_dim = 1,
        num_iters = 3000,
        tolerance = 1e-4,
        observations="input_driven_obs",
        transitions="standard"
):
    """
    Fit a Generalized Linear Model Hidden Markov Model (GLM-HMM) to behavioral data using SSM.

    Args:
        inpts (List[np.ndarray]): List of input feature matrices (n_trials x n_features) per session.
        true_choices (List[np.ndarray]): List of binary response vectors per session (n_trials x 1).
        num_states (int): Number of latent HMM states.
        obs_dim (int): Output dimension (1 for binary choices).
        num_iters (int): Max EM iterations.
        tolerance (float): EM convergence tolerance.
        input_dim (int, optional): Input dimensionality. Inferred from first input if None.
        observations (str): Type of emission model. Default is "input_driven_obs" but could be changed according to preference.
        transitions (str): Type of transition model. Default is "standard" but could also be changed according to preference.
        verbose (bool): Print EM progress if True.

    Returns:
        model (ssm.HMM): Fitted GLM-HMM model.
        fit_ll (np.ndarray): Log-likelihood values per EM iteration.
    """
    if not inpts or not choices:
        raise ValueError("Both 'inpts' and 'true_choices' must be non-empty lists.")


    input_dim = inpts[0].shape[1]

    model = define_glmhmm_model(inpts, num_states, obs_dim, observations,transitions)

    fit_ll = model.fit(
        choices,
        inputs=inpts,
        method="em",
        num_iters=num_iters,
        tolerance=tolerance
    )

    return model, fit_ll



def extract_glmhmm_summary(inpts, choices, num_states, filename="glmhmm_summary.pkl"):
    """
    Extracts and saves core parameters from a trained GLM-HMM model.

    Args:
        model (ssm.HMM): A trained GLM-HMM model.
        inpts (List[np.ndarray]): Input features (X) per session.
        choices (List[np.ndarray]): Binary choices (y) per session.
        filename (str): Name of the file to save the summary (saved to Behaviour/results/).

    Returns:
        summary (dict): GLM-HMM parameters and posterior state probabilities.
    """
    model = define_glmhmm_model(inpts, num_states)


    glm_weights = model.observations.params
    transition_matrix = np.exp(model.transitions.log_Ps)
    posterior_probs = [model.expected_states(data=y, input=X)[0] for y, X in zip(choices, inpts)]
    obs_dim, input_dim = glm_weights[0].shape
    num_states = model.K

    summary = {
        "glm_weights": glm_weights,
        "transition_matrix": transition_matrix,
        "posterior_probs": posterior_probs,
        "num_states": num_states,
        "input_dim": input_dim,
        "obs_dim": obs_dim
    }

    # Only go up to project root, then into Behaviour/results/
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    save_path = os.path.join(project_root, "results", filename)

    # Save summary
    with open(save_path, "wb") as f:
        pickle.dump(summary, f)

    print(f"[INFO] GLM-HMM summary saved in the folder results.")
    return summary


def rebuild_glmhmm_from_summary(
    filename="glmhmm_summary.pkl",
    observations="input_driven_obs",
    transitions="standard"
):
    """
    Rebuild an ssm.HMM model from a saved summary file (without re-fitting).

    Args:
        filename (str): Name of the summary file in the `results/` folder.
        observations (str): Type of observation model used during fitting.
        transitions (str): Type of transition model used during fitting.

    Returns:
        model (ssm.HMM): Reconstructed GLM-HMM model.
        summary (dict): The loaded summary dictionary.
    """
    # Locate results directory relative to project root
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    summary_path = os.path.join(project_root, "results", filename)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No GLM-HMM summary found at: {summary_path}")

    # Load summary
    with open(summary_path, "rb") as f:
        summary = pickle.load(f)

    # Reconstruct the model
    model = ssm.HMM(
        K=summary["num_states"],
        D=summary["obs_dim"],
        M=summary["input_dim"],
        observations=observations,
        transitions=transitions
    )

    model.observations.params = summary["glm_weights"]
    model.transitions.log_Ps = np.log(summary["transition_matrix"])

    print(f"[INFO] GLM-HMM model rebuilt from summary")
    return model, summary


# updating the data with the column predictivestates

def assign_predictive_states(
    data,
    posterior_probs=None,
    filename="predictive_states.pkl",
    load_path=None,
    verbose=True
):
    """
    Assign the most probable latent state (MAP) to each trial in each session,
    optionally saving or loading the predictive states.

    Args:
        data (List[pd.DataFrame]): List of behavioral session DataFrames.
        posterior_probs (List[np.ndarray], optional): Posterior probabilities per session (n_trials x K).
        filename (str): Name of the .pkl file to save the predictive states (under results/).
        load_path (str, optional): If provided, load predictive states from this file instead of computing.
        verbose (bool): Whether to print status messages.

    Returns:
        List[pd.DataFrame]: Updated session list with a 'predictiveStates' column added to each DataFrame.
    """

    if load_path:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Predictive states file not found: {load_path}")
        with open(load_path, "rb") as f:
            predictive_states = pickle.load(f)
        if verbose:
            print(f"[INFO] Loaded predictive states from: {load_path}")

    elif posterior_probs is not None:
        predictive_states = [np.argmax(post_prob, axis=1) for post_prob in posterior_probs]

        # Resolve absolute path to results directory
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        results_dir = os.path.join(project_root, "results")
        save_path = os.path.join(results_dir, filename)

        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

        with open(save_path, "wb") as f:
            pickle.dump(predictive_states, f)
        if verbose:
            print(f"[INFO] Predictive states computed and saved to: {save_path}")

    else:
        raise ValueError("You must provide either `posterior_probs` or `load_path`.")

    # Assign states to each session
    for sess_idx, df in enumerate(data):
        states = predictive_states[sess_idx]
        if len(states) != len(df):
            raise ValueError(f"Length mismatch in session {sess_idx}: "
                             f"{len(states)} states vs {len(df)} trials")
        df['predictiveStates'] = states

    if verbose:
        print("[INFO] 'predictiveStates' successfully assigned to all sessions.")
    return data


def assign_predictive_3_states(
    data,
    posterior_probs=None,
    weight_matrix=None,  # shape: (num_states, 1, num_regressors)
    filename="predictive_states.pkl",
    load_path=None
):
    """
    Assign mapped predictive states (0=motion, 1=color, 2=default) to each trial in each session.

    Args:
        data (List[pd.DataFrame]): List of behavioral session DataFrames.
        posterior_probs (List[np.ndarray], optional): Posterior probabilities per session (n_trials x num_states).
        weight_matrix (np.ndarray, optional): GLM weights (num_states x 1 x input_dim), used to determine state identity.
        filename (str): Filename to save predictive state list to results/ folder.
        load_path (str, optional): If provided, load predictive states from this file.

    Returns:
        List[pd.DataFrame]: Updated session list with a 'predictiveStates' column added to each DataFrame.
    """
    # Load from file
    if load_path:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Predictive states file not found: {load_path}")
        with open(load_path, "rb") as f:
            predictive_states = pickle.load(f)

    # Compute from posterior
    elif posterior_probs is not None and weight_matrix is not None:
        num_sessions = len(data)
        num_states = weight_matrix.shape[0]

        # Step 1: Identify states based on regressor weights
        color_weight_vector = weight_matrix[:, 0, 0]  # index 0 = color regressor
        motion_weight_vector = weight_matrix[:, 0, 1]  # index 1 = motion regressor

        color_state = np.argmax(color_weight_vector)
        motion_state = np.argmax(motion_weight_vector)

        if color_state == motion_state:
            sorted_motion_states = np.argsort(motion_weight_vector)[::-1]
            for alt in sorted_motion_states:
                if alt != color_state:
                    motion_state = alt
                    break

        default_state = list(set(range(num_states)) - {color_state, motion_state})[0]

        print(f"[INFO] State identity → color: {color_state}, motion: {motion_state}, default: {default_state}")

        # Step 2: Map trialwise state predictions to {0,1,2}
        predictive_states = []
        for sess_idx in range(num_sessions):
            most_probable = np.argmax(posterior_probs[sess_idx], axis=1)

            mapped = np.where(most_probable == color_state, 1,
                     np.where(most_probable == motion_state, 0, 2))
            predictive_states.append(mapped)

        # Save if requested
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        save_path = os.path.join(project_root, "results", filename)
        with open(save_path, "wb") as f:
            pickle.dump(predictive_states, f)
        print(f"[INFO] Predictive states saved to the folder results.")

    else:
        raise ValueError("You must provide either `load_path` or both `posterior_probs` and `weight_matrix`.")

    # Step 3: Assign to each session
    for sess_idx, df in enumerate(data):
        states = predictive_states[sess_idx]
        if len(states) != len(df):
            raise ValueError(f"Length mismatch in session {sess_idx}: {len(states)} states vs {len(df)} trials")
        df["predictiveStates"] = states

    print("[INFO] 'predictiveStates' assigned to all sessions.")
    return data
