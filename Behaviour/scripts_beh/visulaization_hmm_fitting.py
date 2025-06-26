import numpy as np
import matplotlib.pyplot as plt


def plot_log_likelihood(fit_ll):
    """
    Plot the log-likelihood over EM iterations.

    """
    plt.figure(figsize=(4, 3))
    plt.plot(fit_ll, label="EM")
    plt.xlabel("EM Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("EM Log-Likelihood Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_glm_weights(model, regressor_labels=None):
    """
    Plot GLM weights per latent state from a fitted GLM-HMM.

    Args:
        model (ssm.HMM): Trained GLM-HMM model using input-driven observation.
        regressor_labels (List[str]): Names of input features (x-axis). If None, default indices are used.
    """
    weights = model.observations.params  # List of [obs_dim x input_dim] arrays
    num_states = model.K
    obs_dim, input_dim = weights[0].shape
    cols = plt.cm.viridis(np.linspace(0, 1, num_states))

    plt.figure(figsize=(6, 3))
    for k in range(num_states):
        for d in range(obs_dim):
            plt.plot(
                range(input_dim),
                weights[k][d],
                label=f"State {k+1}" if obs_dim == 1 else f"State {k+1}, Output {d+1}",
                color=cols[k],
                linestyle="--",
                lw=1.5
            )

    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.ylabel("GLM Weight")
    plt.xlabel("Covariate")
    if regressor_labels:
        plt.xticks(range(input_dim), regressor_labels, rotation=45)
    else:
        plt.xticks(range(input_dim))
    plt.title("GLM Weights by State")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_transition_matrix(model, cmap='YlGnBu'):
    """
    Visualize the transition probability matrix of a fitted HMM.

    Args:
        model (ssm.HMM): A trained HMM-GLM model.
    """
    trans_mat = np.exp(model.transitions.log_Ps)  # shape: [num_states x num_states]
    num_states = trans_mat.shape[0]

    plt.figure(figsize=(3 + num_states * 0.5, 3))
    im = plt.imshow(trans_mat, cmap=cmap, vmin=0, vmax=1)

    for i in range(num_states):
        for j in range(num_states):
            plt.text(j, i, f"{trans_mat[i, j]:.2f}",
                     ha='center', va='center', fontsize=10, color='black')

    plt.xticks(range(num_states), [f"S{k+1}" for k in range(num_states)])
    plt.yticks(range(num_states), [f"S{k+1}" for k in range(num_states)])
    plt.title("State Transition Matrix")
    plt.colorbar(im, label="Probability")
    plt.tight_layout()
    plt.show()

def plot_posterior_probs(model, inpts, choices, sess_id, trial_window=None):
    """
    Plot posterior state probabilities for a given particular session.

    Args:
        model (ssm.HMM): A trained HMM-GLM model.
        inpts (List[np.ndarray]): List of input matrices per session.
        true_choices (List[np.ndarray]): List of observed outputs per session.
        sess_id (int): Index of the session to visualize.
        trial_window (Tuple[int, int], optional): Slice range for trial axis (start, end).
    """
    posterior_probs = [
        model.expected_states(data=y, input=X)[0]
        for y, X in zip(choices, inpts)
    ]

    session_probs = posterior_probs[sess_id]
    cols = plt.cm.viridis(np.linspace(0, 1, model.K))

    plt.figure(figsize=(6, 2.5))
    for k in range(model.K):
        plt.plot(
            session_probs[:, k],
            label=f"State {k+1}",
            color=cols[k],
            lw=2
        )

    if trial_window:
        plt.xlim(*trial_window)
    else:
        plt.xlim(0, session_probs.shape[0])

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Trial")
    plt.ylabel("p(state)")
    plt.title(f"Posterior Probabilities (Session {sess_id})")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
