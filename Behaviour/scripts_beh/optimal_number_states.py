import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
import ssm




def plateau(data, threshold=0.001):
    """
    Find the number of states at which the improvement plateaus.

    Parameters:
    -----------
    data : array-like
        Sequence of values (e.g., test log-likelihoods or bits per trial).
    threshold : float
        Minimum improvement between successive states to be considered meaningful.

    Returns:
    --------
    int
        Optimal number of states based on plateau detection.
    """
    diffs = np.diff(data)
    ind = np.where(diffs > threshold)[0]
    return ind[np.argmax(data[ind + 1])] + 1 if len(ind) > 0 else 1


@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager for tqdm progress bar integration with joblib.

    Parameters:
    -----------
    tqdm_object : tqdm.tqdm
        A tqdm progress bar object.

    Yields:
    -------
    tqdm_object : tqdm.tqdm
        Used to update progress during parallel processing.
    """
    from joblib import parallel
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def MLE_hmm_fit(num_states, training_inpts, training_choices, test_inpts, test_choices, num_iters, tolerance):
    """
    Fit a GLM-HMM model and evaluate performance on test data.

    Parameters:
    -----------
    num_states : int
        Number of hidden states in the HMM.
    training_inpts : list of np.ndarray
        Input regressors for training.
    training_choices : list of np.ndarray
        Output responses for training.
    test_inpts : list of np.ndarray
        Input regressors for testing.
    test_choices : list of np.ndarray
        Output responses for testing.

    Returns:
    --------
    hmm : ssm.HMM
        Trained HMM object.
    train_ll : float
        Log-likelihood per trial on the training set.
    test_ll : float
        Log-likelihood per trial on the test set.
    pred_accuracy : float
        Mean predictive probability across the test set.
    bits_per_trial : float
        Improvement in log-likelihood compared to a baseline Bernoulli model, in bits/trial.
    """
    obs_dim = training_choices[0].shape[1]
    num_categories = len(np.unique(np.concatenate(training_choices)))
    input_dim = training_inpts[0].shape[1]

    hmm = ssm.HMM(
        num_states, obs_dim, input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories),
        transitions="standard"
    )

    train_ll = hmm.fit(
        training_choices,
        inputs=training_inpts,
        method="em",
        num_iters=num_iters,
        tolerance=tolerance
    )

    train_ll = train_ll[-1] / np.concatenate(training_inpts).shape[0]

    LL_test = hmm.log_probability(test_choices, test_inpts)
    n_test = np.concatenate(test_inpts).shape[0]
    test_ll = LL_test / n_test

    test_probs = np.exp(hmm.log_likelihood(test_choices, test_inpts))
    pred_accuracy = np.mean(test_probs)

    y_train = np.concatenate(training_choices).flatten()
    p_right = np.mean(y_train)
    y_test = np.concatenate(test_choices).flatten()
    LL0 = np.sum(y_test * np.log(p_right + 1e-9) + (1 - y_test) * np.log(1 - p_right + 1e-9))

    bits_per_trial = (LL_test - LL0) / (n_test * np.log(2))

    return hmm, train_ll, test_ll, pred_accuracy, bits_per_trial


def run_state_selection(inpts, choices, max_states, n_splits, initializations, num_iters, tolerance):
    """
    Perform model selection for GLM-HMM by cross-validating across different numbers of states.

    Parameters:
    -----------
    inpts : list of np.ndarray
        Input features (e.g., stimulus regressors) per session.
    choices : list of np.ndarray
        Binary responses per session.
    max_states : int
        Maximum number of HMM states to test.
    n_splits : int
        Number of KFold splits for cross-validation.
    initializations : int
        Number of random initializations per model.

    Returns:
    --------
    MLE_train_LL : np.ndarray
        Training log-likelihoods per trial.
    MLE_test_LL : np.ndarray
        Test log-likelihoods per trial.
    MLE_bits_per_trial : np.ndarray
        Bits per trial compared to baseline.
    MLE_HMM : np.ndarray
        Trained HMM objects.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    jobs = [
        (z, s, iK, 
         [choices[i] for i in train_index], [inpts[i] for i in train_index],
         [choices[i] for i in test_index], [inpts[i] for i in test_index])
        for iK, (train_index, test_index) in enumerate(kf.split(choices))
        for s in range(1, max_states + 1)
        for z in range(initializations)
    ]

    shape = (initializations, max_states, n_splits)
    MLE_train_LL = np.full(shape, np.nan)
    MLE_test_LL = np.full(shape, np.nan)
    MLE_bits_per_trial = np.full(shape, np.nan)
    MLE_HMM = np.empty(shape, dtype=object)

    with tqdm_joblib(tqdm(total=len(jobs), desc="Fitting HMMs")):
        results = Parallel(n_jobs=-1)(
            delayed(lambda z, s, iK, tr_ch, tr_in, te_ch, te_in:
                    (z, s, iK, *MLE_hmm_fit(
                        num_states=s,
                        training_inpts=tr_in,
                        training_choices=tr_ch,
                        test_inpts=te_in,
                        test_choices=te_ch,
                        num_iters=num_iters,
                        tolerance=tolerance
                    )))(*job)
            for job in jobs
        )

    for z, s, iK, hmm, train_ll, test_ll, pred_accuracy, bits_per_trial in results:
        MLE_HMM[z, s - 1, iK] = hmm
        MLE_train_LL[z, s - 1, iK] = train_ll
        MLE_test_LL[z, s - 1, iK] = test_ll
        MLE_bits_per_trial[z, s - 1, iK] = bits_per_trial

    return MLE_train_LL, MLE_test_LL, MLE_bits_per_trial, MLE_HMM

