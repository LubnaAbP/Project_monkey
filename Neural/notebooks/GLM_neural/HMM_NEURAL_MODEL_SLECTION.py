from sklearn.svm import SVC
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV as ridge
from sklearn.model_selection import train_test_split
from tqdm import trange,tqdm
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from contextlib import contextmanager
from tqdm.auto import tqdm
import ssm
import matplotlib.pyplot as plt

with open('C:/Users/shahe/PycharmProjects/distributed_SD/Data/predicted_trial_SVM_res_04_04.pkl', 'rb') as f:
    data_neural = pickle.load(f)


def map_predicted_labels_to_raw(data_list):
    new_data_list = []
    for df in data_list:
        # Build mappings from labels to raw values
        color_map = (
            df.dropna(subset=['color', 'color_raw'])
              .drop_duplicates(subset=['color'])
              .set_index('color')['color_raw']
              .to_dict()
        )
        direction_map = (
            df.dropna(subset=['direction', 'direction_raw'])
              .drop_duplicates(subset=['direction'])
              .set_index('direction')['direction_raw']
              .to_dict()
        )
        prev_color_map = (
            df.dropna(subset=['previous_color', 'previous_color_raw'])
              .drop_duplicates(subset=['previous_color'])
              .set_index('previous_color')['previous_color_raw']
              .to_dict()
        )
        prev_direction_map = (
            df.dropna(subset=['previous_direction', 'previous_direction_raw'])
              .drop_duplicates(subset=['previous_direction'])
              .set_index('previous_direction')['previous_direction_raw']
              .to_dict()
        )

        # Define which mappings apply to which predicted variables
        mapping_dict = {
            'predicted_color': color_map,
            'predicted_direction': direction_map,
            'predicted_previous_color': prev_color_map,
            'predicted_previous_direction': prev_direction_map,
        }

        # Collect all new columns in a dict
        new_cols = {}
        for col in df.columns:
            for key, mapping in mapping_dict.items():
                if col.startswith(key):
                    new_cols[col + '_raw'] = df[col].map(mapping)

        # Concatenate all at once
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # Optional: defragment memory (creates a clean copy)
        df = df.copy()
        new_data_list.append(df)

    return new_data_list
mapped_data = map_predicted_labels_to_raw(data_neural)
data_neural = mapped_data




# --- helpers ---
def zscore(s: pd.Series) -> pd.Series:
    """Fast z-score that preserves index and avoids divide-by-zero."""
    m = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return (s - m)  # all zeros; fine for downstream use
    return (s - m) / sd


# --- main processing ---
def process_session_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single session DataFrame.
    Cleans and adds derived variables for analysis with zero fragmentation.
    """
    df = df.copy()
    idx = df.index
    new = {}  # collect ALL new columns here (as Series aligned to idx)

    # ---------- Basic columns ----------
    new['intercept'] = pd.Series(1, index=idx)

    color_z  = zscore(df['color_raw'])
    direc_z  = zscore(df['direction_raw'])
    new['color_z_b']     = color_z
    new['direction_z_b'] = direc_z

    new['p_color_z_b']     = color_z.shift(1).fillna(0)
    new['p_direction_z_b'] = direc_z.shift(1).fillna(0)

    new['dist_prev_current_color']  = (color_z - new['p_color_z_b']).abs()
    new['dist_prev_current_motion'] = (direc_z - new['p_direction_z_b']).abs()

    resp = df['chosenResponse'].fillna(0).astype(int)
    new['response']        = resp
    new['prev_response_b'] = resp.shift(1).fillna(0)

    # Context: map {1->1, 0->-1}
    ctx_b = df['rule'].astype(int).map({1: 1, 0: -1}).fillna(-1)
    new['context_b'] = ctx_b

    new['context_color_b']  = color_z * ctx_b
    new['context_motion_b'] = direc_z * ctx_b

    # Use Series.where so we preserve index (avoid np.where)
    rel_b = color_z.where(ctx_b == 1, direc_z)
    irr_b = direc_z.where(ctx_b == 1, color_z)
    new['relevant_b']   = rel_b
    new['irrelevant_b'] = irr_b
    new['prev_rel_b']   = rel_b.shift(1).fillna(0)
    new['prev_irr_b']   = irr_b.shift(1).fillna(0)

    # Cue one-hots (compact dtype)
    cue_dummies = pd.get_dummies(df['cue'], prefix='cue', drop_first=False).astype('int8')
    cue_dummies = cue_dummies.reindex(idx, fill_value=0)

    # ---------- per-area helper ----------
    def add_area(area: str):
        base = 'predicted_'
        need = [
            f'{base}color_{area}_raw', f'{base}direction_{area}_raw',
            f'{base}previous_color_{area}_raw', f'{base}previous_direction_{area}_raw',
            f'{base}response_{area}', f'{base}previous_response_{area}',
            f'{base}context_response_{area}',
        ]
        if not all(c in df.columns for c in need):
            return

        color  = zscore(df[f'{base}color_{area}_raw'])
        direc  = zscore(df[f'{base}direction_{area}_raw'])
        pcolor = zscore(df[f'{base}previous_color_{area}_raw'])
        pdirec = zscore(df[f'{base}previous_direction_{area}_raw'])

        ctx = df[f'{base}context_response_{area}'].map({1: 1, 0: -1})
        prev_ctx = ctx.shift(1)

        # assign as Series (aligned)
        new[f'color_{area}']       = color
        new[f'direction_{area}']   = direc
        new[f'p_color_{area}']     = pcolor
        new[f'p_direction_{area}'] = pdirec

        new[f'decodedResponse_{area}'] = df[f'{base}response_{area}']
        new[f'prev_response_{area}']   = df[f'{base}previous_response_{area}'].fillna(0)

        new[f'context_{area}']      = ctx
        new[f'prev_context_{area}'] = prev_ctx

        new[f'context_color_{area}']  = color * ctx
        new[f'context_motion_{area}'] = direc * ctx

        rel = color.where(ctx == 1, direc)
        irr = direc.where(ctx == 1, color)
        new[f'relevant_{area}']   = rel
        new[f'irrelevant_{area}'] = irr
        new[f'prev_rel_{area}']   = rel.shift(1).fillna(0)
        new[f'prev_irr_{area}']   = irr.shift(1).fillna(0)

    # Add all areas you use
    for area in ['PFC', 'FEF', 'LIP', 'V4', 'MT', 'IT', 'Parietal']:
        add_area(area)

    # ---------- build new_df in ONE concat (no column-by-column inserts) ----------
    series_list = []
    for k, v in new.items():
        s = v if isinstance(v, pd.Series) else pd.Series(v, index=idx)
        if not s.index.equals(idx):
            s = s.reindex(idx)
        series_list.append(s.rename(k))
    new_df = pd.concat(series_list, axis=1)

    # ---------- final join (single concat) ----------
    out = pd.concat([df, new_df, cue_dummies], axis=1)

    # Optional: defragment memory (creates a compact copy)
    return out.copy()


# --- usage example ---
processed_data_neural = [process_session_data(df) for df in data_neural]


# ------------------------------------------------------------
# Assumes `process_session_data(df)` is already defined above.
# ------------------------------------------------------------

AREA_LIST = ['FEF', 'PFC', 'LIP', 'Parietal', 'V4', 'MT', 'IT']

# ---------- Feature builders (edit here to add/remove predictors) ----------

def build_X_behavior(df: pd.DataFrame) -> np.ndarray:
    """Behavior-only design matrix (negated features as in your code)."""
    relevant_b      = -df[['relevant_b']].fillna(0).to_numpy()
    irrelevant_b    = -df[['irrelevant_b']].fillna(0).to_numpy()
    prev_rel_b      = -df[['prev_rel_b']].fillna(0).to_numpy()
    prev_irr_b      = -df[['prev_irr_b']].fillna(0).to_numpy()
    prev_response_b = -df[['prev_response_b']].fillna(0).to_numpy()
    bias            = -df[['intercept']].fillna(0).to_numpy()
    X = np.hstack([
        relevant_b, irrelevant_b,
        prev_rel_b, prev_irr_b,
        prev_response_b,
        bias
    ])
    return X

def build_X_neural(df: pd.DataFrame, area: str) -> np.ndarray:
    """Neural-only design matrix for an area (relevant/irrelevant/prev & prev_response + bias)."""
    relevant      = -df[[f'relevant_{area}']].fillna(0).to_numpy()
    irrelevant    = -df[[f'irrelevant_{area}']].fillna(0).to_numpy()
    prev_rel      = -df[[f'prev_rel_{area}']].fillna(0).to_numpy()
    prev_irr      = -df[[f'prev_irr_{area}']].fillna(0).to_numpy()
    prev_response = -df[[f'prev_response_{area}']].fillna(0).to_numpy()
    bias          = -df[['intercept']].fillna(0).to_numpy()
    X = np.hstack([
        relevant, irrelevant,
        prev_rel, prev_irr,
        prev_response,
        bias
    ])
    return X

def build_X_combined(df: pd.DataFrame, area: str) -> np.ndarray:
    """Combined behavior + neural (bias only once at the end)."""
    # Behavior (without bias)
    relevant_b      = -df[['relevant_b']].fillna(0).to_numpy()
    irrelevant_b    = -df[['irrelevant_b']].fillna(0).to_numpy()
    prev_rel_b      = -df[['prev_rel_b']].fillna(0).to_numpy()
    prev_irr_b      = -df[['prev_irr_b']].fillna(0).to_numpy()
    prev_response_b = -df[['prev_response_b']].fillna(0).to_numpy()

    # Neural (without bias)
    relevant      = -df[[f'relevant_{area}']].fillna(0).to_numpy()
    irrelevant    = -df[[f'irrelevant_{area}']].fillna(0).to_numpy()
    prev_rel      = -df[[f'prev_rel_{area}']].fillna(0).to_numpy()
    prev_irr      = -df[[f'prev_irr_{area}']].fillna(0).to_numpy()
    prev_response = -df[[f'prev_response_{area}']].fillna(0).to_numpy()

    bias          = -df[['intercept']].fillna(0).to_numpy()

    X = np.hstack([
        # behavior
        relevant_b, irrelevant_b, prev_rel_b, prev_irr_b, prev_response_b,
        # neural
        relevant,   irrelevant,   prev_rel,   prev_irr,   prev_response,
        # single bias
        bias
    ])
    return X

# ---------- Utilities ----------

def y_from_df(df: pd.DataFrame) -> np.ndarray:
    """Binary response as (T,1)."""
    return df['chosenResponse'].fillna(0).astype(int).to_numpy().reshape(-1, 1)

def sessions_with_area(processed_sessions, area: str):
    """Filter sessions that contain the required columns for an area."""
    required = {
        f'relevant_{area}', f'irrelevant_{area}',
        f'prev_rel_{area}', f'prev_irr_{area}',
        f'prev_response_{area}',
        'intercept'
    }
    keep = []
    idxs = []
    for i, df in enumerate(processed_sessions):
        if required.issubset(set(df.columns)):
            keep.append(df)
            idxs.append(i)
    return keep, idxs

def prepare_sessionwise(dfs, X_builder, y_builder=y_from_df):
    """Return session-wise inpts, true_choices, and lengths with a builder function."""
    inpts, true_choices, lengths = [], [], []
    for df in dfs:
        X = X_builder(df)
        y = y_builder(df)
        inpts.append(X)
        true_choices.append(y)
        lengths.append(len(df))
    return inpts, true_choices, lengths

# ---------- Driver that produces all variants you asked for ----------

def build_all_glmhmm_inputs(data_neural_raw, areas=AREA_LIST):
    """
    Returns a results dict with:
      - behavior_all: inpts/y/lengths across ALL sessions
      - per area:
          - indices: indices of original sessions that had that area
          - behavior_matched: behavior-only inpts/y/lengths for those sessions
          - neural: neural-only inpts/y/lengths
          - combined: behavior+neural inpts/y/lengths (only sessions with that area's neural)
    """
    # 1) process all sessions once
    processed_sessions = [process_session_data(df) for df in data_neural_raw]

    # 2) behavior-only across ALL sessions (for global comparison)
    beh_inpts_all, beh_y_all, beh_len_all = prepare_sessionwise(
        processed_sessions, build_X_behavior, y_from_df
    )

    results = {
        'behavior_all': {
            'inpts': beh_inpts_all,
            'true_choices': beh_y_all,
            'session_lengths': beh_len_all,
            'indices': list(range(len(processed_sessions)))
        },
        'areas': {}
    }

    # 3) per-area matched behavior/neural/combined
    for area in areas:
        area_sessions, idxs = sessions_with_area(processed_sessions, area)
        if not area_sessions:
            results['areas'][area] = {
                'indices': [],
                'behavior_matched': {'inpts': [], 'true_choices': [], 'session_lengths': []},
                'neural': {'inpts': [], 'true_choices': [], 'session_lengths': []},
                'combined': {'inpts': [], 'true_choices': [], 'session_lengths': []},
            }
            continue

        # behavior-only but matched to the sessions that have this area's neural data
        beh_inpts, beh_y, beh_len = prepare_sessionwise(area_sessions, build_X_behavior, y_from_df)

        # neural-only for this area
        neu_inpts, neu_y, neu_len = prepare_sessionwise(
            area_sessions, lambda df, a=area: build_X_neural(df, a), y_from_df
        )

        # combined behavior + area neural
        com_inpts, com_y, com_len = prepare_sessionwise(
            area_sessions, lambda df, a=area: build_X_combined(df, a), y_from_df
        )

        results['areas'][area] = {
            'indices': idxs,  # original session indices kept for this area
            'behavior_matched': {
                'inpts': beh_inpts,
                'true_choices': beh_y,
                'session_lengths': beh_len
            },
            'neural': {
                'inpts': neu_inpts,
                'true_choices': neu_y,
                'session_lengths': neu_len
            },
            'combined': {
                'inpts': com_inpts,
                'true_choices': com_y,
                'session_lengths': com_len
            },
        }

    return results




LOG2 = np.log(2.0)

# -----------------------------
# MLE fit for a single (fold, K, init)
# -----------------------------
def MLE_hmm_fit(num_states, training_inpts, training_choices, test_inpts, test_choices, seed=None):
    """
    Fit a GLM-HMM and return the fitted model + train/test log-likelihood
    in **bits per trial**.
    training_* and test_* are lists (one per session).
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure shapes/dtypes
    training_choices = [np.array(y).reshape(-1, 1).astype(int) for y in training_choices]
    test_choices     = [np.array(y).reshape(-1, 1).astype(int) for y in test_choices]

    num_categories = len(np.unique(np.concatenate(training_choices)))
    input_dim = training_inpts[0].shape[1]

    # 1 state: standard transitions; >1 state: sticky transitions
    if num_states == 1:
        hmm = ssm.HMM(
            K=1,
            D=1,
            M=input_dim,
            observations="input_driven_obs",
            observation_kwargs=dict(C=num_categories),
            transitions="standard",
        )
    else:
        hmm = ssm.HMM(
            K=num_states,
            D=1,
            M=input_dim,
            observations="input_driven_obs",
            observation_kwargs=dict(C=num_categories),
            transitions="sticky",
            transition_kwargs=dict(alpha=2.0, kappa=0.0),
        )

    train_lls = hmm.fit(
        training_choices,
        inputs=training_inpts,
        method="em",
        num_iters=3000,
        tolerance=1e-2,
    )

    n_train = np.concatenate(training_inpts).shape[0]
    n_test  = np.concatenate(test_inpts).shape[0]

    # Convert from nats/trial → bits/trial
    train_bpt = (train_lls[-1] / n_train) / LOG2
    test_bpt  = (hmm.log_probability(test_choices, test_inpts) / n_test) / LOG2

    return hmm, train_bpt, test_bpt


# -----------------------------
# TQDM helper for joblib
# -----------------------------
@contextmanager
def tqdm_joblib(tqdm_object):
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


# -----------------------------
# Utilities for model selection
# -----------------------------
def plateau(data_1d, threshold=0.001):
    """
    Given a 1D array of scores over K, return the K at which the last
    improvement above threshold occurs; if none, return 1.
    """
    data_1d = np.asarray(data_1d, dtype=float)
    diffs = np.diff(data_1d)
    idx = np.where(diffs > threshold)[0]
    return int(idx[np.argmax(data_1d[idx + 1])] + 1) if idx.size > 0 else 1

def sign_flip_pvalue(diffs):
    """
    Exact two-sided sign-flip test for the mean of paired differences.
    diffs: per-fold differences (K+1 - K) in bits/trial.
    """
    from itertools import product
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = diffs.size
    if n == 0:
        return np.nan
    t_obs = np.abs(np.mean(diffs))
    count = 0
    total = 0
    for signs in product([-1.0, 1.0], repeat=n):
        total += 1
        t = np.abs(np.mean(diffs * np.array(signs)))
        if t >= t_obs - 1e-15:
            count += 1
    return count / total

def holm_bonferroni(pvals, alpha=0.05):
    pvals = np.array(pvals, dtype=float)
    m = np.sum(np.isfinite(pvals))
    if m == 0:
        return np.zeros_like(pvals, dtype=bool), pvals
    order = np.argsort(np.where(np.isfinite(pvals), pvals, np.inf))
    reject = np.zeros_like(pvals, dtype=bool)
    adj_p = np.full_like(pvals, np.nan, dtype=float)
    for rank, idx in enumerate(order, start=1):
        if not np.isfinite(pvals[idx]):
            continue
        threshold = alpha / (m - rank + 1)
        if pvals[idx] <= threshold:
            reject[idx] = True
        adj_p[idx] = max((m - rank + 1) * pvals[idx], 0.0)
    # enforce monotonicity (cosmetic)
    last = -np.inf
    for idx in order:
        if np.isfinite(adj_p[idx]):
            adj_p[idx] = max(adj_p[idx], last)
            last = adj_p[idx]
    return reject, adj_p


# -----------------------------
# Core CV runner for one dataset (behavior / neural / combined)
# -----------------------------
def run_glmhmm_cv(inpts, true_choices, max_states=4, n_splits=5, initializations=5, base_seed=42, n_jobs=-1, desc="GLM-HMM CV"):
    """
    Run K-fold CV over K=1..max_states with multiple random initializations per fold.
    Returns:
        dict with:
            Best_HMM       : (max_states, n_splits) object array of best-per-fold HMMs
            Best_test_LL   : (max_states, n_splits) bits/trial (best init per fold)
            Best_train_LL  : (max_states, n_splits) bits/trial
            AllInits_test_LL: (initializations, max_states, n_splits) bits/trial
            selection      : dict with K* by various criteria
            summary        : means/SEMs per K
    """
    nsess = len(inpts)
    if nsess < 2:
        raise ValueError(f"Need at least 2 sessions for CV; got {nsess}")
    # Adapt n_splits if needed
    n_splits = min(n_splits, nsess)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=base_seed)

    Best_HMM = np.full((max_states, n_splits), None, dtype=object)
    Best_train_LL = np.full((max_states, n_splits), np.nan, dtype=float)
    Best_test_LL = np.full((max_states, n_splits), np.nan, dtype=float)
    AllInits_test_LL = np.full((initializations, max_states, n_splits), np.nan, dtype=float)

    # Build job list
    jobs = []
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(true_choices)):
        training_choices = [true_choices[i] for i in train_idx]
        test_choices     = [true_choices[i] for i in test_idx]
        training_inpts   = [inpts[i] for i in train_idx]
        test_inpts       = [inpts[i] for i in test_idx]
        for K in range(1, max_states + 1):
            for z in range(initializations):
                seed = base_seed * 100_000 + 10_000 * fold_id + 100 * K + z
                jobs.append((fold_id, K, z, seed, training_inpts, training_choices, test_inpts, test_choices))

    def _run_one(fold_id, K, z, seed, tr_in, tr_ch, te_in, te_ch):
        hmm, tr_ll, te_ll = MLE_hmm_fit(K, tr_in, tr_ch, te_in, te_ch, seed=seed)
        return fold_id, K, z, hmm, tr_ll, te_ll

    with tqdm_joblib(tqdm(total=len(jobs), desc=desc)):
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_run_one)(*args) for args in jobs
        )

    for fold_id, K, z, hmm, tr_ll, te_ll in results:
        AllInits_test_LL[z, K - 1, fold_id] = te_ll
        if np.isnan(Best_test_LL[K - 1, fold_id]) or te_ll > Best_test_LL[K - 1, fold_id]:
            Best_test_LL[K - 1, fold_id] = te_ll
            Best_train_LL[K - 1, fold_id] = tr_ll
            Best_HMM[K - 1, fold_id] = hmm

    # Selection stats over K
    mean_bpt = np.nanmean(Best_test_LL, axis=1)  # (K,)
    sem_bpt  = np.nanstd(Best_test_LL, axis=1, ddof=1) / np.sqrt(
        np.maximum(1, np.sum(np.isfinite(Best_test_LL), axis=1))
    )

    K_max = int(np.nanargmax(mean_bpt)) + 1
    K_plateau = plateau(mean_bpt)

    # 1-SE rule
    best_val = mean_bpt[K_max - 1]
    best_se  = sem_bpt[K_max - 1]
    eligible = np.where(mean_bpt >= (best_val - best_se))[0]
    K_1se = int(eligible[0]) + 1 if eligible.size else K_max

    # Exact sign-flip tests for consecutive pairs
    raw_p = []
    for k in range(max_states - 1):
        diffs = Best_test_LL[k + 1, :] - Best_test_LL[k, :]
        mask = np.isfinite(diffs)
        p = sign_flip_pvalue(diffs[mask]) if np.sum(mask) >= 1 else np.nan
        raw_p.append(p)
    reject_holm, p_adj_holm = holm_bonferroni(raw_p, alpha=0.05)
    sig_pairs = np.where(reject_holm)[0]
    K_last_sig = int(sig_pairs.max() + 2) if sig_pairs.size > 0 else 1

    selection = {
        "K_max": K_max,
        "K_plateau": K_plateau,
        "K_1se": K_1se,
        "K_last_sig": K_last_sig,
        "mean_bpt": mean_bpt,
        "sem_bpt": sem_bpt,
        "raw_p": np.array(raw_p),
        "reject_holm": reject_holm,
        "p_adj_holm": p_adj_holm,
    }

    return {
        "Best_HMM": Best_HMM,
        "Best_test_LL": Best_test_LL,
        "Best_train_LL": Best_train_LL,
        "AllInits_test_LL": AllInits_test_LL,
        "selection": selection,
        "meta": {
            "max_states": max_states,
            "n_splits": n_splits,
            "initializations": initializations,
            "nsessions": nsess,
        }
    }


# -----------------------------
# High-level driver over all areas + behavior
# -----------------------------
def evaluate_all_areas(results_from_builder,
                       areas=("FEF","PFC","LIP","Parietal","V4","MT","IT"),
                       max_states=4, n_splits=5, initializations=5, base_seed=42, n_jobs=-1):
    """
    results_from_builder: dict returned by build_all_glmhmm_inputs(data_neural)
    Returns a dict with CV outcomes for:
        - behavior_all (all sessions)
        - per area: behavior_matched, neural, combined
    """
    out = {"behavior_all": None, "areas": {}}

    # Behavior across ALL sessions
    beh_all = results_from_builder['behavior_all']
    out["behavior_all"] = run_glmhmm_cv(
        beh_all['inpts'], beh_all['true_choices'],
        max_states=max_states, n_splits=n_splits,
        initializations=initializations, base_seed=base_seed,
        n_jobs=n_jobs, desc="CV: Behavior (ALL)"
    )

    # Per area
    for area in areas:
        block = results_from_builder['areas'].get(area, None)
        if block is None or len(block['indices']) == 0:
            out["areas"][area] = {
                "indices": [],
                "behavior_matched": None,
                "neural": None,
                "combined": None,
            }
            continue

        # Behavior matched to this area's sessions
        beh = block['behavior_matched']
        neu = block['neural']
        com = block['combined']

        area_out = {"indices": block['indices']}
        area_out["behavior_matched"] = run_glmhmm_cv(
            beh['inpts'], beh['true_choices'],
            max_states=max_states, n_splits=n_splits,
            initializations=initializations, base_seed=base_seed,
            n_jobs=n_jobs, desc=f"CV: Behavior (matched {area})"
        )
        area_out["neural"] = run_glmhmm_cv(
            neu['inpts'], neu['true_choices'],
            max_states=max_states, n_splits=n_splits,
            initializations=initializations, base_seed=base_seed,
            n_jobs=n_jobs, desc=f"CV: Neural ({area})"
        )
        area_out["combined"] = run_glmhmm_cv(
            com['inpts'], com['true_choices'],
            max_states=max_states, n_splits=n_splits,
            initializations=initializations, base_seed=base_seed,
            n_jobs=n_jobs, desc=f"CV: Combined ({area})"
        )

        out["areas"][area] = area_out

    return out


# 1) Build session-wise inputs
builder_results = build_all_glmhmm_inputs(data_neural)  # from previous step



'''
# 2) Evaluate across all areas
cv_all = evaluate_all_areas(
    builder_results,
    areas=['FEF','PFC','LIP','Parietal','V4','MT','IT'],
    max_states=6,
    n_splits=3,
    initializations=3,
    base_seed=42,
    n_jobs=-1
)

# Examples of what you get:
# Global behavior (all sessions):
cv_all['behavior_all']['selection']['K_last_sig']   # chosen K by "last significant Δ"
cv_all['behavior_all']['selection']['mean_bpt']     # mean bits/trial by K

# For LIP:
cv_all['areas']['LIP']['indices']                   # kept session indices for LIP
cv_all['areas']['LIP']['neural']['selection']['K_max']
cv_all['areas']['LIP']['combined']['selection']['K_1se']
cv_all['areas']['LIP']['behavior_matched']['selection']['K_plateau']


# let's save results
with open('cv_all_results_neural.pkl', 'wb') as f:
    pickle.dump(cv_all, f)



# ---------- helpers ----------
def _summary_from_cvblock(cv_block):
    """Extract Ks, mean_bpt, sem_bpt, delta, and selections from a single cv result block."""
    sel = cv_block["selection"]
    Ks = np.arange(1, cv_block["meta"]["max_states"] + 1)
    mean_bpt = sel["mean_bpt"]
    sem_bpt = sel["sem_bpt"]
    delta = mean_bpt - mean_bpt[0]
    picks = {
        "K_max": sel["K_max"],
        "K_1se": sel["K_1se"],
        "K_last_sig": sel["K_last_sig"],
        "K_plateau": sel["K_plateau"],
        "best_delta": delta[sel["K_max"] - 1],
        "best_se": sem_bpt[sel["K_max"] - 1],
    }
    return Ks, mean_bpt, sem_bpt, delta, picks

def _sig_label_from_p(p):
    if not np.isfinite(p): return "n.s."
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "n.s."


# ---------- plot one block (delta vs K) ----------
def plot_cv_block(cv_block, title="Model selection (Δ bits/trial)"):
    Ks, mean_bpt, sem_bpt, delta, picks = _summary_from_cvblock(cv_block)

    # style
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "grid.color": "0.85",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    ax.plot(Ks, delta, marker='o', linestyle='-', linewidth=2, markersize=6, color='black', label='Δ test bits/trial')
    # markers for selections
    ax.plot(picks["K_max"], delta[picks["K_max"] - 1], marker='s', markersize=9, color='black', linestyle='None', label=f'Max mean: K={picks["K_max"]}')
    ax.plot(picks["K_1se"], delta[picks["K_1se"] - 1], marker='D', markersize=8, color='black', linestyle='None', label=f'1-SE: K={picks["K_1se"]}')
    ax.plot(picks["K_last_sig"], delta[picks["K_last_sig"] - 1], marker='^', markersize=9, color='black', linestyle='None', label=f'Last sig Δ: K={picks["K_last_sig"]}')

    # 1-SE band at best K
    ax.axhspan(picks["best_delta"] - picks["best_se"], picks["best_delta"] + picks["best_se"], color='0.9', alpha=0.6, label='±1 SE (best K)')

    # labels & limits
    y_min, y_max = np.nanmin(delta), np.nanmax(delta)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0
    y_range = max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.3 * y_range)
    ax.set_xlabel('# HMM states')
    ax.set_ylabel('Δ test bits/trial vs K=1')
    ax.set_title(title)
    ax.set_xticks(Ks)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


# ---------- 3-panel per-area comparison ----------
def plot_area_triptych(area_block, area_name):
    """
    area_block: cv_all['areas'][AREA], which contains three cv blocks:
        - behavior_matched, neural, combined
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    titles = ['Behavior (matched)', f'Neural ({area_name})', 'Combined']
    keys   = ['behavior_matched', 'neural', 'combined']

    for ax, key, ttl in zip(axes, keys, titles):
        cv_block = area_block.get(key, None)
        if cv_block is None:
            ax.set_axis_off()
            continue
        Ks, mean_bpt, sem_bpt, delta, picks = _summary_from_cvblock(cv_block)

        ax.plot(Ks, delta, marker='o', linestyle='-', linewidth=2, markersize=6, color='black', label='Δ test bits/trial')
        ax.plot(picks["K_max"], delta[picks["K_max"] - 1], marker='s', markersize=9, color='black', linestyle='None', label=f'Max mean: K={picks["K_max"]}')
        ax.plot(picks["K_1se"], delta[picks["K_1se"] - 1], marker='D', markersize=8, color='black', linestyle='None', label=f'1-SE: K={picks["K_1se"]}')
        ax.plot(picks["K_last_sig"], delta[picks["K_last_sig"] - 1], marker='^', markersize=9, color='black', linestyle='None', label=f'Last sig Δ: K={picks["K_last_sig"]}')
        ax.axhspan(picks["best_delta"] - picks["best_se"], picks["best_delta"] + picks["best_se"], color='0.9', alpha=0.6)

        y_min, y_max = np.nanmin(delta), np.nanmax(delta)
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = 0.0, 1.0
        y_range = max(y_max - y_min, 1e-6)
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.3 * y_range)

        ax.set_title(ttl)
        ax.set_xlabel('K')
        ax.set_xticks(Ks)
        if key == 'behavior_matched':
            ax.set_ylabel('Δ test bits/trial vs K=1')
        ax.grid(True, linestyle='--', linewidth=0.6, color='0.85')

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f'Area: {area_name}', y=1.08, fontsize=14)
    fig.tight_layout()
    return fig, axes


# ---------- overlay mean bits/trial curves (not delta) for a single area ----------
def plot_area_overlay_mean_bpt(area_block, area_name):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = [('behavior_matched', 'Behavior (matched)'), ('neural', f'Neural ({area_name})'), ('combined', 'Combined')]

    for key, label in styles:
        cv_block = area_block.get(key, None)
        if cv_block is None:
            continue
        Ks, mean_bpt, sem_bpt, delta, picks = _summary_from_cvblock(cv_block)
        ax.errorbar(Ks, mean_bpt, yerr=sem_bpt, fmt='-o', linewidth=2, markersize=5, label=label)

    ax.set_xlabel('# HMM states')
    ax.set_ylabel('Mean test bits/trial')
    ax.set_title(f'{area_name}: Mean test bits/trial vs K')
    ax.grid(True, linestyle='--', linewidth=0.6, color='0.85')
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


# ---------- simple text table of selected K per dataset ----------
def selection_table_for_area(area_block, area_name):
    rows = []
    for key, label in [('behavior_matched','Behavior (matched)'), ('neural','Neural'), ('combined','Combined')]:
        cv_block = area_block.get(key, None)
        if cv_block is None:
            continue
        sel = cv_block['selection']
        rows.append({
            'Dataset': label if label != 'Neural' else f'Neural ({area_name})',
            'K_max': sel['K_max'],
            'K_1se': sel['K_1se'],
            'K_last_sig': sel['K_last_sig'],
            'K_plateau': sel['K_plateau'],
            'mean@K* (bits/trial)': round(sel['mean_bpt'][sel['K_max'] - 1], 6),
        })
    return pd.DataFrame(rows)


# ======================
# EXAMPLES OF USAGE
# ======================

# 1) Plot behavior across ALL sessions:
fig, ax = plot_cv_block(cv_all['behavior_all'], title="Behavior (ALL): Δ bits/trial vs K")

# 2) Per-area triptych (behavior matched, neural, combined):
area_name = 'LIP'
fig, axes = plot_area_triptych(cv_all['areas'][area_name], area_name)

# 3) Overlay mean bits/trial curves for one area:
fig, ax = plot_area_overlay_mean_bpt(cv_all['areas'][area_name], area_name)

# 4) Selection table to print:
print(selection_table_for_area(cv_all['areas'][area_name], area_name))



import numpy as np
import matplotlib.pyplot as plt

def _get_summary(block):
    """Return Ks, mean_bpt, sem_bpt from a cv result block (or None if missing)."""
    if block is None:
        return None
    sel = block.get("selection", {})
    meta = block.get("meta", {})
    if "mean_bpt" not in sel or "max_states" not in meta:
        return None
    Ks = np.arange(1, meta["max_states"] + 1)
    mean_bpt = np.asarray(sel["mean_bpt"], dtype=float)
    sem_bpt  = np.asarray(sel["sem_bpt"], dtype=float)
    return Ks, mean_bpt, sem_bpt

def _style(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, color="0.85")
    ax.set_xlabel("# HMM states")
    ax.set_ylabel("Mean test bits/trial")
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

def plot_compare_neural(cv_all, areas=None, title="Neural: mean test bits/trial vs K"):
    """
    Overlays mean bits/trial curves for NEURAL across all brain areas.
    """
    if areas is None:
        areas = list(cv_all["areas"].keys())
    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted = 0

    for area in areas:
        block = cv_all["areas"].get(area, {}).get("neural", None)
        summary = _get_summary(block)
        if summary is None:  # area missing or no sessions
            continue
        Ks, mean_bpt, sem_bpt = summary
        ax.errorbar(Ks, mean_bpt, yerr=sem_bpt, fmt="-o", linewidth=2, markersize=5, label=area)
        plotted += 1

    _style(ax)
    ax.set_title(title)
    if plotted == 0:
        ax.text(0.5, 0.5, "No neural results to display.", transform=ax.transAxes, ha="center")
    else:
        ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    return fig, ax

def plot_compare_combined(cv_all, areas=None, title="Combined (behavior+neural): mean test bits/trial vs K"):
    """
    Overlays mean bits/trial curves for COMBINED across all brain areas.
    """
    if areas is None:
        areas = list(cv_all["areas"].keys())
    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted = 0

    for area in areas:
        block = cv_all["areas"].get(area, {}).get("combined", None)
        summary = _get_summary(block)
        if summary is None:
            continue
        Ks, mean_bpt, sem_bpt = summary
        ax.errorbar(Ks, mean_bpt, yerr=sem_bpt, fmt="-o", linewidth=2, markersize=5, label=area)
        plotted += 1

    _style(ax)
    ax.set_title(title)
    if plotted == 0:
        ax.text(0.5, 0.5, "No combined results to display.", transform=ax.transAxes, ha="center")
    else:
        ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    return fig, ax

def plot_compare_behavior(cv_all, areas=None,
                          title="Behavior (per-area matched) + Behavior (ALL): mean test bits/trial vs K",
                          include_overall=True):
    """
    Overlays mean bits/trial curves for BEHAVIOR across all areas (matched sessions),
    and also overlays the overall behavior (all sessions) if include_overall=True.
    """
    if areas is None:
        areas = list(cv_all["areas"].keys())
    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted = 0

    # Overall behavior (ALL sessions)
    if include_overall and "behavior_all" in cv_all:
        block_all = cv_all["behavior_all"]
        summary = _get_summary(block_all)
        if summary is not None:
            Ks, mean_bpt, sem_bpt = summary
            # thicker line for overall
            ax.errorbar(Ks, mean_bpt, yerr=sem_bpt, fmt="-o", linewidth=2.8, markersize=6,
                        label="Behavior (ALL)")

    # Per-area matched behavior
    for area in areas:
        block = cv_all["areas"].get(area, {}).get("behavior_matched", None)
        summary = _get_summary(block)
        if summary is None:
            continue
        Ks, mean_bpt, sem_bpt = summary
        ax.errorbar(Ks, mean_bpt, yerr=sem_bpt, fmt="--o", linewidth=2, markersize=5,
                    label=f"{area} (matched)")
        plotted += 1

    _style(ax)
    ax.set_title(title)
    if plotted == 0 and not include_overall:
        ax.text(0.5, 0.5, "No behavior results to display.", transform=ax.transAxes, ha="center")
    else:
        ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    return fig, ax
# Choose your areas (optional). Default uses all in cv_all['areas'] keys.
areas = ['FEF','PFC','LIP','Parietal','V4','MT','IT']

# 1) Neural comparison across areas
fig_neu, ax_neu = plot_compare_neural(cv_all, areas)

# 2) Combined comparison across areas
fig_com, ax_com = plot_compare_combined(cv_all, areas)

# 3) Behavior comparison across areas (per-area matched) + overall behavior
fig_beh, ax_beh = plot_compare_behavior(cv_all, areas, include_overall=True)

# Optionally save:
fig_neu.savefig("compare_neural_across_areas.png", dpi=200, bbox_inches="tight")
fig_com.savefig("compare_combined_across_areas.png", dpi=200, bbox_inches="tight")
fig_beh.savefig("compare_behavior_across_areas.png", dpi=200, bbox_inches="tight")

'''

results = build_all_glmhmm_inputs(data_neural)


from sklearn.metrics import accuracy_score  # make sure this is imported
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import ssm

LN2 = np.log(2.0)


# ---------- Fit (standard for K=1, sticky for K>1) ----------
def fit_glmhmm_logits(
    inpts_train, true_choices_train,
    *,
    K=2,
    prior_sigma=2.0,
    prior_alpha=2.0,
    prior_kappa=0.0,
    num_iters=300,
    tolerance=1e-3,
    seed=0,
):
    """Fit a GLM-HMM on lists of sessions (inputs + choices)."""
    inpts_train = [np.asarray(X) for X in inpts_train]
    true_choices_train = [np.asarray(y).reshape(-1, 1).astype(int) for y in true_choices_train]

    M, D, C = inpts_train[0].shape[1], 1, 2
    rng = np.random.default_rng(seed)

    if K == 1:
        hmm = ssm.HMM(
            K=1, D=D, M=M,
            observations="input_driven_obs",
            observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
            transitions="standard",
            rng=rng,
        )
    else:
        hmm = ssm.HMM(
            K=K, D=D, M=M,
            observations="input_driven_obs",
            observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
            transitions="sticky",
            transition_kwargs=dict(alpha=prior_alpha, kappa=prior_kappa),
            rng=rng,
        )

    hmm.fit(true_choices_train, inputs=inpts_train, method="em",
            num_iters=num_iters, tolerance=tolerance)
    return hmm


# ---------- Evaluate held-out bits per trial ----------
def heldout_bits_per_trial(hmm, inpts_test, true_choices_test):
    """Compute normalized log-likelihood (bits per trial) on held-out sessions."""
    total_ll, total_T = 0.0, 0
    for Xs, ys in zip(inpts_test, true_choices_test):
        Xs = np.asarray(Xs)
        ys = np.asarray(ys).reshape(-1, 1).astype(int)
        mask = np.ones_like(ys, dtype=bool)
        ll = hmm.log_probability([ys], inputs=[Xs], masks=[mask], tags=[None])
        total_ll += float(ll)
        total_T  += int(ys.shape[0])
    if total_T == 0:
        return np.nan
    return (total_ll / total_T) / LN2  # bits/trial


# ---------- Predict state-marginalized probabilities for accuracy ----------
def predict_probs_logits_marginalized(hmm, inpts_test, true_choices_test):
    """Return concatenated p(y=1|x) and true labels across test sessions."""
    probs_all, y_all = [], []
    for Xs, ys in zip(inpts_test, true_choices_test):
        Xs = np.asarray(Xs)
        ys = np.asarray(ys).reshape(-1, 1).astype(int)
        mask = np.ones_like(ys, dtype=bool)

        Ez = hmm.expected_states(data=ys, input=Xs, mask=mask)
        Ez = Ez[0] if isinstance(Ez, tuple) else Ez  # (T,K)

        # logits: (T,K,C) → stable softmax
        logits = hmm.observations.calculate_logits(input=Xs)
        logits -= logits.max(axis=2, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=2, keepdims=True)

        p1_k = probs[:, :, 1]            # (T,K)
        p1   = np.sum(Ez * p1_k, axis=1) # (T,)
        probs_all.append(p1)
        y_all.append(ys.ravel())
    return np.concatenate(probs_all), np.concatenate(y_all)


# ---------- CV with many inits; pick best by held-out bits/trial; report accuracy ----------
def cv_glmhmm_bpt_select_best_acc(
    inpts, true_choices,
    *,
    n_splits=5,
    K=2,
    prior_sigma=2.0,
    prior_alpha=2.0,
    prior_kappa=0.0,
    num_iters=3000,
    tolerance=1e-3,
    n_inits=5,
    base_seed=0,
    print_per_fold=True,
):
    """
    Session-block CV:
      - For each fold, fit n_inits models with different seeds.
      - Select the model with the highest held-out bits/trial.
      - Compute accuracy using that selected model on the same held-out sessions.
    """
    inpts = [np.asarray(X) for X in inpts]
    true_choices = [np.asarray(y).reshape(-1, 1).astype(int) for y in true_choices]

    n_sessions = len(inpts)
    if n_sessions != len(true_choices):
        raise ValueError("inpts and true_choices must have the same number of sessions.")
    n_splits = min(n_splits, n_sessions)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=base_seed)

    acc_folds = []
    bpt_folds_best = []
    per_init_bpt = np.full((n_inits, n_splits), np.nan, dtype=float)
    best_models = [None] * n_splits

    for f, (tr_idx, te_idx) in enumerate(kf.split(range(n_sessions)), 1):
        in_tr   = [inpts[i] for i in tr_idx]
        ch_tr   = [true_choices[i] for i in tr_idx]
        in_te   = [inpts[i] for i in te_idx]
        ch_te   = [true_choices[i] for i in te_idx]

        best_bpt = -np.inf
        best_hmm = None

        # Many inits → pick best by held-out bits/trial
        for z in range(n_inits):
            seed = base_seed + 10_000 * f + z
            hmm = fit_glmhmm_logits(
                in_tr, ch_tr, K=K,
                prior_sigma=prior_sigma, prior_alpha=prior_alpha, prior_kappa=prior_kappa,
                num_iters=num_iters, tolerance=tolerance, seed=seed
            )
            bpt = heldout_bits_per_trial(hmm, in_te, ch_te)
            per_init_bpt[z, f-1] = bpt
            if np.isfinite(bpt) and bpt > best_bpt:
                best_bpt = bpt
                best_hmm = hmm

        # Use the best model to compute accuracy on held-out sessions
        probs, y_true = predict_probs_logits_marginalized(best_hmm, in_te, ch_te)
        yhat = (probs >= 0.5).astype(int)
        acc  = accuracy_score(y_true, yhat)

        acc_folds.append(acc)
        bpt_folds_best.append(best_bpt)
        best_models[f-1] = best_hmm

        if print_per_fold:
            n_te = int(sum(len(s) for s in ch_te))
            print(f"Fold {f}/{n_splits} | N={n_te} | best BPT={best_bpt:.6f} | Acc={acc:.3f}")

    res = {
        "acc_folds": acc_folds,
        "acc_mean": float(np.nanmean(acc_folds)),
        "acc_std":  float(np.nanstd(acc_folds)),
        "best_bpt_folds": bpt_folds_best,
        "bpt_mean": float(np.nanmean(bpt_folds_best)),
        "bpt_std":  float(np.nanstd(bpt_folds_best)),
        "per_init_bpt": per_init_bpt,   # shape (n_inits, n_splits)
        "best_models": best_models,     # HMM object per fold
    }

    print("\n=== CV (select best init by held-out bits/trial) ===")
    print(f"Accuracy:  {res['acc_mean']:.3f} ± {res['acc_std']:.3f}")
    print(f"Bits/trial (best per fold): {res['bpt_mean']:.6f} ± {res['bpt_std']:.6f}")
    return res


# ---------------- Example call ----------------
# cv_out = cv_glmhmm_bpt_select_best_acc(
#     inpts, true_choices,
#     n_splits=3,
#     K=2,
#     prior_sigma=2.0,
#     prior_alpha=2.0,
#     prior_kappa=0.0,
#     num_iters=3000,
#     tolerance=1e-3,
#     n_inits=8,
#     base_seed=0,
#     print_per_fold=True,
# )

# ------------------------------------------------------------
# Assumes the following are already defined in your session:
#   - cv_glmhmm_bpt_select_best_acc (from your message)
#   - results = build_all_glmhmm_inputs(data_neural)
#   - cv_all  = evaluate_all_areas(results, ...)
# If you didn't keep `results`, you can still pull inpts/choices from `results`.
# ------------------------------------------------------------

def _run_accuracy_grid(inpts, true_choices, K_grid=(1,2,3,4),
                       n_splits=5, n_inits=5, num_iters=3000, tolerance=1e-3,
                       prior_sigma=2.0, prior_alpha=2.0, prior_kappa=0.0,
                       base_seed=0, quiet=True):
    """
    Run accuracy CV across K for one dataset (inpts/true_choices lists).
    Returns: dict {K: cv_result_dict} where cv_result_dict has acc_mean, acc_std, etc.
    """
    if inpts is None or true_choices is None or len(inpts) == 0:
        return {}
    out = {}
    for K in K_grid:
        if not quiet:
            print(f"\n=== CV accuracy for K={K} ===")
        res = cv_glmhmm_bpt_select_best_acc(
            inpts, true_choices,
            n_splits=n_splits,
            K=K,
            prior_sigma=prior_sigma,
            prior_alpha=prior_alpha,
            prior_kappa=prior_kappa,
            num_iters=num_iters,
            tolerance=tolerance,
            n_inits=n_inits,
            base_seed=base_seed,
            print_per_fold=(not quiet),
        )
        out[K] = res
    return out

def run_accuracy_for_all(results_builder,        # from build_all_glmhmm_inputs(...)
                         areas=('FEF','PFC','LIP','Parietal','V4','MT','IT'),
                         K_grid=(1,2,3,4),
                         n_splits=5, n_inits=5, num_iters=3000, tolerance=1e-3,
                         prior_sigma=2.0, prior_alpha=2.0, prior_kappa=0.0,
                         base_seed=0, quiet=True):
    """
    Compute predictive accuracy (CV over sessions, best init by held-out bits/trial)
    for:
      - Behavior across ALL sessions
      - For each area: behavior_matched, neural, combined
    Returns nested dict with accuracy curves vs K for each.
    """
    out = {"behavior_all": None, "areas": {}}

    # Behavior (ALL sessions)
    beh_all = results_builder['behavior_all']
    out['behavior_all'] = _run_accuracy_grid(
        beh_all['inpts'], beh_all['true_choices'],
        K_grid=K_grid, n_splits=n_splits, n_inits=n_inits,
        num_iters=num_iters, tolerance=tolerance,
        prior_sigma=prior_sigma, prior_alpha=prior_alpha, prior_kappa=prior_kappa,
        base_seed=base_seed, quiet=quiet
    )

    # Per-area datasets
    for area in areas:
        block = results_builder['areas'].get(area, None)
        area_out = {"behavior_matched": {}, "neural": {}, "combined": {}, "indices": []}
        if block is None or len(block['indices']) == 0:
            out['areas'][area] = area_out
            continue

        area_out['indices'] = block['indices']

        # Matched behavior
        beh = block['behavior_matched']
        area_out['behavior_matched'] = _run_accuracy_grid(
            beh['inpts'], beh['true_choices'],
            K_grid=K_grid, n_splits=n_splits, n_inits=n_inits,
            num_iters=num_iters, tolerance=tolerance,
            prior_sigma=prior_sigma, prior_alpha=prior_alpha, prior_kappa=prior_kappa,
            base_seed=base_seed, quiet=quiet
        )

        # Neural-only
        neu = block['neural']
        area_out['neural'] = _run_accuracy_grid(
            neu['inpts'], neu['true_choices'],
            K_grid=K_grid, n_splits=n_splits, n_inits=n_inits,
            num_iters=num_iters, tolerance=tolerance,
            prior_sigma=prior_sigma, prior_alpha=prior_alpha, prior_kappa=prior_kappa,
            base_seed=base_seed, quiet=quiet
        )

        # Combined
        com = block['combined']
        area_out['combined'] = _run_accuracy_grid(
            com['inpts'], com['true_choices'],
            K_grid=K_grid, n_splits=n_splits, n_inits=n_inits,
            num_iters=num_iters, tolerance=tolerance,
            prior_sigma=prior_sigma, prior_alpha=prior_alpha, prior_kappa=prior_kappa,
            base_seed=base_seed, quiet=quiet
        )

        out['areas'][area] = area_out

    return out

# ---------- helpers to flatten accuracy curves to DataFrames ----------
def accuracy_table_from_results(res_by_K):
    """
    Given {K: cv_result_dict} produce a tidy DataFrame with columns:
    ['K','acc_mean','acc_std','bpt_mean','bpt_std']
    """
    rows = []
    for K, d in sorted(res_by_K.items()):
        rows.append({
            "K": int(K),
            "acc_mean": d.get("acc_mean", np.nan),
            "acc_std":  d.get("acc_std",  np.nan),
            "bpt_mean": d.get("bpt_mean", np.nan),
            "bpt_std":  d.get("bpt_std",  np.nan),
        })
    return pd.DataFrame(rows)

def build_all_accuracy_tables(acc_all, areas=('FEF','PFC','LIP','Parietal','V4','MT','IT')):
    """
    Create DataFrames:
      - behavior_all_df
      - per-area dict with behavior_matched_df, neural_df, combined_df
    """
    behavior_all_df = accuracy_table_from_results(acc_all["behavior_all"])

    per_area = {}
    for area in areas:
        block = acc_all["areas"].get(area, None)
        if block is None or 'indices' not in block:
            continue
        per_area[area] = {
            "behavior_matched_df": accuracy_table_from_results(block["behavior_matched"]),
            "neural_df":           accuracy_table_from_results(block["neural"]),
            "combined_df":         accuracy_table_from_results(block["combined"]),
            "indices":             block["indices"],
        }
    return behavior_all_df, per_area


# You already have:
#   results = build_all_glmhmm_inputs(data_neural)
# (if you don't: run that first)

K_grid = (1,2,3,4,5,6)

# 1) Run accuracy CV for all datasets
acc_all = run_accuracy_for_all(
    results_builder=results,
    areas=('FEF','PFC','LIP','Parietal','V4','MT','IT'),
    K_grid=K_grid,
    n_splits=3,
    n_inits=1,
    num_iters=3000,
    tolerance=1e-2,
    base_seed=0,
    quiet=True
)



# 2) Build tidy tables (easy to print / save / plot)
behavior_all_df, per_area_tables = build_all_accuracy_tables(acc_all)


# Save the results
with open('cv_all_accuracy_results.pkl', 'wb') as f:
    pickle.dump({
        'acc_all': acc_all,
        'behavior_all_df': behavior_all_df,
        'per_area_tables': per_area_tables,
    }, f)

print("Behavior (ALL) accuracy:")
print(behavior_all_df)

print("\nExample: LIP tables")
lip_tbls = per_area_tables.get('LIP', None)
if lip_tbls is not None:
    print("LIP - Behavior (matched):")
    print(lip_tbls["behavior_matched_df"])
    print("LIP - Neural:")
    print(lip_tbls["neural_df"])
    print("LIP - Combined:")
    print(lip_tbls["combined_df"])


def plot_compare_accuracy(acc_all, which='neural', areas=None, title=None):
    """
    which: 'neural' | 'combined' | 'behavior_matched' | 'behavior_all'
    """
    if which == 'behavior_all':
        df = accuracy_table_from_results(acc_all['behavior_all'])
        fig, ax = plt.subplots(figsize=(7,5))
        ax.errorbar(df['K'], df['acc_mean'], yerr=df['acc_std'], fmt='-o', linewidth=2, markersize=6, label='Behavior (ALL)')
        ax.set_xlabel('K'); ax.set_ylabel('Accuracy'); ax.grid(True, linestyle='--', color='0.85')
        ax.set_title(title or 'Behavior (ALL): CV Accuracy vs K')
        ax.legend(frameon=False); plt.tight_layout()
        return fig, ax

    if areas is None:
        areas = list(acc_all['areas'].keys())

    fig, ax = plt.subplots(figsize=(9,6))
    for area in areas:
        block = acc_all['areas'].get(area, {})
        resK = block.get(which, {})
        if not resK:
            continue
        df = accuracy_table_from_results(resK)
        ax.errorbar(df['K'], df['acc_mean'], yerr=df['acc_std'], fmt='-o', linewidth=2, markersize=5, label=area)

    ax.set_xlabel('K'); ax.set_ylabel('Accuracy'); ax.grid(True, linestyle='--', color='0.85')
    ttl = {
        'neural': 'Neural: CV Accuracy vs K',
        'combined': 'Combined: CV Accuracy vs K',
        'behavior_matched': 'Behavior (matched): CV Accuracy vs K',
    }.get(which, which)
    ax.set_title(title or ttl)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    return fig, ax

# Examples:
fig1, ax1 = plot_compare_accuracy(acc_all, which='neural', areas=('FEF','PFC','LIP','Parietal','V4','MT','IT'))
fig2, ax2 = plot_compare_accuracy(acc_all, which='combined')
fig3, ax3 = plot_compare_accuracy(acc_all, which='behavior_matched')
fig4, ax4 = plot_compare_accuracy(acc_all, which='behavior_all')
