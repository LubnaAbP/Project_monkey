from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from tqdm import trange,tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.signal
import itertools
import sys, os
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
import itertools
from math import ceil
from sklearn.model_selection import StratifiedKFold

bin_size = 0.1
time = np.arange(-2.5, 3.5, bin_size)[:59]
stim_idx = (time > -0.5) & (time < 0.0)  # Boolean mask

### bigger bins
with open('/home/labdul/Lubna/Projects_monkey/Monkey-Project/DATA/filtered_mocol_categorization_bin_0.1.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
print(loaded_data.keys())


def smoothing(X: np.array, bin_size=bin_size, K=1.0, width=2.0):
    """
    Applies exponential smoothing to spike data (one session).

    Parameters:
    X : np.array
        Spike data (trials x neurons x time bins)
    bin_size : float
        Width of each time bin (in seconds)
    K : float
        Shape parameter for exponential decay
    width : float
        Controls smoothing extent (in seconds)

    Returns:
    X_smoothed : np.array
        Smoothed spike data
    """
    bin_w = int(ceil(width / bin_size))
    win = scipy.signal.windows.exponential(2 * bin_w + 1, tau=bin_w / (2 * K))
    win[:bin_w] = 0
    win /= win.sum() * bin_size  # Normalize for area under the curve

    new_data = np.zeros_like(X)
    convol_fun = lambda x: np.convolve(x, win, mode='same')

    for c, n in itertools.product(range(X.shape[0]), range(X.shape[1])):
        new_data[c, n, :] = convol_fun(X[c, n, :])

    return new_data


def create_y(trial_data):
    # Use original values (in degrees)
    direction = trial_data['direction'].values
    color = trial_data['color'].values

    # Responses
    response = trial_data['chosenResponse']

    # Previous trial info
    prev_direction = np.roll(direction, 1)
    prev_color = np.roll(color, 1)
    prev_response = np.roll(response, 1)

    context_response = trial_data['rule']
    prev_context_response = np.roll(context_response, 1)

    return {
        'direction': direction,
        'color': color,
        'response': response,
        'context_response': context_response,
        'previous_direction': prev_direction,
        'previous_color': prev_color,
        'previous_response': prev_response,
        'previous_context_response': prev_context_response
    }



def decode_one_session_with_real_and_shuffle(session, data, area, variables_to_analyze, time_filter, n_shuffles=10, n_splits=10):
    print(f"\n--- Decoding session {session} for area {area} (real + shuffled, CV {n_splits}-fold) ---")

    area_idx = data['unit'][session]['area'].values == area
    if np.sum(area_idx) < 10:
        print(f"  Skipping session {session}: too few neurons ({np.sum(area_idx)})")
        return session, None, None

    spike_data = data['spikecounts'][session][:, area_idx]
    trial_data = data['trial'][session]
    y_all = create_y(trial_data)

    for key, values in y_all.items():
        trial_data[key] = values

    if time_filter is not None:
        time_filter_indices = np.where(time_filter)[0]
        spike_filtered = np.mean(spike_data[:, :, time_filter_indices], axis=-1)
    else:
        spike_filtered = np.mean(spike_data, axis=-1)

    scaler = StandardScaler()
    spike_scaled = scaler.fit_transform(spike_filtered)

    pca = PCA(n_components=0.9)
    spike_pca = pca.fit_transform(spike_scaled)

    session_shuffled_preds = {}

    for v in variables_to_analyze:
        print(f"    Decoding variable: {v}")
        target = np.array(y_all[v])

        if len(np.unique(target)) < 2:
            print(f"      Skipping {v}: only one class present")
            predictions = np.full((n_shuffles, len(target)), np.nan, dtype=float)
            real_predictions = np.full(len(target), np.nan, dtype=float)
        else:
            predictions = np.zeros((n_shuffles, len(target)), dtype=float)
            real_predictions = np.zeros_like(target, dtype=float)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

            def one_shuffle(_):
                preds = np.zeros_like(target, dtype=float)
                for train_index, test_index in skf.split(spike_pca, target):
                    X_train, X_test = spike_pca[train_index], spike_pca[test_index]
                    y_train = target[train_index]
                    y_train_shuffled = np.random.permutation(y_train)

                    clf = LinearSVC(max_iter=1000, tol=1e-3)
                    clf.fit(X_train, y_train_shuffled)
                    preds[test_index] = clf.predict(X_test)
                return preds

            clf_real = LinearSVC(max_iter=1000, tol=1e-3)
            for train_index, test_index in skf.split(spike_pca, target):
                X_train, X_test = spike_pca[train_index], spike_pca[test_index]
                y_train = target[train_index]
                clf_real.fit(X_train, y_train)
                real_predictions[test_index] = clf_real.predict(X_test)

            pred_col_name = f'predicted_{v}_{area}'
            trial_data[pred_col_name] = real_predictions

            predictions = Parallel(n_jobs=-1)(
                delayed(one_shuffle)(i) for i in range(n_shuffles)
            )
            predictions = np.vstack(predictions)

        session_shuffled_preds[v] = predictions

    print(f"  ✓ Finished decoding session {session} for area {area}")
    return session, session_shuffled_preds, trial_data

def decode_area_cv_parallel_with_real_and_shuffle(data, sessions, area, variables_to_analyze, time_filter, n_shuffles=10, n_splits=10):
    print(f"\n>>> Starting decoding (real + shuffled, CV {n_splits}-fold) for area: {area} (n_sessions={len(sessions)})")

    results = Parallel(n_jobs=-1)(
        delayed(decode_one_session_with_real_and_shuffle)(
            session, data, area, variables_to_analyze, time_filter, n_shuffles, n_splits
        )
        for session in sessions
    )

    predicted_results = defaultdict(dict)
    for session, preds, updated_trial_data in results:
        if preds is not None:
            predicted_results[area][session] = preds
            data['trial'][session] = updated_trial_data

    return data, predicted_results




# === Load Data ===
data = loaded_data
num_sessions = len(data['spikecounts'])
print(f"Number of available sessions: {num_sessions}")

if num_sessions == 0:
    raise ValueError("No sessions found in data!")

# Only decode the first 3 sessions
#sessions = list(range(min(2, num_sessions)))
sessions = list(range(num_sessions))


centered_file_path = 'data_centered_0.1_big_smoithing.pkl'
smoothed_file_path = 'data_smoothed_0.1_big_smoithing.pkl'

# === Step 1: Try loading centered data (spikecounts + unit) ===
if os.path.exists(centered_file_path):
    print('Centered + smoothed data already exists.')
    with open(centered_file_path, 'rb') as f:
        spike_unit_package = pickle.load(f)

    if isinstance(spike_unit_package, dict) and 'spikecounts' in spike_unit_package:
        data['spikecounts'] = spike_unit_package['spikecounts']
        data['unit'] = spike_unit_package['unit']
        print('  Loaded centered + smoothed spikecounts and unit metadata.')
    else:
        raise ValueError("Invalid format in centered file — expected a dict with 'spikecounts' and 'unit'.")

# === Step 2: If centered data doesn't exist, check for smoothed data ===
else:
    if os.path.exists(smoothed_file_path):
        print('Smoothed data found (not centered).')
        with open(smoothed_file_path, 'rb') as f:
            data['spikecounts'] = pickle.load(f)
        print('  Loaded smoothed spikecounts.')
    else:
        print('No smoothed data found — computing smoothing...')
        data['spikecounts'] = [
            smoothing(np.array(session_data), bin_size=bin_size, width=1.5, K=2)
            for session_data in tqdm(data['spikecounts'], desc='Smoothing Data')
        ]
        with open(smoothed_file_path, 'wb') as f:
            pickle.dump(data['spikecounts'], f)
        print('  Saved smoothed data.')

    # === Step 3: Now center the smoothed data ===
    print('Centering smoothed spikecounts...')
    for s, d in enumerate(tqdm(data['spikecounts'], desc='Centering Data')):
        # Reshape to neuron × time
        d_s = np.einsum('ijk -> jik', d).reshape(d.shape[1], -1)

        # Filter neurons with non-zero standard deviation
        stdev = np.std(d_s, axis=1)
        idx = stdev > 0
        stdev = stdev[idx]
        mmean = np.mean(d_s[idx], axis=1)

        # Filter both unit and spikecounts data
        data['unit'][s] = data['unit'][s].loc[idx].reset_index(drop=True)
        data['spikecounts'][s] = (d[:, idx] - mmean[None, :, None]) / stdev[None, :, None]

        # Final safety check
        assert data['spikecounts'][s].shape[1] == len(data['unit'][s]), f"Mismatch in session {s}"

    # === Step 4: Save the centered data (both spikecounts and unit metadata) ===
    with open(centered_file_path, 'wb') as f:
        pickle.dump({'spikecounts': data['spikecounts'], 'unit': data['unit']}, f)
    print('  Saved centered + smoothed spikecounts and unit metadata.')

# === Diagnostic Check ===
print("\nVerifying session alignment:")
for s in range(len(data['spikecounts'])):
    n_spikes = data['spikecounts'][s].shape[1]
    n_units = len(data['unit'][s])
    print(f"Session {s}: spikecounts neurons = {n_spikes}, unit entries = {n_units}")
    assert n_spikes == n_units, f" Mismatch in session {s}"
print("All sessions verified.")


######## clean the data
for i, df in enumerate(data['trial']):
    if 'chosenResponse' in df.columns and 'expectedResponse' in df.columns:
        def clean(val):
            if isinstance(val, np.ndarray):
                return str(val[0]) if len(val) > 0 else None
            return str(val) if val is not None else None

        df['chosenResponse'] = df['chosenResponse'].apply(clean)
        df['expectedResponse'] = df['expectedResponse'].apply(clean)
        df['color'] = df['color']/90
        df['direction'] = df['direction']/90
        data['trial'][i] = df  # Save cleaned DataFrame back
    else:
        print(f"Session {i}: Missing required columns.")


for i, df in enumerate(data['trial']):
    if 'chosenResponse' in df.columns and 'expectedResponse' in df.columns:
        # Map values
        df['chosenResponse'] = df['chosenResponse'].map({'L': 1, 'R': 0})
        df['expectedResponse'] = df['expectedResponse'].map({'L': 1, 'R': 0})
        data['trial'][i] = df  # Save updated DataFrame back
    else:
        print(f"Session {i}: Missing required columns.")

for i, df in enumerate(data['trial']):
    if 'rule' in df.columns:
        # Extract the string from the array and map it to numeric
        df['rule'] = df['rule'].apply(
            lambda x: 1 if isinstance(x, (list, np.ndarray)) and 'color' in x else
                      0 if isinstance(x, (list, np.ndarray)) and 'motion' in x else
                      np.nan  # fallback for unexpected format
        )
        data['trial'][i] = df  # Update cleaned DataFrame

#### binary
for i, df in enumerate(data['trial']):
    if 'color' in df.columns and 'direction' in df.columns:
        # Store raw values
        df['color_raw'] = df['color'].astype(float)
        df['direction_raw'] = df['direction'].astype(float)

        # Binarize
        df['color'] = np.where(df['color_raw'] >= 0, 1, 0)
        df['direction'] = np.where(df['direction_raw'] >= 0, 1, 0)

        # Create previous values using .shift()
        df['previous_color'] = df['color_raw'].shift(1)
        df['previous_direction'] = df['direction_raw'].shift(1)

        # Binarize previous values (after shift)
        df['previous_color'] = np.where(df['previous_color'] >= 0, 1, 0)
        df['previous_direction'] = np.where(df['previous_direction'] >= 0, 1, 0)

        # Save back
        data['trial'][i] = df
'''
for i, df in enumerate(data['trial']):
    if 'color' in df.columns and 'direction' in df.columns:
        # Store raw values
        df['color_raw'] = df['color'].astype(float)
        df['direction_raw'] = df['direction'].astype(float)

        # Assign labels to unique values using category codes
        df['color'] = pd.Categorical(df['color_raw']).codes
        df['direction'] = pd.Categorical(df['direction_raw']).codes

        # Create previous values using .shift()
        df['previous_color_raw'] = df['color_raw'].shift(1)
        df['previous_direction_raw'] = df['direction_raw'].shift(1)

        # Assign labels to previous values
        df['previous_color'] = pd.Categorical(df['previous_color_raw'],
                                              categories=np.sort(df['color_raw'].unique())).codes
        df['previous_direction'] = pd.Categorical(df['previous_direction_raw'],
                                                  categories=np.sort(df['direction_raw'].unique())).codes

'''
# === Define Parameters ===
areas = ['PFC', 'FEF', 'IT', 'MT', 'LIP', 'Parietal', 'V4']
previous_variables = ['previous_color', 'previous_direction', 'previous_response', 'previous_context_response']
current_variables = ['color', 'direction', 'response', 'context_response']
variables_to_analyze = previous_variables + current_variables

n_shuffles = 5
n_splits = 10  # number of CV folds

shuffled_predictions = {}

trial_data_save_path = 'updated_trial_data_SVM_PRESTIM_24_06.pkl'
shuffled_preds_save_path = 'shuffled_predictions_SVM_PRESTIM_24_06.pkl'

if os.path.exists(trial_data_save_path) and os.path.exists(shuffled_preds_save_path):
    print("🔄 Found existing save files. Resuming from previous progress...")
    with open(trial_data_save_path, 'rb') as f:
        trial_data = pickle.load(f)
    with open(shuffled_preds_save_path, 'rb') as f:
        shuffled_predictions = pickle.load(f)
    for i in range(len(trial_data)):
        data['trial'][i] = trial_data[i]
else:
    print("🆕 No previous save found. Starting from scratch...")
    shuffled_predictions = {}

for area in areas:
    if area in shuffled_predictions and len(shuffled_predictions[area]) == len([s for s in sessions if np.sum(data['unit'][s]['area'].values == area) >= 5]):
        print(f"✅ Skipping {area}: already fully decoded and saved.")
        continue

    valid_sessions = [s for s in sessions if np.sum(data['unit'][s]['area'].values == area) >= 5]
    decoded_sessions = set(shuffled_predictions.get(area, {}).keys())
    sessions_to_decode = [s for s in valid_sessions if s not in decoded_sessions]

    if not sessions_to_decode:
        print(f"✅ All sessions for area {area} already decoded.")
        continue

    print(f"==> Decoding {len(sessions_to_decode)} sessions for area {area}")
    start_time = time.time()

    data, area_preds = decode_area_cv_parallel_with_real_and_shuffle(
        data, sessions_to_decode, area, variables_to_analyze, time_filter=stim_idx, n_shuffles=n_shuffles, n_splits=n_splits
    )

    if area not in shuffled_predictions:
        shuffled_predictions[area] = {}
    shuffled_predictions[area].update(area_preds[area])

    elapsed = time.time() - start_time
    print(f"✓ Finished decoding {len(sessions_to_decode)} sessions in area {area} in {elapsed:.2f}s")

    with open(trial_data_save_path, 'wb') as f:
        pickle.dump(data['trial'], f)
    with open(shuffled_preds_save_path, 'wb') as f:
        pickle.dump(shuffled_predictions, f)
    print(f"📏 Progress saved after decoding {area}!")

real_accuracies = {}
for area in areas:
    area_accs = {}
    for var in variables_to_analyze:
        all_acc = []
        for sess_id, df in enumerate(data['trial']):
            pred_col = f'predicted_{var}_{area}'
            if pred_col not in df.columns or var not in df.columns:
                continue
            y_true = df[var].values
            y_pred = df[pred_col].values
            valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
            if np.sum(valid) == 0:
                continue
            acc = np.mean(y_pred[valid] == y_true[valid])
            all_acc.append(acc)
        if all_acc:
            area_accs[var] = {
                'mean_acc': np.mean(all_acc),
                'std_acc': np.std(all_acc)
            }
    if area_accs:
        real_accuracies[area] = area_accs

shuffled_accuracies = {}
for area, area_data in shuffled_predictions.items():
    area_accs = {}
    for var in variables_to_analyze:
        all_acc = []
        for sess_id, preds in area_data.items():
            if var not in preds:
                continue
            y_pred_all = preds[var]
            df = data['trial'][sess_id]
            if var not in df.columns:
                continue
            y_true = df[var].values
            for shuffle_run in range(y_pred_all.shape[0]):
                y_pred = y_pred_all[shuffle_run]
                valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
                if np.sum(valid) == 0:
                    continue
                acc = np.mean(y_pred[valid] == y_true[valid])
                all_acc.append(acc)
        if all_acc:
            area_accs[var] = {
                'mean_acc': np.mean(all_acc),
                'std_acc': np.std(all_acc)
            }
    if area_accs:
        shuffled_accuracies[area] = area_accs

fig, ax = plt.subplots(figsize=(16, 8))
all_vars = variables_to_analyze
x = np.arange(len(all_vars))
width = 0.08
colors = itertools.cycle(plt.cm.tab10.colors)
areas_to_plot = [area for area in areas if area in real_accuracies and area in shuffled_accuracies]

for i, area in enumerate(areas_to_plot):
    color = next(colors)
    shuffled_means = np.array([
        shuffled_accuracies[area].get(var, {'mean_acc': np.nan})['mean_acc'] for var in all_vars
    ])
    shuffled_stds = np.array([
        shuffled_accuracies[area].get(var, {'std_acc': np.nan})['std_acc'] for var in all_vars
    ])
    real_means = np.array([
        real_accuracies[area].get(var, {'mean_acc': np.nan})['mean_acc'] for var in all_vars
    ])
    real_stds = np.array([
        real_accuracies[area].get(var, {'std_acc': np.nan})['std_acc'] for var in all_vars
    ])

    ax.bar(
        x + i * width,
        real_means,
        width,
        label=f'{area}',
        color=color,
        alpha=0.7
    )
    ax.errorbar(
        x + i * width,
        real_means,
        yerr=1.96 * real_stds,
        fmt='none',
        ecolor='black',
        capsize=5,
        linewidth=1,
    )
    for j, (mean, std) in enumerate(zip(shuffled_means, shuffled_stds)):
        if np.isnan(mean):
            continue
        ax.hlines(
            mean,
            x[j] + i * width - width / 2,
            x[j] + i * width + width / 2,
            colors='black',
            linestyles='dashed',
            linewidth=1.5,
            alpha=0.8
        )
        ax.fill_between(
            [x[j] + i * width - width / 2, x[j] + i * width + width / 2],
            mean - 1.96 * std,
            mean + 1.96 * std,
            color='gray',
            alpha=0.2
        )

ax.set_ylabel('Decoding Accuracy')
ax.set_title('Real vs Shuffled Decoding Accuracy per Variable and Area')
ax.set_xticks(x + width * len(areas_to_plot) / 2)
ax.set_xticklabels(all_vars, rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.legend(title='Brain Area')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()