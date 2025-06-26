from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
import numpy as np
from collections import defaultdict
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))  # Two levels up from notebooks
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Neural.scripts_neural.preprocessing_neural import preprocess_data, create_y




### LOO & Training and testing on each timebin
def decode_one_session_timebin(session, data, area, variables_to_analyze):
    print(f"\n--- Decoding session {session} for area {area} ---")

    # Select neurons from the area
    area_idx = data['unit'][session]['area'].values == area
    if np.sum(area_idx) < 10:
        print(f"  Skipping session {session}: too few neurons ({np.sum(area_idx)})")
        return session, None, None

    # Extract spike data and trial data
    spike_data = data['spikecounts'][session][:, area_idx, :]  # trials x neurons x time
    trial_data = data['trial'][session]

    # Generate or retrieve target variables for decoding
    y_all = create_y(trial_data)

    # Optionally update trial_data variables (as some of them might not exist in the dataframe directly)
    for key, values in y_all.items():
        trial_data[key] = values

    session_preds = {}

    n_timebins = spike_data.shape[2]

    # Loop through variables to decode
    for v in variables_to_analyze:
        print(f"    Decoding variable: {v}")
        target = np.array(y_all[v])

        # Skip if only one class is present
        if len(np.unique(target)) < 2:
            print(f"      Skipping {v}: only one class present")
            continue

        preds_timebin = np.full((len(target), n_timebins), np.nan)

        for t in range(n_timebins):
            spike_t = spike_data[:, :, t]  # trials x neurons at timebin t

            # Skip timebin if spike data is all zeros
            if np.all(spike_t == 0):
                continue

            loo = LeaveOneOut()
            clf = Pipeline([
                ('standardize', StandardScaler()),
                ('PCA', PCA(n_components=0.9)),
                ('clf', LinearSVC())
            ])

            predictions = np.zeros_like(target, dtype=float)

            # Train and test on the same timebin using Leave-One-Out cross-validation
            for train_index, test_index in loo.split(spike_t):
                X_train, X_test = spike_t[train_index], spike_t[test_index]
                y_train = target[train_index]

                # Train classifier
                clf.fit(X_train, y_train)

                # Predict
                pred = clf.predict(X_test)
                predictions[test_index[0]] = pred[0]

            preds_timebin[:, t] = predictions

        # Store in session_preds
        session_preds[v] = preds_timebin

    print(f"  ✓ Finished decoding session {session} for area {area}")
    return session, session_preds, trial_data



def decode_area_loo_parallel_timebin(data, sessions, area, variables_to_analyze):
    """
    Run decoding in parallel across sessions for a specific brain area (time-resolved).
    """
    print(f"\n>>> Starting decoding for area: {area} (n_sessions={len(sessions)})")

    results = Parallel(n_jobs=-1)(
        delayed(decode_one_session_timebin)(
            session, data, area, variables_to_analyze
        )
        for session in sessions
    )

    predicted_results = defaultdict(dict)
    for session, preds, updated_trial_data in results:
        if preds is not None:
            predicted_results[area][session] = preds
            data['trial'][session] = updated_trial_data

    return data, predicted_results









