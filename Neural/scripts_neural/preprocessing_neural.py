import numpy as np
import scipy
import scipy.signal
from math import ceil
import itertools



######## clean the data

# Define a function to preprocss
def preprocess_data(data):
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

    return data



#### smoothing the data

def smoothing(X: np.array, bin_size=0.025, K=2, width=1.5):
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


## creating the target variables for the decoder

def create_y(trial_data):

    y = {}

    # Current trial variables
    y['direction'] = trial_data['direction'].values
    y['color'] = trial_data['color'].values
    y['response'] = trial_data['chosenResponse'].values
    y['context'] = trial_data['rule'].values

    # Previous trial variables (already created during preprocessing)
    y['previous_direction'] = trial_data['previous_direction'].values
    y['previous_color'] = trial_data['previous_color'].values

    # Previous response
    if 'previous_response' not in trial_data.columns:
        trial_data['previous_response'] = trial_data['chosenResponse'].shift(1)

    # Previous context
    if 'previous_context' not in trial_data.columns:
        trial_data['previous_context'] = trial_data['rule'].shift(1)

    # === FIX NaNs ===
    # Replace NaNs with a dummy label (e.g., 0), so sklearn won't crash
    for key in ['previous_direction', 'previous_color', 'previous_response', 'previous_context_response']:
        if key in trial_data.columns:
            trial_data[key] = trial_data[key].fillna(0)

    y['previous_response'] = trial_data['previous_response'].values
    y['previous_context'] = trial_data['previous_context'].values

    return y