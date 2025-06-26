import pickle

# preprocessing HMM-GLM
def get_congruency_label(row):
    """
    Determines a trial's congruency label based on the signs of the 'color' and 'direction' features.

    Returns:
        'LL' if both color and direction are ≥ 0
        'RR' if both are < 0
        'LR' if color ≥ 0 and direction < 0
        'RL' if color < 0 and direction ≥ 0
    """
    if row['color'] >= 0 and row['direction'] >= 0:
        return 'LL'
    elif row['color'] < 0 and row['direction'] < 0:
        return 'RR'
    elif row['color'] >= 0:
        return 'LR'
    else:
        return 'RL'


def preprocess_behavioral_session(df):
    """
    Preprocesses a behavioral DataFrame for GLM-HMM fitting:
    - Adds history variables (previous color, motion, response, context)
    - Computes distances between consecutive stimuli
    - Labels congruency type
    - Filters out bad trials
    """
    df = df.copy()
    df['intercept'] = 1

    df['p_color'] = df['color'].shift(1)
    df['p_direction'] = df['direction'].shift(1)
    df['prev_response'] = df['chosenResponse'].shift(1)

    df['context'] = df['rule']
    df['prob_color'] = (df['context'] == 1).astype(int)
    df['prob_motion'] = (df['context'] == 0).astype(int)
    df['prev_context'] = df['context'].shift(1)

    df['dist_prev_current_color'] = (df['color'] - df['p_color']).abs()
    df['dist_prev_current_motion'] = (df['direction'] - df['p_direction']).abs()

    df['correctness_stim_type'] = df.apply(get_congruency_label, axis=1)

    df.fillna(0, inplace=True)

    return df[df['badTimingTrials'] == 0].reset_index(drop=True)


def format_glmhmm_inputs(data, feature_names):
    """
    Converts a list of preprocessed session DataFrames into GLM-HMM input format.

    Args:
        data: list of session DataFrames

    Returns:
        inpts: list of numpy arrays with input features (one per session)
        choices: list of numpy arrays with binary choices (one per session)
    """
    inpts, choices = [], []
    for df in data:

        X = -df[feature_names].to_numpy()
        y = df['chosenResponse'].astype(int).to_numpy().reshape(-1, 1)
        inpts.append(X)
        choices.append(y)
    return inpts, choices

def load_and_prepare_glmhmm_inputs(filepath, feature_names, trial_key="trial"):
    """
    Loads behavioral data, preprocesses each session, and formats inputs for GLM-HMM fitting.

    Args:
        filepath (str): Path to the pickle file containing the data.
        trial_key (str): Key to access trial-level DataFrames inside the dictionary. Default is 'trial'.

    Returns:
        inpts (list of np.ndarray): List of input features per session.
        choices (list of np.ndarray): List of binary choice arrays per session.
        preprocessed (list of pd.DataFrame): List of preprocessed DataFrames (one per session).
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    # Handle both full-dict and list-of-DataFrame cases
    if isinstance(data, dict) and trial_key in data:
        data = data[trial_key]
    elif not isinstance(data, list):
        raise ValueError("Unsupported data format. Expected dict with 'trial' key or list of DataFrames.")

    preprocessed_data = [preprocess_behavioral_session(session_df) for session_df in data]
    inpts, choices = format_glmhmm_inputs(preprocessed_data, feature_names)
    return inpts, choices, preprocessed_data