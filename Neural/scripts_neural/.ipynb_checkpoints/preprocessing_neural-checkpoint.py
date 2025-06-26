
import numpy as np



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

def create_y(trial_data):
    # Use existing cleaned columns
    y = {}

    # Current trial variables
    y['direction'] = trial_data['direction'].values
    y['color'] = trial_data['color'].values
    y['response'] = trial_data['chosenResponse'].values
    y['context_response'] = trial_data['rule'].values

    # Previous trial variables (already created during preprocessing)
    y['previous_direction'] = trial_data['previous_direction'].values
    y['previous_color'] = trial_data['previous_color'].values

    # Previous response
    if 'previous_response' not in trial_data.columns:
        trial_data['previous_response'] = trial_data['chosenResponse'].shift(1)
    prev_response = trial_data['previous_response'].values

    # Previous context
    if 'previous_context_response' not in trial_data.columns:
        trial_data['previous_context_response'] = trial_data['rule'].shift(1)
    prev_context = trial_data['previous_context_response'].values

    # === FIX NaNs ===
    # Replace NaNs with a dummy label (e.g., -1), so sklearn won't crash
    for key in ['previous_direction', 'previous_color', 'previous_response', 'previous_context_response']:
        if key in trial_data.columns:
            trial_data[key] = trial_data[key].fillna(0)  # replace NaNs with 0

    y['previous_response'] = trial_data['previous_response'].values
    y['previous_context_response'] = trial_data['previous_context_response'].values

    return y