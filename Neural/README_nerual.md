# Neural Analysis – README

## Overview

This folder contains the decoding analyses of neural activity across multiple brain areas, aimed at predicting task-relevant behavioral variables. We use a time-resolved decoding approach with leave-one-out cross-validation to assess how well different cortical regions encode current and past trial information during a flexible decision-making task.

All decoding analyses use preprocessed spike data that has been temporally aligned to stimulus.

---

## Notebook

### `main_decoding_neural.ipynb`

This notebook performs time-resolved decoding of behavioral variables from neural activity using Support Vector Machines (SVMs) with leave-one-out cross-validation.

Key components:

- Target variables are prepared and converted to binary (e.g., color ≥ 0 → 1, < 0 → 0).
- Neural data is smoothed before decoding.
- Decoding is performed independently for each time bin.

#### Brain areas analyzed:

```python
areas = ['PFC', 'FEF', 'IT', 'MT', 'LIP', 'Parietal', 'V4']
```

Variables decoded:
- Current trial variables: color, direction, response, context
- Past trial variables: previous color, previous direction, previous response, previous context

The notebook decodes each variable across time using SVMs and plots decoding accuracy over time per area and variable.

Outputs:
- Decoded prediction results are stored in: `Neural/results/decoding_results/`
- Decoding performance figures are saved in: `Neural/Figures/`

