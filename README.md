# Project Monkey: Behavioral and Neural Signatures of Cognitive Flexibility

This repository accompanies the analyses from our CCN 2025 abstract paper:  
**"Task-Relevant Information is Distributed Across the Cortex, but the Past is State-Dependent and Restricted to Frontal Regions."**
https://drive.google.com/file/d/1mIdg724qD1HNBhvAeyGnF9l8woQfKfEI/view?usp=sharing

We analyze behavioral and neural data from monkeys performing a context-dependent decision-making task to study the role of latent cognitive states in flexible decision-making. The repository is structured into two main components: **Behavioral** and **Neural** analysis.

---

## Repository Structure

```text
├── Behaviour/               # Behavioral modeling and results
│   ├── Figures/             # Behavioral figures
│   ├── notebooks/           # Jupyter notebooks for HMM-GLM and plotting
│   ├── results/             # Output of model fitting and metrics
│   └── scripts_beh/         # Scripts for model fitting and visualization
│
├── Neural/                  # Neural decoding
│   ├── Figures/             # Neural figures
│   ├── notebooks/           # Notebooks for decoding
│   ├── results/             # Output of decoding analyses
│   └── scripts_neural/      # Scripts for decoding and regression
│
├── Data/                    # Sample of processed behavioral/neural data
├── environment.yml          # Environmnet requirements for Python
└── README.md                # Project documentation
```
## Reproducibility and Analysis Structure

By running the notebooks in the `Behaviour/notebooks/` and `Neural/notebooks/` directories, you will be able to reproduce all key figures reported in the abstract submitted to CCN 2025.

Each analysis pipeline follows a modular structure:

- **Core functions** are defined in `scripts_beh/` (for behavioral analysis) and `scripts_neural/` (for neural analysis).
- **Results and computed metrics** are saved to the `results/` subdirectory within each main folder.
- **All final plots and panels** are exported to the `Figures/` directory.

This structure ensures a clear separation between code, output, and visualizations, making it easy to trace and reproduce each step of the analysis.

You will find detailed information about Behavioral and Neural analyses in their respective README files.

## Environment Setup

Before beginning, create a Conda environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate Project_monkey
```

We use version 0.0.1 of the Bayesian State Space Modeling (GLM-HMM) framework from Scott Linderman's lab to perform GLM-HMM inference.

Within the glmhmm  environment, install the forked version of the ssm package available here: https://github.com/lindermanlab/ssm. 

In order to install this version of ssm, follow the instructions provided there, namely:

```bash
cd ssm
pip install numpy cython
pip install -e .
```