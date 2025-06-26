# Project Monkey: Behavioral and Neural Signatures of Cognitive Flexibility

This repository contains the code, data, and results for our CCN 2025 paper:  
**"Task-Relevant Information is Distributed Across the Cortex, but the Past is State-Dependent and Restricted to Frontal Regions."**
https://drive.google.com/file/d/1mIdg724qD1HNBhvAeyGnF9l8woQfKfEI/view?usp=sharing

We analyze both behavioral and neural data from monkeys performing a context-dependent decision-making task. Using HMM-GLM and neural decoding approaches, we uncover how internal states shape perception and choice.

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
├── Data/                    # Raw and processed behavioral/neural data
├── environment.yml          # Conda environment definition
└── README.md                # Project documentation
```
## Environment Setup

Before beginning, create a Conda environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate Project_monkey
```

We use version 0.0.1 of the Bayesian State Space Modeling (GLM-HMM) framework from Scott Linderman's lab to perform GLM-HMM inference.
In order to install this version of ssm: https://github.com/lindermanlab/ssm, follow the instructions provided there, namely: 
```bash
cd ssm
pip install numpy cython
pip install -e .
```