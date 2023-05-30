# README

This repository contains code for data analysis, feature selection, benchmarking, and interpretability using machine learning models. The code focuses on two experimental datasets (Exp1 and Exp2) and utilizes the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.

## Prerequisites

- Python 3.7 or higher
- Required Python packages: sklearn, xgboost, numpy, pandas, matplotlib, seaborn

## Getting Started

1. Clone the repository:

```bash
git clone <repository_url>
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```


3. Run the script:

```bash
python main.py
```


## File Structure

- main.py: The main script that executes the data analysis, feature selection, benchmarking, and interpretability code, including utility functions for data processing, reading the ADNI dataset, and other helper functions.
- Figures/: Directory to store generated figures and plots.

## Usage

1. Data Processing and Analysis:
   - The code performs exploratory data analysis (EDA) using violin plots (plot_distrib_violin function).
   - It reads and preprocesses the experimental datasets (Exp1 and Exp2) using the read_data_exp1 and read_data_exp2 functions.
   - EDA plots can be generated using the plot_distrib_violin function.

2. Feature Selection:
   - Feature selection is performed using two methods: feature_selection_xgboost (Method 1) and feature_selection_greedy (Method 2).
   - The selected features are used for further analysis and modeling.

3. Benchmarking:
   - The code includes benchmarking functionality to compare different machine learning models.
   - The compare function evaluates the performance of selected features with different models.

4. Interpretability:
   - The code provides interpretability techniques for machine learning models.
   - The single_tree function generates a decision tree plot and calculates feature importance using SHAP values.

## License

This project is licensed under the MIT License.

