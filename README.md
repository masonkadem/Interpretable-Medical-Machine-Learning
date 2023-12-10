# README

This repository contains code for data analysis, feature selection, benchmarking, and interpretability using machine learning models. The code focuses on the first experiment and utilizes the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.

Abstract:

Despite their necessity in directing patient care worldwide, simple and accurate diagnostic tools for early Alzheimer’s disease (AD) do not exist. To support healthcare decision-making and planning, this research leverages large, multi-site accessible data and state-of-the-art supervised machine learning (XGBoost) to enable rapid, accurate, low-cost, accessible, non-invasive, interpretable, and early clinical evaluation of AD. Machine learning was employed to combine three key features: Everyday Cognition Questionnaire, Alzheimers Disease Assessment Scale, and Delayed Total Recall, achieving area under the receiver operating characteristic curves scores consistently above 97%. The selected features are important because they are non-invasive and easily collected. Low performance on delayed recall alone appears to distinguish most AD patients, consistent with the pathophysiology of AD where individuals having problems storing new information into long-term memory. Distinguishing this research from existing literature was the focus of enhancing the model's interpretability while maintaining performance of more complex and opaque models. The interpretable model enables understanding of the decision process, vital for clinical adoption of machine learning tools in AD evaluation. In summary, we present a methodology which identified accessible and noninvasive features, each with their absolute thresholds, together with a clinically operable decision route, to accurately and rapidly detect, differentiate, and diagnose Alzheimer's disease patients.

## Prerequisites

- Python 3.7 or higher
- Required Python packages: sklearn, xgboost, numpy, pandas, matplotlib, seaborn, shap, mlxtend

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
   - The code booosts interpretability for machine learning models.

## License

This project is licensed under the MIT License.

For the paper: 

Kadem, M. et al. (2023). XGBoost for Interpretable Alzheimer’s Decision Support. Association for the Advancement of Artificial Intelligence.


