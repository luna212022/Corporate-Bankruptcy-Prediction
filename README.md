# UE24CS352A Machine Learning Mini-Project: Corporate Bankruptcy Prediction

## Project Overview
This repository contains the machine learning mini-project for **Corporate Bankruptcy Prediction** using financial indicators from the Polish Companies Bankruptcy dataset. 

Corporate financial distress prediction is critical for lenders, investors, rating agencies, and financial institutions to mitigate economic contagion and insolvency risk. The goal of this project is to develop, evaluate, and benchmark classical machine learning models to identify firms at risk of bankruptcy within a 3-year forecasting horizon.

---

## Team Contributions & Work Split
| Team Member | Phase & Scope | Deliverables |
| :--- | :--- | :--- |
| **Teammate 1** | **Phase 1: EDA & Preprocessing** | • Raw ARFF data ingestion (`data/3year.arff`)<br>• Exploratory Data Analysis & correlation analysis<br>• Missing value analysis & median imputation (no leakage)<br>• Stratified train/test splitting (80/20)<br>• Generated `data/processed/*.csv` and EDA figures (`results/`) |
| **Teammate 2** | **Phase 2: Modeling & Evaluation** | • Imbalance handling strategy (`class_weight='balanced'`)<br>• Scikit-learn pipelines with zero-leakage feature scaling<br>• Implemented 5 diverse ML classifiers (Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting)<br>• Comprehensive evaluation (Recall, F1-Score, ROC-AUC, Confusion Matrices)<br>• Generated `notebooks/model_training.ipynb` and evaluation artifacts |

---

## Dataset Description
- **Dataset:** UCI Polish Companies Bankruptcy Dataset
- **Selected Horizon:** `3year.arff` (Predicting bankruptcy 3 years ahead)
- **Total Records:** 10,503 firms
- **Number of Features:** 64 financial ratios (`Attr1` to `Attr64`) covering liquidity, profitability, leverage, turnover, and cash flow metrics.
- **Target Variable:** `class` (Binary classification: `0` = Solvent / Operating, `1` = Bankrupt)
- **Class Distribution:**
  - **Solvent (`0`):** 10,008 firms (95.29%)
  - **Bankrupt (`1`):** 495 firms (4.71%)
  - **Imbalance Ratio:** ~20:1

---

## Repository Structure
```text
Corporate-Bankruptcy-Prediction/
├── README.md                      # Complete project documentation & execution guide
├── requirements.txt               # Project dependencies
├── data/
│   ├── 1year.arff ... 5year.arff  # Raw Polish bankruptcy datasets across 5 horizons
│   └── processed/                 # Cleaned, imputed & stratified train/test sets
│       ├── X_train.csv            # 8,402 rows x 64 features
│       ├── X_test.csv             # 2,101 rows x 64 features
│       ├── y_train.csv            # 8,402 target labels
│       └── y_test.csv             # 2,101 target labels
├── documents/                     # Course guidelines and literature references
│   ├── Guidelines and Instructions_ML Mini Project Assignment_2026.pdf
│   └── CS229_Project_Report.pdf
├── notebooks/
│   ├── bankruptcy_prediction.ipynb # Phase 1: EDA, Imputation & Splitting
│   └── model_training.ipynb        # Phase 2: Model Training, Evaluation & Benchmarking
└── results/
    ├── class_distribution.png     # Target imbalance plot
    ├── correlation_heatmap.png    # Feature correlation heatmap
    ├── confusion_matrices.png     # Side-by-side confusion matrices for all models
    ├── roc_curves.png             # Multi-model ROC curves with AUC annotations
    ├── model_comparison_bar.png   # Metric comparison bar chart
    └── model_comparison.csv       # Exported quantitative comparison table
```

---

## Setup & Installation Instructions

### 1. Prerequisites
- Python 3.10 or 3.11 installed.
- Git installed.

### 2. Clone Repository & Checkout Branch
```bash
git clone https://github.com/luna212022/Corporate-Bankruptcy-Prediction.git
cd Corporate-Bankruptcy-Prediction
git checkout ritu
```

### 3. Create & Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Option A: Interactive Jupyter Notebooks
Launch Jupyter Lab or Notebook:
```bash
jupyter notebook
```
1. Open and run [`notebooks/bankruptcy_prediction.ipynb`](notebooks/bankruptcy_prediction.ipynb) to inspect EDA and data preprocessing.
2. Open and run [`notebooks/model_training.ipynb`](notebooks/model_training.ipynb) to execute model training, evaluation, plot generation, and metric exports.

### Option B: Command-Line Headless Execution
To re-execute the entire model training notebook programmatically and regenerate all result artifacts:
```bash
python -m nbconvert --to notebook --execute notebooks/model_training.ipynb --inplace
```

---

## Model Evaluation Results

Models were evaluated on an unseen held-out test set of **2,101 firms** (2,002 Solvent, 99 Bankrupt).

| Model | Accuracy | Precision (Bankrupt) | Recall (Bankrupt) | F1-Score (Bankrupt) | ROC-AUC | True Positives (TP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9653** | **0.8095** | 0.3434 | **0.4823** | **0.9113** | 34 | 65 |
| **Random Forest (Balanced)** | 0.9400 | 0.3846 | 0.4545 | 0.4167 | 0.8719 | 45 | 54 |
| **Random Forest (Default)** | 0.9548 | 1.0000 | 0.0404 | 0.0777 | 0.8608 | 4 | 95 |
| **Decision Tree (Balanced)** | 0.7373 | 0.1219 | **0.7374** | 0.2092 | 0.7893 | **73** | 26 |
| **Logistic Regression (Balanced)** | 0.6592 | 0.0935 | 0.7172 | 0.1655 | 0.7419 | 71 | 28 |
| **Logistic Regression (Default)** | 0.9500 | 0.1250 | 0.0101 | 0.0187 | 0.7126 | 1 | 98 |
| **Support Vector Machine (SVC)** | 0.9529 | 0.0000 | 0.0000 | 0.0000 | 0.6353 | 0 | 99 |

*All results recorded from actual runs with `random_state=42`.*

---

## Key Technical Insights & Findings

### 1. The Accuracy Paradox on Imbalanced Financial Data
- In the reference literature (e.g. Stanford CS229 project report), models like unweighted **SVM** were reported as "best" based on **94.7% accuracy**.
- However, as reproduced above, standard unweighted SVM simply predicts the majority class (`0`) for every test instance. It achieves 95.29% accuracy solely because 95.29% of test firms are solvent, but it exhibits **0% Recall** (0 out of 99 bankruptcies detected). In financial risk management, this represents a 100% failure rate on insolvent companies.
- Similarly, default Random Forest misses 95 out of 99 bankrupt companies despite a misleading 95.48% accuracy.

### 2. The Power of Balanced Weighting
- Enabling `class_weight='balanced'` penalizes misclassification of bankrupt firms proportionally to the 20:1 inverse frequency:
  - **Random Forest (Balanced)** boosts bankruptcy detection by **over 11x** (capturing 45 bankrupt firms vs. 4 in default) while maintaining 94% overall accuracy and achieving an **AUC of 0.8719**.
  - **Decision Tree (Balanced)** and **Logistic Regression (Balanced)** capture over **70% of all bankruptcies** (Recall > 71%), suitable for high-sensitivity initial screening where false negatives carry extreme financial penalties.

### 3. Champion Model: Gradient Boosting
- **Gradient Boosting Classifier** achieves the highest overall discriminative performance:
  - **ROC-AUC:** **0.9113** (Outstanding discrimination across thresholds)
  - **Precision:** **80.95%** (Only 8 false alarms across 2,002 solvent firms)
  - **F1-Score:** **0.4823** (Best harmonic mean of precision and recall)
  - Gradient Boosting constructs an ensemble that captures complex, non-linear interactions among financial ratios without requiring synthetic data augmentation.
