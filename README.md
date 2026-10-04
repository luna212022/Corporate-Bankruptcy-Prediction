# Corporate Bankruptcy Prediction

**UE24CS352A - Machine Learning Mini-Project**  
**Team Members:**  
- Ritu Ravish - PES1UG24CS928
- Vennela Shakthi V P - PES1UG24CS525

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Team Work Distribution](#team-work-distribution)
4. [Dataset Description](#dataset-description)
5. [Repository Structure](#repository-structure)
6. [Installation & Setup](#installation--setup)
7. [How to Run the Project](#how-to-run-the-project)
8. [End-to-End Methodology](#end-to-end-methodology)
9. [Experimental Results & Benchmark Table](#experimental-results--benchmark-table)
10. [Key Technical Findings & Analysis](#key-technical-findings--analysis)
11. [Deliverables & Submission Checklist](#deliverables--submission-checklist)

---

## Project Overview
This project focuses on predicting corporate financial distress and bankruptcy using machine learning. Early bankruptcy detection is vital for banks, investors, suppliers, and regulatory bodies to identify failing companies before they default on their obligations.

Using the **Polish Companies Bankruptcy Dataset** from the UCI Machine Learning Repository, we build, evaluate, and compare multiple classical classification models to predict bankruptcy 3 years ahead of time.

---

## Problem Statement
Predicting corporate bankruptcy involves two primary technical challenges:
1. **Severe Class Imbalance:** In real economies, the vast majority of companies remain solvent. In our dataset, only **4.71%** of companies are bankrupt (approx. 1 bankrupt company for every 20 solvent ones).
2. **Asymmetric Cost of Errors:** In credit and lending, missing an impending bankruptcy (**False Negative**) results in massive financial write-offs. Conversely, temporarily flagging a healthy company for audit (**False Positive**) is only a minor operational cost. 

Therefore, standard **Accuracy is deceptive**, and models must be chosen based on **Recall (Bankrupt)**, **F1-Score (Bankrupt)**, and **ROC-AUC**.

---

## Team Work Distribution
The project was divided cleanly between two team members:

| Team Member | Project Phase | Core Responsibilities & Deliverables |
| :--- | :--- | :--- |
| **Ritu Ravish** | **Phase 1: Data Preparation & EDA** | • Ingested raw ARFF dataset (`data/3year.arff`)<br>• Performed Exploratory Data Analysis and correlation checks<br>• Conducted missing-value analysis and median imputation without data leakage<br>• Performed stratified 80/20 train-test split<br>• Created [`notebooks/bankruptcy_prediction.ipynb`](notebooks/bankruptcy_prediction.ipynb) and saved processed CSVs under `data/processed/` |
| **Vennela Shakthi V P** | **Phase 2: Modeling & Evaluation** | • Designed class-weight balancing strategy to address the ~20:1 imbalance<br>• Built leak-free scikit-learn pipelines with standard feature scaling<br>• Implemented 5 diverse ML classifiers (Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting)<br>• Evaluated all models on unseen test data across multiple metrics<br>• Created executed notebook [`notebooks/model_training.ipynb`](notebooks/model_training.ipynb), modular scripts in `src/`, visual results under `results/`, and the project write-up |

---

## Dataset Description
- **Source:** Polish Companies Bankruptcy Dataset (UCI Machine Learning Repository).
- **Forecasting Window:** **3-year horizon (`3year.arff`)**. This horizon provides the largest sample size and a practical 3-year advance warning window for risk management.
- **Total Records:** **10,503 companies**.
- **Number of Features:** **64 continuous financial ratios** (`Attr1` to `Attr64`) measuring liquidity, profitability, leverage, turnover, and cash flow.
- **Target Variable (`class`):**
  - **`0` (Solvent / Operating):** 10,008 companies (95.29%)
  - **`1` (Bankrupt):** 495 companies (4.71%)
- **Data Splits (Stratified 80/20):**
  - **Training Set (`X_train`, `y_train`):** 8,402 companies (8,006 Solvent, 396 Bankrupt)
  - **Testing Set (`X_test`, `y_test`):** 2,101 companies (2,002 Solvent, 99 Bankrupt)

---

## Repository Structure
```text
Corporate-Bankruptcy-Prediction/
├── README.md                           # Project documentation & setup instructions
├── requirements.txt                    # Python package dependencies
├── data/
│   ├── 1year.arff ... 5year.arff       # Raw Polish bankruptcy datasets across 5 horizons
│   └── processed/                      # Preprocessed & imputed train/test sets
│       ├── X_train.csv                 # 8,402 rows x 64 features
│       ├── X_test.csv                  # 2,101 rows x 64 features
│       ├── y_train.csv                 # 8,402 target labels
│       └── y_test.csv                  # 2,101 target labels
├── documents/                          # Guidelines, references, and report write-up
│   ├── Guidelines and Instructions_ML Mini Project Assignment_2026.pdf
│   ├── CS229_Project_Report.pdf        # External literature reference
│   └── Project_Report_Writeup.md       # Content for mandatory 2-page project summary
├── notebooks/
│   ├── bankruptcy_prediction.ipynb     # Phase 1: EDA & data preprocessing
│   └── model_training.ipynb            # Phase 2: Model training, evaluation & plots
├── results/
│   ├── class_distribution.png          # Target imbalance visualization
│   ├── correlation_heatmap.png         # Feature correlation heatmap
│   ├── confusion_matrices.png          # Grid of confusion matrices for all models
│   ├── roc_curves.png                  # Multi-model ROC curves with AUC scores
│   ├── model_comparison_bar.png        # Bar chart comparing key performance metrics
│   └── model_comparison.csv            # Exported quantitative metric table
└── src/                                # Modular, reusable Python source scripts
    ├── preprocessing.py                # Data loading, cleaning, and splitting module
    ├── model.py                        # Model configurations and factory pipelines
    └── evaluate.py                     # Training, evaluation, and plotting CLI script
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or 3.11 installed on your system.
- Git installed.

### 2. Clone the Repository & Checkout the Branch
```bash
git clone https://github.com/luna212022/Corporate-Bankruptcy-Prediction.git
cd Corporate-Bankruptcy-Prediction
git checkout ritu
```

### 3. Create and Activate a Virtual Environment
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

You can run the project in two different ways:

### Option A: Interactive Jupyter Notebooks (Recommended for Live Demo)
Start the notebook server:
```bash
jupyter notebook
```
1. **Phase 1 Notebook:** Open [`notebooks/bankruptcy_prediction.ipynb`](notebooks/bankruptcy_prediction.ipynb) to view the initial exploratory data analysis, missing-value imputation, and train-test split.
2. **Phase 2 Notebook:** Open [`notebooks/model_training.ipynb`](notebooks/model_training.ipynb) to view model setup, training, metric tables, confusion matrices, ROC curves, and key findings.

### Option B: Modular Python Command-Line Execution
You can also run the entire pipeline directly from the terminal using the modular scripts in `src/`:

```bash
# Step 1: Preprocess raw data and regenerate processed CSVs
python src/preprocessing.py

# Step 2: Verify configured models
python src/model.py

# Step 3: Train all models, evaluate on test data, and regenerate results
python src/evaluate.py
```

---

## End-to-End Methodology

```
┌────────────────────────────────────────────────────────┐
│            1. Raw Data Ingestion (ARFF)                │
│    Loaded 10,503 records x 64 features (3year.arff)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│        2. Stratified Train-Test Split (80 / 20)        │
│   Preserves exact 4.71% bankruptcy ratio in both sets  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│     3. Median Imputation (Strictly on Training Set)    │
│  Fitted on X_train only; transformed X_test (No Leak)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│        4. Model Pipelines & Class-Weight Balancing     │
│   Linear/SVM: StandardScaler inside Pipeline           │
│   Tree/Ensemble: Balanced Class Weights (20:1 penalty) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│       5. Comprehensive Evaluation on 2,101 Test Set    │
│  Accuracy, Precision, Recall, F1, ROC-AUC, Conf. Matrix│
└────────────────────────────────────────────────────────┘
```

1. **Handling Missing Values:** Raw data contained 9,888 missing entries. Because financial ratios have extreme outliers (e.g. division by near-zero equity), we used **Median Imputation** rather than Mean.
2. **Guaranteed Zero Data Leakage:** The imputer was fitted **only** on `X_train` (`fit_transform`) and applied to `X_test` (`transform`). For Logistic Regression and SVM, `StandardScaler` was wrapped inside `Pipeline` objects so scaling statistics are never computed on the test set.
3. **Handling Class Imbalance:** We utilized **cost-sensitive balancing (`class_weight='balanced'`)** to penalize misclassifications of the minority bankrupt class inversely to their ~20:1 frequency.

---

## Experimental Results & Benchmark Table

All models were evaluated on the exact same unseen test set of **2,101 companies (2,002 solvent, 99 bankrupt)** using `random_state=42`.

| Model Name | Accuracy | Precision | Recall (Bankrupt) | F1-Score | ROC-AUC | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9653** | **0.8095** | 0.3434 | **0.4823** | **0.9113** | 34 | 65 | 8 | 1994 |
| **Random Forest (Balanced)** | 0.9400 | 0.3846 | 0.4545 | 0.4167 | 0.8719 | 45 | 54 | 72 | 1930 |
| **Random Forest (Default)** | 0.9548 | 1.0000 | 0.0404 | 0.0777 | 0.8608 | 4 | 95 | 0 | 2002 |
| **Decision Tree (Balanced)** | 0.7373 | 0.1219 | **0.7374** | 0.2092 | 0.7893 | **73** | 26 | 526 | 1476 |
| **Logistic Regression (Balanced)** | 0.6592 | 0.0935 | 0.7172 | 0.1655 | 0.7419 | 71 | 28 | 688 | 1314 |
| **Logistic Regression (Default)** | 0.9500 | 0.1250 | 0.0101 | 0.0187 | 0.7126 | 1 | 98 | 7 | 1995 |
| **Support Vector Machine (SVC)** | 0.9529 | 0.0000 | 0.0000 | 0.0000 | 0.6353 | 0 | 99 | 0 | 2002 |

*All results recorded from actual runs and exported to [`results/model_comparison.csv`](results/model_comparison.csv).*

---

## Key Technical Findings & Analysis

### 1. The "Accuracy Trap" in Imbalanced Data
A critical observation in this project is that **accuracy alone is completely misleading**:
- The standard **Support Vector Machine (SVC)** achieved **95.29% accuracy**, but predicted that **zero** companies would go bankrupt ($TP=0, FN=99$). In some past literature, this model was mistakenly called the "best" based on accuracy, a gap we identified and addressed. In reality, it has a **0% Recall** and provides zero predictive value.
- Similarly, standard unweighted **Random Forest (Default)** achieved **95.48% accuracy**, but missed 95 out of 99 bankrupt companies ($Recall = 4.04\%$).

### 2. Why Class Weight Balancing Matters
- By setting `class_weight='balanced'`, tree models and linear models are forced to pay attention to bankrupt companies.
- In **Random Forest (Balanced)**, bankruptcy detection increased from 4 companies to **45 companies** (an **11.3x increase in recall**) while retaining an overall accuracy of **94.00%** and an **ROC-AUC of 0.8719**.

### 3. Best Model Selection
- **Overall Champion: Gradient Boosting Classifier**
  - **Highest ROC-AUC (0.9113):** Excellent separation between solvent and failing firms across all probability thresholds.
  - **Highest Precision (80.95%):** Only 8 false alarms across 2,002 solvent firms.
  - **Highest F1-Score (0.4823):** Optimal harmonic balance of precision and recall.
  - Best suited as the primary scoring engine for automated credit assessment.
- **Best High-Sensitivity Screening Models: Decision Tree & Logistic Regression (Balanced)**
  - Both models detect **over 71% to 73% of all bankrupt companies**, making them ideal for initial regulatory screening where false negatives cannot be tolerated.

---

## Deliverables & Submission Checklist
Aliged with the **UE24CS352A Mini-Project Guidelines**:

- [x] **Source Code in GitHub:** Hosted on private repository under branch `ritu`.
- [x] **README File:** Clear, comprehensive setup and execution instructions.
- [x] **Code Functionality & Modularity:** Tested notebooks and clean scripts in `src/`.
- [x] **Evaluation Visuals:** High-resolution confusion matrices, ROC curves, and metric charts saved under `results/`.
- [x] **Two-Page Write-Up Content:** Detailed markdown summary prepared in [`documents/Project_Report_Writeup.md`](documents/Project_Report_Writeup.md).
- [x] **Zero Data Leakage:** Preprocessing and scaling verified leak-free.
