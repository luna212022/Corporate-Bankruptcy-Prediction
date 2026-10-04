# Mini-Project Report: Corporate Bankruptcy Prediction

**Course:** UE24CS352A - Machine Learning  
**Project Title:** Corporate Bankruptcy Prediction  
**Team Members:** Team of Two (Teammate 1 & Teammate 2)  
**Repository:** [Corporate-Bankruptcy-Prediction (branch: `ritu`)](https://github.com/luna212022/Corporate-Bankruptcy-Prediction)  

---

## 1. Problem Statement
Corporate bankruptcy prediction is the task of forecasting whether a company will face insolvency or financial collapse based on its historical accounting and balance-sheet records. When a company goes bankrupt, it causes significant losses for lenders, suppliers, employees, and investors. Early warning systems help financial institutions take protective measures before a company defaults.

From a machine learning perspective, this is a **supervised binary classification problem on tabular financial data**. However, it poses two distinct practical challenges:
1. **Severe Class Imbalance:** In real-world economies, only a small percentage of companies go bankrupt in any given year. In our dataset, only ~4.7% of firms are bankrupt, while ~95.3% are solvent.
2. **Asymmetric Misclassification Costs:** In financial risk assessment, missing a company that is going bankrupt (**False Negative / Type II Error**) is far more damaging than flagging a healthy company for closer review (**False Positive / Type I Error**). Therefore, standard accuracy is a misleading metric, and models must be evaluated primarily on their ability to detect actual bankruptcies (**Recall, F1-Score, and ROC-AUC**).

---

## 2. Dataset Details
We used the **Polish Companies Bankruptcy Dataset** from the UCI Machine Learning Repository. The data was originally gathered from the Emerging Markets Information Service (EMIS) and contains financial statements of Polish manufacturing companies.

The dataset includes 5 separate files corresponding to different prediction horizons (from 1 to 5 years before bankruptcy). Following established benchmarks in literature, our project focuses on the **third-year forecasting horizon (`3year.arff`)**:
- **Sample Size:** 10,503 company records.
- **Forecasting Window:** Financial health evaluated 3 years in advance, giving managers and creditors sufficient time to act.
- **Features:** Exactly **64 continuous financial ratios** (`Attr1` to `Attr64`) covering liquidity (e.g., current assets to short-term liabilities), profitability (e.g., return on assets, gross profit), leverage, cash flow, and operating efficiency.
- **Target Variable (`class`):**
  - **`0` (Solvent / Non-bankrupt):** 10,008 companies (95.29%)
  - **`1` (Bankrupt):** 495 companies (4.71%)
  - **Imbalance Ratio:** Approximately **20:1**.
- **Missing Data:** The raw data contained 9,888 missing values across 44 financial ratios, with `Attr37` missing ~45% of its entries due to differing financial reporting requirements across company sizes.

---

## 3. Approach Taken to Solve the Problem
Our team divided the project into two collaborative phases:

### Phase 1: Data Preparation & Leakage Prevention (Teammate 1)
1. **Data Ingestion:** Extracted raw records from ARFF format and converted target labels from raw byte strings into binary integers (`0` and `1`).
2. **Stratified Train-Test Split:** Split the dataset into an **80% training set (8,402 firms)** and a **20% testing set (2,101 firms)** using `stratify=y` with a fixed `random_state=42`. Stratification ensures the exact ~4.7% bankruptcy proportion is preserved in both sets.
3. **Median Imputation Without Data Leakage:** Because financial ratios contain extreme positive and negative values (due to division by small equity or cash amounts), the median is much more reliable than the mean. The imputer was fitted **only** on the training set (`fit_transform`) and then applied to the test set (`transform`), preventing any test information from influencing data cleaning.

### Phase 2: Model Training, Imbalance Handling & Evaluation (Teammate 2)
1. **Handling Class Imbalance:** Instead of artificially creating synthetic data, we used **cost-sensitive learning (`class_weight='balanced'`)**. This assigns higher penalty weights to misclassifying the minority bankrupt class inversely proportional to their class frequency.
2. **Leakage-Free Feature Scaling:** For linear and distance-based classifiers (Logistic Regression and SVM), we used `StandardScaler` inside `sklearn.pipeline.Pipeline` objects, ensuring test data is never used to calculate scaling means or variances.
3. **Multi-Model Benchmarking:** We evaluated 5 diverse classical machine learning model families:
   - **Logistic Regression:** Linear decision boundary baseline (both default and balanced).
   - **Decision Tree:** Non-linear rule-based tree model (`max_depth=6`).
   - **Random Forest:** Bagging ensemble of 150 de-correlated trees (both default and balanced).
   - **Support Vector Machine (SVC):** Margin-based non-linear classifier with RBF kernel.
   - **Gradient Boosting Classifier:** Sequential boosting ensemble optimizing deviance loss.
4. **Comprehensive Evaluation Metrics:** We tracked Accuracy, Precision, **Recall**, **F1-Score**, **ROC-AUC**, and full confusion matrices.

---

## 4. Implementation Overview & Pipeline Architecture

```
[ Raw ARFF Data (3year.arff) ]
             │
             ▼
[ Target Encoding & Stratified Split (80% Train / 20% Test) ]
             │
             ▼
[ Median Imputation (Fitted on Train Only) ]
             │
             ▼
┌────────────────────────────────────────────────────────┐
│               Model Training & Pipelines               │
├──────────────────────────┬─────────────────────────────┤
│   Tree & Ensemble Models │   Pipeline Scaled Models    │
│   (No Scaling Needed)    │   (StandardScaler + Clf)    │
│  - Decision Tree         │  - Logistic Regression      │
│  - Random Forest         │  - Support Vector Machine   │
│  - Gradient Boosting     │                             │
└──────────────────────────┴─────────────────────────────┘
             │
             ▼
[ Comprehensive Evaluation on 2,101 Unseen Test Companies ]
             │
             ▼
[ Exported Deliverables: CSV Metrics, ROC Curves, Confusion Matrices ]
```

The codebase is organized into two execution paths:
- **Interactive Notebooks:** [`notebooks/bankruptcy_prediction.ipynb`](../notebooks/bankruptcy_prediction.ipynb) (EDA & Preprocessing) and [`notebooks/model_training.ipynb`](../notebooks/model_training.ipynb) (Model Training & Evaluation).
- **Modular Python Source Scripts:** [`src/preprocessing.py`](../src/preprocessing.py), [`src/model.py`](../src/model.py), and [`src/evaluate.py`](../src/evaluate.py) enabling full headless command-line execution.

---

## 5. Experimental Results & Performance Comparison

The models were evaluated on the held-out test set consisting of **2,101 companies (2,002 solvent, 99 bankrupt)**.

| Model Name | Accuracy | Precision | Recall (Bankrupt) | F1-Score | ROC-AUC | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9653** | **0.8095** | 0.3434 | **0.4823** | **0.9113** | 34 | 65 | 8 | 1994 |
| **Random Forest (Balanced)** | 0.9400 | 0.3846 | 0.4545 | 0.4167 | 0.8719 | 45 | 54 | 72 | 1930 |
| **Random Forest (Default)** | 0.9548 | 1.0000 | 0.0404 | 0.0777 | 0.8608 | 4 | 95 | 0 | 2002 |
| **Decision Tree (Balanced)** | 0.7373 | 0.1219 | **0.7374** | 0.2092 | 0.7893 | **73** | 26 | 526 | 1476 |
| **Logistic Regression (Balanced)** | 0.6592 | 0.0935 | 0.7172 | 0.1655 | 0.7419 | 71 | 28 | 688 | 1314 |
| **Logistic Regression (Default)** | 0.9500 | 0.1250 | 0.0101 | 0.0187 | 0.7126 | 1 | 98 | 7 | 1995 |
| **Support Vector Machine (SVC)** | 0.9529 | 0.0000 | 0.0000 | 0.0000 | 0.6353 | 0 | 99 | 0 | 2002 |

*All results obtained with fixed seeds (`random_state=42`) and saved to `results/model_comparison.csv`.*

---

## 6. Key Findings & Discussion

### 1. Uncovering the "Accuracy Trap"
A central finding of this project is that **overall accuracy is completely deceptive when working with imbalanced data**:
- Standard **Support Vector Machine (SVC)** achieved **95.29% accuracy**, but it predicted that **zero** companies would go bankrupt ($TP=0, FN=99$). In literature (such as the reference Stanford CS229 project), this model was mistakenly declared the "best" based on accuracy alone. In reality, it has a **0% Recall** and fails at the primary business goal.
- Standard **Random Forest (Default)** similarly achieved **95.48% accuracy**, but missed 95 out of 99 bankrupt companies ($Recall = 4.04\%$).

### 2. The Practical Benefit of Class Weight Balancing
- Applying `class_weight='balanced'` penalizes errors on the bankrupt class proportionally to the ~20:1 ratio.
- In **Random Forest (Balanced)**, bankruptcy detection jumped from 4 companies to **45 companies** (an **11.3x improvement in recall**) while maintaining an overall accuracy of **94.00%** and achieving an **ROC-AUC of 0.8719**.

### 3. Champion Model: Gradient Boosting
- **Gradient Boosting Classifier** delivered the best overall discriminative performance:
  - **Highest ROC-AUC (0.9113):** Excellent ability to distinguish between solvent and failing firms across all probability thresholds.
  - **Highest Precision (80.95%):** Out of all companies it flagged as bankrupt, more than 80% were truly bankrupt, producing only 8 false alarms across 2,002 healthy companies.
  - **Highest F1-Score (0.4823):** Best harmonic balance between identifying failing firms and avoiding false alarms.

### 4. Specialized Screening Models
- If an organization's primary objective is **risk screening** where no failing firm can be missed (even at the cost of more manual audits), **Decision Tree (Balanced)** and **Logistic Regression (Balanced)** are the most sensitive, successfully detecting **over 71% to 73% of all bankrupt firms**.

---

## 7. Conclusions
1. Predicting corporate bankruptcy requires careful evaluation beyond raw accuracy due to the ~20:1 class imbalance.
2. Leakage-free preprocessing (fitting median imputation and feature scaling strictly on training folds) is vital for realistic evaluation.
3. Cost-sensitive weighting (`class_weight='balanced'`) dramatically improves minority recall in tree ensembles without needing synthetic data generation.
4. **Gradient Boosting Classifier is the champion model** for automated bankruptcy scoring with an **ROC-AUC of 0.9113** and **80.95% Precision**, while **Random Forest (Balanced)** provides a reliable operational alternative with **45.45% Recall** at 94% accuracy.
5. All code, modular scripts, executed notebooks, and visualization artifacts are fully documented, reproducible, and ready for evaluation.
