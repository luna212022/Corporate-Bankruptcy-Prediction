"""
Pre-processing module for Corporate Bankruptcy Prediction.
Handles raw ARFF loading, target label encoding, train-test splitting,
and median missing-value imputation without data leakage.
"""

import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def load_arff_data(arff_path: str) -> pd.DataFrame:
    """Load ARFF file and convert byte-string target to integers (0 and 1)."""
    if not os.path.exists(arff_path):
        raise FileNotFoundError(f"File not found: {arff_path}")
    
    data, _ = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    
    # Extract digit from byte-string/object class column (e.g. b'0' -> 0, b'1' -> 1)
    df["class"] = df["class"].astype(str).str.extract(r"(\d)").astype(int)
    return df


def preprocess_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Split dataset into train and test sets, then perform median imputation.
    Imputer is fitted strictly on X_train to prevent data leakage.
    """
    X = df.drop(columns=["class"])
    y = df["class"]

    # Stratified split to preserve the minority bankruptcy class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Impute missing values using median (robust against extreme financial outliers)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_imp, X_test_imp, y_train, y_test


def save_processed_datasets(X_train, X_test, y_train, y_test, output_dir: str):
    """Save processed train and test sets to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Processed datasets successfully saved to: {output_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_file = os.path.join(base_dir, "data", "3year.arff")
    out_dir = os.path.join(base_dir, "data", "processed")

    print(f"Loading raw data from: {raw_file}")
    df_raw = load_arff_data(raw_file)
    print(f"Loaded dataset shape: {df_raw.shape}")

    X_tr, X_te, y_tr, y_te = preprocess_data(df_raw)
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print(f"Bankrupt cases in train: {y_tr.sum()}, in test: {y_te.sum()}")

    save_processed_datasets(X_tr, X_te, y_tr, y_te, out_dir)
