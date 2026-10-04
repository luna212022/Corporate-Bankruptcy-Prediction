"""
Model definition and factory module for Corporate Bankruptcy Prediction.
Provides standardized, leak-free scikit-learn pipelines and models with
both default and cost-sensitive (class-weighted) configurations.
"""

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of all machine learning models configured for the project.
    Linear and distance-based models are wrapped in a Pipeline with StandardScaler
    to guarantee zero data leakage during training and testing.
    """
    models = {
        "Logistic Regression (Balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_state
            ))
        ]),
        "Logistic Regression (Default)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000, random_state=random_state
            ))
        ]),
        "Decision Tree (Balanced)": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=20,
            random_state=random_state
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Random Forest (Default)": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        ),
        "Support Vector Machine (SVC)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", C=1.0, random_state=random_state
            ))
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state
        )
    }
    return models


def train_model(model: Any, X_train, y_train):
    """Fits the given model or pipeline on the training data."""
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    configured_models = get_models()
    print(f"Successfully initialized {len(configured_models)} machine learning models:")
    for name in configured_models:
        print(f" - {name}")
