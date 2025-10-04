"""workforceAnalytics
Improved version: uses sklearn's ColumnTransformer for one-hot encoding.
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ---------------------------------------------------------------------
# Default data path
# ---------------------------------------------------------------------
DATA_PATH = Path("data") / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(Path(path))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Map Attrition Yes/No → 1/0
    - Drop obvious identifier or leaky columns.
    (Categorical encoding is handled later by ColumnTransformer.)
    """
    df = df.copy()

    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Drop non-predictive or leaky columns
    drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def split_and_encode(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer, List[str]]:
    """
    Split into train/test and encode features (numeric + one-hot for categoricals).

    Returns
    -------
    X_train : np.ndarray
    X_test  : np.ndarray
    y_train : pd.Series
    y_test  : pd.Series
    preprocessor : ColumnTransformer
    feature_names : list[str]
    """
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # Separate column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", 
                                  handle_unknown="ignore"), cat_cols),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fit on training set only
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # Extract feature names for interpretability
    feature_names = list(num_cols)
    if cat_cols:
        # get_feature_names_out exists in sklearn >=1.0
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
        feature_names = num_cols + cat_features.tolist()

    return X_train_enc, X_test_enc, y_train, y_test, preprocessor, feature_names


def train_model(
    X_train: np.ndarray, y_train: pd.Series, random_state: int = 42
) -> GradientBoostingClassifier:
    """Train and return a GradientBoostingClassifier."""
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    
    # Load & clean
    df = load_data()
    print("Shape:", df.shape)
    df = clean_data(df)

    # Encode & split
    X_train, X_test, y_train, y_test, preprocessor, feature_names = split_and_encode(df)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
