"""
train_model_pipeline.py

Train an HR attrition model with a full preprocessing pipeline and save it as model.pkl.
Run:
    python train_model_pipeline.py
"""

from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_PATH = Path("data") / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = Path("model.pkl")

# ---------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------
if not DATA_PATH.exists():
    raise SystemExit(f"Data file not found at {DATA_PATH}. Please place the CSV here.")

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Map target
if "Attrition" not in df.columns:
    raise SystemExit("CSV must contain an 'Attrition' column with Yes/No values.")
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Drop useless/leaky columns
drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
df = df.drop(columns=drop_cols, errors="ignore")

# Separate features/target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Identify column types
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ---------------------------------------------------------------------
# Preprocessing + model
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Fitting preprocessing + model...")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_proc, y_train)

# ---------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "feature_names": num_cols
            + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)),
        },
        f,
    )

print(f"Model and preprocessing pipeline saved to {MODEL_PATH}")
