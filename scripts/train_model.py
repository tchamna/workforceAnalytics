# scripts/train_model.py
from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
from sklearn.metrics import make_scorer, f1_score

DATA_PATH = Path("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = Path("../model.pkl")

# ---------------------------
# Load & preprocess
# ---------------------------
df = pd.read_csv(DATA_PATH)
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df.drop(["EmployeeNumber","EmployeeCount","StandardHours","Over18"], axis=1, errors="ignore", inplace=True)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = X.select_dtypes(include=[int,float]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# ---------------------------
# Models + parameter grids
# ---------------------------
models_and_grids = [
    (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"C":[0.01,0.1,1,10]}
    ),
    (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {"n_estimators":[200,500],"max_depth":[None,10,20]}
    ),
    (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators":[200,500],"learning_rate":[0.05,0.1],"max_depth":[3,5]}
    ),
    (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"n_estimators":[200,400],
         "max_depth":[3,5],
         "learning_rate":[0.05,0.1],
         "subsample":[0.8,1.0],
         "colsample_bytree":[0.8,1.0],
         "scale_pos_weight":[ (y_train==0).sum()/max((y_train==1).sum(),1) ]}
    )
]

if HAS_LGBM:
    models_and_grids.append(
        (
            LGBMClassifier(random_state=42, class_weight="balanced"),
            {"n_estimators":[200,400],
             "learning_rate":[0.05,0.1],
             "num_leaves":[31,63]}
        )
    )

scorer = make_scorer(f1_score, pos_label=1)
best_model = None
best_score = -1
best_name = ""

for model, grid in models_and_grids:
    print(f"🔍 Tuning {model.__class__.__name__}...")
    search = GridSearchCV(model, grid, scoring=scorer, cv=3, n_jobs=-1)
    search.fit(X_train_p, y_train)
    print(f"   Best {model.__class__.__name__}: {search.best_score_:.3f} F1")
    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_model = search.best_estimator_
        best_name = model.__class__.__name__

print(f"\n✅ Best model: {best_name} with F1={best_score:.3f}")

# ---------------------------
# Save
# ---------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        {"model": best_model,
         "preprocessor": preprocessor,
         "feature_names": num_cols
         + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))},
        f,
    )
print(f"Model saved to {MODEL_PATH}")
