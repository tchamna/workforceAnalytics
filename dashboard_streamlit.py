import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

from sklearn.metrics import classification_report, roc_auc_score, f1_score, make_scorer

try:
    import shap
except Exception:
    shap = None

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = Path("model.pkl")

# ---------------------------------------------------------------------
# Page configuration & styles
# ---------------------------------------------------------------------
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide", page_icon="💼")
st.markdown(
    """
    <style>
        h1, h2, h3 {text-align: center;}
        .stApp {background-color: #f7f9fb;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="color:#2b8cbe;">💼 HR Attrition Dashboard</h1>
        <p style="font-size:16px; color:#555;">
            Interactive analytics & explainability for employee attrition
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

@st.cache_data
def clean_data(df: pd.DataFrame):
    df = df.copy()
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df.drop(["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"], axis=1, errors="ignore", inplace=True)
    return df

@st.cache_data
def train_best_model(df: pd.DataFrame):
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    scorer = make_scorer(f1_score, pos_label=1)

    models = [
        (LogisticRegression(max_iter=1000, class_weight="balanced"),
         {"C":[0.01,0.1,1,10]}),
        (RandomForestClassifier(class_weight="balanced", random_state=42),
         {"n_estimators":[200,500], "max_depth":[None,10,20]}),
        (GradientBoostingClassifier(random_state=42),
         {"n_estimators":[200,500], "learning_rate":[0.05,0.1], "max_depth":[3,5]}),
        (XGBClassifier(eval_metric="logloss", random_state=42),
         {"n_estimators":[200,400],
          "max_depth":[3,5],
          "learning_rate":[0.05,0.1],
          "subsample":[0.8,1.0],
          "colsample_bytree":[0.8,1.0],
          "scale_pos_weight":[(y_train==0).sum()/max((y_train==1).sum(),1)]})
    ]
    if HAS_LGBM:
        models.append(
            (LGBMClassifier(random_state=42, class_weight="balanced"),
             {"n_estimators":[200,400], "learning_rate":[0.05,0.1], "num_leaves":[31,63]})
        )

    best_model, best_score, best_name = None, -1, ""
    for model, grid in models:
        with st.spinner(f"Tuning {model.__class__.__name__}..."):
            search = GridSearchCV(model, grid, scoring=scorer, cv=3, n_jobs=-1)
            search.fit(X_train_p, y_train)
        if search.best_score_ > best_score:
            best_score, best_model, best_name = search.best_score_, search.best_estimator_, model.__class__.__name__

    st.success(f"✅ Best model: {best_name} (F1={best_score:.3f})")

    feature_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))

    import pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model,
                     "preprocessor": preprocessor,
                     "feature_names": feature_names},
                    f)

    return best_model, preprocessor, X_test_p, y_test, feature_names

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.image("data/resulam_logo_egg.png", width=100)
    st.markdown("<h2 style='text-align: center; margin-top: 0;'>⚙️ Controls</h2>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    show_shap = st.toggle("Show SHAP (advanced)", value=True)
    st.markdown("---")
    st.caption("Upload new data; the app trains a model if no model.pkl exists.")
    st.markdown(
        """
        <hr style="margin-top:20px; margin-bottom:10px;">
        <div style="text-align: center;font-size: 13px;line-height: 1.4;color: #444;">
            <b>Author:</b> Shck Tchamna<br>
            Founder @ <a href="https://www.resulam.com" target="_blank" style="color:#2b8cbe;text-decoration:none;">Resulam</a><br>
            Data Scientist | Developer<br>
            <a href="mailto:tchamna@gmail.com" style="color:#2b8cbe;text-decoration:none;">tchamna@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif not DATA_PATH.exists():
    st.error(f"Data file not found at {DATA_PATH}.")
    st.stop()
else:
    raw = load_data()

st.subheader("📊 Dataset Sample")
st.dataframe(raw.head(5))
df = clean_data(raw)

col1, col2, col3 = st.columns(3)
with col1: st.metric("👥 Rows", f"{df.shape[0]:,}")
with col2: st.metric("🔄 Attrition rate", f"{df['Attrition'].mean():.2%}")
with col3: st.metric("🧩 Features", df.shape[1] - 1)
st.markdown("---")

# ---------------------------------------------------------------------
# Model: load or train
# ---------------------------------------------------------------------
if MODEL_PATH.exists():
    import pickle
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]
    preprocessor = obj["preprocessor"]
    feature_names = obj["feature_names"]

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    _, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_test_p = preprocessor.transform(X_test)
else:
    st.info("No model.pkl found — training best model now…")
    model, preprocessor, X_test_p, y_test, feature_names = train_best_model(df)

y_prob = model.predict_proba(X_test_p)[:,1]
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
y_pred = (y_prob >= threshold).astype(int)

# ---------------------------------------------------------------------
# Evaluation (styled)
# ---------------------------------------------------------------------
st.markdown("## 🧮 Model Evaluation")

roc_value = roc_auc_score(y_test, y_prob)
col_left, col_right = st.columns([1, 3])
with col_left:
    st.markdown(
        f"""
        <div style="
            background-color:#ffffff;
            border:2px solid #2b8cbe;
            border-radius:10px;
            padding:20px;
            text-align:center;
        ">
            <h3 style="color:#2b8cbe;">ROC&nbsp;AUC</h3>
            <p style="font-size:40px; font-weight:700; margin:0;">{roc_value:.3f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)
rows_to_keep = ["0", "1", "accuracy", "weighted avg"]
report_df = report_df.loc[[r for r in rows_to_keep if r in report_df.index]]
report_df = report_df[["precision","recall","f1-score","support"]]
row_names = {"0": "Stayed (0)", "1": "Left (1)", "accuracy": "Accuracy", "weighted avg": "Weighted Avg"}
report_df.rename(index=row_names, inplace=True)

styled = (
    report_df.style
    .format(precision=2)
    .set_table_styles([
        {"selector": "thead th",
         "props": [("background-color", "#f0f2f6"), ("color", "#1E2B3C"),
                   ("font-weight", "bold"), ("text-align", "center")]},
        {"selector": "tbody td",
         "props": [("text-align", "center")]},
        {"selector": "tbody th",
         "props": [("text-align", "left"), ("font-weight", "bold")]},
    ])
)
with col_right:
    st.dataframe(styled, use_container_width=True)

# ---- Detailed explanation (your old style)
accuracy = report_dict["accuracy"]
precision_0 = report_dict["0"]["precision"]
recall_0 = report_dict["0"]["recall"]
f1_0 = report_dict["0"]["f1-score"]
support_0 = report_dict["0"]["support"]

precision_1 = report_dict["1"]["precision"]
recall_1 = report_dict["1"]["recall"]
f1_1 = report_dict["1"]["f1-score"]
support_1 = report_dict["1"]["support"]

macro_precision = report_dict.get("macro avg",{}).get("precision",0)
macro_recall = report_dict.get("macro avg",{}).get("recall",0)
macro_f1 = report_dict.get("macro avg",{}).get("f1-score",0)

weighted_precision = report_dict["weighted avg"]["precision"]
weighted_recall = report_dict["weighted avg"]["recall"]
weighted_f1 = report_dict["weighted avg"]["f1-score"]

st.markdown(
    f"""
    <div style="
        background-color:#E6EEF5;
        padding:15px;
        border-radius:8px;
        margin-top:10px;
        line-height:1.6;
    ">
    <b>📊 Understanding the results on your test set:</b><br><br>

    • <b>Accuracy:</b> The model correctly predicted attrition status for
      <b>{accuracy:.0%}</b> of all {int(support_0 + support_1)} employees.<br><br>

    • <b>Class 0 (Stayed):</b> Out of {int(support_0)} employees who stayed, the model
      correctly identified <b>{recall_0:.0%}</b> of them (recall) and when it predicted
      someone would stay, it was correct <b>{precision_0:.0%}</b> of the time (precision).
      Combined, this gives an F1-score of <b>{f1_0:.2f}</b>.<br><br>

    • <b>Class 1 (Left):</b> Out of {int(support_1)} employees who actually left, the model
      correctly caught <b>{recall_1:.0%}</b> of them (recall) and when it predicted someone
      would leave, it was correct <b>{precision_1:.0%}</b> of the time (precision).
      F1-score for leavers is <b>{f1_1:.2f}</b>.<br><br>

    • <b>Macro avg:</b> Averaging each class equally (regardless of how many samples there are)
      gives Precision <b>{macro_precision:.2f}</b>, Recall <b>{macro_recall:.2f}</b>, and
      F1-score <b>{macro_f1:.2f}</b>.<br>

    • <b>Weighted avg:</b> Taking into account that most employees stayed (class imbalance),
      the overall weighted Precision is <b>{weighted_precision:.2f}</b>, Recall
      <b>{weighted_recall:.2f}</b>, and F1-score <b>{weighted_f1:.2f}</b>.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------
if show_shap:
    if shap is None:
        st.warning("SHAP not installed. Install `shap` to enable interactive explanations.")
    else:
        st.subheader("🔍 Attrition SHAP Summary Plot")
        try:
            explainer = shap.Explainer(model, X_test_p)
            sample = X_test_p[:min(200, X_test_p.shape[0])]
            shap_values = explainer(sample)
            plt.figure(figsize=(10,5))
            shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"SHAP plotting failed: {e}")
