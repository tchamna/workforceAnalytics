import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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
# Page configuration & custom styles
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
# Utilities
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

@st.cache_data
def preprocess(df: pd.DataFrame):
    """Basic cleaning and one-hot encoding"""
    df = df.copy()
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
    df = df.drop(columns=drop_cols, errors="ignore")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Attrition"]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

@st.cache_data
def train_model(df: pd.DataFrame, random_state=42):
    """Train a Gradient Boosting model and return key outputs"""
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_names": X.columns.tolist(),
        "X_train_for_shap": X_train_s,
    }

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    fi = getattr(model, "feature_importances_", None)
    if fi is None:
        return None
    fi_series = pd.Series(fi, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(fi_series))))
    sns.barplot(
        x=fi_series.values,
        y=fi_series.index,
        ax=ax,
        # palette=sns.color_palette("Reds", n_colors=len(fi_series)),
         palette=sns.color_palette("Blues", n_colors=len(fi_series))[::-1],  # reverse
    )
    ax.set_title(f"Top {len(fi_series)} Features", fontsize=13, color="#2b8cbe")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Sidebar controls (centered logo + title)
# # ---------------------------------------------------------------------
# with st.sidebar:
#     col_empty_left, col_center, col_empty_right = st.columns([1, 3, 1])
#     with col_center:
#         st.image("data/resulam_logo_egg.png", width=100)

#     st.markdown(
#         "<h2 style='text-align: center; margin-top: 0px;'>⚙️ Controls</h2>",
#         unsafe_allow_html=True
#     )

#     uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
#     # show_shap = st.toggle("Show SHAP (advanced)")
#     show_shap = st.toggle("Show SHAP (advanced)", value=True)

#     retrain = st.button("🔄 Retrain model")
#     st.markdown("---")
#     st.caption("Upload new data, toggle SHAP explanations, or retrain the model.")

#     # ---- Author info card ----
#     st.markdown(
#         """
#         <hr style="margin-top:20px; margin-bottom:10px;">
#         <div style="
#             text-align: center;
#             font-size: 13px;
#             line-height: 1.4;
#             color: #444;
#         ">
#             <b>Author:</b> Shck Tchamna<br>
#             Founder @ <a href="https://www.resulam.com" target="_blank" style="color:#2b8cbe;text-decoration:none;">Resulam</a><br>
#             Data Scientist | Developer<br>
#             <a href="mailto:tchamna@gmail.com" style="color:#2b8cbe;text-decoration:none;">tchamna@gmail.com</a>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# ---------------------------------------------------------------------
# Sidebar controls (true centered logo, works on mobile)
# ---------------------------------------------------------------------
with st.sidebar:
    # Create three columns and put the image in the middle
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.image("data/resulam_logo_egg.png", width=100)

    # Center the header
    st.markdown(
        "<h2 style='text-align: center; margin-top: 0;'>⚙️ Controls</h2>",
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    show_shap = st.toggle("Show SHAP (advanced)", value=True)
    retrain = st.button("🔄 Retrain model")
    st.markdown("---")
    st.caption("Upload new data, toggle SHAP explanations, or retrain the model.")

    # ---- Author info card ----
    st.markdown(
        """
        <hr style="margin-top:20px; margin-bottom:10px;">
        <div style="
            text-align: center;
            font-size: 13px;
            line-height: 1.4;
            color: #444;
        ">
            <b>Author:</b> Shck Tchamna<br>
            Founder @ <a href="https://www.resulam.com" target="_blank" style="color:#2b8cbe;text-decoration:none;">Resulam</a><br>
            Data Scientist | Developer<br>
            <a href="mailto:tchamna@gmail.com" style="color:#2b8cbe;text-decoration:none;">tchamna@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif not DATA_PATH.exists():
    st.error(f"Data file not found at {DATA_PATH}. Place the CSV there or upload a file.")
    st.stop()
else:
    raw = load_data()

st.subheader("📊 Dataset Sample")
st.dataframe(raw.head(5))

df = preprocess(raw)

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("👥 Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("🔄 Attrition rate", f"{df['Attrition'].mean():.2%}")
with col3:
    st.metric("🧩 Features", df.shape[1] - 1)

st.markdown("---")

# ---------------------------------------------------------------------
# Data distribution
# ---------------------------------------------------------------------
left, right = st.columns([1, 2])
with left:
    st.subheader("Attrition Balance")
    balance = df["Attrition"].value_counts().rename(index={0: "No", 1: "Yes"})
    st.bar_chart(balance)

with right:
    st.subheader("Selected Feature Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Attrition"]
    if numeric_cols:
        sel = st.selectbox("Choose numeric feature", numeric_cols, index=0)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df[sel], kde=True, ax=ax, color="#2b8cbe")
        ax.set_title(sel)
        st.pyplot(fig)

# ---------------------------------------------------------------------
# Model: load or train
# ---------------------------------------------------------------------
if retrain:
    st.cache_data.clear()

if MODEL_PATH.exists():
    import pickle

    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]
    scaler = obj["scaler"]
    feature_names = obj.get("feature_names") or obj.get("features")

    # Build test split to evaluate
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    results = {
        "model": model,
        "scaler": scaler,
        "X_test": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_names": feature_names,
        "X_train_for_shap": None,
    }
else:
    st.warning("No pretrained model found — training a temporary model. "
               "Run `python scripts/train_model.py` to persist a model.pkl for production.")
    results = train_model(df)


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# 🌟 Model evaluation (clean, no colors, no macro avg)
# ---------------------------------------------------------------------
from sklearn.metrics import classification_report

st.markdown("## 🧮 Model Evaluation")

# --- KPI Card for ROC AUC
roc_value = roc_auc_score(results["y_test"], results["y_prob"])
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

# --- Classification report
report_dict = classification_report(results["y_test"], results["y_pred"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)

# Drop macro avg (keep only 0,1,accuracy,weighted avg)
rows_to_keep = ["0", "1", "accuracy", "weighted avg"]
report_df = report_df.loc[[r for r in rows_to_keep if r in report_df.index]]

# Reorder columns nicely
report_df = report_df[["precision", "recall", "f1-score", "support"]]

# Rename rows to be more human-friendly
row_names = {"0": "Stayed (0)", "1": "Left (1)", "accuracy": "Accuracy", "weighted avg": "Weighted Avg"}
report_df.rename(index=row_names, inplace=True)

# Style clean: center numbers, bold header
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


st.markdown("---")

# ---- Dynamic explanation card ----
report_dict = classification_report(results["y_test"], results["y_pred"], output_dict=True)
accuracy = report_dict["accuracy"]
precision_0 = report_dict["0"]["precision"]
recall_0 = report_dict["0"]["recall"]
f1_0 = report_dict["0"]["f1-score"]
support_0 = report_dict["0"]["support"]

precision_1 = report_dict["1"]["precision"]
recall_1 = report_dict["1"]["recall"]
f1_1 = report_dict["1"]["f1-score"]
support_1 = report_dict["1"]["support"]

macro_precision = report_dict["macro avg"]["precision"]
macro_recall = report_dict["macro avg"]["recall"]
macro_f1 = report_dict["macro avg"]["f1-score"]

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

st.markdown(
    """
    <div style="
        background-color:#E6EEF5;
        padding:15px;
        border-radius:8px;
        margin-top:10px;
    ">
    <b>📊 How to read these metrics:</b><br><br>
    • <b>Precision</b> – Of all employees the model predicted as attrition, how many really left.<br>
    • <b>Recall</b> – Of all employees who truly left, how many the model correctly flagged.<br>
    • <b>F1-score</b> – Harmonic mean of precision and recall (balances the two).<br>
    • <b>Support</b> – Number of true samples of each class in the test set.<br><br>
    • <b>Accuracy</b> – Overall % of correct predictions.<br>
    • <b>Macro avg</b> – Simple average of precision/recall/F1 across classes (treats each class equally).<br>
    • <b>Weighted avg</b> – Average weighted by the number of samples per class (better overall summary when classes are imbalanced).<br>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------
if show_shap:
    if shap is None:
        st.warning("SHAP not installed. Install `shap` to enable interactive explanations.")
    else:
        st.subheader("🔍 SHAP Summary Plot")
        try:
            explainer = shap.Explainer(results["model"], results.get("X_train_for_shap"))
            sample = results["X_test"][: min(200, results["X_test"].shape[0])]
            shap_values = explainer(sample)
            plt.figure(figsize=(10, 5))
            shap.summary_plot(shap_values, sample, feature_names=results["feature_names"], show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"SHAP plotting failed: {e}")
