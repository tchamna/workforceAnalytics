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

DATA_PATH = Path('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
MODEL_PATH = Path('model.pkl')

st.set_page_config(page_title='HR Attrition Dashboard', layout='wide')

_HEADER_STYLE = """
<style>
h1 {text-align: center;}
.stApp {background-color: #f7f9fb}
</style>
"""

st.markdown(_HEADER_STYLE, unsafe_allow_html=True)
st.title('HR Attrition Dashboard')

# Attempt to download model if MODEL_URL is set and model.pkl is missing
try:
    from utils.model_fetch import download_model_if_missing
except Exception:
    download_model_if_missing = None


@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)


@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    drop_cols = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
    df = df.drop(columns=drop_cols, errors='ignore')
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'Attrition']
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


@st.cache_data
def train_model(df: pd.DataFrame, random_state=42):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
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
    results = {
        'model': model,
        'scaler': scaler,
        'X_test': X_test_s,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_names': X.columns.tolist(),
        'X_train_for_shap': X_train_s,
    }
    return results


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    fi = getattr(model, 'feature_importances_', None)
    if fi is None:
        return None
    fi_series = pd.Series(fi, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(fi_series))))
    sns.barplot(x=fi_series.values, y=fi_series.index, ax=ax, palette='viridis')
    ax.set_xlabel('Feature importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {len(fi_series)} Feature Importances')
    fig.tight_layout()
    return fig


# Sidebar controls
with st.sidebar:
    st.header('Controls')
    uploaded = st.file_uploader('Upload CSV (optional)', type=['csv'])
    show_shap = st.checkbox('Show SHAP (advanced)', value=False)
    retrain = st.button('Retrain model')


if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif not DATA_PATH.exists():
    st.error(f'Data file not found at {DATA_PATH}. Place the CSV at this path or upload a file.')
    st.stop()
else:
    raw = load_data()

st.subheader('Dataset sample')
st.dataframe(raw.head(10))

df = preprocess(raw)

# Top-level KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Rows', f"{df.shape[0]:,}")
with col2:
    attr_rate = df['Attrition'].mean()
    st.metric('Attrition rate', f"{attr_rate:.2%}")
with col3:
    st.metric('Features', df.shape[1] - 1)

st.markdown('---')

# Class balance and distributions
left, right = st.columns([1, 2])
with left:
    st.subheader('Attrition balance')
    balance = df['Attrition'].value_counts().rename(index={0: 'No', 1: 'Yes'})
    st.bar_chart(balance)

with right:
    st.subheader('Selected feature distributions')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Attrition']
    if numeric_cols:
        sel = st.selectbox('Choose numeric feature', numeric_cols, index=0)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df[sel], kde=True, ax=ax, color='#2b8cbe')
        ax.set_title(sel)
        st.pyplot(fig)

# Use pretrained model if available, otherwise train locally (but indicate that
# the model is ephemeral unless the user runs scripts/train_model.py to save it.)
if not MODEL_PATH.exists() and download_model_if_missing is not None:
    # try to fetch model from MODEL_URL (if configured)
    try:
        download_model_if_missing(MODEL_PATH)
    except Exception:
        # swallow download errors here; we will handle absence below
        pass

if MODEL_PATH.exists():
    import pickle

    with open(MODEL_PATH, 'rb') as f:
        obj = pickle.load(f)
    model = obj['model']
    scaler = obj['scaler']
    feature_names = obj['features']

    # We need X_test and y_test for evaluation; create a test split from df
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    results = {
        'model': model,
        'scaler': scaler,
        'X_test': X_test_s,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_names': feature_names,
        'X_train_for_shap': None,
    }
else:
    if retrain:
        st.cache_data.clear()
    st.warning('No pretrained model found at model.pkl — training a temporary model. Run `python scripts/train_model.py` to create and persist a model.pkl for production deployments.')
    results = train_model(df)

st.subheader('Model evaluation')
roc = roc_auc_score(results['y_test'], results['y_prob'])
st.metric('ROC AUC', f"{roc:.3f}")
st.text('Classification report:')
st.text(classification_report(results['y_test'], results['y_pred']))

fig_cm = plot_confusion_matrix(results['y_test'], results['y_pred'])
st.pyplot(fig_cm)

fig_fi = plot_feature_importance(results['model'], results['feature_names'])
if fig_fi is not None:
    st.pyplot(fig_fi)

if show_shap:
    if shap is None:
        st.warning('SHAP not installed. Install `shap` to enable interactive explanations.')
    else:
        st.subheader('SHAP summary (sample)')
        try:
            # Use a small sample to keep it responsive
            explainer = shap.Explainer(results['model'], results['X_train_for_shap'])
            sample = results['X_test'][: min(200, results['X_test'].shape[0])]
            shap_values = explainer(sample)
            # Render SHAP summary into matplotlib figure and display
            plt.figure(figsize=(8, 4))
            shap.summary_plot(shap_values, sample, feature_names=results['feature_names'], show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.clf()
        except Exception as e:
            st.error(f'SHAP plotting failed: {e}')
        st.warning('SHAP not installed. Install `shap` to enable interactive explanations.')
