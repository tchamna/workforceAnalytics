"""Train a model and save as model.pkl

Usage:
    python scripts/train_model.py
"""
from pathlib import Path
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = Path('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
MODEL_PATH = Path('model.pkl')

if not DATA_PATH.exists():
    raise SystemExit(f'Data file not found at {DATA_PATH}. Please place the CSV here.')

print('Loading data...')
df = pd.read_csv(DATA_PATH)
df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})
drop_cols = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
df = df.drop(columns=drop_cols, errors='ignore')
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != 'Attrition']
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

print('Training model...')
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_s, y_train)

print('Saving model to', MODEL_PATH)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'features': X.columns.tolist()}, f)

print('Done.')
