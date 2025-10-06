#!/usr/bin/env python3
"""
Streamlit app: Student Score Prediction
Features:
 - Upload dataset (CSV) or use built-in sample
 - Data cleaning (dropna, encode categoricals)
 - EDA: scatter, heatmap, distributions
 - Train/test split controls
 - Linear Regression and Polynomial Regression (select degree)
 - Feature selection widget
 - Metrics: MSE, RMSE, MAE, R2
 - Plots: actual vs predicted, residuals
 - Download predictions

Requirements: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn
Run: streamlit run student_score_streamlit_app.py
"""

import io
import base64
from typing import Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="Student Score Prediction", layout="wide")

# ------------------------- Helpers -------------------------

def load_sample_data() -> pd.DataFrame:
    # Small synthetic sample that mimics a typical student performance dataset
    rng = np.random.RandomState(42)
    n = 200
    study = np.clip(rng.normal(5, 2, n), 0, 12)
    sleep = np.clip(rng.normal(7, 1, n), 4, 10)
    attendance = np.clip(rng.normal(85, 10, n), 40, 100)
    participation = np.clip(rng.poisson(3, n), 0, 10)
    # final score roughly correlated with study, attendance
    score = (3.5 * study) + (0.08 * attendance) + (0.5 * participation) + rng.normal(0, 5, n)
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        'StudyHours': study,
        'SleepHours': sleep,
        'Attendance': attendance,
        'Participation': participation,
        'ExamScore': score
    })
    return df


def preprocess(df: pd.DataFrame, drop_na=True) -> pd.DataFrame:
    df = df.copy()
    if drop_na:
        df = df.dropna()
    # If categorical columns exist, do one-hot encoding
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def train_models(X_train, X_test, y_train, y_test, model_type='linear', poly_degree=2, scale=False):
    results = {}

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results['model'] = model
        results['y_pred'] = y_pred
        results['X_test'] = X_test

    elif model_type == 'polynomial':
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_train_p = poly.fit_transform(X_train)
        X_test_p = poly.transform(X_test)
        if scale:
            scaler_p = StandardScaler()
            X_train_p = scaler_p.fit_transform(X_train_p)
            X_test_p = scaler_p.transform(X_test_p)
            results['scaler_poly'] = scaler_p
        model = LinearRegression()
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        results['model'] = model
        results['y_pred'] = y_pred
        results['poly'] = poly
        results['X_test'] = X_test_p

    # metrics
    mse = mean_squared_error(y_test, results['y_pred'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, results['y_pred'])
    r2 = r2_score(y_test, results['y_pred'])

    results['mse'] = mse
    results['rmse'] = rmse
    results['mae'] = mae
    results['r2'] = r2
    return results


def plot_scatter_with_line(df, x_col, y_col, model=None, poly=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    if model is not None and poly is None:
        # show predicted line for simple linear model based on x_col
        xs = np.linspace(df[x_col].min(), df[x_col].max(), 100).reshape(-1, 1)
        ys = model.predict(xs)
        ax.plot(xs, ys, color='red', linewidth=2)
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


# ------------------------- App Layout -------------------------

st.title("📘 Student Score Prediction — Streamlit")
st.markdown("Upload your student dataset (CSV) or use the sample dataset. The app supports linear and polynomial regression and basic EDA.")

# Sidebar: data input and options
with st.sidebar:
    st.header("Data & Model Settings")
    data_source = st.radio("Data source", options=['Sample dataset', 'Upload CSV'], index=0)
    uploaded_file = None
    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    st.markdown("---")
    test_size = st.slider("Test set size (fraction)", 0.05, 0.5, 0.2, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    st.markdown("---")
    model_type = st.selectbox("Model", ['linear', 'polynomial'])
    poly_degree = 2
    if model_type == 'polynomial':
        poly_degree = st.slider("Polynomial degree", 2, 5, 2)
    scale_features = st.checkbox("Scale features (StandardScaler)", value=False)
    st.markdown("---")
    st.write("Export")
    export_predictions = st.checkbox("Allow download of predictions", value=True)

# Load data
if data_source == 'Sample dataset' or uploaded_file is None:
    df = load_sample_data()
    st.info("Using built-in sample dataset — replace with your CSV by choosing 'Upload CSV' in the sidebar.")
else:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded CSV loaded successfully.")
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head())

# Basic cleaning options
with st.expander("Data cleaning & preprocessing options", expanded=False):
    drop_na = st.checkbox("Drop rows with missing values", value=True)
    st.write("Categorical columns (if any) will be one-hot encoded automatically.")

# Preprocess
df_clean = preprocess(df, drop_na=drop_na)

st.write(f"Dataset shape after preprocessing: {df_clean.shape}")

# Select target and features
all_columns = df_clean.columns.tolist()
if 'ExamScore' in all_columns:
    target_default = 'ExamScore'
else:
    target_default = all_columns[-1]

col1, col2 = st.columns([2, 1])
with col1:
    target_col = st.selectbox("Target column (what to predict)", options=all_columns, index=all_columns.index(target_default))
with col2:
    st.write("\n")
    st.write("\n")
    st.write("Tip: Choose a numeric target.")

feature_cols = [c for c in all_columns if c != target_col]
selected_features = st.multiselect("Select feature columns", options=feature_cols, default=feature_cols[:2])

if len(selected_features) == 0:
    st.warning("Select at least one feature to train the model.")
    st.stop()

X = df_clean[selected_features]
y = df_clean[target_col]

# Show correlation heatmap
with st.expander("Show correlation heatmap", expanded=False):
    corr = df_clean[[*selected_features, target_col]].corr()
    fig_h, ax_h = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_h)
    st.pyplot(fig_h)

# Simple EDA for first selected feature
with st.expander("Feature vs Target scatter (first selected feature)", expanded=True):
    f0 = selected_features[0]
    fig_sc = plot_scatter_with_line(df_clean, f0, target_col)
    st.pyplot(fig_sc)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

st.markdown("---")
st.subheader("Model training & evaluation")
st.write(f"Training samples: {len(X_train)} — Testing samples: {len(X_test)}")

# Train model
results = train_models(X_train.values, X_test.values, y_train.values, y_test.values,
                       model_type=model_type, poly_degree=poly_degree, scale=scale_features)

colm1, colm2 = st.columns(2)
with colm1:
    st.metric("R² (test)", f"{results['r2']:.4f}")
    st.metric("RMSE (test)", f"{results['rmse']:.4f}")
    st.metric("MAE (test)", f"{results['mae']:.4f}")
with colm2:
    st.write("Model details")
    st.write(results.get('model'))

# Actual vs Predicted plot
pred_df = pd.DataFrame({
    'Actual': y_test.values.flatten(),
    'Predicted': results['y_pred'].flatten()
}).reset_index(drop=True)

fig_ap, ax_ap = plt.subplots(figsize=(6, 4))
sns.scatterplot(x='Actual', y='Predicted', data=pred_df, ax=ax_ap)
ax_ap.plot([pred_df['Actual'].min(), pred_df['Actual'].max()], [pred_df['Actual'].min(), pred_df['Actual'].max()], '--', linewidth=1.5)
ax_ap.set_title('Actual vs Predicted')
st.pyplot(fig_ap)

# Residuals
residuals = pred_df['Actual'] - pred_df['Predicted']
fig_res, ax_res = plt.subplots(figsize=(6, 4))
sns.histplot(residuals, kde=True, ax=ax_res)
ax_res.set_title('Residuals distribution (Actual - Predicted)')
st.pyplot(fig_res)

# Download predictions
if export_predictions:
    download_df = X_test.reset_index(drop=True).copy()
    download_df[target_col + '_Actual'] = y_test.reset_index(drop=True)
    download_df[target_col + '_Predicted'] = results['y_pred']
    csv_bytes = to_csv_bytes(download_df)
    b64 = base64.b64encode(csv_bytes).decode()
    href = f"data:file/csv;base64,{b64}"
    st.markdown(f"[Download predictions as CSV]({href})")

# Show a preview table of actual vs predicted
with st.expander("Preview: actual vs predicted", expanded=True):
    st.dataframe(download_df.head(20))

st.markdown("---")
st.markdown("### Notes & next steps")
st.markdown(
    "- Try adding/removing features to see effect on R² and RMSE.\n"
    "- For polynomial models, beware of overfitting at high degrees — check train vs test metrics.\n"
    "- Consider k-fold cross-validation for more robust estimation.\n"
    "- If you have categorical inputs with many levels, consider embedding or target encoding for advanced workflows."
)

st.write("Made with ❤️ — ask me if you want additional features (cross-val, hyperparameter tuning, visualization customization).")
