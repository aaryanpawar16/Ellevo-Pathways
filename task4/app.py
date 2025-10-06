# streamlit_loan_approval_app.py
# Auto-training Streamlit App for Loan Approval Prediction
# This version preprocesses and trains models automatically on load (no buttons needed)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

st.set_page_config(page_title="Loan Approval Predictor (Auto-train)", layout="wide")
st.title("Loan Approval Prediction — Auto-train Streamlit App")

DEFAULT_FILENAME = "train_u6lujuX_CVtuZ9i (1).csv"

# --------------------- Helpers ---------------------
@st.cache_data
def load_data_from_file(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_data_from_upload(uploaded_file):
    return pd.read_csv(uploaded_file)

def preprocess(df, drop_id=True):
    df = df.copy()
    # common fixes for this dataset
    if drop_id and 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    # Strip whitespace in object columns
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    # Replace empty strings with NaN
    df.replace({'': np.nan, 'NA': np.nan, 'nan': np.nan}, inplace=True)

    # Fill missing values
    # categorical modes
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c != 'Loan_Status']
    for c in cat_cols:
        if df[c].isna().sum() > 0:
            df[c].fillna(df[c].mode()[0], inplace=True)

    # numeric columns
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for c in num_cols:
        if df[c].isna().sum() > 0:
            df[c].fillna(df[c].median(), inplace=True)

    # Special handling
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(float)

    # Encode categorical variables using LabelEncoder where appropriate
    encoders = {}
    for c in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    return df, encoders

# --------------------- UI: Data load ---------------------
st.sidebar.header("Load dataset")
use_default = st.sidebar.checkbox(f"Load default file: {DEFAULT_FILENAME}", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload CSV file", type=['csv'])

# Options
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.4, 0.2)
apply_smote = st.sidebar.checkbox("Apply SMOTE to training set (recommended)", value=True)
show_raw = st.sidebar.checkbox("Show raw data preview", value=False)

# Load data
df = None
if uploaded_file is not None:
    try:
        df = load_data_from_upload(uploaded_file)
        st.sidebar.success("Loaded uploaded file")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")

elif use_default:
    try:
        df = load_data_from_file(DEFAULT_FILENAME)
        st.sidebar.success(f"Loaded {DEFAULT_FILENAME}")
    except FileNotFoundError:
        st.sidebar.error(f"Default file not found in working directory: {DEFAULT_FILENAME}")

if df is None:
    st.info("Upload a CSV or enable the default file in the sidebar to proceed.")
    st.stop()

if show_raw:
    st.subheader("Raw data (first 10 rows)")
    st.dataframe(df.head(10))
st.write(f"Dataset shape: {df.shape}")

# --------------------- Preprocess & Train automatically ---------------------
with st.spinner("Preprocessing and training models — this may take a few seconds..."):
    try:
        df_proc, encoders = preprocess(df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    if 'Loan_Status' not in df_proc.columns:
        st.error("Target column 'Loan_Status' not found in dataset. Make sure your CSV contains 'Loan_Status'.")
        st.stop()

    X = df_proc.drop('Loan_Status', axis=1)
    y = df_proc['Loan_Status']

    # Show class balance
    st.subheader("Class distribution (target: Loan_Status)")
    counts = y.value_counts()
    st.bar_chart(counts)
    st.write(counts)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # SMOTE handling
    if apply_smote and not IMBLEARN_AVAILABLE:
        st.warning("imblearn not installed — proceeding without SMOTE. To enable SMOTE install imbalanced-learn (pip install imbalanced-learn)")
        apply_smote_effective = False
    else:
        apply_smote_effective = apply_smote

    if apply_smote_effective:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train.copy(), y_train.copy()

    # Feature scaling for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train_res)
    y_pred_log = log_reg.predict(X_test_scaled)
    y_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1] if hasattr(log_reg, 'predict_proba') else None

    # Train Decision Tree
    dtree = DecisionTreeClassifier(random_state=42, max_depth=6)
    dtree.fit(X_train_res, y_train_res)
    y_pred_tree = dtree.predict(X_test)
    y_proba_tree = dtree.predict_proba(X_test)[:, 1] if hasattr(dtree, 'predict_proba') else None

# --------------------- Display results ---------------------
st.subheader("Evaluation — Logistic Regression")
st.text(classification_report(y_test, y_pred_log, digits=4))
cm1 = confusion_matrix(y_test, y_pred_log)
fig1, ax1 = plt.subplots()
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp1.plot(ax=ax1)
st.pyplot(fig1)
if y_proba_log is not None:
    try:
        auc_log = roc_auc_score(y_test, y_proba_log)
        st.write(f"Logistic ROC AUC: {auc_log:.4f}")
        fpr, tpr, _ = roc_curve(y_test, y_proba_log)
        fig_roc1, axroc1 = plt.subplots()
        axroc1.plot(fpr, tpr)
        axroc1.set_title('Logistic ROC')
        axroc1.set_xlabel('FPR')
        axroc1.set_ylabel('TPR')
        st.pyplot(fig_roc1)
    except Exception:
        pass

st.subheader("Evaluation — Decision Tree")
st.text(classification_report(y_test, y_pred_tree, digits=4))
cm2 = confusion_matrix(y_test, y_pred_tree)
fig2, ax2 = plt.subplots()
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot(ax=ax2)
st.pyplot(fig2)
if y_proba_tree is not None:
    try:
        auc_tree = roc_auc_score(y_test, y_proba_tree)
        st.write(f"Decision Tree ROC AUC: {auc_tree:.4f}")
        fpr, tpr, _ = roc_curve(y_test, y_proba_tree)
        fig_roc2, axroc2 = plt.subplots()
        axroc2.plot(fpr, tpr)
        axroc2.set_title('Decision Tree ROC')
        axroc2.set_xlabel('FPR')
        axroc2.set_ylabel('TPR')
        st.pyplot(fig_roc2)
    except Exception:
        pass

# Feature importance
st.subheader("Decision Tree — Feature importances")
importances = pd.Series(dtree.feature_importances_, index=X.columns).sort_values(ascending=False)
st.write(importances.head(15))
fig_imp, ax_imp = plt.subplots()
importances.head(15).plot(kind='bar', ax=ax_imp)
ax_imp.set_ylabel('Importance')
st.pyplot(fig_imp)

# Prediction UI
st.subheader("Make a prediction on a test row")
row_idx = st.number_input("Test set row index (0-based)", min_value=0, max_value=max(0, X_test.shape[0]-1), value=0)
if st.button("Predict this row"):
    sample = X_test.iloc[[row_idx]]
    sample_scaled = scaler.transform(sample)
    pred_log = log_reg.predict(sample_scaled)[0]
    pred_tree = dtree.predict(sample)[0]
    st.write("Logistic predicted label:", int(pred_log))
    st.write("Decision Tree predicted label:", int(pred_tree))
    st.write("True label:", int(y_test.iloc[row_idx]))

st.info("Models were trained automatically when the app loaded. Use the sidebar to change test-size, toggle SMOTE, or upload a different dataset.")
