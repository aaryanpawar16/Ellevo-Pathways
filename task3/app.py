# streamlit_cover_type_app.py
# Streamlit app: Forest Cover Type Classification (Covertype)
# Behavior changed: "Train selected models" now directly runs both RandomForest and XGBoost (or HistGB fallback)
# Safeguards: sampling default=0.1, single-threaded training (n_jobs=1), clear progress and tracebacks

import streamlit as st
import zipfile
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import time
import traceback

# Try importing xgboost if available; otherwise fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

st.set_page_config(page_title="Forest Cover Type — Streamlit Classifier", layout="wide")

# ---------------- Helpers ----------------
@st.cache_data
def extract_csvs_from_zip(zip_bytes: bytes):
    csvs = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.filename.lower().endswith('.csv'):
                with z.open(info) as f:
                    csvs[info.filename] = f.read()
    return csvs

@st.cache_data
def load_dataframe_from_bytes(csv_bytes: bytes):
    try:
        return pd.read_csv(io.BytesIO(csv_bytes))
    except Exception:
        return pd.read_csv(io.BytesIO(csv_bytes), sep=';')

@st.cache_data
def infer_feature_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, cat_cols

@st.cache_data
def preprocess_dataframe(df: pd.DataFrame, target_col: str):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols, cat_cols = infer_feature_types(X)

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([('num', num_pipe, numeric_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')

    X_processed = preprocessor.fit_transform(X)

    feature_names = []
    feature_names.extend(numeric_cols)
    if len(cat_cols) > 0:
        try:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
        except Exception:
            feature_names.extend(cat_cols)

    return X_processed, y.values, preprocessor, feature_names

# ---------------- UI ----------------
st.title('🌲 Forest Cover Type — Streamlit Classifier')
st.write('Upload a ZIP with CSV(s). The app will preprocess and **directly run both** RandomForest and XGBoost (or fallback) when you click Train.')

uploaded = st.file_uploader('Upload ZIP (contains CSV)', type=['zip'])

if uploaded:
    csvs = extract_csvs_from_zip(uploaded.read())
    if not csvs:
        st.error('No CSV found inside ZIP.')
    else:
        selected = st.selectbox('CSV inside ZIP', list(csvs.keys()))
        df = load_dataframe_from_bytes(csvs[selected])
        st.subheader('Data preview')
        st.dataframe(df.head())

        cols = df.columns.tolist()
        suggested = [c for c in cols if c.lower() in ('cover_type', 'cover_type_id', 'target', 'class')]
        target_col = st.selectbox('Select target', cols, index=cols.index(suggested[0]) if suggested else 0)

        st.write('Shape:', df.shape)

        if st.checkbox('Show column types / missing'):
            st.dataframe(pd.DataFrame({'dtype': df.dtypes, 'missing': df.isna().sum()}))

        # Small dataset visualizations
        st.markdown('### Dataset visuals')
        try:
            fig, ax = plt.subplots(figsize=(6,3))
            df[target_col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Class distribution')
            st.pyplot(fig)
        except Exception:
            st.warning('Could not plot class distribution')

        if st.button('Preprocess & Train (runs both models)'):
            with st.spinner('Preprocessing...'):
                try:
                    X, y, preprocessor, feature_names = preprocess_dataframe(df, target_col)
                except Exception as e:
                    st.error(f'Preprocessing failed: {e}')
                    st.stop()

            # Fix label issues and rare classes by grouping rare into 'Other' if necessary
            y_series = pd.Series(y)
            counts = y_series.value_counts()
            rare = counts[counts < 2].index.tolist()
            if rare:
                st.warning(f'Grouping rare classes {rare} into "Other" to allow splitting')
                y_series = y_series.astype(object)
                y_series.loc[y_series.isin(rare)] = 'Other'
                y = y_series.values

            # Train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Sampling safeguard (default 10% to avoid OOM)
            sample_frac = st.slider('Train fraction (sampling)', 0.01, 1.0, 0.1)
            if sample_frac < 1.0:
                n = int(X_train.shape[0] * sample_frac)
                idx = np.random.RandomState(42).choice(X_train.shape[0], n, replace=False)
                X_train_use = X_train[idx]
                y_train_use = y_train[idx]
            else:
                X_train_use, y_train_use = X_train, y_train

            st.info(f'Training on {X_train_use.shape[0]} samples. Models will run sequentially (single-threaded).')

            results = {}
            progress = st.progress(0)
            step = 0

            # --- Random Forest ---
            try:
                step += 1
                progress.progress(int(100 * step / 3))
                rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=1, random_state=42)
                rf.fit(X_train_use, y_train_use)
                y_pred_rf = rf.predict(X_test)
                acc_rf = accuracy_score(y_test, y_pred_rf)
                results['RandomForest'] = {'model': rf, 'accuracy': acc_rf, 'y_pred': y_pred_rf}
                st.success(f'RandomForest done — accuracy: {acc_rf:.4f}')
            except Exception:
                st.error('RandomForest training failed:')
                st.text(traceback.format_exc())

            # --- XGBoost (or fallback) ---
            try:
                step += 1
                progress.progress(int(100 * step / 3))
                if HAS_XGBOOST:
                    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='mlogloss')
                    # use a small eval_set for early stopping
                    xgb_clf.fit(X_train_use, y_train_use, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
                    y_pred_xgb = xgb_clf.predict(X_test)
                    acc_xgb = accuracy_score(y_test, y_pred_xgb)
                    results['XGBoost'] = {'model': xgb_clf, 'accuracy': acc_xgb, 'y_pred': y_pred_xgb}
                    st.success(f'XGBoost done — accuracy: {acc_xgb:.4f}')
                else:
                    # fallback to HistGradientBoosting
                    hgb = HistGradientBoostingClassifier(random_state=42)
                    hgb.fit(X_train_use, y_train_use)
                    y_pred_hgb = hgb.predict(X_test)
                    acc_hgb = accuracy_score(y_test, y_pred_hgb)
                    results['HistGB'] = {'model': hgb, 'accuracy': acc_hgb, 'y_pred': y_pred_hgb}
                    st.success(f'HistGradientBoosting done — accuracy: {acc_hgb:.4f}')
            except Exception:
                st.error('XGBoost/HistGB training failed:')
                st.text(traceback.format_exc())

            progress.progress(100)
            time.sleep(0.3)
            progress.empty()

            # --- Evaluation visuals ---
            st.subheader('Evaluation')
            for name, res in results.items():
                st.write(f'### {name} — accuracy: {res["accuracy"]:.4f}')
                try:
                    st.text(classification_report(y_test, res['y_pred']))
                except Exception:
                    st.warning('Could not compute classification report')

                try:
                    cm = confusion_matrix(y_test, res['y_pred'])
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    ax.set_title(f'{name} — Confusion Matrix')
                    st.pyplot(fig)
                except Exception:
                    st.warning('Could not plot confusion matrix')

            # --- Feature importances (RF or fallback) ---
            st.subheader('Feature importance')
            for name, res in results.items():
                model = res['model']
                try:
                    if hasattr(model, 'feature_importances_'):
                        imp = model.feature_importances_
                    elif HAS_XGBOOST and hasattr(model, 'get_booster'):
                        raw = model.get_booster().get_score(importance_type='weight')
                        imp = np.zeros(len(feature_names))
                        for k, v in raw.items():
                            if k.startswith('f'):
                                idx = int(k[1:])
                                if idx < len(imp):
                                    imp[idx] = v
                    else:
                        st.write(f'No feature importance for {name}')
                        continue

                    fi = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False).head(25)
                    fig2, ax2 = plt.subplots(figsize=(6,4))
                    ax2.barh(fi['feature'][::-1], fi['importance'][::-1])
                    ax2.set_title(f'{name} — Top importances')
                    st.pyplot(fig2)
                except Exception:
                    st.warning(f'Could not compute feature importance for {name}')

            st.success('All done — models ran sequentially.')

# Footer
st.markdown('---')
st.caption('Notes: training large datasets can still be slow. Use the Train fraction slider to reduce sample size for quick runs.')
