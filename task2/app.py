import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import io, os

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Customer Segmentation using Clustering")

# ------------------ DATA LOADING ------------------

def load_sample_data(n=200) -> pd.DataFrame:
    """Load the Mall_Customers dataset or fallback to a small embedded sample."""
    sample_path = 'Mall_Customers.csv'
    if os.path.exists(sample_path):
        try:
            df = pd.read_csv(sample_path)
            # Normalize column names to match expected naming
            df.columns = df.columns.str.strip()
            df.rename(columns={'Spending Score (1-100)': 'SpendingScore(1-100)'}, inplace=True)
            return df
        except Exception:
            pass

    # Embedded sample dataset
    csv_text = """
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
11,Male,67,19,14
12,Female,35,19,99
13,Female,58,20,15
14,Female,24,20,77
15,Male,37,20,13
16,Male,22,20,79
17,Female,35,21,35
18,Male,20,21,66
19,Male,52,23,29
20,Male,35,23,98
"""
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = df.columns.str.strip()
    df.rename(columns={'Spending Score (1-100)': 'SpendingScore(1-100)'}, inplace=True)
    return df

# ------------------ SIDEBAR ------------------

st.sidebar.header("Data & Settings")
uploaded = st.sidebar.file_uploader("Upload Mall_Customers.csv", type=['csv'])

def make_unique_columns(cols):
    """Return a list of column names where duplicates are made unique by appending _1, _2, ..."""
    counts = {}
    new_cols = []
    for c in cols:
        if c in counts:
            counts[c] += 1
            new_cols.append(f"{c}_{counts[c]}")
        else:
            counts[c] = 0
            new_cols.append(c)
    return new_cols

if uploaded is not None:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()
    # normalize common column names
    if 'Spending Score (1-100)' in df.columns:
        df.rename(columns={'Spending Score (1-100)': 'SpendingScore(1-100)'}, inplace=True)
    # make duplicate column names unique (e.g., if dataset has repeated headers)
    if df.columns.duplicated().any():
        df.columns = make_unique_columns(list(df.columns))
else:
    df = load_sample_data()

st.write("### Dataset Preview", df.head())

# ------------------ DATA CLEANING ------------------

df = df.dropna()

# Select numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'CustomerID' in num_cols:
    num_cols.remove('CustomerID')

st.sidebar.write("Numeric Columns:", num_cols)
features = st.sidebar.multiselect("Select features for clustering", num_cols, default=['Annual Income (k$)', 'SpendingScore(1-100)'])

X = df[features]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ K-MEANS ------------------

st.subheader("🔹 K-Means Clustering")
max_k = st.sidebar.slider("Max K for Elbow Method", 2, 10, 6)

inertia = []
for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(2, max_k + 1), inertia, marker='o')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method')
st.pyplot(fig)

optimal_k = st.sidebar.number_input("Select optimal K", min_value=2, max_value=max_k, value=5)
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster_KMeans'] = kmeans_final.fit_predict(X_scaled)

st.write(f"### K-Means Cluster Centers ({optimal_k} clusters)")
st.dataframe(pd.DataFrame(kmeans_final.cluster_centers_, columns=features))

fig_km = px.scatter(df, x=features[0], y=features[1], color=df['Cluster_KMeans'].astype(str), title="K-Means Clusters", color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig_km, use_container_width=True)

avg_spend = df.groupby('Cluster_KMeans')['SpendingScore(1-100)'].mean()
st.write("### 🧾 Average Spending per Cluster")
st.bar_chart(avg_spend)

# ------------------ DBSCAN ------------------

st.subheader("🔸 DBSCAN Clustering")
eps_val = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 1.5)
min_samples_val = st.sidebar.slider("min_samples", 2, 10, 5)

dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
db_clusters = dbscan.fit_predict(X_scaled)
df['Cluster_DBSCAN'] = db_clusters

fig_db = px.scatter(df, x=features[0], y=features[1], color=df['Cluster_DBSCAN'].astype(str), title="DBSCAN Clusters", color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig_db, use_container_width=True)

# ------------------ SUMMARY ------------------

st.write("### Cluster Summary (K-Means)")
# Ensure we don't select duplicate columns (features may already include SpendingScore)
selection_cols = []
for c in features + ['SpendingScore(1-100)']:
    if c not in selection_cols and c in df.columns:
        selection_cols.append(c)

# If no valid selection columns, fall back to features
if not selection_cols:
    selection_cols = [c for c in features if c in df.columns]

summary = df.groupby('Cluster_KMeans')[selection_cols].mean().round(2)
# Make sure summary has unique column names (pyarrow/streamlit requires this)
if summary.columns.duplicated().any():
    # reuse make_unique_columns if available, otherwise append suffixes
    try:
        summary.columns = make_unique_columns(list(summary.columns))
    except Exception:
        counts = {}
        new_cols = []
        for c in summary.columns:
            counts[c] = counts.get(c, 0) + 1
            if counts[c] > 1:
                new_cols.append(f"{c}_{counts[c]-1}")
            else:
                new_cols.append(c)
        summary.columns = new_cols

st.dataframe(summary)

st.success("✅ Analysis complete! Adjust sidebar settings to explore different clustering outcomes.")
