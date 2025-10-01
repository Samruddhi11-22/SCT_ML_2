import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("Mall_Customers.csv")

# Clean column names
df = df.rename(columns={
    'Annual Income (k$)': 'AnnualIncome',
    'Spending Score (1-100)': 'SpendingScore',
    'Genre': 'Gender'
})

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍 Mall Customer Segmentation Dashboard")
st.write("This app groups customers based on *Annual Income* and *Spending Score* using KMeans clustering.")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

# Sidebar for user input
st.sidebar.header("Clustering Settings")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 5)

# ---------------------------
# KMeans Clustering
# ---------------------------
X = df[['AnnualIncome', 'SpendingScore']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------------------
# Cluster Plot
# ---------------------------
st.subheader("📊 Cluster Visualization")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='tab10', data=df, s=80, ax=ax)
ax.set_title(f"Customer Clusters (k={k})")
st.pyplot(fig)

# ---------------------------
# Cluster Summary
# ---------------------------
st.subheader("📋 Cluster Summary")
summary = df.groupby('Cluster').agg(
    Count=('CustomerID','count'),
    AvgAge=('Age','mean'),
    AvgIncome=('AnnualIncome','mean'),
    AvgSpendingScore=('SpendingScore','mean')
).reset_index()

st.dataframe(summary)

# ---------------------------
# Download Results
# ---------------------------
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Clustered Data", csv, "mall_customers_with_clusters.csv", "text/csv")