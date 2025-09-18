import streamlit as st
from data_utils import wrangle, create_rfm
from rfm_analysis import cluster_rfm
from tree_models import tree_interpret
from anomaly import anomaly_detection

st.title("Customer Segmentation Dashboard")

file = st.file_uploader("Upload Excel (online_retail.xlsx)", type="xlsx")
if file:
    df = wrangle(file)
    st.write("Cleaned Transactions", df.head())

    rfm_df = create_rfm(df)
    st.write("RFM Table", rfm_df.head())

    st.subheader("Clustering & PCA")
    labels = cluster_rfm(rfm_df)

    st.subheader("Tree-based Interpretation")
    tree_interpret(rfm_df[["Recency","Frequency","Monetary"]], labels)

    st.subheader("Anomaly/Outlier Detection")
    st.caption(
        "Outliers in this project represent customers whose spend or timing is abnormally high/low given their recency and monetary value.")
    anomaly_detection(rfm_df)

