import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pandas as pd
import streamlit as st

def cluster_rfm(rfm_df: pd.DataFrame):
    X = rfm_df[["Recency","Frequency","Monetary"]]
    s_scaler = StandardScaler()

    inertia_scores, silhouette_scores = [], []
    ks = range(2,11)
    for k in ks:
        km = make_pipeline(s_scaler, KMeans(n_clusters=k, random_state=42))
        km.fit(X)
        inertia_scores.append(km.named_steps["kmeans"].inertia_)
        silhouette_scores.append(silhouette_score(X, km.named_steps["kmeans"].labels_))

    # Plot Inertia
    fig1 = px.line(x=list(ks), y=inertia_scores,
                   title="Inertia vs K",
                   labels={"x":"Clusters (k)","y":"Inertia"})
    st.plotly_chart(fig1)

    # Plot Silhouette
    fig2 = px.line(x=list(ks), y=silhouette_scores,
                   title="Silhouette vs K",
                   labels={"x":"Clusters (k)","y":"Silhouette"})
    st.plotly_chart(fig2)

    # Final KMeans with k=3 (example)
    final_model = make_pipeline(s_scaler, KMeans(n_clusters=3, random_state=42))
    final_model.fit(X)
    labels = final_model.named_steps["kmeans"].labels_

    # PCA for 2-D visualization
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    pca_df = pd.DataFrame(pcs, columns=["PC1","PC2"])
    pca_df["Cluster"] = labels
    fig3 = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                      title="PCA representation of clusters")
    st.plotly_chart(fig3)

    return labels
