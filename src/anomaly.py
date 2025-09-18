import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import streamlit as st

def anomaly_detection(rfm_df):
    X = rfm_df[["Recency","Frequency","Monetary"]]
    X_scaled = StandardScaler().fit_transform(X)

    iso_full = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    rfm_df["Anomaly_ISO"] = iso_full.fit_predict(X_scaled)
    rfm_df["Anomaly_ISO"] = rfm_df["Anomaly_ISO"].map({1:0,-1:1})

    iso_rf = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    rfm_df["Anomaly_ISO(RF)"] = iso_rf.fit_predict(rfm_df[["Recency", "Frequency"]])
    rfm_df["Anomaly_ISO(RF)"] = rfm_df["Anomaly_ISO(RF)"].map({1: 0, -1: 1})

    iso_fm = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    rfm_df["Anomaly_ISO(FM)"] = iso_fm.fit_predict(rfm_df[["Frequency", "Monetary"]])
    rfm_df["Anomaly_ISO(FM)"] = rfm_df["Anomaly_ISO(FM)"].map({1: 0, -1: 1})

    fig, ax = plt.subplots()
    sns.scatterplot(data=rfm_df, x="Recency", y="Monetary",
                    hue="Anomaly_ISO", palette={0:"blue",1:"red"}, ax=ax)
    ax.set_title("Isolation Forest Outliers (Recency vs Monetary)")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary Value(Millions)")
    st.subheader("Isolation Forest Outliers (Recency vs Monetary)")
    st.pyplot(fig)
    st.caption("Outliers(Red points): Customers who either spend unusually high or low amounts and have an atypical time since last purchase.")
    st.caption("Business view: Could be one-off bulk buyers (very high spend but shop rarely) or dormant accounts that suddenly made a large purchase.")
    st.caption("Action: Review for possible fraud or special-order patterns and decide whether to treat them as VIPs or investigate.")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=rfm_df, x="Frequency", y="Monetary",
                    hue="Anomaly_ISO(FM)", palette={0: "blue", 1: "red"}, ax=ax2)
    ax2.set_title("Isolation Forest Outliers (Frequency vs Monetary)")
    ax2.set_xlabel("Purchase Frequency")
    ax2.set_ylabel("Monetary Value(Millions)")
    st.subheader("Isolation Forest Outliers (Frequency vs Monetary)")
    st.pyplot(fig2)
    st.caption("Outliers(Red points): Customers whose spend is out of proportion to their purchase count.")
    st.caption("e.g; very high spend with very few orders (bulk/wholesale) "
               "or, very high order count but tiny total spend (possible micro-orders or errors)")
    st.caption("Business view: Highlights atypical shopping behaviour—either high-value wholesale accounts or suspicious “low-value high-frequency” patterns.")
    st.caption("Action: Segment separately for tailored pricing or flag for fraud prevention.")

    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=rfm_df, x="Recency", y="Frequency",
                    hue="Anomaly_ISO(RF)", palette={0: "blue", 1: "red"}, ax=ax1)
    ax1.set_title("Isolation Forest Outliers (Recency vs Frequency)")
    ax1.set_xlabel("Recency (days)")
    ax1.set_ylabel("Purchase Frequency")
    st.subheader("Isolation Forest Outliers (Recency vs Frequency)")
    st.pyplot(fig1)
    st.caption("Outliers(Red points): Customers who either spend unusually high or low amounts and have an atypical time since last purchase.")
    st.caption("e.g; very frequent buyers who haven’t purchased in a long time, or one-time buyers who purchased many times in a short window.")
    st.caption("Business view: May signal subscription-style shoppers who suddenly churned, or data-entry issues.")
    st.caption("Action: Target for win-back campaigns or data quality checks.")

    return rfm_df





