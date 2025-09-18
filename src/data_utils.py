import pandas as pd

def wrangle(filename: str) -> pd.DataFrame:
    df = pd.read_excel(filename).dropna()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df_uk = df[df["Country"] == "United Kingdom"].copy()
    df_uk["TotalCost"] = df_uk["Quantity"] * df_uk["UnitPrice"]
    df_uk = df_uk[(df_uk["TotalCost"] <= 500) & (df_uk["Quantity"] <= 100)]
    return df_uk.reset_index(drop=True)

def create_rfm(df: pd.DataFrame) -> pd.DataFrame:
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
          .agg(Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
               Frequency=("InvoiceNo", "nunique"),
               Monetary=("TotalCost", "sum"))
          .reset_index()
    )
    return rfm
