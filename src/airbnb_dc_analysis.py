import os
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")

CRIME_CSV_URL = (
    "https://opendata.dc.gov/api/download/v1/items/"
    "dc3289eab3d2400ea49c154863312434/csv?layers=8"
)


# ---------------------------------------------------------------------------
# AIRBNB DATA
# ---------------------------------------------------------------------------

def load_airbnb_data(path: str) -> pd.DataFrame:
    """
    Load Airbnb listings dataset.
    Expects a CSV file at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded AIRBNB dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def clean_airbnb_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Convert price to numeric
    - Remove extreme outliers
    - Cast relevant columns to numeric
    - Optionally compute amenities_count
    """
    df = df.copy()

    # --- Clean price ---
    if "price" not in df.columns:
        raise KeyError("The dataset does not contain a 'price' column.")

    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    before_rows = df.shape[0]
    df = df[df["price"].notna() & (df["price"] > 0)]
    print(f"Removed {before_rows - df.shape[0]} rows with invalid price")

    before_rows = df.shape[0]
    df = df[df["price"] < 1000]
    print(f"Removed {before_rows - df.shape[0]} rows with very high price outliers")

    # Columns that are usually numeric in InsideAirbnb summary data
    numeric_candidates: List[str] = [
        "accommodates",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Amenities count ---
    if "amenities" in df.columns:
        df["amenities_count"] = (
            df["amenities"]
            .astype(str)
            .str.strip("{}")
            .replace("nan", "")
            .apply(lambda x: 0 if x.strip() == "" else len(x.split(",")))
        )

    # --- Review score ---
    if "review_scores_rating" in df.columns:
        df["review_scores_rating"] = pd.to_numeric(
            df["review_scores_rating"], errors="coerce"
        )

    print(f"After cleaning AIRBNB: {df.shape[0]} rows")
    return df


def get_neighbourhood_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a reasonable neighbourhood column name.
    """
    candidates = [
        "neighbourhood_cleansed",
        "neighbourhood_group_cleansed",
        "neighbourhood",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def airbnb_summary_stats(df: pd.DataFrame) -> None:
    print("\n=== BASIC AIRBNB PRICE STATISTICS ===")
    print(df["price"].describe())

    neigh_col = get_neighbourhood_column(df)
    if neigh_col is not None:
        print(f"\nTop 10 neighbourhoods by average nightly price ({neigh_col}):")
        print(
            df.groupby(neigh_col)["price"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
            .head(10)
        )

    if "room_type" in df.columns:
        print("\nAverage price by room type:")
        print(
            df.groupby("room_type")["price"]
            .mean()
            .round(2)
            .sort_values(ascending=False)
        )


def airbnb_plots(df: pd.DataFrame, figures_dir: str = "figures") -> None:
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], bins=50, kde=True)
    plt.title("Distribution of Airbnb Nightly Prices (DC)")
    plt.xlabel("Price (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "price_distribution.png"))
    plt.close()
    print("Saved price_distribution.png")

    # 2. Price by neighbourhood (boxplot)
    neigh_col = get_neighbourhood_column(df)
    if neigh_col is not None:
        top_neigh = df[neigh_col].value_counts().head(10).index
        df_top = df[df[neigh_col].isin(top_neigh)]

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_top, x=neigh_col, y="price")
        plt.xticks(rotation=45, ha="right")
        plt.title("Price by Neighbourhood (Top 10 by Listing Count)")
        plt.xlabel("Neighbourhood")
        plt.ylabel("Price (USD)")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "price_by_neighbourhood.png"))
        plt.close()
        print("Saved price_by_neighbourhood.png")
    else:
        print("No neighbourhood column found; skipping neighbourhood boxplot.")

    # 3. Correlation heatmap of numeric columns
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in ["id", "host_id"]
    ]

    if "price" in numeric_cols and len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap (Numeric Airbnb Features)")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "airbnb_correlation_heatmap.png"))
        plt.close()
        print("Saved airbnb_correlation_heatmap.png")
    else:
        print("Not enough numeric columns for a correlation heatmap.")


def build_price_model(df: pd.DataFrame, figures_dir: str = "figures") -> None:
    """
    Build a simple RandomForest model to predict nightly price.
    Uses numeric features + one-hot encoded room_type (if available).
    """
    # Candidate features that are usually present
    candidate_features = [
        "accommodates",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "amenities_count",
        "review_scores_rating",
    ]
    numeric_features = [f for f in candidate_features if f in df.columns]

    if not numeric_features:
        print("\n[MODEL] No numeric features available for modeling; skipping.")
        return

    X = df[numeric_features].copy()

    # One-hot encode room type if present
    if "room_type" in df.columns:
        dummies = pd.get_dummies(df["room_type"], prefix="room_type")
        X = pd.concat([X, dummies], axis=1)

    y = df["price"]

    # Drop rows with missing values in X or y
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    if len(X) < 200:
        print("\n[MODEL] Not enough rows after cleaning to train a model; skipping.")
        return

    # Log-transform price to reduce skew
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)

    print("\n=== RANDOM FOREST PRICE MODEL ===")
    print(f"Test RMSE: ${rmse:0.2f}")
    print(f"Test R^2 : {r2:0.3f}")

    # Feature importances
    feature_importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 model features by importance:")
    print(feature_importances.head(10).round(4))

    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    feature_importances.head(10).iloc[::-1].plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_feature_importances.png"))
    plt.close()
    print("Saved model_feature_importances.png")


# ---------------------------------------------------------------------------
# CRIME DATA (PUBLIC API)
# ---------------------------------------------------------------------------

def fetch_crime_data(path: str = os.path.join("data", "crime_last30.csv")) -> Optional[pd.DataFrame]:
    """
    Download DC 'Crime Incidents in the Last 30 Days' CSV from the public API
    if it doesn't already exist locally.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(f"\nCrime data already exists at {path}, loading from disk.")
        return pd.read_csv(path)

    print("\nDownloading crime incidents (last 30 days) from DC Open Data API...")
    try:
        resp = requests.get(CRIME_CSV_URL, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to download crime data: {e}")
        return None

    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"Saved crime data to {path}")

    return pd.read_csv(path)


def get_crime_neighbourhood_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a neighbourhood / neighborhood cluster column in the crime data.
    """
    candidates = [
        "NEIGHBORHOODCLUSTER",
        "NEIGHBORHOOD_CLUSTER",
        "NEIGHBORHOOD_CLUSTER_LABEL",
        "NEIGHBORHOOD",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def crime_summary(df: pd.DataFrame, figures_dir: str = "figures") -> None:
    """
    Print and plot basic crime statistics by neighborhood cluster.
    """
    if df is None or df.empty:
        print("\nNo crime data loaded; skipping crime analysis.")
        return

    print(f"\nLoaded CRIME dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    cluster_col = get_crime_neighbourhood_column(df)
    if cluster_col is None:
        print("Could not identify a neighbourhood cluster column in crime data.")
        print(f"Available columns: {list(df.columns)[:20]} ...")
        return

    crime_counts = (
        df.groupby(cluster_col)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    print(f"\nTop 10 neighbourhood clusters by crime count (last 30 days) [{cluster_col}]:")
    print(crime_counts)

    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    crime_counts.iloc[::-1].plot(kind="barh")
    plt.xlabel("Crime Count (Last 30 Days)")
    plt.ylabel("Neighbourhood Cluster")
    plt.title("Top 10 DC Neighbourhood Clusters by Crime Incidents")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "crime_by_neighbourhood_cluster.png"))
    plt.close()
    print("Saved crime_by_neighbourhood_cluster.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    airbnb_path = os.path.join("data", "listings.csv")
    airbnb_df_raw = load_airbnb_data(airbnb_path)
    airbnb_df = clean_airbnb_data(airbnb_df_raw)

    airbnb_summary_stats(airbnb_df)
    airbnb_plots(airbnb_df)
    build_price_model(airbnb_df)

    # Crime API integration
    crime_df = fetch_crime_data()
    if crime_df is not None:
        crime_summary(crime_df)


if __name__ == "__main__":
    main()
