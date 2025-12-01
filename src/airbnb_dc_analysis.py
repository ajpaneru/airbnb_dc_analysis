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

# Global visual theme
sns.set_theme(
    style="whitegrid",
    palette="deep",
    rc={
        "figure.figsize": (10, 6),
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    }
)

CRIME_CSV_URL = (
    "https://opendata.dc.gov/api/download/v1/items/"
    "dc3289eab3d2400ea49c154863312434/csv?layers=8"
)

# ---------------------------------------------------------------------------
# AIRBNB DATA
# ---------------------------------------------------------------------------

def load_airbnb_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded AIRBNB dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def clean_airbnb_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # price
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

    if "amenities" in df.columns:
        df["amenities_count"] = (
            df["amenities"]
            .astype(str)
            .str.strip("{}")
            .replace("nan", "")
            .apply(lambda x: 0 if x.strip() == "" else len(x.split(",")))
        )

    if "review_scores_rating" in df.columns:
        df["review_scores_rating"] = pd.to_numeric(
            df["review_scores_rating"], errors="coerce"
        )

    if "bedrooms" in df.columns:
        df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")

    print(f"After cleaning AIRBNB: {df.shape[0]} rows")
    return df


def get_neighbourhood_column(df: pd.DataFrame) -> Optional[str]:
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

# ---------------------------------------------------------------------------
# PLOTTING HELPERS (UPGRADED)
# ---------------------------------------------------------------------------

def plot_price_distribution(df: pd.DataFrame, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.histplot(
        df["price"],
        bins=40,
        kde=True,
        color="#1f77b4",
        alpha=0.85
    )
    plt.title("Distribution of Airbnb Prices in Washington, DC", pad=20)
    plt.xlabel("Price (USD)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "price_distribution.png"), dpi=300)
    plt.close()
    print("Saved price_distribution.png")


def plot_price_by_neighbourhood(df: pd.DataFrame, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    neigh_col = get_neighbourhood_column(df)
    if neigh_col is None:
        print("No neighbourhood column found; skipping neighbourhood plot.")
        return

    top10 = df[neigh_col].value_counts().nlargest(10).index
    df_top = df[df[neigh_col].isin(top10)]

    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=df_top,
        x=neigh_col,
        y="price",
        showfliers=False,
        linewidth=1.5,
        palette="viridis"
    )
    plt.title("Airbnb Prices by Neighbourhood (Top 10 by Listing Count)", pad=20)
    plt.xlabel("Neighbourhood")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "price_by_neighbourhood.png"), dpi=300)
    plt.close()
    print("Saved price_by_neighbourhood.png")


def plot_bedrooms_vs_price(df: pd.DataFrame, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    if "bedrooms" not in df.columns:
        print("No bedrooms column; skipping bedrooms vs price plot.")
        return

    plt.figure(figsize=(12, 6))
    sns.regplot(
        data=df,
        x="bedrooms",
        y="price",
        scatter_kws={"alpha": 0.25, "color": "#2ca02c"},
        line_kws={"color": "red", "linewidth": 2},
        ci=None
    )
    plt.title("Relationship Between Bedrooms and Price", pad=20)
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "bedrooms_vs_price.png"), dpi=300)
    plt.close()
    print("Saved bedrooms_vs_price.png")


def plot_correlation_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        print("Not enough numeric cols for heatmap; skipping.")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.7}
    )
    plt.title("Correlation Heatmap of Numeric Airbnb Features", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "airbnb_correlation_heatmap.png"), dpi=300)
    plt.close()
    print("Saved airbnb_correlation_heatmap.png")


def plot_feature_importance(importances: pd.Series, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(
        kind="barh",
        color="#ff7f0e",
        edgecolor="black"
    )
    plt.title("Random Forest Feature Importances", pad=20)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_feature_importances.png"), dpi=300)
    plt.close()
    print("Saved model_feature_importances.png")

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def build_price_model(df: pd.DataFrame, figures_dir: str = "figures") -> None:
    candidate_features = [
        "accommodates",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "amenities_count",
        "review_scores_rating",
        "bedrooms",
    ]
    numeric_features = [f for f in candidate_features if f in df.columns]

    if not numeric_features:
        print("\n[MODEL] No numeric features available for modeling; skipping.")
        return

    X = df[numeric_features].copy()

    if "room_type" in df.columns:
        dummies = pd.get_dummies(df["room_type"], prefix="room_type")
        X = pd.concat([X, dummies], axis=1)

    y = df["price"]
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    if len(X) < 200:
        print("\n[MODEL] Not enough rows after cleaning to train a model; skipping.")
        return

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

    feature_importances = pd.Series(
        model.feature_importances_, index=X.columns
    )

    print("\nTop 10 model features by importance:")
    print(feature_importances.sort_values(ascending=False).head(10).round(4))

    plot_feature_importance(feature_importances, figures_dir)

# ---------------------------------------------------------------------------
# CRIME DATA
# ---------------------------------------------------------------------------

def fetch_crime_data(path: str = os.path.join("data", "crime_last30.csv")) -> Optional[pd.DataFrame]:
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
    plt.figure(figsize=(12, 7))
    crime_counts.iloc[::-1].plot(kind="barh", color="#d62728", edgecolor="black")
    plt.xlabel("Crime Count (Last 30 Days)")
    plt.ylabel("Neighbourhood Cluster")
    plt.title("Top 10 DC Neighbourhood Clusters by Crime Incidents", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "crime_by_neighbourhood_cluster.png"), dpi=300)
    plt.close()
    print("Saved crime_by_neighbourhood_cluster.png")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    figures_dir = "figures"

    airbnb_path = os.path.join("data", "listings.csv")
    airbnb_df_raw = load_airbnb_data(airbnb_path)
    airbnb_df = clean_airbnb_data(airbnb_df_raw)

    airbnb_summary_stats(airbnb_df)

    plot_price_distribution(airbnb_df, figures_dir)
    plot_price_by_neighbourhood(airbnb_df, figures_dir)
    plot_bedrooms_vs_price(airbnb_df, figures_dir)
    plot_correlation_heatmap(airbnb_df, figures_dir)

    build_price_model(airbnb_df, figures_dir)

    crime_df = fetch_crime_data()
    if crime_df is not None:
        crime_summary(crime_df, figures_dir)


if __name__ == "__main__":
    main()
