import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data(path: str) -> pd.DataFrame:
    """
    Load Airbnb listings dataset.
    Expects a CSV file at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Convert price to numeric
    - Remove extreme outliers
    - Make numeric versions of bedrooms, bathrooms
    - Add amenities_count if 'amenities' column exists
    - Cast review_scores_rating to numeric
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

    # Remove rows with missing or non-positive price
    before_rows = df.shape[0]
    df = df[df["price"].notna() & (df["price"] > 0)]
    print(f"Removed {before_rows - df.shape[0]} rows with invalid price")

    # Remove extreme outliers (e.g., > $1000 per night)
    before_rows = df.shape[0]
    df = df[df["price"] < 1000]
    print(f"Removed {before_rows - df.shape[0]} rows with very high price outliers")

    # --- Bedrooms / Bathrooms ---
    if "bedrooms" in df.columns:
        df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
    if "bathrooms" in df.columns:
        df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")

    # --- Amenities count ---
    if "amenities" in df.columns:
        # amenities are usually a string like: "{TV, Wifi, Kitchen, ...}"
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

    print(f"After cleaning: {df.shape[0]} rows")
    return df


def get_neighbourhood_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a reasonable neighbourhood column name.
    Different InsideAirbnb datasets sometimes use:
    - 'neighbourhood_cleansed'
    - 'neighbourhood_group_cleansed'
    - 'neighbourhood'
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


def basic_plots(df: pd.DataFrame, figures_dir: str = "figures") -> None:
    """
    Generate and save basic exploratory plots:
    - Price distribution
    - Price by neighbourhood (boxplot)
    - Bedrooms vs price (scatter)
    - Correlation heatmap
    """
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
        # Use top 10 neighbourhoods by listing count
        top_neigh = (
            df[neigh_col]
            .value_counts()
            .head(10)
            .index
        )
        df_top = df[df[neigh_col].isin(top_neigh)]

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_top,
            x=neigh_col,
            y="price",
        )
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

    # 3. Bedrooms vs price (scatter)
    if "bedrooms" in df.columns:
        df_bed = df.dropna(subset=["bedrooms"])
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=df_bed,
            x="bedrooms",
            y="price",
            alpha=0.3,
        )
        plt.title("Bedrooms vs Price")
        plt.xlabel("Number of Bedrooms")
        plt.ylabel("Price (USD)")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "bedrooms_vs_price.png"))
        plt.close()
        print("Saved bedrooms_vs_price.png")
    else:
        print("No 'bedrooms' column; skipping bedrooms vs price plot.")

    # 4. Correlation heatmap of numeric columns
    numeric_cols = [
        col
        for col in [
            "price",
            "bedrooms" if "bedrooms" in df.columns else None,
            "bathrooms" if "bathrooms" in df.columns else None,
            "amenities_count" if "amenities_count" in df.columns else None,
            "review_scores_rating" if "review_scores_rating" in df.columns else None,
        ]
        if col is not None
    ]

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap (Key Numeric Features)")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"))
        plt.close()
        print("Saved correlation_heatmap.png")
    else:
        print("Not enough numeric columns for a correlation heatmap.")


def print_summary_stats(df: pd.DataFrame) -> None:
    """
    Print some basic summary stats to the console for interpretation.
    """
    print("\n=== BASIC SUMMARY STATISTICS ===")
    print(df["price"].describe())

    if "bedrooms" in df.columns:
        print("\nAverage price by number of bedrooms:")
        print(
            df.groupby("bedrooms")["price"]
            .mean()
            .sort_index()
            .round(2)
            .head(10)
        )

    neigh_col = get_neighbourhood_column(df)
    if neigh_col is not None:
        print(f"\nTop 10 neighbourhoods by average price ({neigh_col}):")
        print(
            df.groupby(neigh_col)["price"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
            .head(10)
        )


def main() -> None:
    data_path = os.path.join("data", "listings.csv")
    df_raw = load_data(data_path)
    df_clean = clean_data(df_raw)
    print_summary_stats(df_clean)
    basic_plots(df_clean)


if __name__ == "__main__":
    main()
