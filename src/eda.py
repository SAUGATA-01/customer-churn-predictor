"""Exploratory data analysis for the churn dataset. Saves charts to reports/figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/processed/churn_clean.csv")
FIG_DIR = Path("reports/figures")
TARGET = "churn_value"

CATEGORICAL_COLS = [
    "contract",
    "internet_service",
    "payment_method",
    "tech_support",
    "online_security",
    "paperless_billing",
    "senior_citizen",
]

sns.set_theme(style="whitegrid")


def save_fig(name: str) -> None:
    """Save the current figure to the figures folder and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Bar chart showing how many customers stayed vs churned."""
    counts = df[TARGET].map({0: "Stayed", 1: "Churned"}).value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Customer churn distribution")
    ax.set_ylabel("Customers")
    save_fig("01_target_distribution.png")


def plot_churn_rate(df: pd.DataFrame, col: str, index: int) -> None:
    """Bar chart of churn rate (%) for each category of a column."""
    rates = df.groupby(col)[TARGET].mean().mul(100).sort_values(ascending=False)
    print(f"\nChurn rate (%) by {col}:")
    print(rates.round(1).to_string())

    ax = sns.barplot(x=rates.index.astype(str), y=rates.values)
    ax.axhline(df[TARGET].mean() * 100, color="red", linestyle="--", label="Overall churn rate")
    ax.set_title(f"Churn rate by {col}")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(col)
    plt.xticks(rotation=20, ha="right")
    ax.legend()
    save_fig(f"{index:02d}_churn_by_{col}.png")


def plot_tenure(df: pd.DataFrame) -> None:
    """Histogram of tenure and churn rate per tenure bucket."""
    labels = df[TARGET].map({0: "Stayed", 1: "Churned"})
    sns.histplot(data=df, x="tenure_months", hue=labels, bins=30, multiple="stack")
    plt.title("Tenure distribution by churn")
    save_fig("10_tenure_distribution.png")

    buckets = pd.cut(
        df["tenure_months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    )
    rates = df.groupby(buckets, observed=True)[TARGET].mean().mul(100)
    print("\nChurn rate (%) by tenure bucket (months):")
    print(rates.round(1).to_string())

    ax = sns.barplot(x=rates.index.astype(str), y=rates.values)
    ax.set_title("Churn rate by tenure bucket")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate (%)")
    save_fig("11_churn_by_tenure_bucket.png")


def plot_monthly_charges(df: pd.DataFrame) -> None:
    """Boxplot comparing monthly charges of churned and retained customers."""
    labels = df[TARGET].map({0: "Stayed", 1: "Churned"})
    sns.boxplot(x=labels, y=df["monthly_charges"])
    plt.title("Monthly charges by churn")
    plt.xlabel("")
    save_fig("12_monthly_charges_boxplot.png")


def plot_correlation(df: pd.DataFrame) -> None:
    """Correlation heatmap of numeric columns."""
    numeric = df[["tenure_months", "monthly_charges", "total_charges", "cltv", TARGET]]
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation of numeric features")
    save_fig("13_correlation_heatmap.png")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Overall churn rate: {df[TARGET].mean():.2%}")

    plot_target_distribution(df)
    for i, col in enumerate(CATEGORICAL_COLS, start=2):
        plot_churn_rate(df, col, i)
    plot_tenure(df)
    plot_monthly_charges(df)
    plot_correlation(df)

    print(f"\nAll charts saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()