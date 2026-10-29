"""Load, clean and save the IBM Telco churn dataset."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/Telco_customer_churn.xlsx")
PROCESSED_PATH = Path("data/processed/churn_clean.csv")

# Columns that leak the target or reveal the outcome after the customer left
LEAKAGE_COLS = ["churn_label", "churn_score", "churn_reason"]

# Columns with no predictive value (IDs, constants, high-cardinality geography)
USELESS_COLS = [
    "customerid", "count", "country", "state",
    "city", "zip_code", "lat_long", "latitude", "longitude",
]

TARGET = "churn_value"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw Excel file and standardize column names to snake_case."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data types, handle missing values and drop unwanted columns."""
    df = df.copy()

    # total_charges has blank strings for customers with tenure 0
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    missing = df["total_charges"].isna().sum()
    print(f"Missing total_charges before fix: {missing}")

    # New customers (tenure 0) have not been billed yet, so 0 is the right value
    df.loc[df["tenure_months"] == 0, "total_charges"] = 0.0
    print(f"Missing total_charges after fix: {df['total_charges'].isna().sum()}")

    # Remove leakage and useless columns
    df = df.drop(columns=LEAKAGE_COLS + USELESS_COLS)

    # Strip whitespace in text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def main() -> None:
    df_raw = load_raw()
    print(f"Raw shape: {df_raw.shape}")

    df_clean = clean(df_raw)
    print(f"Clean shape: {df_clean.shape}")
    print(f"Churn rate: {df_clean[TARGET].mean():.2%}")
    print(f"Duplicate rows: {df_clean.duplicated().sum()}")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()