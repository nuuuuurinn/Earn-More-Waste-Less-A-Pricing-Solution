import pandas as pd
import numpy as np


# =========================================================
# 1. LOAD AND PREP DATA
# =========================================================

def load_data(filepath: str, sheet_name: str = "Bakery sales") -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ["date", "time", "item", "quantity", "unit_price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean item names
    df = df[df["item"].notna()].copy()
    df["item"] = df["item"].astype(str).str.strip().str.upper()

    # Remove weird / non-product / grouped entries
    df = df[df["item"] != ""]
    df = df[df["item"] != "."]
    df = df[~df["item"].str.contains("ARTICLE",  na=False)]
    df = df[~df["item"].str.contains("DIVERS",   na=False)]
    df = df[~df["item"].str.contains("BOISSON",  na=False)]
    df = df[~df["item"].str.contains("CAFE",     na=False)]
    df = df[~df["item"].str.contains("THE",      na=False)]
    df = df[~df["item"].str.contains("PLAT",     na=False)]
    df = df[~df["item"].str.contains("FORMULE",  na=False)]
    df = df[~df["item"].str.contains("TRAITEUR", na=False)]

    # Keep only valid rows
    df = df[(df["quantity"] > 0) & (df["unit_price"] >= 0)]

    # Build datetime
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    if df["datetime"].isna().any():
        bad_rows = df[df["datetime"].isna()][["date", "time"]]
        raise ValueError(f"Some date/time values could not be parsed:\n{bad_rows.head()}")

    df["hour"]    = df["datetime"].dt.hour
    df["day"]     = df["datetime"].dt.date
    df["revenue"] = df["quantity"] * df["unit_price"]

    # ── Day-of-week segmentation ──
    df["day_of_week"] = pd.to_datetime(df["day"]).dt.dayofweek  # 0=Mon, 6=Sun
    df["day_type"]    = df["day_of_week"].apply(
        lambda x: "Weekend" if x >= 5 else "Weekday"
    )

    return df