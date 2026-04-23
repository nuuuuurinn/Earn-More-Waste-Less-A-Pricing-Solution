import pandas as pd
import numpy as np

STATES           = ["Fresh", "Discounted", "Sold", "Waste"]
START_STATE      = "Fresh"
ABSORBING_STATES = ["Sold", "Waste"]
DAY_TYPES        = ["Weekday", "Weekend"]


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def average_sale_time(df: pd.DataFrame, item_name: str) -> float:
    item_df = df[df["item"] == item_name].copy()
    if item_df.empty:
        raise ValueError(f"Item '{item_name}' not found.")
    return (item_df["hour"] * item_df["quantity"]).sum() / item_df["quantity"].sum()


def build_hourly_baseline(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["item", "hour"], as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            avg_quantity=("quantity",   "mean"),
            avg_price=("unit_price",    "mean"),
            total_revenue=("revenue",   "sum")
        )
    )


def build_daily_item_sales(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["day", "item", "hour"], as_index=False)
        .agg(
            quantity_sold=("quantity",   "sum"),
            avg_price=("unit_price",     "mean")
        )
    )


# =========================================================
# 2b. PRICE ELASTICITY ESTIMATION
# =========================================================

def estimate_price_elasticity(
    df: pd.DataFrame,
    item_name: str,
    day_type: str = None,
    min_observations: int = 10,
    default_elasticity: float = -1.5
) -> float:
    """
    Log-log OLS regression of daily quantity on daily avg price.
    Optionally filtered to a specific day_type ("Weekday" or "Weekend").
    """
    item_df = df[df["item"] == item_name].copy()
    if day_type and "day_type" in item_df.columns:
        item_df = item_df[item_df["day_type"] == day_type]

    daily = item_df.groupby("day").agg(
        quantity=("quantity",   "sum"),
        avg_price=("unit_price","mean")
    ).reset_index()
    daily = daily[(daily["quantity"] > 0) & (daily["avg_price"] > 0)]

    if len(daily) < min_observations:
        return default_elasticity

    price_cv = daily["avg_price"].std() / daily["avg_price"].mean()
    if price_cv < 0.01:
        return default_elasticity

    log_q = np.log(daily["quantity"])
    log_p = np.log(daily["avg_price"])
    var_p = np.var(log_p)

    if var_p == 0:
        return default_elasticity

    elasticity = np.cov(log_p, log_q)[0, 1] / var_p
    return float(np.clip(elasticity, -5.0, -0.1))


def estimate_all_elasticities(df: pd.DataFrame) -> dict:
    """
    Returns { (item, day_type): elasticity } for all items × day types.
    """
    items   = sorted(df["item"].dropna().unique())
    result  = {}
    for item in items:
        for day_type in DAY_TYPES:
            result[(item, day_type)] = estimate_price_elasticity(
                df, item, day_type=day_type
            )
    return result


def sell_through_from_elasticity(
    base_factor: float,
    elasticity: float,
    discount_pct: float
) -> float:
    """
    Adjusts sell-through factor using price elasticity and discount depth.
    volume_multiplier = (1 - discount_pct/100) ^ elasticity
    Since elasticity < 0 and price ratio < 1 → multiplier > 1 (more sales).
    """
    if discount_pct <= 0:
        return base_factor
    price_ratio       = 1.0 - (discount_pct / 100.0)
    volume_multiplier = price_ratio ** elasticity
    return float(np.clip(base_factor * volume_multiplier, base_factor, 1.0))


# =========================================================
# 3. MARKOV INPUTS
# =========================================================

def estimate_markov_inputs_with_proxy(
    df: pd.DataFrame,
    item_name: str,
    discount_start_hour: int = 15,
    inventory_factor: float = 1.10,
    discount_sell_through_factor: float = 0.70,
    day_type: str = None        # ← NEW: "Weekday", "Weekend", or None (all)
) -> dict:
    item_df = df[df["item"] == item_name].copy()

    # Filter by day_type if specified
    if day_type and "day_type" in item_df.columns:
        item_df = item_df[item_df["day_type"] == day_type]

    if item_df.empty:
        raise ValueError(f"No data for '{item_name}' / day_type='{day_type}'.")

    daily_totals    = item_df.groupby("day")["quantity"].sum()
    avg_daily_sales = daily_totals.mean()

    if pd.isna(avg_daily_sales) or avg_daily_sales == 0:
        raise ValueError(f"No valid daily sales for '{item_name}' / '{day_type}'.")

    hourly_total  = item_df.groupby("hour")["quantity"].sum()
    total_sales   = hourly_total.sum()

    if total_sales == 0:
        raise ValueError(f"No sales for '{item_name}' / '{day_type}'.")

    fresh_sales_observed          = hourly_total[hourly_total.index < discount_start_hour].sum()
    after_discount_sales_observed = hourly_total[hourly_total.index >= discount_start_hour].sum()

    p_sell_fresh  = fresh_sales_observed / total_sales
    p_to_discount = 1 - p_sell_fresh

    observed_after_discount_share = after_discount_sales_observed / total_sales
    p_sell_discount = min(
        observed_after_discount_share * discount_sell_through_factor,
        p_to_discount
    )
    p_waste = max(0.0, 1 - p_sell_fresh - p_sell_discount)

    return {
        "item":             item_name,
        "discount_hour":    discount_start_hour,
        "day_type":         day_type or "All",
        "p_sell_fresh":     p_sell_fresh,
        "p_to_discount":    p_to_discount,
        "p_sell_discount":  p_sell_discount,
        "p_waste":          p_waste
    }


# =========================================================
# 4. MARKOV CHAIN
# =========================================================

def create_bakery_matrix(
    p_sell_fresh: float,
    p_to_discount: float,
    p_sell_discount: float,
    p_waste: float
) -> np.ndarray:
    fresh_stay    = max(0.0, 1 - p_sell_fresh - p_to_discount)
    discount_stay = max(0.0, 1 - p_sell_discount - p_waste)
    return np.array([
        [fresh_stay,   p_to_discount, p_sell_fresh,    0.0],
        [0.0,          discount_stay, p_sell_discount,  p_waste],
        [0.0,          0.0,           1.0,              0.0],
        [0.0,          0.0,           0.0,              1.0]
    ])


# =========================================================
# 5. EVALUATE ONE ITEM ACROSS HOURS
# =========================================================

def evaluate_discount_hours_with_proxy(
    df: pd.DataFrame,
    item_name: str,
    discount_hours: list,
    inventory_factor: float = 1.10,
    discount_sell_through_factor: float = 0.70,
    day_type: str = None
) -> pd.DataFrame:
    results = []
    for hour in discount_hours:
        params = estimate_markov_inputs_with_proxy(
            df=df,
            item_name=item_name,
            discount_start_hour=hour,
            inventory_factor=inventory_factor,
            discount_sell_through_factor=discount_sell_through_factor,
            day_type=day_type
        )
        matrix = create_bakery_matrix(
            p_sell_fresh=params["p_sell_fresh"],
            p_to_discount=params["p_to_discount"],
            p_sell_discount=params["p_sell_discount"],
            p_waste=params["p_waste"]
        )
        results.append({**params, "transition_matrix": matrix})
    return pd.DataFrame(results)


# =========================================================
# 6. EVALUATE ALL ITEMS × DAY TYPES
# =========================================================

def evaluate_all_items_with_proxy(
    df: pd.DataFrame,
    discount_hours: list,
    inventory_factor: float = 1.10,
    discount_sell_through_factor: float = 0.70
) -> pd.DataFrame:
    """
    Runs Markov evaluation for each item × discount_hour × day_type.
    Returns a DataFrame with a 'day_type' column.
    """
    all_results = []
    items = sorted(df["item"].dropna().unique())

    for day_type in DAY_TYPES:
        df_segment = df[df["day_type"] == day_type]
        for item in items:
            try:
                item_results = evaluate_discount_hours_with_proxy(
                    df=df_segment,
                    item_name=item,
                    discount_hours=discount_hours,
                    inventory_factor=inventory_factor,
                    discount_sell_through_factor=discount_sell_through_factor,
                    day_type=day_type
                )
                all_results.append(item_results)
            except Exception as e:
                pass   # silently skip items with insufficient data for a segment

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)