import pandas as pd
import numpy as np

# The Single Source of Truth for the whole project
STATES = ["Fresh", "Discounted", "Sold", "Waste"]
START_STATE = "Fresh"
ABSORBING_STATES = ["Sold", "Waste"]

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def average_sale_time(df: pd.DataFrame, item_name: str) -> float:
    item_df = df[df["item"] == item_name].copy()

    if item_df.empty:
        raise ValueError(f"Item '{item_name}' not found.")

    weighted_avg_hour = (
        (item_df["hour"] * item_df["quantity"]).sum()
        / item_df["quantity"].sum()
    )
    return weighted_avg_hour


def build_hourly_baseline(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["item", "hour"], as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            avg_quantity=("quantity", "mean"),
            avg_price=("unit_price", "mean"),
            total_revenue=("revenue", "sum")
        )
    )


def build_daily_item_sales(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["day", "item", "hour"], as_index=False)
        .agg(
            quantity_sold=("quantity", "sum"),
            avg_price=("unit_price", "mean")
        )
    )


# =========================================================
# 3. MARKOV INPUTS WITH NONZERO WASTE PROXY
# =========================================================

def estimate_markov_inputs_with_proxy(
    df: pd.DataFrame,
    item_name: str,
    discount_start_hour: int = 15,
    inventory_factor: float = 1.10,
    discount_sell_through_factor: float = 0.70
) -> dict:
    """
    Assumptions:
    - estimated inventory = avg_daily_sales * inventory_factor
    - not everything that reaches discount gets sold
    - discount_sell_through_factor controls how much of the observed
      after-discount sales potential actually converts into sell-through

    Interpretation:
    - p_sell_fresh: share of observed sales before discount hour
    - p_to_discount: share still unsold by the time discount begins
    - p_sell_discount: discounted-stage sell-through proxy
    - p_waste: leftover share after fresh + discount sales
    """

    item_df = df[df["item"] == item_name].copy()

    if item_df.empty:
        raise ValueError(f"Item '{item_name}' not found in dataset.")

    # Average daily sales
    daily_totals = item_df.groupby("day")["quantity"].sum()
    avg_daily_sales = daily_totals.mean()

    if pd.isna(avg_daily_sales) or avg_daily_sales == 0:
        raise ValueError(f"No valid daily sales found for item '{item_name}'.")

    # Assumed inventory (used as waste proxy base)
    estimated_inventory = avg_daily_sales * inventory_factor

    # Observed hourly sales timing
    hourly_total = item_df.groupby("hour")["quantity"].sum()
    total_sales = hourly_total.sum()

    if total_sales == 0:
        raise ValueError(f"No sales found for item '{item_name}'.")

    fresh_sales_observed = hourly_total[hourly_total.index < discount_start_hour].sum()
    after_discount_sales_observed = hourly_total[hourly_total.index >= discount_start_hour].sum()

    # Probability of selling before discount
    p_sell_fresh = fresh_sales_observed / total_sales

    # Probability of surviving until discount time
    p_to_discount = 1 - p_sell_fresh

    # Observed share of sales after discount hour
    observed_after_discount_share = after_discount_sales_observed / total_sales

    # Only a fraction of after-discount potential actually converts to sell-through
    p_sell_discount = min(
        observed_after_discount_share * discount_sell_through_factor,
        p_to_discount
    )

    # Leftover becomes waste
    p_waste = max(0.0, 1 - p_sell_fresh - p_sell_discount)

    return {
        "item": item_name,
        "discount_hour": discount_start_hour,
        "p_sell_fresh": p_sell_fresh,
        "p_to_discount": p_to_discount,
        "p_sell_discount": p_sell_discount,
        "p_waste": p_waste
    }


# =========================================================
# 4. MARKOV CHAIN FORMULATION
# =========================================================

def create_bakery_matrix(
    p_sell_fresh: float,
    p_to_discount: float,
    p_sell_discount: float,
    p_waste: float
) -> np.ndarray:
    """
    States:
    [Fresh, Discounted, Sold, Waste]
    """

    fresh_stay = max(0.0, 1 - p_sell_fresh - p_to_discount)
    discount_stay = max(0.0, 1 - p_sell_discount - p_waste)

    P = np.array([
        [fresh_stay,      p_to_discount, p_sell_fresh,    0.0],
        [0.0,             discount_stay, p_sell_discount, p_waste],
        [0.0,             0.0,           1.0,             0.0],
        [0.0,             0.0,           0.0,             1.0]
    ])

    return P
# =========================================================
# 5. EVALUATE MULTIPLE DISCOUNT TIMES FOR ONE ITEM
# =========================================================

def evaluate_discount_hours_with_proxy(
    df: pd.DataFrame,
    item_name: str,
    discount_hours: list,
    inventory_factor: float = 1.10,
    discount_sell_through_factor: float = 0.70
) -> pd.DataFrame:
    results = []

    for hour in discount_hours:
        params = estimate_markov_inputs_with_proxy(
            df=df,
            item_name=item_name,
            discount_start_hour=hour,
            inventory_factor=inventory_factor,
            discount_sell_through_factor=discount_sell_through_factor
        )

        matrix = create_bakery_matrix(
            p_sell_fresh=params["p_sell_fresh"],
            p_to_discount=params["p_to_discount"],
            p_sell_discount=params["p_sell_discount"],
            p_waste=params["p_waste"]
        )

        results.append({
            "item": params["item"],
            "discount_hour": params["discount_hour"],
            "p_sell_fresh": params["p_sell_fresh"],
            "p_to_discount": params["p_to_discount"],
            "p_sell_discount": params["p_sell_discount"],
            "p_waste": params["p_waste"],
            "transition_matrix": matrix
        })

    return pd.DataFrame(results)


# =========================================================
# 6. EVALUATE ALL ITEMS
# =========================================================

def evaluate_all_items_with_proxy(
    df: pd.DataFrame,
    discount_hours: list,
    inventory_factor: float = 1.10,
    discount_sell_through_factor: float = 0.70
) -> pd.DataFrame:
    all_results = []

    items = sorted(df["item"].dropna().unique())

    for item in items:
        try:
            item_results = evaluate_discount_hours_with_proxy(
                df=df,
                item_name=item,
                discount_hours=discount_hours,
                inventory_factor=inventory_factor,
                discount_sell_through_factor=discount_sell_through_factor
            )
            all_results.append(item_results)
        except Exception as e:
            print(f"Skipping {item}: {e}")

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)
