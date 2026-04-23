import pandas as pd
import numpy as np

from markov_formulation import (
    estimate_markov_inputs_with_proxy,
    create_bakery_matrix,
    sell_through_from_elasticity,
    DAY_TYPES
)
from simulation import run_monte_carlo


# =========================================================
# 7. Financial Impact Model
# =========================================================

def run_financial_impact_model(
    simulation_df,
    results_df,
    item_prices,
    cost_ratio=0.3,
    discount_ratio=0.9
):
    """
    Merges simulation results with Markov probabilities and computes
    profit gain over baseline. Works with day_type-segmented data.
    """
    merge_cols = ["item", "discount_hour", "day_type"]
    results = pd.merge(
        simulation_df,
        results_df[merge_cols + ["p_sell_fresh", "p_to_discount"]],
        on=merge_cols
    )

    results["full_price"]     = results["item"].map(item_prices)
    results["prod_cost"]      = results["full_price"] * cost_ratio
    results["discount_price"] = results["full_price"] * discount_ratio

    results["baseline_profit"] = (
        results["p_sell_fresh"] * results["full_price"]
    ) - (results["p_to_discount"] * results["prod_cost"])

    results["final_profit"] = (
        results["sold_rate"] * results["discount_price"]
    ) - (results["waste_rate"] * results["prod_cost"])

    results["profit_gain"]   = results["final_profit"] - results["baseline_profit"]
    results["is_profitable"] = results["profit_gain"] > 0
    results = results.sort_values(by="profit_gain", ascending=False)

    return results


# =========================================================
# 8. Elasticity-driven Sensitivity Analysis
# =========================================================

def run_sensitivity_analysis(
    df,
    results_df,
    item_prices,
    elasticity_dict,
    cost_ratio=0.3,
    base_sell_through_factor=0.70,
    discount_rates=None,
    n_simulations=1000
):
    """
    Sweeps discount rates 5%–40% per item × hour × day_type.
    Uses price elasticity (segmented by day_type) to adjust sell-through,
    re-runs simulation, then computes financial impact.
    """
    if discount_rates is None:
        discount_rates = [round(r, 2) for r in np.arange(0.05, 0.45, 0.05)]

    rows  = []
    items = results_df["item"].unique()
    hours = results_df["discount_hour"].unique()

    total = len(items) * len(hours) * len(DAY_TYPES) * len(discount_rates)
    print(f"Sensitivity sweep: {len(items)} items × {len(hours)} hours × "
          f"2 day types × {len(discount_rates)} rates = {total} runs...")

    for day_type in DAY_TYPES:
        df_segment = df[df["day_type"] == day_type]

        for item in items:
            elasticity = elasticity_dict.get((item, day_type), -1.5)
            full_price = item_prices.get(item, None)
            if full_price is None:
                continue

            prod_cost = full_price * cost_ratio

            # Baseline profit reference (from first available hour for this segment)
            ref = results_df[
                (results_df["item"] == item) &
                (results_df["day_type"] == day_type)
            ]
            if ref.empty:
                continue

            baseline_profit = (
                ref.iloc[0]["p_sell_fresh"] * full_price
            ) - (ref.iloc[0]["p_to_discount"] * prod_cost)

            for hour in hours:
                ref_row = ref[ref["discount_hour"] == hour]
                if ref_row.empty:
                    continue

                for discount_pct_decimal in discount_rates:
                    discount_pct = round(discount_pct_decimal * 100)

                    # Elasticity-adjusted sell-through
                    adjusted_factor = sell_through_from_elasticity(
                        base_factor=base_sell_through_factor,
                        elasticity=elasticity,
                        discount_pct=discount_pct
                    )

                    # Recompute Markov with adjusted factor
                    try:
                        params = estimate_markov_inputs_with_proxy(
                            df=df_segment,
                            item_name=item,
                            discount_start_hour=int(hour),
                            discount_sell_through_factor=adjusted_factor,
                            day_type=day_type
                        )
                    except Exception:
                        continue

                    P   = create_bakery_matrix(**{
                        k: params[k] for k in
                        ["p_sell_fresh","p_to_discount","p_sell_discount","p_waste"]
                    })
                    sim = run_monte_carlo(P, n_simulations=n_simulations)

                    discount_price = full_price * (1.0 - discount_pct_decimal)
                    final_profit   = (
                        sim["sold_rate"] * discount_price
                    ) - (sim["waste_rate"] * prod_cost)

                    rows.append({
                        "item":                item,
                        "discount_hour":       hour,
                        "day_type":            day_type,
                        "discount_rate":       discount_pct,
                        "elasticity":          round(elasticity, 3),
                        "sell_through_factor": round(adjusted_factor, 3),
                        "sold_rate":           round(sim["sold_rate"], 4),
                        "waste_rate":          round(sim["waste_rate"], 4),
                        "final_profit":        round(final_profit, 4),
                        "baseline_profit":     round(baseline_profit, 4),
                        "profit_gain":         round(final_profit - baseline_profit, 4),
                        "is_profitable":       (final_profit - baseline_profit) > 0
                    })

    sensitivity = pd.DataFrame(rows)
    if sensitivity.empty:
        return sensitivity, pd.DataFrame()

    idx     = sensitivity.groupby(["item", "discount_hour", "day_type"])["profit_gain"].idxmax()
    optimal = sensitivity.loc[idx, [
        "item", "discount_hour", "day_type",
        "discount_rate", "profit_gain", "elasticity"
    ]].copy().rename(columns={
        "discount_rate": "optimal_discount_pct",
        "profit_gain":   "max_profit_gain"
    })

    print("Sensitivity analysis complete.")
    return sensitivity, optimal