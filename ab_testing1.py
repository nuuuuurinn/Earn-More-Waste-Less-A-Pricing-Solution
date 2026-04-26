import pandas as pd

# =========================================================
# PERSON 1: PREPARE A/B TESTING FILES
# =========================================================

def prepare_ab_testing_data(financial_results: pd.DataFrame):
    """
    Creates:
    1. results_A.csv  -> baseline model output
    2. results_B.csv  -> dynamic model output
    3. merged_ab_results.csv -> comparison dataset for Person 2 and Person 3
    """

    # -----------------------------
    # Model A = Baseline / Control
    # -----------------------------
    results_A = financial_results[[
        "item",
        "discount_hour",
        "baseline_profit",
        "p_sell_fresh",
        "p_to_discount"
    ]].copy()

    results_A = results_A.rename(columns={
        "baseline_profit": "profit",
        "p_sell_fresh": "sellthrough",
        "p_to_discount": "waste"
    })

    results_A["model"] = "A"

    # -----------------------------
    # Model B = Dynamic / Treatment
    # -----------------------------
    results_B = financial_results[[
        "item",
        "discount_hour",
        "final_profit",
        "sold_rate",
        "waste_rate"
    ]].copy()

    results_B = results_B.rename(columns={
        "final_profit": "profit",
        "sold_rate": "sellthrough",
        "waste_rate": "waste"
    })

    results_B["model"] = "B"

    # -----------------------------
    # Save separate outputs
    # -----------------------------
    results_A.to_csv("results_A.csv", index=False)
    results_B.to_csv("results_B.csv", index=False)

    # -----------------------------
    # Merge A and B for comparison
    # -----------------------------
    merged = pd.merge(
        results_A,
        results_B,
        on=["item", "discount_hour"],
        suffixes=("_A", "_B"),
        how="inner"
    )

    if merged.empty:
        raise ValueError("Merged dataset is empty. Check matching item and discount_hour values.")

    # -----------------------------
    # Calculate differences (B - A)
    # -----------------------------
    merged["profit_diff"] = merged["profit_B"] - merged["profit_A"]
    merged["waste_diff"] = merged["waste_B"] - merged["waste_A"]
    merged["sellthrough_diff"] = merged["sellthrough_B"] - merged["sellthrough_A"]

    # -----------------------------
    # Optional interpretation labels
    # -----------------------------
    def higher_is_better(x):
        if x > 0:
            return "B better"
        elif x < 0:
            return "A better"
        return "Same"

    def lower_is_better(x):
        if x < 0:
            return "B better"
        elif x > 0:
            return "A better"
        return "Same"

    merged["profit_result"] = merged["profit_diff"].apply(higher_is_better)
    merged["waste_result"] = merged["waste_diff"].apply(lower_is_better)
    merged["sellthrough_result"] = merged["sellthrough_diff"].apply(higher_is_better)

    # -----------------------------
    # Save merged comparison file
    # -----------------------------
    merged.to_csv("merged_ab_results.csv", index=False)

    print("Saved results_A.csv")
    print("Saved results_B.csv")
    print("Saved merged_ab_results.csv")

    print("\nQuick Summary:")
    print("Average profit difference (B - A):", merged["profit_diff"].mean())
    print("Average waste difference (B - A):", merged["waste_diff"].mean())
    print("Average sell-through difference (B - A):", merged["sellthrough_diff"].mean())

    return results_A, results_B, merged

## nurin's part

def run_historical_ab_test(df, sensitivity_df, results_df):
    """
    Uses the project's actual Markov and Simulation outputs to 
    calculate profit, removing all hardcoded placeholders.
    """
    import numpy as np

    # 1. Prepare Data
    df['date'] = pd.to_datetime(df['date'])
    split_date = pd.to_datetime("2022-01-01")
    test_df = df[df['date'] >= split_date].copy()
    
    # Identify the optimal strategy for each item from the sensitivity analysis
    # We find the row with the highest profit_gain for each item
    idx = sensitivity_df.groupby(['item', 'day_type'])['profit_gain'].idxmax()
    best_strategies = sensitivity_df.loc[idx].copy()

    # 2. SCENARIO A: Baseline
    # Use the cost_ratio from your financial_model (0.3)
    test_df['prod_cost'] = test_df['unit_price'] * 0.30
    
    # Merge with results_df to get the ACTUAL Markov 'p_to_discount' (Estimated Baseline Waste)
    # We take the mean p_to_discount for the item to represent its general waste behavior
    baseline_waste_map = results_df.groupby('item')['p_to_discount'].mean()
    
    total_revenue_A = (test_df['quantity'] * test_df['unit_price']).sum()
    total_waste_A = (test_df['quantity'].sum() * test_df['item'].map(baseline_waste_map)).sum()
    total_waste_cost_A = total_waste_A * test_df['prod_cost'].mean()
    
    profit_A = total_revenue_A - total_waste_cost_A

    # 3. SCENARIO B: Model (Applying Simulation Results)
    # Merge test data with the specific simulation results for the optimal strategy
    merged_test = pd.merge(test_df, best_strategies[['item', 'discount_hour', 'discount_rate', 'waste_rate']], on='item', how='left')
    
    merged_test['sale_hour'] = pd.to_datetime(merged_test['time'], format='%H:%M:%S').dt.hour
    merged_test['discount_decimal'] = merged_test['discount_rate'] / 100.0

    # Apply the discount price to items sold after the trigger hour
    merged_test['applied_price'] = np.where(
        (merged_test['sale_hour'] >= merged_test['discount_hour']) & (merged_test['discount_decimal'].notna()), 
        merged_test['unit_price'] * (1 - merged_test['discount_decimal']), 
        merged_test['unit_price']
    )

    total_revenue_B = (merged_test['quantity'] * merged_test['applied_price']).sum()
    
    # Use the actual 'waste_rate' calculated by your Monte Carlo simulation
    total_waste_B = (merged_test['quantity'].sum() * merged_test['waste_rate']).sum()
    total_waste_cost_B = total_waste_B * merged_test['prod_cost'].mean()
    
    profit_B = total_revenue_B - total_waste_cost_B

    print(f"\n--- HISTORICAL A/B TEST (Calculated Results) ---")
    print(f"Baseline Profit (Scenario A): ${profit_A:,.2f}")
    print(f"Model Profit (Scenario B):    ${profit_B:,.2f}")
    print(f"Net Gain from using Model:    ${profit_B - profit_A:,.2f}\n")
    
    return profit_A, profit_B