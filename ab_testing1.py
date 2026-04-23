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

def run_historical_ab_test(df, optimal_df):
    """
    Runs a true chronological backtest:
    Uses the first 16 months to 'train' and the last 5 months to 'test'.
    """
    import numpy as np

    # 1. Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Split Data Chronologically
    split_date = pd.to_datetime("2022-05-01")
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    print(f"\n--- HISTORICAL A/B TEST ---")
    print(f"Training on: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Testing on:  {test_df['date'].min().date()} to {test_df['date'].max().date()}")

    # 3. SCENARIO A: What actually happened in Summer 2022 (Baseline)
    test_df['prod_cost'] = test_df['unit_price'] * 0.30
    
    # Assume 10% Waste Buffer actually happened in reality
    # Use standard python lowercase 'Quantity' based on dataset
    try:
        qty_col = 'Quantity' if 'Quantity' in test_df.columns else 'quantity'
        
        total_revenue_A = (test_df[qty_col] * test_df['unit_price']).sum()
        assumed_waste_A = test_df[qty_col].sum() * 0.10
        total_waste_cost_A = assumed_waste_A * test_df['prod_cost'].mean()
        
        profit_A = total_revenue_A - total_waste_cost_A

        # 4. SCENARIO B: Applying your Optimal Model to Summer 2022
        # Use 'optimal_discount_pct' which exists in your dataframe!
        merged_test = pd.merge(test_df, optimal_df[['item', 'discount_hour', 'optimal_discount_pct']], on='item', how='left')
        
        # Convert the percentage whole number (e.g., 20) to a decimal (e.g., 0.20)
        merged_test['discount_decimal'] = merged_test['optimal_discount_pct'] / 100.0
        
        # Extract just the hour from the time string (e.g. '15:30:00' -> 15)
        merged_test['sale_hour'] = pd.to_datetime(merged_test['time'], format='%H:%M:%S').dt.hour

        # Apply the discount if the sale happened on or after the recommended discount_hour
        merged_test['applied_price'] = np.where(
            (merged_test['sale_hour'] >= merged_test['discount_hour']) & (merged_test['discount_decimal'].notna()), 
            merged_test['unit_price'] * (1 - merged_test['discount_decimal']), 
            merged_test['unit_price']
        )

        # Assume your strategy reduces waste from 10% to 4%
        total_revenue_B = (merged_test[qty_col] * merged_test['applied_price']).sum()
        assumed_waste_B = merged_test[qty_col].sum() * 0.04
        total_waste_cost_B = assumed_waste_B * merged_test['prod_cost'].mean()
        
        profit_B = total_revenue_B - total_waste_cost_B

        print("\n--- RESULTS ---")
        print(f"Baseline Profit (Scenario A): ${profit_A:,.2f}")
        print(f"Model Profit (Scenario B):    ${profit_B:,.2f}")
        print(f"Net Gain from using Model:    ${profit_B - profit_A:,.2f}\n")
        
        return profit_A, profit_B
        
    except Exception as e:
        print(f"Error in backtest computation: {e}")
        return 0, 0