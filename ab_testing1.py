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