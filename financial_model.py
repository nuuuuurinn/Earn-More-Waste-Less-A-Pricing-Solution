import pandas as pd


# =========================================================
# 7. Calculate Financial Impact
# =========================================================

def run_financial_impact_model(simulation_df, results_df, item_prices, cost_ratio=0.3, discount_ratio=0.5): #possible to change cost and discount assumptions here
    # Merge simulation rates with historical fresh sales probabilities
    results = pd.merge(simulation_df, results_df[['item', 'discount_hour', 'p_sell_fresh', 'p_to_discount']], on=['item', 'discount_hour'])
    
    results['full_price'] = results['item'].map(item_prices)
    results['prod_cost'] = results['full_price'] * cost_ratio
    results['discount_price'] = results['full_price'] * discount_ratio

    # 1. BASELINE: Throw away everything that doesn't sell fresh
    # Profit = (Fresh Sales * Price) - (Unsold Items * Cost)
    results['baseline_profit'] = (results['p_sell_fresh'] * results['full_price']) - \
                                 (results['p_to_discount'] * results['prod_cost'])

    # 2. INTERVENTION: Use the simulation results (sold_rate) 
    # This includes both Fresh and Discounted sales
    # Profit = (Total Sold * Discount Price) - (Simulation Waste * Cost)
    # Note: Using discount_price here for ALL sales simulates aggressive cannibalization
    results['final_profit'] = (results['sold_rate'] * results['discount_price']) - \
                               (results['waste_rate'] * results['prod_cost'])
    
    # 3. THE TIPPING POINT: Intervention Profit - Baseline Profit
    results['profit_gain'] = results['final_profit'] - results['baseline_profit']
    results['is_profitable'] = results['profit_gain'] > 0

    results = results.sort_values(by='profit_gain', ascending=False)
    
    return results

