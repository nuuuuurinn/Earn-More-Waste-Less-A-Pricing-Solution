import data_processing
import markov_formulation
import simulation
import financial_model
import server
import ab_testing1

def run_project():
    # STAGE 1: Load data (now includes day_type column)
    df = data_processing.load_data("Datasets.xlsx")

    # STAGE 2-4: Markov → Simulation → Financial model (segmented by day_type)
    results_df  = markov_formulation.evaluate_all_items_with_proxy(df, [15, 16, 17])
    sim_df      = simulation.run_all_simulations(results_df)
    item_prices = df.groupby("item")["unit_price"].mean()
    final_report = financial_model.run_financial_impact_model(sim_df, results_df, item_prices)

    # STAGE 5: Estimate price elasticity per item × day_type
    print("Estimating price elasticities...")
    elasticity_dict = markov_formulation.estimate_all_elasticities(df)

    # STAGE 6: Sensitivity analysis with elasticity + day_type
    sensitivity_df, optimal_df = financial_model.run_sensitivity_analysis(
        df=df,
        results_df=results_df,
        item_prices=item_prices,
        elasticity_dict=elasticity_dict,
        n_simulations=1000
    )

    # A/B testing outputs
    results_A, results_B, merged_ab_results = ab_testing1.prepare_ab_testing_data(final_report)
    
    rofit_A, profit_B = ab_testing1.run_historical_ab_test(df, sensitivity_df, results_df)

    return final_report, sensitivity_df, optimal_df

if __name__ == "__main__":
    final_report, sensitivity_df, optimal_df = run_project()
    server.set_data(final_report, sensitivity_df, optimal_df)
    server.app.run(debug=False, port=5000)