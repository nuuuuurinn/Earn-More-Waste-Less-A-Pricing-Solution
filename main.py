import data_processing
import markov_formulation
import simulation
import financial_model
import dashboard

def run_project():
    # STAGES 1-4 (The Math)
    df = data_processing.load_data("Datasets.xlsx")
    results_df = markov_formulation.evaluate_all_items_with_proxy(df, [15, 16, 17])
    sim_df = simulation.run_all_simulations(results_df)
    item_prices = df.groupby('item')['unit_price'].mean()
    final_report = financial_model.run_financial_impact_model(sim_df, results_df, item_prices)

    # STAGE 5 (The Delivery)
    dashboard.print_executive_summary(final_report)
    dashboard.create_managerial_dashboard(final_report)

if __name__ == "__main__":
    run_project()