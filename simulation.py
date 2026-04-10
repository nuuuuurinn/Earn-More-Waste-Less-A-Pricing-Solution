import numpy as np
import pandas as pd
from markov_formulation import STATES, START_STATE, ABSORBING_STATES

def simulate_one_item(transition_matrix, states=STATES, start_state=START_STATE, absorbing_states=ABSORBING_STATES):
    current_state = start_state
    state_index = states.index(current_state)

    while current_state not in absorbing_states:
        probs = transition_matrix[state_index]
        
        # "Flip the coin" → choose next state
        next_state_index = np.random.choice(len(states), p=probs)
        current_state = states[next_state_index]
        state_index = next_state_index

    return current_state


def run_monte_carlo(P, n_simulations=10000):
    sold = 0
    waste = 0

    for _ in range(n_simulations):
        result = simulate_one_item(P)

        if result == "Sold":
            sold += 1
        else:
            waste += 1

    return {
        "sold": sold,
        "waste": waste,
        "sold_rate": sold / n_simulations,
        "waste_rate": waste / n_simulations
    }

def run_all_simulations(results_df, n_simulations=10000):
    simulation_results = []

    for _, row in results_df.iterrows():
        P = row["transition_matrix"]

        sim_result = run_monte_carlo(P, n_simulations)

        simulation_results.append({
            "item": row["item"],
            "discount_hour": row["discount_hour"],
            "sold_rate": sim_result["sold_rate"],
            "waste_rate": sim_result["waste_rate"]
        })

    return pd.DataFrame(simulation_results)