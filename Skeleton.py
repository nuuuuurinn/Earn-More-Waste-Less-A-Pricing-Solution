import numpy as np
import pandas as pd

# 1. LOAD DATA
df = pd.read_csv('Datasets(Bakery sales).csv')
df['hour'] = pd.to_datetime(df['time'], format='%I:%M:%S %p').dt.hour

# 2. THE FUNCTIONS (The Logic)
def get_item_baseline(data, item_name):
    item_df = data[data['item'] == item_name]
    if item_df.empty:
        return None, f"Item '{item_name}' not found."

    # Calculate average daily sales and hourly probability
    avg_daily_stock = item_df.groupby('date')['Quantity'].sum().mean()
    hourly_avg = item_df.groupby('hour')['Quantity'].mean()
    
    # We look at core business hours (8am - 5pm) to find the baseline
    p_sell_fresh = hourly_avg[(hourly_avg.index >= 8) & (hourly_avg.index <= 17)].mean() / avg_daily_stock
    return p_sell_fresh, avg_daily_stock
  

# p_sell_fresh: Prob. of selling a fresh item
# p_to_discount: Prob. of an item moving from fresh to the discount window
# p_sell_discount: Prob. of selling once the price is lowered

def create_bakery_matrix(p_sell_fresh, p_to_discount, p_sell_discount, p_waste):
    # Matrix Structure: [Fresh, Discount, Sold, Waste]
    P = np.array([
        # Row 0: What happens to a FRESH item?
        # We subtract sales and aging from 1.0 to find the "Stay Fresh" chance.
        [1 - p_sell_fresh - p_to_discount, p_to_discount, p_sell_fresh, 0.0],  # From Fresh
        
        # Row 1: What happens to a DISCOUNTED item?
        # Now we use 'p_sell_discount' instead of a hard-coded number.
        [0.0, 1 - p_sell_discount - p_waste, p_sell_discount, p_waste],       # From Discount
        
        # Rows 2 & 3: The item is either Sold or Wasted (Game Over states)
        [0.0, 0.0, 1.0, 0.0],                                                 # From Sold (Absorbing)
        [0.0, 0.0, 0.0, 1.0]                                                  # From Waste (Absorbing)
    ])
    
    return P

target_item = 'BAGUETTE'  # <--- CHANGE THIS (e.g., 'CROISSANT', 'PAIN AU CHOCOLAT')
 
  

# Get the "Knowns" from the dataset
p_fresh, stock = get_item_baseline(df, target_item)

if p_fresh:
    # Set your "What-If" numbers
    p_aging = 0.10      # How fast it moves to the discount shelf
    p_discount_sale = 0.8  # How much you think it will sell at a lower price
    p_waste_risk = 0.05     # Risk of it going bad

    # Build the Matrix
    matrix = create_bakery_matrix(p_fresh, p_aging, p_discount_sale, p_waste_risk)

    print(f"Results for: {target_item}")
    print(f"Historical p_sell_fresh: {p_fresh:.4%}")
    print(f"\nMatrix: \n{matrix}")
    
