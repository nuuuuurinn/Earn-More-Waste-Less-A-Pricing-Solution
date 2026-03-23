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

    # From Fresh: 
    # stays Fresh, move to Discounted, or become Sold 
    fresh_stay = 1 - (p_sell_fresh + p_to_discount) #letfover probability for items to stay fresh

    # From Discounted: 
    # stays Discounted, become Sold or become Waste
    discount_stay = 1 - (p_sell_discount + p_waste) #leftover probability for items to stay discounted

    P = np.array([
        [fresh_stay, p_to_discount, p_sell_fresh, 0.0], #what if item is currently fresh
        [0.0, discount_stay, p_sell_discount, p_waste], #what if item is currently discounted
        [0.0, 0.0, 1.0, 0.0],   # Sold is absorbing
        [0.0, 0.0, 0.0, 1.0]    # Waste is absorbing
    ])

    return P 