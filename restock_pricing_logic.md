# Restock-Aware Dynamic Pricing Logic

This document details the hackathon-ready design for integrating a Supply-Chain Restock Timeline into the APEX dynamic pricing engine.

## 1. Clear Decision Logic
The decision mechanism uses a tiered evaluation combining **current stock level** and **restock ETA**:
*   **Healthy Stock:** If `inventory > 20`, ignore the restock timeline. Supply is sufficient.
*   **Low Stock + Short Restock (< 3 days):** Imminent restock. Apply a minimal scarcity effect. We don't want to kill the conversion rate right before new inventory arrives.
*   **Low Stock + Medium Restock (4-14 days):** Supply is tightening. Apply a moderate scarcity premium to slow sales slightly and capture higher margins.
*   **Low Stock + Long Restock (> 14 days):** Severe scarcity. Apply a maximum scarcity premium to heavily throttle the burn rate and maximize profit on the remaining units.

## 2. Rule-Based Pricing Formula Design
A simplified, explicit formula to append to the existing pricing algorithm:

```python
# Constants
LOW_STOCK_THRESHOLD = 20
PREMIUM_SHORT = 0.02  # +2%
PREMIUM_MEDIUM = 0.05 # +5%
PREMIUM_LONG = 0.10   # +10%

def calculate_scarcity_multiplier(inventory, days_to_restock):
    if inventory > LOW_STOCK_THRESHOLD:
        return 1.0  # No premium
    
    if days_to_restock <= 3:
        return 1.0 + PREMIUM_SHORT
    elif days_to_restock <= 14:
        return 1.0 + PREMIUM_MEDIUM
    else:
        return 1.0 + PREMIUM_LONG

# Integration into final price:
# final_price = base_price * demand_multiplier * scarcity_multiplier * user_segment_multiplier
```

## 3. What New Field Should Be Stored
A single new explicit field should be added to the `PRODUCTS` dictionary (in `server.py` data loading):
*   **Field Name:** `days_to_restock` (Type: `int`)
*   **Storage:** Added to each product record. 
*   **Hackathon Tip:** You can simulate this during your `_load_product_catalog` routine by generating a random integer between `1` and `30` for products with low inventory.

## 4. How This Should Affect Final Price
This acts as a localized **Scarcity Surge Multiplier**. Unlike demand velocity (which is driven by users), this is a supply-side constraint. It will push the final dynamic price *upwards* (by 2% to 10%) independent of user behavior, acting as a natural brake on inventory depletion while capturing higher willingness-to-pay.

## 5. Example Scenarios
| Scenario | Current Inventory | Days to Restock | Scarcity Premium | Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| **A** | `150` | `28 days` | **+0%** | Inventory is healthy. Restock ETA doesn't matter yet. |
| **B** | `12` | `2 days` | **+2%** | Stock is low, but the truck arrives soon. Keep sales flowing, tiny markup. |
| **C** | `8` | `10 days` | **+5%** | Moderate risk of stock-out. Increase price to slow burn rate. |
| **D** | `3` | `21 days` | **+10%** | Critical supply shock. Maximize margins on remaining stock. |

## 6. Explanation Text to Show in Frontend
To build urgency and trust on the frontend (Phase 6 Explainability UI), map the state to these strings:
*   **Days <= 3:** *"Low stock, but more arriving in {N} days! Lock in your price now."*
*   **Days 4-14:** *"Only {X} left! Restock expected in {N} days. High demand expected."*
*   **Days > 14:** *"🔥 Only {X} left! Next shipment is {N} weeks away. Secure yours before prices rise further."*

## 7. Judge-Friendly Wording (For Demo/Pitch)
*"We upgraded our ML pricing engine to be true Supply-Chain Aware. We recognized that low inventory isn't a static problem—it depends heavily on when the next truck arrives. By injecting a real-time 'Restock ETA' metric, our algorithm automatically throttles pricing. It prevents premature stock-outs when shipments are delayed by surging prices heavily, but enables aggressive sell-throughs when replenishment is inbound. This bridges the gap between marketing demand and supply chain reality."*
