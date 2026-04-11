# Explicit Price Parity Guardrail Design

This document details the hackathon-ready design for extracting Price Parity into a standalone, explicit business rule within the APEX dynamic pricing engine. By separating this from the core ML weights, the system becomes highly interpretable and safe.

## 1. Parity Rule Definition
The **Price Parity Rule** enforces that our dynamic price remains competitive without triggering a dangerous "race to the bottom."
*   **Parity Floor:** We will not price lower than `Competitor_Price * 0.95` (We stay at most 5% cheaper than the competition).
*   **Parity Ceiling:** We will not price higher than `Competitor_Price * 1.05` (We charge at most a 5% premium over competition, justified by our brand).
*   **Condition:** This rule only applies when live competitor data exists for the `sku_id`. If omitted, the guardrail allows the price to pass through.

## 2. Clamping / Guardrail Flow
The Parity logic operates as a mathematical clamp:
```python
def apply_parity_guardrail(raw_price, competitor_price):
    if competitor_price is None:
        return raw_price
        
    parity_floor = competitor_price * 0.95
    parity_ceiling = competitor_price * 1.05
    
    # Clamp price within the allowed parity band
    clamped_price = max(parity_floor, min(raw_price, parity_ceiling))
    
    return clamped_price
```

## 3. Where Parity Sits in the Pricing Pipeline
Parity must be treated as a **Market Constraint** and should sit between the raw ML outputs and the absolute **Financial Constraints** (Margin Floors). 

1.  **ML Engine** generates an unconstrained predictive price based on demand/scarcity.
2.  **Parity Guardrail** aligns the price with the outside market reality (Competitor Clamp).
3.  **Business Caps** enforce standard limits (Max 20% discount from MSRP).
4.  **Margin Floor** (Absolute final check to ensure we do not sell at a loss to match a competitor).

## 4. Example Scenarios

| Scenario | Unconstrained ML Price | Competitor Price | Parity Band [95% - 105%] | Clamped Parity Price | Action Taken |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A (Over-priced)** | `$115.00` (High Demand) | `$100.00` | `[$95.00, $105.00]` | **`$105.00`** | **Capped.** ML pushed too high; clamped down to prevent losing sale to competitor. |
| **B (Under-priced)** | `$85.00` (Low Demand)| `$100.00` | `[$95.00, $105.00]` | **`$95.00`** | **Floored.** We were giving away too much margin; clamped up to 5% below competitor. |
| **C (In-Band)** | `$102.00` (Normal) | `$100.00` | `[$95.00, $105.00]` | **`$102.00`** | **Passes.** Price is competitive and reasonable. |

## 5. Explanation Text for Users and Judges

**For Frontend UI (Users):**
*   If Price is pulled down to Parity Ceiling: *"Market Match Applied! We've adjusted our price to keep you competitive."*
*   If Price is pushed up to Parity Floor: *"Best Value Guaranteed! Checking daily competitor metrics."*

**For Pitch/Presentation (Judges):**
*"We architected our guardrails to reflect real enterprise compliance. Instead of hoping our ML model internally learns pricing boundaries, we extracted 'Price Parity' into an explicit, hard-coded micro-rule. Our ML engine generates the optimal demand-based price, but our Parity Guardrail instantly clamps it to a 5% delta against live competitor benchmarks. This completely eliminates the risk of an algorithmic anomaly sparking an unrecoverable price war, while maintaining transparent explainability."*

## 6. Final Pricing Pipeline (All Guardrails in Order)

Below is the exact execution order to guarantee safety during the hackathon demo:

```python
def final_pricing_pipeline(base_price, cost_price, competitor_price, ml_multiplier):
    # Step 1: Raw Predictive Pricing
    raw_price = base_price * ml_multiplier
    
    # Step 2: Price Parity Guardrail (Market Constraint)
    if competitor_price:
        parity_floor = competitor_price * 0.95
        parity_ceiling = competitor_price * 1.05
        raw_price = max(parity_floor, min(raw_price, parity_ceiling))
        
    # Step 3: Business Logic Caps (Sane retail rules)
    max_price = base_price * 1.15  # Max 15% surge
    min_price = base_price * 0.80  # Max 20% discount
    raw_price = max(min_price, min(raw_price, max_price))
    
    # Step 4: Absolute Minimum Margin Floor (Survival Constraint - overrides all)
    hard_floor = cost_price * 1.05 # Always keep 5% margin minimum
    final_price = max(raw_price, hard_floor)
    
    return round(final_price, 2)
```
