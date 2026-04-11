# A/B Testing Metrics Redesign

This document outlines the rigorous, statistically sound design for APEX's A/B Testing framework. By cleaning up the denominators and strictly adhering to session-level causal boundaries, we ensure the hackathon demo stands up to deep scrutiny from data-literate judges.

## 1. Correct Metric Definitions
Mixing up denominators is the biggest risk in prototype engines. Here is the mathematically correct standard:

*   **Conversion Rate (CVR):** $\frac{\text{Conversions}}{\text{Unique Sessions}}$
    *   *Definition:* The percentage of sessions that resulted in at least one purchase. (Do not use page views or total clicks as the denominator).
*   **Average Order Value (AOV):** $\frac{\text{Total Revenue}}{\text{Total Number of Purchases}}$
    *   *Definition:* How much users spend *on average* when they decide to buy.
*   **Revenue Per Session (RPS):** $\frac{\text{Total Revenue}}{\text{Total Unique Sessions}}$
    *   *Definition:* The ultimate North Star metric. RPS combines both CVR and AOV into one number, showing the true financial yield of your traffic.
*   **Total Revenue:** The sheer sum of all purchase event values for a variant.

## 2. Session-Level Schema
To track metrics correctly without double-counting, you **must track data at the session grain**, not the event grain. 

```python
# A dictionary holding aggregated state for each session
SESSION_SCHEMA = {
    "session_id": "sess_8941",
    "assigned_variant": "B",       # 'A' (Static) or 'B' (Dynamic)
    "first_interaction": 1698230193, # Timestamp
    "total_events": 14,
    "has_purchase": True,          # Boolean flag (critical for CVR)
    "num_purchases": 2,            # Number of items/orders
    "total_revenue": 145.98        # Summed value of purchases
}
```

## 3. Aggregation Logic
When asked for dashboard stats, loop over the session objects, calculate raw sums, and then apply final divisions:

```python
def aggregate_metrics(sessions_list):
    total_sessions = len(sessions_list)
    converted_sessions = sum(1 for s in sessions_list if s["has_purchase"])
    total_orders = sum(s["num_purchases"] for s in sessions_list)
    total_revenue = sum(s["total_revenue"] for s in sessions_list)

    return {
        "Total Sessions": total_sessions,
        "Total Revenue": total_revenue,
        "Conversion Rate": (converted_sessions / total_sessions) if total_sessions else 0.0,
        "Average Order Value": (total_revenue / total_orders) if total_orders else 0.0,
        "Revenue Per Session": (total_revenue / total_sessions) if total_sessions else 0.0
    }
```

## 4. Experiment Logging Flow
1.  **Incoming Event:** User `sess_8941` sends a click via `POST /event`.
2.  **Assignment (Crucial):** If `sess_8941` is missing from the A/B tracker, dynamically assign it precisely once using deterministic hashing. `group = "A" if hash("sess_8941") % 2 == 0 else "B"`.
3.  **Update Counters:** Increment `total_events`.
4.  **Purchase Logic:** If `event_type == 'purchase'`, set `has_purchase = True`, update `num_purchases += 1`, and add the `final_price` to `total_revenue`.

## 5. Dashboard Metrics to Show
On the UI, display a side-by-side comparison matrix:
*   **Total Sessions**: $Control=1,200$ | $Treatment=1,215$ *(Shows the traffic split is balanced and healthy)*
*   **Conversion Rate**: $Control=3.8\%$ | $Treatment=3.9\%$ *(+0.1% lift)*
*   **AOV**: $Control=\$42.00$ | $Treatment=\$46.80$ *(+11.4% lift, highlights pricing algo success)*
*   **RPS:** $Control=\$1.60$ | $Treatment=\$1.82$ *(+13.7% True Financial Lift!)*

## 6. Common Mistakes to Avoid (Checklist)
*   🚫 **Variant Flipping:** Assigning users arbitrarily so they jump between Group A and B on page refreshes. (Fix: Hash the `session_id`).
*   🚫 **Inflated Session Denominator:** Incrementing your "Total Sessions" variable on *every* click event instead of counting unique session IDs. This severely artificially lowers CVR.
*   🚫 **AOV vs RPS Confusion:** Dividing Total Revenue by Total Sessions and calling it AOV. That is RPS.

## 7. How to Explain Statistical Rigor in Demo
*To the Judges:* 
"When building our ML pricing engine, we recognized that bad tracking data destroys the credibility of dynamic models. We rebuilt our A/B tracking layer using rigorous **Session-Level Allocation**. Instead of relying on raw click streams, we bind users deterministically to treatment groups and calculate our primary metric as **Revenue Per Session (RPS)**. RPS is crucial because it balances both Conversion Rate and Average Order Value—proving that our algorithm doesn't just raise prices at the cost of abandoned carts, but genuinely increases the final net yield of our traffic."
