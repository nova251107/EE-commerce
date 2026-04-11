# Data Engine Concept & Pipeline Breakdown

Understanding what each dataset brings to the table is the foundation of building a high-accuracy ML prototype. Here is the blueprint of what each dataset natively controls in the APEX engine, and how the features map back to your target models.

## 1. Primary Datasets Overview
*   **Clickstream Events:** The behavioral heartbeat. Logs *who, what, and when* interaction happens (views, clicks, carts, purchases).
*   **Product Catalog:** The internal supply limits. Logs intrinsic values of the goods (base price, cost, current inventory, categorical clusters).
*   **User/Session Behavior (Segments):** The buyer proxy. Logs user-level static tendencies (willingness to pay, abandonment rates).
*   **Competitor Pricing (Optional but critical):** The external market proxy. Provides market bounds (so your ML doesn’t accidentally overprice beyond market limits).

## 2. Useful Columns by Model Target
Each model in your system looks for different predictive triggers:

*   **Pricing Model Target:** Predicts maximum extractable value (Willingness to Pay / Conversion vs Price Drop).
    *   *Useful:* `inventory_count` (Scarcity), `price_seen_usd`, `competitor_price`, `willingness_to_pay_multiplier`.
*   **Recommendation Model Target:** Predicts affinity/next-item engagement (CTR / Co-Occurrence).
    *   *Useful:* `category`, `user_segment`, `event_type` (what was clicked lastly), `rating`.
*   **Demand Prediction Target:** Predicts sheer trend velocity to forecast stock depletion.
    *   *Useful:* `hour_of_day`, `day_of_week`, running aggregates of `click` / `add_to_cart` streams, `review_count` (historical proxy).

## 3. Data Dictionary: Column Types
To build your ML matrices effectively, split the columns conceptually so your models don't accidentally train on an ID column (preventing overfitting) or target column (preventing data leakage).

*   **ID Columns (Drop before training):** `user_id`, `sku_id`, `session_id`.
*   **Time Columns (Use cyclically or for ordering, drop before training):** `timestamp`, `date_of_event`.
*   **Feature Columns (Inputs X):**
    *   *Numeric:* `cost_price_usd`, `base_price_usd`, `inventory_count`, `sessions_per_month`.
    *   *Categorical (Require Encoding):* `category`, `sub_category`, `device_type`, `segment`.
    *   *Derived / Temporal:* `hour_sin`, `hour_cos`, `day_sin`, `is_weekend`, `click_velocity_60s`.
*   **Label Columns (Target Y - Must be cleanly dropped from Input X):**
    *   *Pricing:* `label_conversion` (1 if purchased, 0 if not).
    *   *Recommendation:* `label_clicked` (1 if item was engaged with, 0 if only passively viewed).

## 4. Missing Columns You Should Create
For a judge-friendly ML hackathon architecture, your pipeline is currently "reacting" to raw integers. You need derived feature-engineered columns to give the ML models contextual memory:

1.  **`demand_velocity` (Running aggregate):** You cannot just use total clicks. You must engineer a rolling 60-second or 5-minute count of clicks for an item. Why? 50 clicks in the last minute implies a viral spike; 50 clicks over a year is a dead product.
2.  **`session_intent_score`:** A rolling, weighted score calculated client-side or on ingestion. (E.g., Cart Add = +5, Click = +2). A user deeply invested in a session is less price-sensitive.
3.  **`days_to_restock` (Crucial for Pricing):** Add this randomly to your product catalog catalog. Low inventory + short restock ETAs means the price stays stable. Low inventory + long restock ETA triggers massive surge pricing.
4.  **`price_delta_pct`:** The percentage gap between your `base_price` and the `competitor_price`. Models train far better on normalized percentages (e.g., `-0.05` for 5% cheaper) than raw dollar amounts.

*(Note: The critical bug in your `data_preparation.py` that destroyed session logic during random row-sampling has been natively fixed.)*
