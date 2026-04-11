# APEX ML Pipeline Design — Dataset Analysis, Cleaning & Feature Engineering

Comprehensive reference document for hackathon judges.
Based on the 4 real datasets from Problem Statement 3.

---

## Part 1: Dataset Analysis & Column Classification

### 1A. Clickstream Events (10.2M rows × 21 cols)
The behavioral core. Every user interaction is logged here.

| Column | Type | Role | Notes |
|--------|------|------|-------|
| `event_id` | string | 🔑 ID | Unique per event. Drop before training. |
| `session_id` | string | 🔑 ID | Groups events into browsing sessions. Drop before training. |
| `user_id` | string | 🔑 ID | Links to user segments. Drop before training. |
| `event_type` | string | 📊 Feature / 🎯 Label source | 9 types: page_view, product_view, search, add_to_cart, add_to_wishlist, checkout_start, purchase, remove_from_cart, page_exit |
| `timestamp` | string (ISO-8601) | ⏰ Time | Range: 2024-01-01 to 2024-06-29. Parse to datetime. |
| `sku_id` | string | 🔑 ID (join key) | Links to product catalog. |
| `category` | string | 📊 Feature | Product category at interaction time. |
| `page_url` | string | 📊 Feature (low priority) | Can extract page-type signal. |
| `referral_source` | string | 📊 Feature | organic, paid, social, email, direct. |
| `device_type` | string | 📊 Feature | desktop, mobile, tablet. |
| `user_segment` | string | 📊 Feature | Denormalized from user profiles. |
| `price_seen_usd` | float | 📊 Feature | The price displayed to the user at interaction time. |
| `quantity` | int | 📊 Feature | Relevant for purchase events. |
| `search_query` | string | 📊 Feature (NLP) | Raw text; useful for intent but complex to encode. |
| `session_duration_s` | float | 📊 Feature | Total session length in seconds. |
| `scroll_depth_pct` | float | 📊 Feature | How far the user scrolled (0–100). Engagement signal. |
| `time_on_page_s` | float | 📊 Feature | Dwell time. Higher = more interest. |
| `is_mobile` | bool | 📊 Feature | Binary device flag. |
| `ab_group` | string | 📊 Feature | A/B test assignment (control vs treatment). |
| `hour_of_day` | int (0–23) | ⏰ Time → Feature | Encode cyclically (sin/cos). |
| `day_of_week` | int (0–6) | ⏰ Time → Feature | Encode cyclically (sin/cos). |

---

### 1B. Product Catalog (5,368 rows × 18 cols)
The supply-side truth. Intrinsic product attributes.

| Column | Type | Role | Notes |
|--------|------|------|-------|
| `sku_id` | string | 🔑 ID (join key) | Primary key. |
| `product_name` | string | 🔑 ID | Human label. Drop before training. |
| `category` | string | 📊 Feature | Encode with LabelEncoder. |
| `subcategory` | string | 📊 Feature | Finer product grouping. |
| `brand` | string | 📊 Feature | High cardinality — encode or group. |
| `base_price_usd` | float | 📊 Feature | MSRP. Core pricing input. |
| `cost_price_usd` | float | 📊 Feature | Floor for margin guardrail. |
| `current_price_usd` | float | ⚠️ **LEAKAGE RISK** | This IS the price outcome. Exclude from pricing model inputs. |
| `min_price_usd` | float | 📊 Feature | Business floor constraint. |
| `max_price_usd` | float | 📊 Feature | Business ceiling constraint. |
| `inventory_count` | int | 📊 Feature | Scarcity signal. Range: 0–500. |
| `restock_days` | float | 📊 Feature | Days until next shipment. **3,658 nulls (68%)** — fill with median (16). |
| `avg_rating` | float | 📊 Feature | Quality signal. Range: 1.0–5.0. |
| `review_count` | int | 📊 Feature | Popularity proxy. |
| `weight_kg` | float | 📊 Feature | Shipping cost proxy. |
| `is_active` | bool | 📊 Feature / Filter | Drop inactive products from training. |
| `launch_date` | string | ⏰ Time | Product age signal. |
| `tags` | string (JSON list) | 📊 Feature (NLP) | Optional multi-label encoding. |

---

### 1C. User Segment Profiles (500K rows × 19 cols)
Static user-level attributes. Join by `user_id`.

| Column | Type | Role | Notes |
|--------|------|------|-------|
| `user_id` | string | 🔑 ID (join key) | Primary key. |
| `segment` | string | 📊 Feature | budget_conscious, premium, etc. Encode. |
| `country` | string | ❌ **EXCLUDE** | Fairness: geographic price discrimination risk. |
| `device_type` | string | ❌ **EXCLUDE** | Fairness: device-based pricing risk. |
| `os` | string | ❌ **EXCLUDE** | Fairness: OS-based pricing risk. |
| `primary_referral` | string | 📊 Feature | Acquisition channel. |
| `registration_date` | string | ⏰ Time | Account age signal. |
| `last_seen_date` | string | ⏰ Time | Recency signal. |
| `lifetime_value_usd` | float | ⚠️ **LEAKAGE RISK** | Encodes future spend. Use for analysis only, not real-time features. |
| `avg_order_value_usd` | float | 📊 Feature | Spending tendency. |
| `sessions_per_month` | float | 📊 Feature | Activity level → session_intensity. |
| `purchase_frequency` | float | 📊 Feature | Conversion tendency. |
| `cart_abandonment_rate` | float | 📊 Feature | Price sensitivity proxy. |
| `willingness_to_pay_multiplier` | float | 📊 Feature | Direct WTP signal. |
| `preferred_categories` | string (JSON) | 📊 Feature | Category affinity. |
| `email_opt_in` | bool | 📊 Feature (low priority) | Marketing engagement. |
| `push_opt_in` | bool | 📊 Feature (low priority) | Marketing engagement. |
| `age_group` | string | ❌ **EXCLUDE** | Fairness: age discrimination. |
| `gender` | string | ❌ **EXCLUDE** | Fairness: gender discrimination. |

---

### 1D. Competitor Pricing Feed (1.69M rows × 10 cols)
External market benchmark. Join by `sku_id` + `date`.

| Column | Type | Role | Notes |
|--------|------|------|-------|
| `date` | string | ⏰ Time (join key) | Use latest record per SKU. |
| `sku_id` | string | 🔑 ID (join key) | Links to product catalog. |
| `competitor` | string | 📊 Feature | Which competitor. |
| `competitor_price` | float | 📊 Feature | Core parity benchmark. |
| `our_base_price` | float | 📊 Feature | Our MSRP at that date (for delta calc). |
| `price_delta_pct` | float | 📊 Feature | Pre-computed gap (%). Directly usable. |
| `is_on_promotion` | bool | 📊 Feature | Competitor running a sale? |
| `promo_discount_pct` | float | 📊 Feature | How deep the competitor's discount is. |
| `in_stock` | bool | 📊 Feature | Competitor stock status. |
| `scraped_at` | string | ⏰ Time | Data freshness indicator. |

---

### 1E. Column Usefulness by Model

| Column / Derived Feature | Pricing Model | Recommendation Model | Demand Prediction |
|--------------------------|:---:|:---:|:---:|
| `inventory_count` | ✅ scarcity | — | ✅ supply constraint |
| `base_price_usd` | ✅ anchor | — | ✅ price sensitivity |
| `competitor_price` | ✅ parity | — | ✅ market position |
| `price_delta_pct` | ✅ gap signal | — | ✅ competitiveness |
| `restock_days` | ✅ scarcity timeline | — | ✅ supply risk |
| `demand_velocity` (derived) | ✅ surge pricing | — | ✅ core target proxy |
| `event_type` | ✅ conversion signal | ✅ engagement signal | ✅ action weight |
| `category` | ✅ segment pricing | ✅ affinity | ✅ category trends |
| `avg_rating` | — | ✅ quality filter | ✅ virality proxy |
| `review_count` | — | ✅ popularity | ✅ trend velocity |
| `session_duration_s` | ✅ intent depth | ✅ engagement | — |
| `scroll_depth_pct` | ✅ interest signal | ✅ engagement | — |
| `cart_abandonment_rate` | ✅ price sensitivity | — | — |
| `willingness_to_pay` | ✅ WTP ceiling | — | — |
| `hour_of_day`, `day_of_week` | ✅ time-of-day pricing | — | ✅ temporal patterns |

---

## Part 2: Data Cleaning Pipeline

### Step-by-Step Workflow

#### Step 1 — Detect Missing Values
| Dataset | Column | Missing Count | Strategy |
|---------|--------|:---:|---------|
| Products | `restock_days` | 3,658 (68%) | Fill with **median = 16 days** (conservative). |
| Clickstream | — | 0 | No action needed. |
| Users | — | 0 | No action needed. |
| Competitor | — | 0 | No action needed. |

#### Step 2 — Remove Duplicates
| Dataset | Dedup Key | Strategy |
|---------|----------|---------|
| Clickstream | `event_id` | Exact dedup on primary key. |
| Products | `sku_id` | Exact dedup on primary key. |
| Users | `user_id` | Exact dedup on primary key. |
| Competitor | `date` + `sku_id` + `competitor` | Composite key dedup. |

#### Step 3 — Standardize Event Types
Map all event names to a canonical set:
```
Canonical:  page_view, product_view, search, add_to_cart,
            add_to_wishlist, checkout_start, purchase,
            remove_from_cart, page_exit

Aliases:    view → page_view, click → product_view,
            cart → add_to_cart, buy/bought → purchase,
            wishlist → add_to_wishlist, checkout → checkout_start,
            exit → page_exit
```

#### Step 4 — Validate IDs
- Ensure `user_id`, `sku_id`, `session_id` are non-null, non-empty strings.
- **Cross-reference check:** Every `sku_id` in clickstream must exist in product catalog. Drop orphan events.
- **Cross-reference check:** Every `user_id` in clickstream should ideally exist in user segments (soft warning, don't drop).

#### Step 5 — Fix Timestamp Format
- Raw format: ISO-8601 strings (`"2024-04-08T06:42:57Z"`).
- Parse to `datetime64[ns, UTC]` using `pd.to_datetime(..., utc=True)`.
- Validate range: expect 2024-01-01 through 2024-06-29.
- Clamp any future timestamps to `now()`.

#### Step 6 — Detect & Cap Outliers
| Column | Floor | Ceiling | Strategy |
|--------|:---:|:---:|---------|
| `base_price_usd` | $1.00 | $5,000 | Winsorize (clip). |
| `cost_price_usd` | $1.00 | $5,000 | Winsorize. Also clamp if cost > base. |
| `competitor_price` | $1.00 | $5,000 | Winsorize. |
| `price_seen_usd` | $0.50 | $5,000 | Winsorize. |
| `inventory_count` | 0 | 1,000 | Cap at warehouse max. |
| Anomaly: `cost > base` | — | — | Reset cost = base × 0.6. |
| Anomaly: `min > max` | — | — | Swap values. |

#### Step 7 — Prevent Data Leakage
| Risk | Column | Mitigation |
|------|--------|-----------|
| Target leaks into features | `current_price_usd` | **Exclude** from pricing model inputs (it is the outcome). |
| Future information | `lifetime_value_usd` | **Exclude** from real-time features (encodes future spend). |
| Temporal contamination | Train/test split | Apply **temporal split** (first 80% by time → train, last 20% → test). Never shuffle. |
| Fairness violation | `age_group`, `gender`, `country`, `os` | **Exclude** from pricing model per Zero-Discrimination Policy. |

---

## Part 3: Feature Engineering for Dynamic Pricing

### Goal
Predict **conversion probability** (binary: purchased or not) given product context, user signals, and market state.

### 3A. Engineered Features

#### Demand Signals (from Clickstream aggregation)
| Feature | Formula | Latency | Notes |
|---------|---------|---------|-------|
| `demand_velocity_60s` | Count of clicks on SKU in last 60s | Real-time (O(1) deque) | Short-window spike detector. |
| `demand_velocity_300s` | Count of clicks on SKU in last 5min | Real-time | Sustained interest detector. |
| `cart_velocity_60s` | Count of add_to_cart on SKU in last 60s | Real-time | Stronger intent signal (weighted 5×). |
| `demand_score_proxy` | `0.5 × (1 − inventory_ratio) + 0.3 × review_popularity + 0.2 × rating_score` | Batch | Composite scarcity + popularity. Range: 0–1. |

#### Inventory & Supply Signals (from Product Catalog)
| Feature | Formula | Notes |
|---------|---------|-------|
| `inventory_ratio` | `inventory_count / max(inventory_count)` | Normalized 0–1. Low = scarce. |
| `scarcity_flag` | `1 if inventory_count < 20 else 0` | Binary trigger for pricing rules. |
| `restock_urgency` | `1 / max(restock_days, 1)` | Higher when restock is far away. |

#### Price & Market Signals (from Catalog + Competitor)
| Feature | Formula | Notes |
|---------|---------|-------|
| `price_delta_pct` | `(our_price − competitor_price) / competitor_price` | Negative = we're cheaper. Range: typically −0.3 to +0.3. |
| `margin_ratio` | `(base_price − cost_price) / base_price` | Available profit margin. |
| `price_seen_norm` | `(price_seen − min) / (max − min)` | Min-max normalized. |

#### User & Session Signals (from Clickstream + User Profiles)
| Feature | Formula | Notes |
|---------|---------|-------|
| `session_intensity` | `sessions_per_month / max(sessions_per_month)` | Normalized activity level. 0–1. |
| `click_frequency` | `purchase_frequency / max(purchase_frequency)` | Normalized buying tendency. 0–1. |
| `price_sensitivity` | `cart_abandonment_rate` (clipped 0–1) | High abandonment = very price sensitive. |
| `intent_score` | `Σ(event_weight × recency_decay)` over last 10 session actions | Weighted sum. Purchase=10, Cart=5, Click=2, View=1. |
| `scroll_engagement` | `scroll_depth_pct / 100` | Normalized. Deep scroll = high interest. |
| `dwell_signal` | `min(time_on_page_s / 120, 1.0)` | Capped at 2 minutes. |

#### Time Signals (from `hour_of_day`, `day_of_week`)
| Feature | Formula | Notes |
|---------|---------|-------|
| `hour_sin` | `sin(2π × hour / 24)` | Cyclical encoding preserving hour 23 ≈ hour 0. |
| `hour_cos` | `cos(2π × hour / 24)` | Pair with sin for full circle. |
| `day_sin` | `sin(2π × day / 7)` | Weekly cycle. |
| `day_cos` | `cos(2π × day / 7)` | Pair with sin. |
| `is_peak_hour` | `1 if 18 ≤ hour ≤ 22 else 0` | Binary flag for evening shopping surge. |
| `is_weekend` | `1 if day ∈ {5, 6} else 0` | Weekend shopping pattern. |

### 3B. Normalization Plan
| Strategy | Applied To | Method |
|----------|-----------|--------|
| Min-Max → [0, 1] | All price columns, inventory, review_count | `(x − min) / (max − min)` with stored min/max for inference. |
| Clip + Scale | Velocity features | Clip to 99th percentile, then min-max. Prevents spike distortion. |
| As-is (already bounded) | `price_sensitivity`, `intent_score`, `rating_score` | Already in 0–1 range. |

### 3C. Encoding Plan
| Column | Cardinality | Method |
|--------|:-----------:|--------|
| `category` | ~20 | LabelEncoder → integer. |
| `subcategory` | ~80 | LabelEncoder → integer. |
| `event_type` | 9 | LabelEncoder → integer. |
| `segment` | ~5 | LabelEncoder → integer. |
| `ab_group` | 2 | LabelEncoder (A=0, B=1). |
| `brand` | ~500 | Target encoding or drop (too high cardinality for one-hot). |

### 3D. Final Feature Vector (24 dimensions)

```
X = [
  # Demand (4)
  demand_velocity_60s, demand_velocity_300s,
  cart_velocity_60s, demand_score_proxy,

  # Supply (3)
  inventory_ratio, scarcity_flag, restock_urgency,

  # Price & Market (3)
  price_delta_pct, margin_ratio, price_seen_norm,

  # User & Session (6)
  session_intensity, click_frequency, price_sensitivity,
  intent_score, scroll_engagement, dwell_signal,

  # Time (6)
  hour_sin, hour_cos, day_sin, day_cos,
  is_peak_hour, is_weekend,

  # Categorical (2)
  category_encoded, segment_encoded,
]

y = label_conversion   # 1 = purchased, 0 = did not
```

### 3E. Latency Budget
| Component | Target | Strategy |
|-----------|--------|---------|
| Feature extraction | < 5ms | O(1) deque append/evict. Pre-computed rolling windows. |
| Demand scoring | < 10ms | Cached per-SKU. Refresh every 1–3 seconds in background. |
| Model inference | < 20ms | SGDClassifier or lightweight XGBoost. No GPU needed. |
| Total pipeline | < 50ms | Well within the 200ms SLA. |
