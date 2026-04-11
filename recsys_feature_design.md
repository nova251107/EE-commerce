# Recommendation System — Feature Engineering Design

Goal: rank candidate products for the **current active session**.
Latency target: < 15ms per recommendation request.

---

## Real Session Profile (from 97K sessions)

| Metric | Value |
|--------|-------|
| Events per session | mean 5.1, median 5, max 19 |
| Unique products per session | mean 5.1, median 5 |
| Categories per session | mean 3.2, median 3 |
| Unique categories in catalog | 5 (Electronics, Clothing, Home & Kitchen, Books & Media, Beauty & Health) |
| Price range | Q25=$467, Q50=$930, Q75=$1415 |

Sessions are **short and focused** — the recommender must work well with just 3–5 signals.

---

## 1. Session-Aware Features

These capture what the user is doing **right now**, computed incrementally as events arrive.

| Feature | Formula | Range | Update Cost |
|---------|---------|-------|-------------|
| `session_action_count` | Total events in this session | 0–19 | O(1) counter |
| `session_depth` | Unique products viewed in session | 1–19 | O(1) set size |
| `session_duration_s` | `now − session_start_time` | 0–∞ | O(1) subtraction |
| `session_cart_count` | add_to_cart events in session | 0–N | O(1) counter |
| `session_purchase_count` | purchases in session | 0–N | O(1) counter |
| `last_event_type` | Most recent event type (encoded) | 0–8 | O(1) overwrite |
| `last_3_skus` | Deque of last 3 product SKUs | list | O(1) deque |
| `session_price_mean` | Running mean of `price_seen_usd` | float | O(1) incremental mean |
| `intent_score` | Weighted sum of recent action types (purchase=10, cart=5, view=1) | 0–100 | O(1) add/evict |

**Data structure:** One `deque(maxlen=10)` per session storing `EventRecord` objects. All features derivable from the deque in O(1) without full recomputation.

---

## 2. Category Affinity Features

With only 5 categories, affinity is a compact 5-dimensional vector per session.

| Feature | Formula | Notes |
|---------|---------|-------|
| `affinity_vector[5]` | Count of events per category in session, normalized to sum=1 | `[0.0, 0.4, 0.0, 0.6, 0.0]` = 40% Clothing, 60% Electronics |
| `dominant_category` | `argmax(affinity_vector)` | The category the user is focusing on |
| `category_focus_ratio` | `max(affinity_vector)` | 1.0 = single category session, 0.2 = browsing everything equally |
| `candidate_in_dominant_cat` | `1 if candidate.category == dominant_category else 0` | Binary match signal (strongest ranking feature) |

**Why this works:** With median 3 categories per session, `category_focus_ratio` reliably identifies single-category shopping intent (e.g., user clearly shopping for electronics).

---

## 3. Product Similarity Signals

For each candidate product being scored, compute similarity to the active session context.

### 3A. Category Match (strongest signal)

| Feature | Formula | Notes |
|---------|---------|-------|
| `cat_match` | `1.0 if candidate.category == session_dominant_category else 0.0` | Direct hit. Highest weight in ranking. |
| `cat_affinity_score` | `affinity_vector[candidate.category_index]` | Graded version: 0.0 to 1.0 based on how much the user browsed this category. |

### 3B. Price Similarity

| Feature | Formula | Notes |
|---------|---------|-------|
| `price_distance` | `abs(candidate.price − session_price_mean) / session_price_mean` | Low = good match. User browsing $900 items shouldn't see $50 items. |
| `price_match_score` | `1.0 − min(price_distance, 1.0)` | Inverted and capped: 1.0 = perfect price match, 0.0 = wildly different. |

### 3C. Co-Occurrence Signal (Collaborative Filtering lite)

| Feature | Formula | Notes |
|---------|---------|-------|
| `co_view_score` | `co_occurrence_graph[last_viewed_sku][candidate_sku]` normalized to 0–1 | "Users who viewed X also viewed Y." Built incrementally at runtime. |

**Graph structure:**
```
co_occurrence = defaultdict(Counter)

# On every event:
for past_sku in session.last_3_skus:
    co_occurrence[past_sku][current_sku] += event_weight
```

Retrieval: `co_occurrence[active_sku].most_common(20)` → candidate pool.

---

## 4. Trending / Popularity Features

Global signals independent of the individual session.

| Feature | Formula | Notes |
|---------|---------|-------|
| `demand_velocity` | Click count on this SKU in last 60s (from demand predictor) | Real-time spike detector. |
| `trending_flag` | `1 if demand_score increased > 10 pts in last refresh` | Binary trend indicator. |
| `global_popularity` | `product.views / max(all_products.views)` normalized 0–1 | Baseline popularity from historical clickstream. |
| `review_popularity` | `product.review_count / max(review_count)` normalized 0–1 | Social proof signal. |

---

## 5. Time-of-Day Features

| Feature | Formula | Notes |
|---------|---------|-------|
| `hour_sin` | `sin(2π × hour / 24)` | Cyclical encoding. |
| `hour_cos` | `cos(2π × hour / 24)` | Pair with sin. |
| `is_peak_hour` | `1 if 18 ≤ hour ≤ 22` | Evening shopping surge. |
| `is_weekend` | `1 if day_of_week ∈ {5, 6}` | Weekend pattern. |

These are shared across all candidates in a single scoring call (compute once, reuse).

---

## 6. Price Preference Features

| Feature | Formula | Notes |
|---------|---------|-------|
| `user_price_tier` | Bucket from session_price_mean: budget (<$300), mid ($300–$1000), premium (>$1000) | Encoded 0/1/2. |
| `candidate_price_tier` | Same buckets applied to candidate product | Encoded 0/1/2. |
| `tier_match` | `1 if user_price_tier == candidate_price_tier` | Users stay in their price lane. |
| `relative_price` | `candidate.price / session_price_mean` | <1.0 = cheaper than browsed, >1.0 = more expensive. |

---

## 7. Cold-Start Features

When the session or product has insufficient history, these signals fill the gap.

### Cold Session (user has 0–1 events)
| Feature | Value | Rationale |
|---------|-------|-----------|
| `affinity_vector` | `[0.2, 0.2, 0.2, 0.2, 0.2]` (uniform) | No dominant category yet. |
| `intent_score` | `0.0` | No behavioral signal. |
| `session_price_mean` | Global median ($930) | Best neutral guess. |
| **Ranking fallback** | Sort by `(0.5 × global_popularity) + (0.3 × trending_flag) + (0.2 × rating_score)` | Pure popularity + trending. |

### Cold Product (new item, 0 views/co-occurrences)
| Feature | Value | Rationale |
|---------|-------|-----------|
| `co_view_score` | `0.0` | No collaborative data. |
| `global_popularity` | `0.0` | No history. |
| **Injection rule** | Insert into bottom 2 of 5 recommendation slots if `candidate.category == session_dominant_category` | Forces exploration to gather initial signals. |

---

## 8. Final Recommendation Feature Vector (per candidate)

```
candidate_features = [
    # Session context (5)
    intent_score,
    session_depth,
    session_cart_count,
    category_focus_ratio,
    last_event_type_encoded,

    # Category match (2)
    cat_match,
    cat_affinity_score,

    # Price match (2)
    price_match_score,
    tier_match,

    # Collaborative (1)
    co_view_score,

    # Popularity (3)
    demand_velocity_norm,
    global_popularity,
    review_popularity,

    # Time (2)
    is_peak_hour,
    is_weekend,
]
# Total: 15 dimensions per candidate
```

---

## 9. Ranking Formula (Lightweight, No Neural Net)

For hackathon speed, use a weighted linear combination instead of a learned model:

```python
def score_candidate(features):
    return (
        0.30 * features['co_view_score']       +   # collaborative signal
        0.25 * features['cat_affinity_score']   +   # session category match
        0.15 * features['price_match_score']    +   # price similarity
        0.10 * features['demand_velocity_norm'] +   # trending
        0.10 * features['global_popularity']    +   # social proof
        0.05 * features['intent_score'] / 100   +   # user engagement depth
        0.05 * features['review_popularity']        # quality signal
    )
```

**Top-5 selection:** Score all candidates, sort descending, return top 5.
**Latency:** ~2ms for scoring 200 candidates (pure arithmetic, no model loading).

---

## 10. Summary

```
┌──────────────────────────────────────────────────┐
│  RECOMMENDATION FEATURE ENGINEERING              │
│                                                  │
│  Candidate scoring: 15-dim feature vector        │
│  Ranking: weighted linear combination            │
│  Latency: < 5ms for 200 candidates               │
│                                                  │
│  Signal priority:                                │
│    1. Co-view collaborative (30%)                │
│    2. Category affinity (25%)                    │
│    3. Price match (15%)                          │
│    4. Trending + Popularity (20%)                │
│    5. Session intent + Reviews (10%)             │
│                                                  │
│  Cold-start: popularity + trending fallback      │
│  Cold-product: category-match injection          │
│                                                  │
│  Data structures: deque(10), Counter, 5-dim vec  │
│  All updates: O(1) incremental                   │
└──────────────────────────────────────────────────┘
```
