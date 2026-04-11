# Pricing Model — Label Design

Target: predict **purchase probability** (binary classification).

---

## 1. Target Label Definition

```
label_conversion = 1   if the user purchased in this session-product interaction
label_conversion = 0   if the user did NOT purchase
```

**Grain of prediction:** One row = one unique `(user_id, sku_id, session_id)` interaction.
This means: "Given that user U saw product P during session S, did they buy it?"

---

## 2. Event → Label Mapping

Not all events generate training rows. Only **meaningful product impressions** create labeled samples.

| Event Type | Generates Row? | Label | Rationale |
|-----------|:-:|:-:|-----------|
| `page_view` | ❌ No | — | Generic page hit, no product context. |
| `product_view` | ✅ Yes | 0 (default) | User saw the product. Core negative sample. |
| `search` | ❌ No | — | Intent signal, but no specific product impression. |
| `add_to_wishlist` | ✅ Yes | 0 | Interest shown but no conversion. |
| `add_to_cart` | ✅ Yes | 0 | Strong intent but not a purchase yet. |
| `checkout_start` | ✅ Yes | 0 | Near-conversion but didn't complete. |
| `purchase` | ✅ Yes | **1** | Conversion confirmed. |
| `remove_from_cart` | ❌ No | — | Negative signal, but not a product impression. |
| `page_exit` | ❌ No | — | Bounced. No product context. |

**Critical rule:** If the same `(user, product, session)` has BOTH a `product_view` AND a `purchase`, the **label is 1**. The purchase overrides all earlier non-purchase events for that same tuple.

---

## 3. Positive vs Negative Sample Definition

| | Positive (label=1) | Negative (label=0) |
|---|---|---|
| **Definition** | User purchased product P in session S | User viewed/carted product P in session S but did NOT purchase |
| **What it means** | This price + context combination converted | This price + context combination failed to convert |
| **Count in dataset** | ~20,022 (4.01%) | ~479,543 (95.99%) |
| **Imbalance ratio** | 1 : 24 | |

---

## 4. Labeling Logic (Step by Step)

```
Step 1:  Filter clickstream to impression events only
         Keep: product_view, add_to_wishlist, add_to_cart,
               checkout_start, purchase

Step 2:  Deduplicate to unique (user_id, sku_id, session_id) rows
         Keep the LAST event and the max price_seen_usd per tuple

Step 3:  Build purchase lookup
         Find all (user_id, sku_id, session_id) tuples where
         event_type == 'purchase'

Step 4:  Left-join
         All impression rows LEFT JOIN purchase lookup
         If match exists → label = 1
         If no match    → label = 0

Step 5:  Join features
         Attach product features (price, inventory, category)
         Attach user features (sensitivity, session_intensity)
         Attach time features (hour_sin, is_peak, is_weekend)
```

---

## 5. Handling Class Imbalance (1:24 ratio)

The dataset is heavily skewed toward non-purchases. Three practical strategies, ranked by hackathon friendliness:

### Strategy A: Class Weights (Recommended — zero cost)
Most sklearn/XGBoost models accept `class_weight='balanced'` or `scale_pos_weight=24`.
The model internally upweights positive samples during gradient computation.
```python
# sklearn
model = SGDClassifier(class_weight='balanced')

# XGBoost
model = XGBClassifier(scale_pos_weight=24)
```

### Strategy B: SMOTE Oversampling (if precision matters)
Synthetically generate positive samples to reach ~15-20% positive ratio.
```python
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(sampling_strategy=0.2).fit_resample(X_train, y_train)
```

### Strategy C: Negative Downsampling (fastest, loses data)
Randomly drop 80% of negative samples to reach ~1:5 ratio.
Only use if training time is a constraint (it is not here).

**Recommendation for hackathon:** Use **Strategy A** (`class_weight='balanced'`). Zero implementation cost, no data manipulation, works out of the box.

---

## 6. Common Labeling Mistakes to Avoid

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Labeling `add_to_cart` as positive | Cart ≠ purchase. ~55% of carts are abandoned in real e-commerce. | Only `purchase` event = label 1. |
| Using `page_view` as negative sample | Page views have no product context. The model learns noise. | Only use events where a specific product was shown. |
| Counting same user-product twice in one session | Inflates the dataset with correlated duplicates. Model overfits. | Deduplicate to unique `(user, sku, session)` first. |
| Labeling at event level instead of session level | A user who viewed 5 products and bought 1 creates 5 training rows. The 4 "negative" rows are correct, the 1 positive is correct. But if you label at event level, the SAME purchase generates multiple positive rows for the same sku (one per event type). | Deduplicate per grain, then apply label from purchase lookup. |
| Using future purchase data in features | If demand_velocity or engagement_score includes the purchase event itself, you're leaking the answer into the input. | Compute features using events BEFORE the prediction moment only. |

---

## 7. Summary

```
┌─────────────────────────────────────────────────┐
│  LABEL DESIGN                                   │
│                                                 │
│  Grain:    (user_id, sku_id, session_id)        │
│  Target:   label_conversion ∈ {0, 1}            │
│  Positive: purchase event exists for this tuple  │
│  Negative: product seen but not purchased        │
│  Ratio:    4% positive / 96% negative (1:24)    │
│  Fix:      class_weight='balanced'               │
│                                                 │
│  Excluded from row generation:                   │
│    page_view, search, remove_from_cart, page_exit│
│                                                 │
│  Leakage guard:                                  │
│    Never include the purchase event itself       │
│    in feature computation for that same row.     │
└─────────────────────────────────────────────────┘
```
