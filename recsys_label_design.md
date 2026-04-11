# Recommendation Model — Label Design

Goal: predict **engagement probability** — will the user interact with a recommended product?

---

## 1. Best Label Choice

Three options considered:

| Label Option | Positive Rate | Imbalance | Verdict |
|-------------|:---:|:---:|---------|
| **Purchase only** | ~4% | 1:24 | ❌ Too sparse. Recommendation needs to predict *interest*, not just transactions. Most good recommendations don't end in immediate purchase. |
| **Click only** (`product_view`) | ~39% | 1:1.6 | ⚠️ Usable but misses deeper intent (cart, wishlist). |
| **Any engagement** (view + cart + wishlist + checkout + purchase) | **38.57%** | **1:1.6** | ✅ **Best.** Captures the full engagement funnel. Balanced enough for training without resampling. |

### Chosen Label

```
label_engaged = 1   if user interacted beyond a passive page_view
                    (product_view, add_to_cart, add_to_wishlist,
                     checkout_start, or purchase)

label_engaged = 0   if user only had a page_view impression
                    and never clicked through to the product
```

**Why this works:** The recommendation system's job is to surface products the user will *engage with* — not just buy. A user who clicks, wishlists, or carts a recommended product is a success. This gives a healthy **38.57% positive rate (1:1.6 ratio)** — no class imbalance handling needed.

---

## 2. Positive vs Negative Samples

### Grain
One row = one unique `(user_id, sku_id, session_id)` impression.

### Positive Sample (label = 1)
The user **actively engaged** with product P during session S:
- Clicked to view the product detail page (`product_view`)
- Added to cart (`add_to_cart`)
- Added to wishlist (`add_to_wishlist`)
- Started checkout (`checkout_start`)
- Purchased (`purchase`)

Count: **109,974 rows (38.57%)**

### Negative Sample (label = 0)
The user saw product P in a listing/search result during session S but **did not click through**:
- Only a `page_view` event exists (product appeared on page but wasn't selected)

Count: **175,174 rows (61.43%)**

---

## 3. Event → Label Mapping

| Event Type | Generates Row? | Label | Reasoning |
|-----------|:-:|:-:|-----------|
| `page_view` | ✅ Yes | **0** | Product shown but ignored. Core negative signal. |
| `product_view` | ✅ Yes | **1** | User clicked to see details. Clear engagement. |
| `add_to_wishlist` | ✅ Yes | **1** | Saved for later. Strong interest. |
| `add_to_cart` | ✅ Yes | **1** | Intent to buy. |
| `checkout_start` | ✅ Yes | **1** | Deep funnel engagement. |
| `purchase` | ✅ Yes | **1** | Strongest possible engagement. |
| `search` | ❌ No | — | No specific product impression. |
| `remove_from_cart` | ❌ No | — | Negative signal but not an impression event. |
| `page_exit` | ❌ No | — | Session ended, no product context. |

**Override rule:** If the same `(user, sku, session)` has both a `page_view` (label=0) AND a `product_view` (label=1), the label is **1**. Engagement overrides impression.

---

## 4. Noise Reduction

| Noise Source | Problem | Fix |
|-------------|---------|-----|
| **Bot traffic** | Automated scrapers create fake page_views with 0s scroll/dwell time | Filter: drop rows where `session_duration_s < 2` AND `scroll_depth_pct == 0` |
| **Duplicate impressions** | Same product shown 3× in one session inflates negatives | Deduplicate to unique `(user, sku, session)` before labeling |
| **Accidental clicks** | User clicks a product and immediately bounces (<1s dwell) | Accept as positive — even a "mistake" click is better than no click. Over-filtering creates label noise. |
| **Self-engagement** | User views a product via direct URL, not recommendation | No way to distinguish in this dataset — accept all as valid |

---

## 5. Labeling Logic (Step by Step)

```
Step 1:  Load cleaned clickstream

Step 2:  Split into impression pool and engagement pool
           impression_events = {page_view, product_view}
           engage_events     = {product_view, add_to_cart,
                                add_to_wishlist, checkout_start, purchase}

Step 3:  Build impression rows
           Deduplicate impression_events to unique (user, sku, session)
           These are ALL the products the user was exposed to

Step 4:  Build engagement lookup
           Deduplicate engage_events to unique (user, sku, session)
           Flag each tuple with engaged=1

Step 5:  Left-join
           impression_rows LEFT JOIN engagement_lookup
           on (user, sku, session)
           Fill missing → label_engaged = 0

Step 6:  Attach features
           Join product features (category, price, rating)
           Compute session features (affinity, intent, depth)
           Add time features (hour_sin, is_peak)
```

---

## 6. Why No Imbalance Handling Needed

| Metric | Value |
|--------|-------|
| Positive rate | 38.57% |
| Negative rate | 61.43% |
| Ratio | 1 : 1.6 |

This is **near-balanced**. For reference:
- Ratios below 1:3 are generally considered balanced
- `class_weight='balanced'` can be added as a safety net but is not required
- No SMOTE, no downsampling, no synthetic data needed

This is a significant advantage over the pricing label (which was 1:24) — the recommendation model will train faster and more stably.

---

## 7. Comparison: Pricing vs Recommendation Labels

| Aspect | Pricing Label | Recommendation Label |
|--------|:---:|:---:|
| Target | `label_conversion` | `label_engaged` |
| Positive event | purchase only | any engagement |
| Positive rate | 4.01% | 38.57% |
| Imbalance | 1:24 (severe) | 1:1.6 (balanced) |
| Needs rebalancing | Yes (`class_weight`) | No |
| Grain | (user, sku, session) | (user, sku, session) |

---

## 8. Summary

```
┌──────────────────────────────────────────────────┐
│  RECOMMENDATION LABEL DESIGN                     │
│                                                  │
│  Grain:     (user_id, sku_id, session_id)        │
│  Target:    label_engaged ∈ {0, 1}               │
│  Positive:  user engaged (view/cart/wish/buy)     │
│  Negative:  product shown but ignored             │
│                                                  │
│  Rate:      38.57% positive (109,974 rows)       │
│  Ratio:     1 : 1.6 — naturally balanced         │
│  Imbalance: none needed                           │
│                                                  │
│  Excluded events:                                │
│    search, remove_from_cart, page_exit            │
│                                                  │
│  Noise filter:                                    │
│    Drop bot sessions (duration < 2s + 0% scroll) │
│    Deduplicate per (user, sku, session)           │
└──────────────────────────────────────────────────┘
```
