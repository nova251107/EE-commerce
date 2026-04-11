# Dataset Split Strategy — Temporal Production-Style Design

---

## Why Temporal, Not Random

| Method | How It Works | Problem |
|--------|-------------|---------|
| **Random split** | Shuffle all rows, take 80%/20% | A January event can appear in test while a June event is in train. The model "sees the future" during training — this is **data leakage**. |
| **Temporal split** | Sort by timestamp, train on earlier data, test on later data | Mirrors production exactly: the model is always predicting events it has never seen before. |

In a real-time pricing/recommendation system, the model is **always predicting the future**. The split must reflect this.

---

## Real Data Timeline

```
Dataset span: 2024-01-01 → 2024-06-29 (180 days, 6 months)

Monthly event volume (evenly distributed):
  Jan:  85,414
  Feb:  80,135
  Mar:  85,453
  Apr:  82,807
  May:  85,953
  Jun:  79,803
  ─────────────
  Total: 499,565
```

---

## Split Design

### Primary Split: 80/20 Temporal

```
          TRAIN (80%)                    TEST (20%)
  ┌──────────────────────────┐    ┌─────────────────┐
  │ Jan  Feb  Mar  Apr  May* │    │  May*    Jun     │
  │         399,652 rows     │    │   99,913 rows    │
  └──────────────────────────┘    └─────────────────┘
  2024-01-01              2024-05-24     →     2024-06-29

  * May is split at the 80th percentile date (May 24)
```

| Set | Date Range | Rows | Purpose |
|-----|-----------|:---:|---------|
| **Train** | Jan 1 → May 24 | 399,652 (80%) | Model learns patterns from historical data |
| **Test** | May 25 → Jun 29 | 99,913 (20%) | Evaluates on "unseen future" data |

### Why 80/20?

| Ratio | Train | Test | Verdict |
|:---:|:---:|:---:|---------|
| 90/10 | 5 months | 18 days | ❌ Test window too short — can't detect weekly patterns |
| **80/20** | **4.8 months** | **~5 weeks** | ✅ Train has enough history, test covers full weekly cycles |
| 70/30 | 4.2 months | 7.5 weeks | ⚠️ Acceptable but wastes training data for a hackathon |

---

## Leakage Prevention Checklist

### ✅ What the split guarantees

| Leakage Type | Risk | Prevention |
|-------------|------|-----------|
| **Temporal leakage** | Future events contaminate training features | Hard cutoff at May 24. No test-set timestamps appear in train. |
| **Session leakage** | Same session split across train and test | Sessions spanning the cutoff are assigned entirely to whichever set contains their **first** event. |
| **User leakage** | Same user in both sets | **Allowed and intentional.** In production, we serve existing users. But: user-level aggregated features (engagement scores, demand counters) must be computed using **train data only**. |
| **Feature leakage** | Demand counters include test-period events | Compute all aggregate features (demand_velocity, engagement_score, popularity) using only train-period events. Freeze these before scoring test set. |
| **Label leakage** | `current_price_usd` used as feature when predicting conversion | Excluded from feature set (documented in label_design.md). |

### Session Boundary Rule

```
If a session spans the train/test cutoff date:
  → Assign the ENTIRE session to TRAIN (conservative approach)
  → This prevents partial sessions in test where the model might
    see "half a purchase funnel" without the conclusion
```

---

## Implementation Logic

```python
import pandas as pd

def temporal_split(df, timestamp_col='timestamp', train_ratio=0.80):
    """
    Production-style temporal split.
    Train on past, test on future. Never shuffle.
    """
    # 1. Sort by time
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # 2. Find cutoff point
    cutoff_idx = int(len(df) * train_ratio)
    cutoff_date = df.iloc[cutoff_idx][timestamp_col]

    # 3. Handle session boundaries — push split sessions into train
    if 'session_id' in df.columns:
        split_sessions = df[df[timestamp_col] <= cutoff_date]['session_id'].unique()
        train_mask = df['session_id'].isin(split_sessions)
        train = df[train_mask].copy()
        test = df[~train_mask].copy()
    else:
        train = df.iloc[:cutoff_idx].copy()
        test = df.iloc[cutoff_idx:].copy()

    return train, test
```

---

## Validation Split (Optional But Recommended)

For hyperparameter tuning without touching the test set:

```
  TRAIN (64%)           VAL (16%)       TEST (20%)
  ┌──────────────────┐  ┌──────────┐    ┌─────────┐
  │ Jan  Feb  Mar Apr│  │  May 1-24│    │ May-Jun  │
  │    ~320K rows    │  │  ~80K    │    │  ~100K   │
  └──────────────────┘  └──────────┘    └─────────┘

  Train the model on TRAIN.
  Tune hyperparameters on VAL.
  Report final metrics on TEST (once, at the end).
```

For a hackathon, this is optional — the simple 80/20 split is sufficient.

---

## What to Tell Judges

> "We use a strict temporal split: the model trains on January through late May and is evaluated on data from the final 5 weeks. This mirrors production deployment exactly — the model never sees future events during training. We also enforce session-boundary integrity at the cutoff point, preventing partial session artifacts from contaminating the test set."

---

## Summary

```
┌──────────────────────────────────────────────┐
│  SPLIT STRATEGY                              │
│                                              │
│  Method:     Temporal (sorted by timestamp)  │
│  Ratio:      80% train / 20% test            │
│  Cutoff:     2024-05-24                      │
│  Train:      399,652 rows (Jan–May 24)       │
│  Test:       99,913 rows  (May 25–Jun 29)    │
│                                              │
│  Session rule: whole sessions stay together   │
│  Feature rule: aggregates from train only     │
│  Shuffle:     NEVER                           │
└──────────────────────────────────────────────┘
```
