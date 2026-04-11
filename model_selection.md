# Pricing Model Selection

Task: binary classification — predict purchase probability (label_conversion).
Dataset: ~400K train rows, 24 features, 4% positive rate (1:24 imbalance).

---

## Head-to-Head Comparison

| Criterion | Logistic Regression | Random Forest | XGBoost | Neural Network |
|-----------|:---:|:---:|:---:|:---:|
| **Inference latency** | ~0.01ms | ~1–5ms | ~0.1–0.5ms | ~2–10ms |
| **Training time (400K rows)** | ~2s | ~30–60s | ~10–20s | ~60–300s |
| **Explainability** | ✅ Full (coefficients) | ⚠️ Partial (feature importance) | ⚠️ Partial (SHAP needed) | ❌ Black box |
| **Handles imbalance** | ✅ `class_weight` | ✅ `class_weight` | ✅ `scale_pos_weight` | ⚠️ Manual loss weighting |
| **Handles non-linearity** | ❌ Linear only | ✅ Natively | ✅ Natively | ✅ Natively |
| **Overfitting risk** | Low | Medium (deep trees) | Medium (tunable) | High |
| **Dependency weight** | sklearn only | sklearn only | xgboost (~50MB) | torch/tf (~500MB+) |
| **Incremental learning** | ✅ `SGDClassifier` | ❌ Full retrain | ❌ Full retrain | ✅ `partial_fit` possible |
| **Hackathon friendliness** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

---

## Best Model: XGBoost

### Why XGBoost Wins for This Task

1. **Non-linear pricing dynamics.** Price sensitivity, demand velocity, and scarcity interact non-linearly. A user browsing premium electronics with high intent at peak hour during low stock is a fundamentally different situation than any single feature suggests alone. Logistic Regression cannot capture these interactions without manual feature crosses.

2. **Built-in imbalance handling.** With a 1:24 class ratio, setting `scale_pos_weight=24` is a single parameter — no SMOTE, no custom loss functions.

3. **Fast enough for real-time.** Inference is ~0.1–0.5ms per prediction with a 100-tree ensemble. Well within the 50ms pipeline budget.

4. **Training in seconds.** On 400K rows × 24 features, XGBoost trains in 10–20 seconds. You can retrain between demo rounds if needed.

5. **Feature importance is free.** `model.feature_importances_` gives you a ranked list instantly — critical for the explainability layer and judge Q&A.

---

## Runner-Up: Logistic Regression (SGDClassifier)

**Use this alongside XGBoost, not instead of it.**

Your existing `ml_engine.py` already runs an SGDClassifier for incremental online learning (updating with each event via `partial_fit`). This is the **real-time warm model** that stays current between full XGBoost retrains.

### Recommended Dual-Model Architecture

```
┌────────────────────────────────────────────────┐
│          DUAL-MODEL PRICING PIPELINE           │
│                                                │
│  SGDClassifier (online, incremental)           │
│    → Updates with every event via partial_fit  │
│    → Captures last-minute demand shifts        │
│    → Latency: 0.01ms                           │
│                                                │
│  XGBoost (batch, retrained periodically)       │
│    → Deeper non-linear patterns                │
│    → Higher accuracy on complex interactions   │
│    → Latency: 0.3ms                            │
│                                                │
│  Final score = 0.6 × XGBoost + 0.4 × SGD      │
│    → Hybrid: accuracy + freshness              │
└────────────────────────────────────────────────┘
```

---

## Why NOT the Others

### Random Forest — Rejected
- Inference is 5–10× slower than XGBoost (must traverse all trees independently, no early stopping).
- No native `scale_pos_weight` — requires `class_weight='balanced'` which is less granular.
- Cannot do incremental learning. In a real-time system, this is a hard limitation.
- Accuracy is typically equal to or slightly below XGBoost on tabular data.

### Neural Network — Rejected
- Massive dependency (PyTorch/TensorFlow = 500MB+). Risky on a hackathon demo machine.
- Training is 10–30× slower. Hyperparameter tuning (learning rate, layers, dropout) is a time sink.
- Inference latency is 10–50× higher than XGBoost.
- Explainability requires SHAP or LIME — additional complexity.
- On structured tabular data with 24 features, neural nets [rarely beat gradient boosting](https://arxiv.org/abs/2207.08815).
- **High risk, zero reward for this task size.**

---

## Recommended Configuration

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,         # 100 trees (fast inference)
    max_depth=6,              # prevent overfitting
    learning_rate=0.1,        # standard
    scale_pos_weight=24,      # handle 1:24 imbalance
    subsample=0.8,            # row sampling for regularization
    colsample_bytree=0.8,    # feature sampling
    eval_metric='aucpr',      # AUC-PR is better than AUC-ROC for imbalanced data
    random_state=42,
    n_jobs=-1,                # use all CPU cores
)
```

---

## Trade-Off Summary

| You Get | You Give Up |
|---------|------------|
| Non-linear interaction capture | Full coefficient interpretability (use feature_importances_ instead) |
| 0.3ms inference | Not as fast as pure logistic (0.01ms) — but 0.3ms is fine |
| 10s training | Cannot do `partial_fit` — pair with SGD for online updates |
| Native imbalance handling | One extra dependency (`pip install xgboost`) |

---

## What to Tell Judges

> "We evaluated four model families against three constraints: latency, explainability, and training speed. XGBoost won decisively for our tabular dataset — it captures the non-linear interactions between demand velocity, inventory scarcity, and price sensitivity that a linear model misses, while maintaining sub-millisecond inference. We pair it with an online SGDClassifier that incrementally updates with every streaming event, giving us both deep accuracy and real-time freshness."
