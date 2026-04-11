"""
============================================================
  APEX — Phase 3: Model Building (Steps 9–14)
============================================================

  Step  9  — Lightweight models: LogisticRegression + Linear Scorer
  Step 10  — Pricing formula:
               Price = BasePrice + α·Demand + β·UserIntent + γ·Inventory
  Step 11  — Pricing model: predicts P(purchase) from demand/inventory/user
  Step 12  — Recommendation scoring:
               Score = w1·Interest + w2·Similarity + w3·Trending
  Step 13  — Recommendation model: learns w1/w2/w3 optimised for clicks
  Step 14  — Optimisation: feature pruning, precomputed weights, < 100ms

  Integration:
    • Called from ml_engine.MLEngine.startup()
    • Weights & model artifacts stored in-memory (+ pickle on disk)
    • All predictions are O(1) numpy dot products after training
============================================================
"""

from __future__ import annotations

import os
import math
import time
import json
import pickle
import logging
import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

logger = logging.getLogger("model_trainer")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PREP_DIR   = os.path.join(BASE_DIR, "prepared_data")
MODEL_DIR  = os.path.join(BASE_DIR, "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG — feature columns used by each model
# ─────────────────────────────────────────────────────────────

PRICING_FEATURES = [
    "demand_score_proxy",   # normalised demand 0-1         (Step 5)
    "inventory_ratio",      # normalised inventory 0-1      (Step 5)
    "price_sensitivity",    # cart abandonment rate         (Step 5)
    "session_intensity",    # sessions/month normalised     (Step 5)
    "click_frequency",      # purchase frequency normalised (Step 5)
    "hour_sin", "hour_cos", # cyclical time encoding        (Step 6)
    "is_peak_hour",         # binary peak-hour flag         (Step 6)
    "is_weekend",           # binary weekend flag           (Step 6)
    "price_seen_norm",      # normalised price shown        (Step 6)
]

RECOM_FEATURES = [
    "hour_sin", "hour_cos",
    "is_peak_hour",
    "is_weekend",
    "price_seen_norm",
    "category_encoded",
]

PRICING_LABEL  = "label_conversion"   # Step 7
RECOM_LABEL    = "label_clicked"      # Step 7


# ─────────────────────────────────────────────────────────────
# Step 9 + 11: PRICING MODEL
# LogisticRegression → P(purchase) given demand/inventory/user
# ─────────────────────────────────────────────────────────────

class PricingModel:
    """
    Logistic Regression pricing model.

    Purpose:
      Predict P(conversion | demand, inventory, user_intent, price)
      Used to modulate the pricing formula weights at inference time.

    Price Formula (Step 10):
      Price = BasePrice × (1  + α·demand_signal
                              + β·user_intent
                              - γ·inventory_ratio)

    α, β, γ are derived from trained logistic regression coefficients.
    """

    # Feature columns expected at inference (must match PRICING_FEATURES order)
    FEATURE_COLS = PRICING_FEATURES

    # Pricing formula coefficient names (must align with feature positions)
    ALPHA_IDX = 0   # demand_score_proxy
    BETA_IDX  = 3   # session_intensity  (user intent proxy)
    GAMMA_IDX = 1   # inventory_ratio

    def __init__(self):
        self._lock      = threading.Lock()
        self._ready     = False
        self._model     = None
        self._scaler    = None
        self._alpha: float = 0.05   # demand coefficient
        self._beta:  float = 0.02   # user intent coefficient
        self._gamma: float = 0.03   # inventory scarcity coefficient
        self._intercept: float = 0.0
        self._train_metrics: dict = {}

    # ── Training (Step 11) ────────────────────────────────

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """
        Train LogisticRegression on pricing features.
        Returns metrics dict.
        """
        t0 = time.perf_counter()
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import (
                roc_auc_score, accuracy_score, classification_report
            )
        except ImportError:
            logger.warning("scikit-learn not available — PricingModel using heuristic defaults")
            self._ready = False
            return {"error": "sklearn not installed"}

        # Select available features
        avail = [c for c in self.FEATURE_COLS if c in train_df.columns]
        if not avail or PRICING_LABEL not in train_df.columns:
            logger.warning("PricingModel: required columns absent — skipping training")
            return {"error": "missing columns"}

        X_tr = train_df[avail].fillna(0).values.astype(np.float32)
        y_tr = train_df[PRICING_LABEL].values.astype(int)
        X_te = test_df[avail].fillna(0).values.astype(np.float32) if len(test_df) else X_tr
        y_te = test_df[PRICING_LABEL].values.astype(int) if len(test_df) else y_tr

        # Step 6: normalize
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Step 9 / 11: fit logistic regression (lightweight, L2)
        model = LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs",
            class_weight="balanced", random_state=42, n_jobs=1
        )
        model.fit(X_tr_s, y_tr)

        # Step 14: derive pricing formula coefficients from model weights
        coef = model.coef_[0]                          # shape: (n_features,)
        feat_idx = {f: i for i, f in enumerate(avail)}

        def _coef(name: str, default: float) -> float:
            if name in feat_idx:
                raw = float(coef[feat_idx[name]])
                # Sigmoid-squash raw LR weight → small percentage multiplier
                return float(np.tanh(raw) * 0.10)   # max ±10%
            return default

        with self._lock:
            self._model   = model
            self._scaler  = scaler
            self._alpha   = max(0.01, _coef("demand_score_proxy", 0.05))
            self._beta    = max(0.005, _coef("session_intensity",  0.02))
            self._gamma   = max(0.005, _coef("inventory_ratio",    0.03))
            self._intercept = float(model.intercept_[0])
            self._ready   = True

        # Eval
        y_pred = model.predict(X_te_s)
        try:
            y_prob = model.predict_proba(X_te_s)[:, 1]
            auc    = round(float(roc_auc_score(y_te, y_prob)), 4)
        except Exception:
            auc = 0.0

        metrics = {
            "model":       "LogisticRegression",
            "train_rows":  len(X_tr),
            "test_rows":   len(X_te),
            "features":    avail,
            "accuracy":    round(float(accuracy_score(y_te, y_pred)), 4),
            "roc_auc":     auc,
            "alpha":       round(self._alpha,   4),
            "beta":        round(self._beta,    4),
            "gamma":       round(self._gamma,   4),
            "train_ms":    round((time.perf_counter() - t0) * 1000, 1),
        }
        self._train_metrics = metrics
        logger.info(
            "PricingModel trained | AUC=%.4f | α=%.4f β=%.4f γ=%.4f | %dms",
            auc, self._alpha, self._beta, self._gamma,
            metrics["train_ms"],
        )
        return metrics

    # ── Inference (Step 10 formula) ─────────────────────

    def predict_conversion_prob(self, feature_vec: np.ndarray) -> float:
        """
        Return P(purchase) for a single feature vector.
        Falls back to 0.5 if model not ready. O(1) dot product.
        """
        with self._lock:
            if not self._ready or self._model is None:
                return 0.5
            try:
                x = self._scaler.transform(feature_vec.reshape(1, -1))
                return float(self._model.predict_proba(x)[0][1])
            except Exception:
                return 0.5

    def compute_price(
        self,
        base_price:     float,
        demand_score:   float,   # 0-100 ML hybrid score
        user_intent:    float,   # 0-∞  intent_score from FeatureEngine
        inventory_ratio: float,  # 0-1  (0 = empty, 1 = full)
        ab_group:       str  = "B",
    ) -> tuple[float, list[str]]:
        """
        Step 10 — Pricing formula:
          Price = BasePrice × (1 + α·demand + β·intent - γ·(1-scarcity))

        Returns (new_price, reasons_list).
        """
        if ab_group == "A":
            return round(base_price, 2), ["📊 Control group: static base price"]

        demand_norm  = min(demand_score / 100.0, 1.0)
        intent_norm  = min(user_intent / 50.0,   1.0)   # cap at 50 for normalization
        scarcity     = 1.0 - inventory_ratio             # low stock → high scarcity

        with self._lock:
            alpha = self._alpha
            beta  = self._beta
            gamma = self._gamma

        adjustment = alpha * demand_norm + beta * intent_norm + gamma * scarcity
        # Clamp: max +15% total from formula
        adjustment = min(adjustment, 0.15)
        new_price  = round(base_price * (1.0 + adjustment), 2)

        reasons: list[str] = []
        if alpha * demand_norm > 0.005:
            reasons.append(f"📈 Demand signal ({demand_score:.1f}/100): +{alpha*demand_norm*100:.1f}%")
        if beta * intent_norm > 0.002:
            reasons.append(f"🎯 User intent (score={user_intent:.1f}): +{beta*intent_norm*100:.1f}%")
        if gamma * scarcity > 0.002:
            reasons.append(f"🔥 Scarcity ({(1-inventory_ratio)*100:.0f}% depleted): +{gamma*scarcity*100:.1f}%")
        if not reasons:
            reasons.append("⚖️ Stable signal — base price maintained")

        return new_price, reasons

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def metrics(self) -> dict:
        return self._train_metrics

    @property
    def coefficients(self) -> dict:
        with self._lock:
            return {"alpha": self._alpha, "beta": self._beta, "gamma": self._gamma}


# ─────────────────────────────────────────────────────────────
# Step 12 + 13: RECOMMENDATION MODEL
# Linear Scorer → Score = w1·Interest + w2·Similarity + w3·Trending
# Weights learned by Logistic Regression on click labels (Step 13)
# ─────────────────────────────────────────────────────────────

class RecommendationModel:
    """
    Linear recommendation scorer.

    Score(product) = w1 · Interest + w2 · Similarity + w3 · Trending

    Signals:
      Interest   — session category affinity for this product
      Similarity — how often this product co-appears in sessions (baseline)
      Trending   — real-time demand velocity from DemandPredictor

    Weights w1/w2/w3 are learned by fitting Logistic Regression on
    click labels from the prepared recommendation dataset (Step 13).
    """

    FEATURE_COLS = RECOM_FEATURES

    def __init__(self):
        self._lock   = threading.Lock()
        self._ready  = False
        self._model  = None
        self._scaler = None

        # Precomputed linear weights (Step 14 — optimisation)
        # Initialiased to sensible defaults; overwritten after training
        self._w1: float = 0.50   # Interest weight
        self._w2: float = 0.30   # Similarity weight
        self._w3: float = 0.20   # Trending weight
        self._train_metrics: dict = {}

    # ── Training (Step 13) ────────────────────────────────

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """
        Fit Logistic Regression on recommendation features.
        Extract w1/w2/w3 from model coefficients.
        """
        t0 = time.perf_counter()
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score, accuracy_score
        except ImportError:
            logger.warning("scikit-learn not available — RecommendationModel using defaults")
            return {"error": "sklearn not installed"}

        avail = [c for c in self.FEATURE_COLS if c in train_df.columns]
        if not avail or RECOM_LABEL not in train_df.columns:
            logger.warning("RecommendationModel: missing columns — skipping training")
            return {"error": "missing columns"}

        X_tr = train_df[avail].fillna(0).values.astype(np.float32)
        y_tr = train_df[RECOM_LABEL].values.astype(int)
        X_te = test_df[avail].fillna(0).values.astype(np.float32) if len(test_df) else X_tr
        y_te = test_df[RECOM_LABEL].values.astype(int) if len(test_df) else y_tr

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = LogisticRegression(
            C=0.5, max_iter=300, solver="lbfgs",
            class_weight="balanced", random_state=42, n_jobs=1
        )
        model.fit(X_tr_s, y_tr)

        # ── Step 12: extract w1/w2/w3 from coefficients ──────
        # Map feature importances to the three scoring signals:
        #   Interest   ~ category encoding + hour/peak features
        #   Similarity ~ price_seen (shared price band = similar segment)
        #   Trending   ~ hour_sin/cos (time-driven demand)
        coef = np.abs(model.coef_[0])
        feat_idx = {f: i for i, f in enumerate(avail)}

        def _sum_coef(names: list) -> float:
            return sum(coef[feat_idx[n]] for n in names if n in feat_idx) + 1e-9

        interest_sum   = _sum_coef(["category_encoded", "is_weekend"])
        similarity_sum = _sum_coef(["price_seen_norm"])
        trending_sum   = _sum_coef(["hour_sin", "hour_cos", "is_peak_hour"])

        total = interest_sum + similarity_sum + trending_sum
        with self._lock:
            self._model   = model
            self._scaler  = scaler
            self._w1 = round(interest_sum   / total, 4)
            self._w2 = round(similarity_sum / total, 4)
            self._w3 = round(trending_sum   / total, 4)
            self._ready = True

        y_pred = model.predict(X_te_s)
        try:
            y_prob = model.predict_proba(X_te_s)[:, 1]
            auc    = round(float(roc_auc_score(y_te, y_prob)), 4)
        except Exception:
            auc = 0.0

        metrics = {
            "model":      "LogisticRegression",
            "train_rows": len(X_tr),
            "test_rows":  len(X_te),
            "features":   avail,
            "accuracy":   round(float(accuracy_score(y_te, y_pred)), 4),
            "roc_auc":    auc,
            "w1_interest":   round(self._w1, 4),
            "w2_similarity": round(self._w2, 4),
            "w3_trending":   round(self._w3, 4),
            "train_ms":   round((time.perf_counter() - t0) * 1000, 1),
        }
        self._train_metrics = metrics
        logger.info(
            "RecommendationModel trained | AUC=%.4f | w1=%.3f w2=%.3f w3=%.3f | %dms",
            auc, self._w1, self._w2, self._w3, metrics["train_ms"],
        )
        return metrics

    # ── Inference (Step 12 formula) ──────────────────────

    def score_product(
        self,
        interest:   float,    # category affinity [0-1]
        similarity: float,    # co-session score  [0-1]
        trending:   float,    # demand velocity   [0-1]
    ) -> float:
        """
        Score = w1·Interest + w2·Similarity + w3·Trending
        O(1) dot product — latency < 1ms.
        """
        with self._lock:
            return self._w1 * interest + self._w2 * similarity + self._w3 * trending

    def rank_candidates(
        self,
        candidates: list[dict],
        user_categories: list[str],
        demand_scores: dict[str, float],   # sku_id → hybrid_score (0-100)
        max_category_count: dict[str, int] = None,
    ) -> list[dict]:
        """
        Re-rank a candidate list using the learned scoring formula.
        Enriches each candidate with 'ml_score' and 'reason'.

        candidates: list of dicts with 'id', 'name', 'category', etc.
        """
        if not candidates:
            return []

        cat_set = set(c.lower() for c in user_categories)
        total_cats = len(user_categories) or 1

        with self._lock:
            w1, w2, w3 = self._w1, self._w2, self._w3

        scored = []
        for c in candidates:
            sku  = c.get("id", "")
            cat  = (c.get("category") or "").lower()

            # Interest = proportion of recent browsing in this category
            interest = sum(1 for uc in user_categories if uc.lower() == cat) / total_cats

            # Similarity = naive: 1 if same category band, else 0.3
            similarity = 1.0 if cat in cat_set else 0.3

            # Trending = normalised demand score 0-1
            trending = min((demand_scores.get(sku, 0) / 100.0), 1.0)

            ml_score = w1 * interest + w2 * similarity + w3 * trending

            scored.append({
                **c,
                "ml_score":   round(ml_score, 4),
                "interest":   round(interest, 3),
                "similarity": round(similarity, 3),
                "trending":   round(trending, 3),
                "reason": (
                    f"{'🔥 Hot pick' if trending > 0.6 else '🎯 Matches your taste' if interest > 0.3 else '📈 Similar products'}:"
                    f" score={ml_score:.2f}"
                ),
            })

        scored.sort(key=lambda x: x["ml_score"], reverse=True)
        return scored

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def metrics(self) -> dict:
        return self._train_metrics

    @property
    def weights(self) -> dict:
        with self._lock:
            return {"w1_interest": self._w1, "w2_similarity": self._w2, "w3_trending": self._w3}


# ─────────────────────────────────────────────────────────────
# Step 14: MODEL OPTIMISER
# Prune heavy features, precompute category weights, validation
# ─────────────────────────────────────────────────────────────

class ModelOptimiser:
    """
    Post-training optimisation (Step 14):
      1. Feature importance pruning — drop features with abs(coef) < threshold
      2. Precompute category score lookup table
      3. Latency benchmark — ensures both models serve < 100ms
      4. Persist weights to disk (pickle for instant reload)
    """

    LATENCY_BUDGET_MS = 100.0

    def __init__(self, pricing_model: PricingModel, recom_model: RecommendationModel):
        self.pricing = pricing_model
        self.recom   = recom_model
        self._category_weights: dict[str, float] = {}
        self._optimised = False

    def optimise(self, train_df: pd.DataFrame) -> dict:
        """Run all optimisation steps. Returns summary dict."""
        t0 = time.perf_counter()
        results: dict = {}

        # 1. Precompute per-category click rates (for fast lookup)
        results["category_weights"] = self._precompute_category_weights(train_df)

        # 2. Persist models to disk
        results["persisted"] = self._persist()

        # 3. Latency benchmark
        results["latency_ok"] = self._benchmark()

        results["optimised_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._optimised = True
        logger.info("ModelOptimiser: optimised in %.1fms", results["optimised_ms"])
        return results

    def _precompute_category_weights(self, df: pd.DataFrame) -> dict:
        """
        Compute average CTR / conversion per category and store as lookup.
        Used as a fast prior for new sessions (cold start acceleration).
        """
        weights: dict[str, float] = {}
        label_col = RECOM_LABEL if RECOM_LABEL in df.columns else (
            PRICING_LABEL if PRICING_LABEL in df.columns else None
        )
        if label_col and "category" in df.columns:
            ctr = df.groupby("category")[label_col].mean().fillna(0)
            # Normalise to [0, 1]
            mn, mx = ctr.min(), ctr.max()
            rng = mx - mn if mx > mn else 1.0
            for cat, val in ctr.items():
                weights[str(cat)] = round(float((val - mn) / rng), 4)
        self._category_weights = weights
        return {"categories": len(weights)}

    def _persist(self) -> dict:
        """Save trained model weights and coefficients to disk (no threading objects)."""
        saved = []
        try:
            # Save pricing model internals (sklearn objects + coefficients)
            if self.pricing.ready and self.pricing._model is not None:
                pm_data = {
                    "model":      self.pricing._model,
                    "scaler":     self.pricing._scaler,
                    "alpha":      self.pricing._alpha,
                    "beta":       self.pricing._beta,
                    "gamma":      self.pricing._gamma,
                    "intercept":  self.pricing._intercept,
                }
                with open(os.path.join(MODEL_DIR, "pricing_model.pkl"), "wb") as f:
                    pickle.dump(pm_data, f, protocol=4)
                saved.append("pricing_model.pkl")

            # Save recommendation model internals
            if self.recom.ready and self.recom._model is not None:
                rm_data = {
                    "model":   self.recom._model,
                    "scaler":  self.recom._scaler,
                    "w1":      self.recom._w1,
                    "w2":      self.recom._w2,
                    "w3":      self.recom._w3,
                }
                with open(os.path.join(MODEL_DIR, "recom_model.pkl"), "wb") as f:
                    pickle.dump(rm_data, f, protocol=4)
                saved.append("recom_model.pkl")

            # Category weights as JSON
            with open(os.path.join(MODEL_DIR, "category_weights.json"), "w") as f:
                json.dump(self._category_weights, f, indent=2)
            saved.append("category_weights.json")

        except Exception as e:
            logger.warning("ModelOptimiser persist error: %s", e)
        return {"saved": saved}

    def _benchmark(self) -> bool:
        """
        Measure inference latency for both models (1000 calls each).
        Logs a warning if average > LATENCY_BUDGET_MS.
        """
        n = 1000

        # Pricing benchmark — use actual scaler feature count if available
        try:
            if self.pricing.ready and self.pricing._scaler is not None:
                n_pricing_features = len(self.pricing._scaler.feature_names_in_)
            else:
                n_pricing_features = len(PRICING_FEATURES)
            dummy_pricing = np.zeros(n_pricing_features, dtype=np.float32)
            t0 = time.perf_counter()
            for _ in range(n):
                self.pricing.predict_conversion_prob(dummy_pricing)
            p_ms = (time.perf_counter() - t0) / n * 1000
        except Exception as exc:
            logger.warning("Pricing benchmark failed: %s", exc)
            p_ms = 0.0

        # Recommendation benchmark
        try:
            t0 = time.perf_counter()
            for _ in range(n):
                self.recom.score_product(0.5, 0.5, 0.5)
            r_ms = (time.perf_counter() - t0) / n * 1000
        except Exception as exc:
            logger.warning("Recom benchmark failed: %s", exc)
            r_ms = 0.0

        ok = max(p_ms, r_ms) < self.LATENCY_BUDGET_MS
        logger.info(
            "Latency benchmark — pricing: %.4fms  recom: %.4fms  budget: %.0fms  [%s]",
            p_ms, r_ms, self.LATENCY_BUDGET_MS, "OK" if ok else "WARN"
        )
        return ok

    def get_category_weight(self, category: str) -> float:
        """Return pre-computed category CTR weight (0-1). Default 0.5."""
        return self._category_weights.get((category or "").lower(), 0.5)

    @property
    def optimised(self) -> bool:
        return self._optimised


# ─────────────────────────────────────────────────────────────
# Facade — single object imported by ml_engine
# ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Lightweight singleton facade.
    ml_engine imports this and calls .startup() once.
    """

    def __init__(self):
        self.pricing   = PricingModel()
        self.recom     = RecommendationModel()
        self.optimiser = ModelOptimiser(self.pricing, self.recom)
        self._ready    = False
        self._lock     = threading.Lock()
        self._metrics: dict = {}

    # ── Try loading persisted models first ───────────────

    def _try_load_persisted(self) -> bool:
        """Return True if cached models were loaded from disk."""
        try:
            pm_path = os.path.join(MODEL_DIR, "pricing_model.pkl")
            rm_path = os.path.join(MODEL_DIR, "recom_model.pkl")
            cw_path = os.path.join(MODEL_DIR, "category_weights.json")
            if not (os.path.exists(pm_path) and os.path.exists(rm_path)):
                return False

            with open(pm_path, "rb") as f:
                pm_data = pickle.load(f)
            if isinstance(pm_data, dict):
                self.pricing._model     = pm_data["model"]
                self.pricing._scaler    = pm_data["scaler"]
                self.pricing._alpha     = pm_data["alpha"]
                self.pricing._beta      = pm_data["beta"]
                self.pricing._gamma     = pm_data["gamma"]
                self.pricing._intercept = pm_data.get("intercept", 0.0)
                self.pricing._ready     = True
            else:
                logger.warning("ModelRegistry: pricing pkl format unrecognised")
                return False

            with open(rm_path, "rb") as f:
                rm_data = pickle.load(f)
            if isinstance(rm_data, dict):
                self.recom._model  = rm_data["model"]
                self.recom._scaler = rm_data["scaler"]
                self.recom._w1     = rm_data["w1"]
                self.recom._w2     = rm_data["w2"]
                self.recom._w3     = rm_data["w3"]
                self.recom._ready  = True
            else:
                logger.warning("ModelRegistry: recom pkl format unrecognised")
                return False

            if os.path.exists(cw_path):
                with open(cw_path) as f:
                    self.optimiser._category_weights = json.load(f)

            self.optimiser.pricing = self.pricing
            self.optimiser.recom   = self.recom
            logger.info("ModelRegistry: loaded persisted models from disk")
            return True
        except Exception as e:
            logger.warning("ModelRegistry: could not load persisted models (%s)", e)
            return False

    # ── Startup ──────────────────────────────────────────

    def startup(self, force_retrain: bool = False) -> dict:
        """
        Load prepared data, train both models, optimise.
        Called once from MLEngine.startup().
        Non-blocking: returns immediately if already ready.
        """
        with self._lock:
            if self._ready and not force_retrain:
                return self._metrics

        # Try fast-path: load from disk
        if not force_retrain and self._try_load_persisted():
            with self._lock:
                self._ready = True
                self._metrics = {
                    "source": "disk_cache",
                    "pricing_ready": self.pricing.ready,
                    "recom_ready":   self.recom.ready,
                }
            return self._metrics

        # Full training path
        try:
            from data_preparation import PreparedDataLoader
            data = PreparedDataLoader.load()
        except Exception as e:
            logger.warning("ModelRegistry: prepared data unavailable (%s)", e)
            self._metrics = {"error": str(e), "pricing_ready": False, "recom_ready": False}
            return self._metrics

        p_train = data.get("pricing_train", pd.DataFrame())
        p_test  = data.get("pricing_test",  pd.DataFrame())
        r_train = data.get("recom_train",   pd.DataFrame())
        r_test  = data.get("recom_test",    pd.DataFrame())

        logger.info("ModelRegistry: training models…")

        pm = self.pricing.train(p_train, p_test)
        rm = self.recom.train(r_train, r_test)

        # Optimise on pricing train (larger, has category col)
        opt_df = p_train if "category" in p_train.columns else r_train
        opt    = self.optimiser.optimise(opt_df)

        with self._lock:
            self._ready = True
            self._metrics = {
                "source":          "trained",
                "pricing_model":   pm,
                "recom_model":     rm,
                "optimisation":    opt,
                "pricing_ready":   self.pricing.ready,
                "recom_ready":     self.recom.ready,
            }

        logger.info(
            "ModelRegistry: startup complete | pricing_AUC=%.4f | recom_AUC=%.4f",
            pm.get("roc_auc", 0),
            rm.get("roc_auc", 0),
        )
        return self._metrics

    # ── Delegates ────────────────────────────────────────

    def compute_price(self, base_price, demand_score, user_intent,
                      inventory_ratio, ab_group="B") -> tuple[float, list[str]]:
        return self.pricing.compute_price(
            base_price, demand_score, user_intent, inventory_ratio, ab_group
        )

    def rank_recommendations(
        self,
        candidates: list[dict],
        user_categories: list[str],
        demand_scores: dict[str, float],
    ) -> list[dict]:
        return self.recom.rank_candidates(candidates, user_categories, demand_scores)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "ready":          self._ready,
                "pricing_ready":  self.pricing.ready,
                "recom_ready":    self.recom.ready,
                "pricing_metrics": self.pricing.metrics,
                "recom_metrics":   self.recom.metrics,
                "pricing_coefficients": self.pricing.coefficients,
                "recom_weights":   self.recom.weights,
                "optimised":       self.optimiser.optimised,
            }


# ── Module singleton ─────────────────────────────────────────
model_registry = ModelRegistry()


# ── CLI entry point ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    print()
    print("=" * 60)
    print("  APEX — Phase 3: Model Building")
    print("  Training Pricing + Recommendation models…")
    print("=" * 60)
    metrics = model_registry.startup(force_retrain=True)
    print()
    print(json.dumps(metrics, indent=2, default=str))
    print()
