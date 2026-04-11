"""
============================================================
  APEX — Phase 4: Real-Time Decision Engine (Steps 15–18)
============================================================

  Step 15  — Live Pricing Decision
             Get real-time demand → apply model → output price instantly

  Step 16  — Dynamic Adjustment Logic
             High demand  → +5% to +15%
             Low demand   → −10% to −30%
             Low stock    → urgency boost
             Competitor   → match / undercut

  Step 17  — Recommendation Generation
             Rank all candidates by Score = w1·Interest + w2·Similarity + w3·Trending
             Return top 5 with explainability tags

  Step 18  — Cold Start Handling
             New users → use device type + hour-of-day + trending products
             No session history needed

  Design:
    - All decisions < 50ms guaranteed (pure in-memory O(1) / O(n) lookups)
    - Thread-safe (read-only access to shared state, immutable snapshots)
    - Fully decoupled from server.py — import and call, no tight coupling
    - Falls back gracefully if model_trainer or ml_engine not ready
============================================================
"""

from __future__ import annotations

import math
import time
import logging
import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("decision_engine")


# ═══════════════════════════════════════════════════════════
# OUTPUT DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class PricingDecision:
    """
    Full pricing decision output (Step 15 / 16).
    All fields serialisable to dict for JSON response.
    """
    sku_id:           str
    base_price:       float
    final_price:      float
    adjustment_pct:   float          # signed %, e.g. +8.2 or -12.0
    tier:             str            # "surge" | "normal" | "discount" | "clearance"
    demand_score:     float          # 0–100
    demand_level:     str            # "high" | "medium" | "low"
    inventory_status: str            # "scarce" | "normal" | "surplus"
    competitor_gap:   Optional[float]  # our_price − competitor_price (None if unknown)
    reasons:          List[str]
    ab_group:         str
    latency_ms:       float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["adjustment_pct"] = round(self.adjustment_pct, 2)
        return d


@dataclass
class RecommendationItem:
    """Single recommendation item (Step 17)."""
    rank:       int
    sku_id:     str
    name:       str
    category:   str
    price:      float
    rating:     float
    ml_score:   float          # composite score 0–1
    interest:   float          # w1 signal
    similarity: float          # w2 signal
    trending:   float          # w3 signal
    cold_start: bool           # True if generated via cold-start (Step 18)
    tag:        str            # "🔥 Trending" | "🎯 For You" | "💡 New Arrival" etc.
    reason:     str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionBundle:
    """
    Full real-time decision bundle returned by DecisionEngine.decide().
    One object contains both pricing + recommendations + metadata.
    """
    pricing:         PricingDecision
    recommendations: List[RecommendationItem]
    cold_start:      bool
    session_id:      str
    user_id:         str
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pricing":          self.pricing.to_dict(),
            "recommendations":  [r.to_dict() for r in self.recommendations],
            "cold_start":       self.cold_start,
            "session_id":       self.session_id,
            "user_id":          self.user_id,
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


# ═══════════════════════════════════════════════════════════
# STEP 16: DYNAMIC ADJUSTMENT CALCULATOR
# ═══════════════════════════════════════════════════════════

class DynamicAdjustmentCalculator:
    """
    Step 16 — Tiered demand-driven price adjustment logic.

    Tiers (demand_score 0–100):
      Surge     [75–100]:  +10% to +15%
      High      [55–74 ]:  +5%  to +10%
      Normal    [35–54 ]:  0%   to +4%
      Low       [20–34 ]:  −10% (recovery)
      Very Low  [ 0–19 ]:  −20% to −30%

    Additional boosts (multiplicative on top of tier):
      Scarcity  (inventory < 20):   +3% to +5% urgency
      Surplus   (inventory > 400):  additional −3% to −5%
      Competitor undercut (>5% cheaper): neutralise gap up to −3%
    """

    TIERS = [
        # (min_score, max_score, base_adj, label)
        (75, 100, 0.12,  "surge"),
        (55,  74, 0.07,  "high"),
        (35,  54, 0.02,  "normal"),
        (20,  34, -0.10, "low"),
        ( 0,  19, -0.22, "clearance"),
    ]

    # Hard limits (never exceed regardless of signals)
    CEILING_PCT = 0.15
    FLOOR_PCT   = -0.30

    def calculate(
        self,
        base_price:       float,
        demand_score:     float,    # 0–100 hybrid score from DemandPredictor
        inventory:        int,
        user_intent:      float,    # 0–∞ intent score from FeatureEngine
        competitor_price: Optional[float] = None,
        ab_group:         str = "B",
        model_coefficients: Optional[Dict] = None,   # α, β, γ from PricingModel
    ) -> Tuple[float, float, str, List[str], str, str, Optional[float]]:
        """
        Compute (final_price, adjustment_pct, tier, reasons, demand_level).
        All arithmetic is O(1).
        """
        if ab_group == "A":
            return (
                round(base_price, 2), 0.0, "normal",
                ["📊 Control group (A): static base price"],
                "medium",
            )

        # ── Pick tier base adjustment ──────────────────────────
        tier_adj = 0.02   # default: normal
        tier_label = "normal"
        for lo, hi, adj, label in self.TIERS:
            if lo <= demand_score <= hi:
                # Interpolate within tier range for smoother transitions
                span  = hi - lo or 1
                frac  = (demand_score - lo) / span
                tier_adj   = adj * (0.8 + 0.4 * frac)    # 80%–120% of tier center
                tier_label = label
                break

        demand_level = (
            "high"   if demand_score >= 55 else
            "medium" if demand_score >= 35 else
            "low"
        )

        reasons: List[str] = []

        # ── Use trained model coefficients if available ─────────
        alpha = (model_coefficients or {}).get("alpha", 0.05)
        beta  = (model_coefficients or {}).get("beta",  0.02)
        gamma = (model_coefficients or {}).get("gamma", 0.03)

        # ── Demand adjustment ────────────────────────────────────
        demand_norm = demand_score / 100.0
        demand_adj  = tier_adj + alpha * demand_norm
        if demand_score >= 55:
            reasons.append(
                f"📈 {'Surge' if demand_score>=75 else 'High'} demand "
                f"({demand_score:.1f}/100): {demand_adj*100:+.1f}%"
            )
        elif demand_score < 35:
            reasons.append(
                f"📉 {'Very low' if demand_score<20 else 'Low'} demand "
                f"({demand_score:.1f}/100): {tier_adj*100:+.1f}%"
            )

        # ── Inventory adjustment (Step 16) ───────────────────────
        inv_adj = 0.0
        inv_status = "normal"

        if inventory < 10:
            inv_adj    = 0.05    # +5% critical scarcity
            inv_status = "scarce"
            reasons.append(f"🚨 Critical stock ({inventory} left): +5% urgency")
        elif inventory < 20:
            inv_adj    = 0.03    # +3% scarcity boost
            inv_status = "scarce"
            reasons.append(f"🔥 Low stock ({inventory} left): +3% urgency")
        elif inventory > 500:
            inv_adj    = -0.05   # −5% clearance
            inv_status = "surplus"
            reasons.append(f"📦 Surplus stock ({inventory} units): −5% to clear")
        elif inventory > 200:
            inv_adj    = -0.02   # −2% mild surplus
            inv_status = "surplus"
            reasons.append(f"📦 High stock ({inventory} units): −2%")

        # ── User-intent micro-boost (β coefficient) ──────────────
        intent_adj = 0.0
        if user_intent > 20:
            intent_adj = beta * min(user_intent / 50.0, 1.0)
            reasons.append(f"🎯 High-intent user (score={user_intent:.1f}): +{intent_adj*100:.1f}%")

        # ── Competitor-aware adjustment ──────────────────────────
        comp_gap   = None
        comp_adj   = 0.0
        if competitor_price and competitor_price > 0:
            temp_price = base_price * (1 + demand_adj + inv_adj + intent_adj)
            comp_gap   = round(temp_price - competitor_price, 2)
            if competitor_price < temp_price * 0.95:  # they're >5% cheaper
                comp_adj = -min(abs(comp_gap) / temp_price * 0.5, 0.03)
                reasons.append(
                    f"🏪 Competitor is ${abs(comp_gap):.2f} cheaper → "
                    f"{comp_adj*100:.1f}% match adjustment"
                )
            elif competitor_price > temp_price * 1.10:  # we're way cheaper
                reasons.append(f"💚 We're ${comp_gap:.2f} below market — maintaining advantage")

        # ── Total adjustment & price ────────────────────────────
        total_adj = demand_adj + inv_adj + intent_adj + comp_adj
        # Hard guardrails
        total_adj = max(self.FLOOR_PCT, min(total_adj, self.CEILING_PCT))

        final_price = round(base_price * (1.0 + total_adj), 2)

        if not reasons:
            reasons.append("⚖️ Balanced market signals — base price maintained")

        return final_price, total_adj * 100, tier_label, reasons, demand_level, inv_status, comp_gap


# ═══════════════════════════════════════════════════════════
# STEP 18: COLD START HANDLER
# ═══════════════════════════════════════════════════════════

class ColdStartHandler:
    """
    Step 18 — Cold-start recommendation for new users.

    Signals used (no session history needed):
      1. Device type   → mobile users prefer lower-priced, visual items
      2. Hour of day   → morning = productivity, evening = entertainment/fashion
      3. Day of week   → weekdays vs weekend behaviour
      4. Trending      → top-N globally trending products from DemandPredictor

    Returns a scored list of dicts (same shape as warm-start recommendations).
    """

    # Device-to-category preference map
    DEVICE_PREFS: Dict[str, List[str]] = {
        "mobile":  ["Fashion", "Beauty", "Electronics", "Sports"],
        "tablet":  ["Home & Kitchen", "Books", "Electronics"],
        "desktop": ["Electronics", "Books", "Home & Kitchen", "Sports"],
        "other":   [],
    }

    # Hour → category affinity boost
    HOUR_PREFS: Dict[Tuple, str] = {
        (6,  10): "Sports",         # morning — fitness
        (10, 12): "Electronics",    # mid-morning — productivity
        (12, 14): "Books",          # lunch — content
        (14, 18): "Home & Kitchen", # afternoon — home shopping
        (18, 22): "Fashion",        # evening — social / fashion
        (22, 24): "Beauty",         # night — self-care
    }

    def get_time_category(self) -> Optional[str]:
        hour = datetime.datetime.now().hour
        for (lo, hi), cat in self.HOUR_PREFS.items():
            if lo <= hour < hi:
                return cat
        return None

    def score_for_cold_start(
        self,
        product:       Dict,
        device_type:   str,
        hour_category: Optional[str],
        demand_score:  float,
        dow:           int,     # day of week 0=Mon
    ) -> float:
        """Return cold-start score for a product (0–1 scale)."""
        score = 0.0
        cat = product.get("category", "")

        # Device affinity (0.4 weight)
        device_cats = self.DEVICE_PREFS.get(device_type.lower(), [])
        if cat in device_cats:
            score += 0.4 * (1 - device_cats.index(cat) / max(len(device_cats), 1))

        # Hour affinity (0.2 weight)
        if hour_category and cat == hour_category:
            score += 0.2

        # Weekend boost for Fashion / Beauty (0.1 weight)
        if dow >= 5 and cat in ("Fashion", "Beauty"):
            score += 0.10

        # Trending signal (0.3 weight) — normalised demand 0–100 → 0–0.3
        score += 0.30 * (demand_score / 100.0)

        # Rating quality signal (0.1 weight) — prefer highly rated
        score += 0.10 * (product.get("rating", 0) / 5.0)

        return min(score, 1.0)

    def generate(
        self,
        all_products:   Dict[str, Dict],
        demand_scores:  Dict[str, float],   # sku_id → hybrid_score
        device_type:    str = "desktop",
        n:              int = 5,
    ) -> List[Dict]:
        """Generate cold-start top-N recommendations."""
        hour_cat = self.get_time_category()
        dow      = datetime.datetime.now().weekday()

        cold_recs = []
        for sku, product in all_products.items():
            ds = demand_scores.get(sku, 0)
            cs = self.score_for_cold_start(product, device_type, hour_cat, ds, dow)
            cold_recs.append({
                "id":         sku,
                "name":       product.get("name", ""),
                "category":   product.get("category", ""),
                "price":      product.get("current_price", 0),
                "rating":     product.get("rating", 0),
                "ml_score":   round(cs, 4),
                "interest":   0.0,    # no session history
                "similarity": 0.0,
                "trending":   round(ds / 100.0, 3),
                "cold_start": True,
                "reason":     (
                    f"{'🔥 Trending in' if ds > 50 else '⭐ Popular in'} "
                    f"{product.get('category', 'all categories')}"
                    + (f" · perfect for {device_type}" if device_type != "desktop" else "")
                    + (f" · {hour_cat} pick" if hour_cat and product.get('category') == hour_cat else "")
                ),
            })

        cold_recs.sort(key=lambda x: x["ml_score"], reverse=True)
        return cold_recs[:n]


# ═══════════════════════════════════════════════════════════
# STEP 15 + 17: REAL-TIME DECISION ENGINE (MAIN CLASS)
# ═══════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Phase 4 — Real-Time Decision Engine.

    Orchestrates Steps 15–18:
      step15()  → PricingDecision   (< 50ms)
      step17()  → List[RecommendationItem]  (< 20ms)
      step18()  → cold-start variant (< 20ms)
      decide()  → full DecisionBundle (< 100ms total)

    Designed to be called from:
      - POST  /event          (ml_engine.execute_ml_pipeline)
      - POST  /decide         (new direct endpoint)
      - GET   /price          (fallback pricing path)
      - GET   /recommendations (fallback rec path)
    """

    def __init__(self):
        self._adjustment_calc = DynamicAdjustmentCalculator()
        self._cold_start      = ColdStartHandler()
        self._model_registry  = None   # injected from ml_engine at runtime

    # ── Dependency injection ───────────────────────────────

    def set_model_registry(self, registry) -> None:
        self._model_registry = registry

    # ── Step 15: Live Pricing Decision ─────────────────────

    def step15_pricing(
        self,
        product:          Dict,
        demand_score_obj, # DemandScore dataclass from DemandPredictor
        feature_vec_obj,  # FeatureVector dataclass from FeatureEngine
        ab_group:         str = "B",
        competitor_data:  Optional[Dict] = None,
    ) -> PricingDecision:
        """
        Step 15: Live Pricing Decision.
        1. Pull real-time demand score from DemandPredictor.
        2. Apply DynamicAdjustmentCalculator (Step 16).
        3. Return PricingDecision with full explainability.
        """
        t0 = time.perf_counter()

        base_price    = product.get("base_price", 0.0)
        inventory     = product.get("inventory", 100)
        sku_id        = product.get("sku_id", "")
        demand_score  = getattr(demand_score_obj, "hybrid_score", 50.0)
        user_intent   = getattr(feature_vec_obj, "intent_score", 0.0)
        comp_price    = competitor_data.get("competitor_price") if competitor_data else None

        # Fetch coefficients from Phase 3 model (if trained)
        model_coefs = None
        if self._model_registry and self._model_registry.ready:
            model_coefs = self._model_registry.pricing.coefficients

        # Step 16: apply dynamic adjustment
        (
            final_price, adj_pct, tier,
            reasons, demand_level, inv_status, comp_gap
        ) = self._adjustment_calc.calculate(
            base_price        = base_price,
            demand_score      = demand_score,
            inventory         = inventory,
            user_intent       = user_intent,
            competitor_price  = comp_price,
            ab_group          = ab_group,
            model_coefficients= model_coefs,
        )

        # Enforce catalog min/max
        final_price = max(
            product.get("min_price", base_price * 0.70),
            min(final_price, product.get("max_price", base_price * 1.30))
        )
        final_price = round(final_price, 2)

        return PricingDecision(
            sku_id           = sku_id,
            base_price       = base_price,
            final_price      = final_price,
            adjustment_pct   = round(adj_pct, 2),
            tier             = tier,
            demand_score     = round(demand_score, 2),
            demand_level     = demand_level,
            inventory_status = inv_status,
            competitor_gap   = comp_gap,
            reasons          = reasons,
            ab_group         = ab_group,
            latency_ms       = round((time.perf_counter() - t0) * 1000, 3),
        )

    # ── Step 17: Recommendation Generation ─────────────────

    def step17_recommend(
        self,
        session:        Dict,              # session dict from SESSION_STORE
        all_products:   Dict[str, Dict],
        all_categories: Dict[str, List[str]],
        demand_scores:  Dict[str, float],  # sku_id → hybrid_score 0–100
        n:              int = 5,
    ) -> List[RecommendationItem]:
        """
        Step 17: Recommendation Generation.
        Rank candidates using Score = w1·Interest + w2·Similarity + w3·Trending.
        Pulls w1/w2/w3 from Phase 3 RecommendationModel (falls back to defaults).
        """
        t0 = time.perf_counter()

        clicked      = set(session.get("clicks", []))
        recent_cats  = session.get("categories", [])[-10:]  # last 10 categories
        cat_set      = set(c.lower() for c in recent_cats)
        total_cats   = len(recent_cats) or 1
        seen_skus: set = set()

        # Pull weights from Phase 3 model
        w1, w2, w3 = 0.40, 0.25, 0.35  # sensible defaults
        if self._model_registry and self._model_registry.ready and self._model_registry.recom.ready:
            w = self._model_registry.recom.weights
            w1 = w.get("w1_interest",   0.40)
            w2 = w.get("w2_similarity",  0.25)
            w3 = w.get("w3_trending",    0.35)

        # Build candidates: category-affinity first
        raw_candidates: List[Dict] = []
        for cat in recent_cats:
            for c_sku in all_categories.get(cat, []):
                if c_sku not in clicked and c_sku not in seen_skus and c_sku in all_products:
                    raw_candidates.append({"sku": c_sku, "from_cat": cat})
                    seen_skus.add(c_sku)

        # Fill with all remaining products (sorted by views for speed)
        if len(raw_candidates) < 20:
            trending_sorted = sorted(
                all_products.values(),
                key=lambda p: p.get("views", 0),
                reverse=True,
            )
            for p in trending_sorted:
                if p["sku_id"] not in clicked and p["sku_id"] not in seen_skus:
                    raw_candidates.append({"sku": p["sku_id"], "from_cat": None})
                    seen_skus.add(p["sku_id"])
                if len(raw_candidates) >= 50:
                    break

        # Score candidates (vectorised loop, O(n) in candidates)
        scored: List[Tuple[float, float, float, float, Dict]] = []
        for c in raw_candidates:
            sku     = c["sku"]
            product = all_products.get(sku, {})
            cat     = (product.get("category") or "").lower()

            # w1 — Interest: proportion of session in this category
            interest = sum(1 for rc in recent_cats if rc.lower() == cat) / total_cats

            # w2 — Similarity: in browsed category set?
            similarity = 1.0 if cat in cat_set else 0.3

            # w3 — Trending: normalised demand score
            trending = min(demand_scores.get(sku, 0) / 100.0, 1.0)

            ml_score = w1 * interest + w2 * similarity + w3 * trending
            scored.append((ml_score, interest, similarity, trending, product))

        # Sort by composite score descending, take top-N
        scored.sort(key=lambda x: x[0], reverse=True)

        items = []
        for rank, (ms, interest, sim, trend, product) in enumerate(scored[:n], start=1):
            ds = demand_scores.get(product.get("sku_id", ""), 0)
            # Generate explainability tag
            if trend > 0.6:
                tag = "🔥 Trending"
            elif interest > 0.3:
                tag = "🎯 For You"
            elif sim > 0.5:
                tag = "📦 Similar"
            else:
                tag = "💡 Discover"

            reason = (
                f"{tag}: "
                + (f"active in {product.get('category', '')} " if interest > 0.1 else "")
                + (f"· demand={ds:.0f}/100 " if trend > 0.3 else "")
                + f"· score={ms:.3f}"
            )

            items.append(RecommendationItem(
                rank       = rank,
                sku_id     = product.get("sku_id", ""),
                name       = product.get("name", ""),
                category   = product.get("category", ""),
                price      = product.get("current_price", 0),
                rating     = product.get("rating", 0),
                ml_score   = round(ms, 4),
                interest   = round(interest, 3),
                similarity = round(sim, 3),
                trending   = round(trend, 3),
                cold_start = False,
                tag        = tag,
                reason     = reason,
            ))

        logger.debug("step17_recommend: %d items in %.2fms",
                     len(items), (time.perf_counter() - t0) * 1000)
        return items

    # ── Step 18: Cold Start ─────────────────────────────────

    def step18_cold_start(
        self,
        all_products:  Dict[str, Dict],
        demand_scores: Dict[str, float],
        device_type:   str = "desktop",
        n:             int = 5,
    ) -> List[RecommendationItem]:
        """
        Step 18: Cold Start Handling.
        Uses device type + hour + trending when no session history exists.
        """
        raw = self._cold_start.generate(all_products, demand_scores, device_type, n)
        items = []
        for rank, r in enumerate(raw, start=1):
            ds = demand_scores.get(r["id"], 0)
            tag = "🔥 Trending" if ds > 50 else "🌟 Popular"
            items.append(RecommendationItem(
                rank       = rank,
                sku_id     = r["id"],
                name       = r["name"],
                category   = r["category"],
                price      = r["price"],
                rating     = r["rating"],
                ml_score   = r["ml_score"],
                interest   = r["interest"],
                similarity = r["similarity"],
                trending   = r["trending"],
                cold_start = True,
                tag        = tag,
                reason     = r["reason"],
            ))
        return items

    # ── Main entry: full DecisionBundle ────────────────────

    def decide(
        self,
        product:          Dict,
        session:          Dict,
        demand_score_obj, # DemandScore from DemandPredictor
        feature_vec_obj,  # FeatureVector from FeatureEngine
        all_products:     Dict[str, Dict],
        all_categories:   Dict[str, List[str]],
        demand_scores:    Dict[str, float],    # sku_id → hybrid_score 0–100
        ab_group:         str = "B",
        competitor_data:  Optional[Dict] = None,
        device_type:      str = "desktop",
        user_id:          str = "anonymous",
        session_id:       str = "anon",
    ) -> DecisionBundle:
        """
        Full real-time decision: pricing + recommendations + cold-start detection.
        Target latency: < 100ms total.
        """
        t0 = time.perf_counter()

        # Step 15 + 16: Pricing
        pricing = self.step15_pricing(
            product         = product,
            demand_score_obj= demand_score_obj,
            feature_vec_obj = feature_vec_obj,
            ab_group        = ab_group,
            competitor_data = competitor_data,
        )

        # Step 17 or 18: Recommendations
        is_cold_start = len(session.get("clicks", [])) == 0

        if is_cold_start:
            # Step 18: new user — device/time/trending-based
            recs = self.step18_cold_start(
                all_products  = all_products,
                demand_scores = demand_scores,
                device_type   = device_type,
                n             = 5,
            )
        else:
            # Step 17: warm user — session-aware ranking
            recs = self.step17_recommend(
                session        = session,
                all_products   = all_products,
                all_categories = all_categories,
                demand_scores  = demand_scores,
                n              = 5,
            )

        total_ms = (time.perf_counter() - t0) * 1000
        logger.debug("decide(): cold_start=%s total=%.2fms", is_cold_start, total_ms)

        return DecisionBundle(
            pricing          = pricing,
            recommendations  = recs,
            cold_start       = is_cold_start,
            session_id       = session_id,
            user_id          = user_id,
            total_latency_ms = round(total_ms, 2),
        )


# ── Module singleton ─────────────────────────────────────────
decision_engine = DecisionEngine()
