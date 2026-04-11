"""
============================================================
  APEX — Phase 6: Explainability Engine (Step 22)
============================================================

  Generates structured, human-readable explanations for:
    • Every price change (why did it go up/down?)
    • Every recommendation (why is this product shown?)
    • Challenge solutions (how the system handled cold-start, latency, etc.)

  Design:
    - Template-based (O(1) per explanation)
    - Multi-level: one-liner + detailed breakdown + badge
    - Fully JSON-serialisable (safe for API responses)
    - Integrated into /event, /decide, /simulate endpoints
============================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════
# OUTPUT DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class PriceExplanation:
    """Step 22 — Full pricing explainability output."""
    headline:        str            # one-liner: "Price ↑ due to high demand"
    badge:           str            # emoji badge: "🔥 Surge" | "📉 Discount" etc.
    badge_color:     str            # hex color for UI badge
    factors:         List[Dict]     # ordered list of factor dicts
    summary:         str            # paragraph-length explanation
    challenge_note:  Optional[str]  # active challenge mitigation note

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationExplanation:
    """Step 22 — Recommendation explainability for one item."""
    headline:    str
    badge:       str
    badge_color: str
    factors:     List[Dict]
    cold_start:  bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════
# FACTOR BUILDERS (internal helpers)
# ═══════════════════════════════════════════════════════════

def _factor(icon: str, label: str, value: str, impact: str, color: str) -> Dict:
    """impact: 'positive' | 'negative' | 'neutral'"""
    return {"icon": icon, "label": label, "value": value, "impact": impact, "color": color}


# ═══════════════════════════════════════════════════════════
# STEP 22A: PRICING EXPLAINABILITY
# ═══════════════════════════════════════════════════════════

class PricingExplainer:
    """
    Attach rich explainability to every pricing decision.

    Covers all 5 challenge mitigations mentioned in Phase 6:
      C1: Data Delay    → "Using real-time demand score (no historical lag)"
      C2: Overfitting   → "Simple linear model keeps pricing stable"
      C3: Latency       → "Precomputed weights: decision in <1ms"
      C4: Cold Start    → "Trending signal used for new products"
      C5: Unstable Prices → "Min/max guardrails enforced (±15%)"
    """

    # Badge definitions: (demand_level, inventory_status) → (label, emoji, hex_color)
    BADGES = {
        ("high",   "scarce"):  ("🚀 Surge Pricing",   "#ff4757"),
        ("high",   "normal"):  ("📈 Dynamic Boost",   "#ff6b35"),
        ("high",   "surplus"): ("📊 Demand Driven",   "#ffa502"),
        ("medium", "scarce"):  ("🔥 Urgency",         "#ff6348"),
        ("medium", "normal"):  ("⚖️ Optimal",         "#2ed573"),
        ("medium", "surplus"): ("📦 Clearance Prep",  "#747d8c"),
        ("low",    "scarce"):  ("💎 Value Hold",      "#5352ed"),
        ("low",    "normal"):  ("📉 Recovery Mode",   "#1e90ff"),
        ("low",    "surplus"): ("🏷️ Clearance",       "#70a1ff"),
    }

    def explain(
        self,
        base_price:      float,
        final_price:     float,
        demand_score:    float,    # 0–100
        demand_level:    str,      # "high"|"medium"|"low"
        inventory:       int,
        inventory_status: str,     # "scarce"|"normal"|"surplus"
        user_intent:     float,
        ab_group:        str,
        competitor_gap:  Optional[float],
        raw_reasons:     List[str],   # from DecisionEngine
        latency_ms:      float = 0.0,
        is_new_product:  bool = False,
    ) -> PriceExplanation:
        """Generate full pricing explanation."""

        pct_change = ((final_price - base_price) / base_price) * 100 if base_price > 0 else 0
        badge_key  = (demand_level, inventory_status)
        badge_text, badge_color = self.BADGES.get(
            badge_key, ("⚖️ Balanced", "#2ed573")
        )

        # ── Headline ─────────────────────────────────────────
        if ab_group == "A":
            headline = "📊 Control group — static base price (no ML adjustment)"
        elif pct_change > 8:
            headline = f"🚀 Price increased {pct_change:+.1f}% — surge demand & low stock"
        elif pct_change > 2:
            headline = f"📈 Price increased {pct_change:+.1f}% — demand above threshold"
        elif pct_change < -10:
            headline = f"🏷️ Price reduced {pct_change:.1f}% — clearance mode active"
        elif pct_change < -2:
            headline = f"📉 Price reduced {pct_change:.1f}% — low demand recovery"
        else:
            headline = "⚖️ Price stable — balanced supply-demand signals"

        # ── Factor list ──────────────────────────────────────
        factors: List[Dict] = []

        # Demand factor
        demand_bar = min(int(demand_score), 100)
        demand_col = "#ff4757" if demand_score > 70 else ("#ffa502" if demand_score > 40 else "#1e90ff")
        factors.append(_factor(
            "📊", "Demand Score",
            f"{demand_score:.1f}/100",
            "positive" if demand_score > 50 else "negative",
            demand_col,
        ))

        # Inventory factor
        stock_col = "#ff4757" if inventory < 20 else ("#2ed573" if inventory > 200 else "#ffa502")
        factors.append(_factor(
            "📦", "Inventory",
            f"{inventory} units ({inventory_status})",
            "positive" if inventory_status == "scarce" else ("negative" if inventory_status == "surplus" else "neutral"),
            stock_col,
        ))

        # User intent factor
        if user_intent > 0:
            intent_col = "#ff6b35" if user_intent > 15 else "#747d8c"
            factors.append(_factor(
                "🎯", "User Intent Score",
                f"{user_intent:.1f} (session actions)",
                "positive" if user_intent > 10 else "neutral",
                intent_col,
            ))

        # Competitor factor
        if competitor_gap is not None:
            comp_col = "#2ed573" if competitor_gap < 0 else "#ff4757"
            factors.append(_factor(
                "🏪", "vs Competitor",
                f"${abs(competitor_gap):.2f} {'cheaper than us' if competitor_gap > 0 else 'above competitor'}",
                "negative" if competitor_gap > 5 else "positive",
                comp_col,
            ))

        # A/B group factor
        factors.append(_factor(
            "🔬", "A/B Group",
            f"Group {ab_group} ({'Static' if ab_group == 'A' else 'ML Dynamic'})",
            "neutral",
            "#747d8c",
        ))

        # Price guardrails factor
        factors.append(_factor(
            "🛡️", "Guardrails",
            f"±15% cap · floor=${base_price*0.85:.2f} · ceil=${base_price*1.15:.2f}",
            "neutral",
            "#5352ed",
        ))

        # ── Challenge mitigations note ───────────────────────
        notes = []
        if latency_ms < 10:
            notes.append("⚡ C3 solved: precomputed weights → decision in <1ms")
        if is_new_product:
            notes.append("🌟 C4 solved: new product — trending signal used (cold-start)")
        if inventory_status == "scarce" or inventory_status == "surplus":
            notes.append("🛡️ C5 solved: min/max price bounds enforced (±15% cap)")

        # ── Summary paragraph ────────────────────────────────
        summary_parts = []
        if ab_group == "A":
            summary_parts.append("This user is in the Control Group (A) — base price is shown without ML adjustment to measure baseline conversion.")
        else:
            summary_parts.append(
                f"The ML engine analysed real-time signals and adjusted the price from "
                f"${base_price:.2f} to ${final_price:.2f} ({pct_change:+.1f}%). "
            )
            if demand_score > 60:
                summary_parts.append(f"Demand is high ({demand_score:.0f}/100), indicating strong buying intent in this session window. ")
            elif demand_score < 35:
                summary_parts.append(f"Demand is low ({demand_score:.0f}/100) — a recovery discount is applied to stimulate conversion. ")
            if inventory < 20:
                summary_parts.append(f"Stock is critically low ({inventory} units), triggering an urgency boost. ")
            elif inventory > 200:
                summary_parts.append(f"High inventory ({inventory} units) applies downward pressure to accelerate sell-through. ")
            if competitor_gap and competitor_gap > 5:
                summary_parts.append(f"Competitor is cheaper by ${competitor_gap:.2f} — partial match applied. ")

        return PriceExplanation(
            headline       = headline,
            badge          = badge_text,
            badge_color    = badge_color,
            factors        = factors,
            summary        = "".join(summary_parts),
            challenge_note = " | ".join(notes) if notes else None,
        )


# ═══════════════════════════════════════════════════════════
# STEP 22B: RECOMMENDATION EXPLAINABILITY
# ═══════════════════════════════════════════════════════════

class RecommendationExplainer:
    """Attach rich explanations to each recommendation item."""

    def explain_item(
        self,
        name:        str,
        category:    str,
        interest:    float,    # 0–1
        similarity:  float,    # 0–1
        trending:    float,    # 0–1
        ml_score:    float,    # composite
        cold_start:  bool,
        session_cats: List[str],
        device_type:  str = "desktop",
    ) -> RecommendationExplanation:
        """Generate explanation for one recommendation item."""

        factors: List[Dict] = []

        # Interest
        if interest > 0:
            factors.append(_factor(
                "🎯", "Category Match",
                f"{interest*100:.0f}% of your session in {category}",
                "positive", "#2ed573",
            ))
        elif cold_start:
            factors.append(_factor(
                "📱", "Device Context",
                f"Top pick for {device_type} users",
                "neutral", "#747d8c",
            ))

        # Trending signal
        factors.append(_factor(
            "🔥", "Trending Score",
            f"{trending*100:.0f}/100 demand velocity",
            "positive" if trending > 0.5 else "neutral",
            "#ff6b35" if trending > 0.5 else "#747d8c",
        ))

        # Similarity
        if similarity > 0.5:
            factors.append(_factor(
                "📦", "Category Similarity",
                f"Similar category to recently browsed items",
                "positive", "#1e90ff",
            ))

        # ML score breakdown
        factors.append(_factor(
            "🧠", "ML Score",
            f"{ml_score:.3f} (w1·Interest + w2·Similarity + w3·Trending)",
            "positive" if ml_score > 0.5 else "neutral",
            "#5352ed",
        ))

        # Headline
        if cold_start:
            headline = f"🌟 Trending pick for {device_type} users — personalisation warming up"
        elif interest > 0.4:
            headline = f"🎯 Recommended because you browsed {category} products"
        elif trending > 0.6:
            headline = f"🔥 Trending in {category} — high demand right now"
        elif similarity > 0.7:
            headline = f"📦 Similar to items you've clicked on"
        else:
            headline = f"💡 Discover {category} — popular with shoppers like you"

        # Badge
        if trending > 0.7:
            badge, color = "🔥 Hot Right Now", "#ff4757"
        elif interest > 0.4:
            badge, color = "🎯 Personalised",  "#2ed573"
        elif cold_start:
            badge, color = "🌟 Popular",       "#ffa502"
        else:
            badge, color = "💡 Suggested",     "#5352ed"

        return RecommendationExplanation(
            headline    = headline,
            badge       = badge,
            badge_color = color,
            factors     = factors,
            cold_start  = cold_start,
        )


# ═══════════════════════════════════════════════════════════
# STEP 23: PRICE SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════

# Scenario matrix from the spec
SIMULATION_SCENARIOS = [
    # label,  demand_level, demand_score, inventory, inv_status, expected_direction
    ("High Demand, Low Stock",   "high",   82, 8,   "scarce",  "up_surge"),
    ("High Demand, High Stock",  "high",   75, 800, "surplus", "up_moderate"),
    ("Low Demand,  High Stock",  "low",    18, 700, "surplus", "down_clearance"),
    ("Low Demand,  Low Stock",   "low",    22, 12,  "scarce",  "hold"),
    ("Medium Demand, Normal",    "medium", 50, 150, "normal",  "stable"),
    ("Surge + Critical Stock",   "high",   95, 3,   "scarce",  "up_max"),
]


class PriceSimulator:
    """
    Step 23 — Simulate pricing intelligence across demand×inventory scenarios.

    Shows system intelligence:
      High demand × Low stock  → price up (urgency)
      High demand × High stock → price up (but less, surplus dampens)
      Low demand  × High stock → price down (clearance)
      Low demand  × Low stock  → hold (protect margin despite low demand)
    """

    def __init__(self):
        try:
            from decision_engine import DynamicAdjustmentCalculator
            self._calc = DynamicAdjustmentCalculator()
        except ImportError:
            self._calc = None

        self._explainer = PricingExplainer()

    def run_scenario(
        self,
        base_price:       float,
        demand_score:     float,
        inventory:        int,
        user_intent:      float = 5.0,
        competitor_price: Optional[float] = None,
        ab_group:         str = "B",
        model_coefs:      Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Run one simulation scenario and return full output."""
        if self._calc is None:
            return {"error": "DecisionEngine not available"}

        final_price, adj_pct, tier, reasons, demand_level, inv_status, comp_gap = \
            self._calc.calculate(
                base_price        = base_price,
                demand_score      = demand_score,
                inventory         = inventory,
                user_intent       = user_intent,
                competitor_price  = competitor_price,
                ab_group          = ab_group,
                model_coefficients= model_coefs,
            )

        expl = None
        if self._explainer:
            expl = self._explainer.explain(
                base_price       = base_price,
                final_price      = final_price,
                demand_score     = demand_score,
                demand_level     = demand_level,
                inventory        = inventory,
                inventory_status = inv_status,
                user_intent      = user_intent,
                ab_group         = ab_group,
                competitor_gap   = comp_gap,
                raw_reasons      = reasons,
            )

        return {
            "base_price":       base_price,
            "final_price":      final_price,
            "adjustment_pct":   round(adj_pct, 2),
            "tier":             tier,
            "demand_score":     demand_score,
            "demand_level":     demand_level,
            "inventory":        inventory,
            "inventory_status": inv_status,
            "reasons":          reasons,
            "explanation":      expl.to_dict() if expl else None,
        }

    def run_matrix(
        self,
        base_price: float = 1000.0,
        model_coefs: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Run the full 6-scenario simulation matrix.
        Returns list of scenario result dicts.
        """
        results = []
        for label, dlevel, dscore, inv, inv_status, direction in SIMULATION_SCENARIOS:
            result = self.run_scenario(
                base_price   = base_price,
                demand_score = dscore,
                inventory    = inv,
                user_intent  = 8.0,
                model_coefs  = model_coefs,
            )
            result["scenario_label"]     = label
            result["expected_direction"] = direction
            results.append(result)
        return results


# ── Module singletons ──
pricing_explainer        = PricingExplainer()
recommendation_explainer = RecommendationExplainer()
price_simulator          = PriceSimulator()
