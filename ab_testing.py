"""
============================================================
  APEX — Phase 5: A/B Testing System (Steps 19–21)
============================================================

  Step 19  — User Split
             Group A → static base price  (control)
             Group B → dynamic ML price   (treatment)
             Deterministic hash split (same user always same group)

  Step 20  — Metrics Tracking
             Conversion rate  — purchases / sessions
             Revenue per user — total revenue / unique users
             CTR              — clicks / impressions
             Price uplift     — avg(B price) − avg(A price)
             Latency          — p50 / p95 / p99 per group

  Step 21  — Decision Rule
             If ML (Group B) conversion > Control (Group A) by threshold:
               → "deploy" (apply ML pricing to 100%)
             Elif B revenue_per_user > A by threshold:
               → "deploy_revenue" (revenue-driven deployment)
             Elif A wins on both metrics:
               → "adjust_weights" (rebalance model coefficients)
             Else:
               → "continue" (not enough data yet)
============================================================
"""

from __future__ import annotations

import math
import time
import hashlib
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ab_testing")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

MIN_SAMPLE_SIZE      = 30    # min events per group before decision
CONV_WIN_THRESHOLD   = 0.005 # B conversion rate must beat A by ≥ 0.5pp
REV_WIN_THRESHOLD    = 0.02  # B revenue/user must beat A by ≥ 2%
SIGNIFICANCE_LEVEL   = 0.05  # p-value threshold for statistical test


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class GroupMetrics:
    """Real-time metrics for one A/B group."""
    group:           str
    sessions:        int   = 0
    impressions:     int   = 0   # views/clicks shown
    clicks:          int   = 0   # product click events
    conversions:     int   = 0   # purchase events
    total_revenue:   float = 0.0
    total_latency_ms: float = 0.0
    latency_samples: int   = 0
    price_sum:       float = 0.0
    price_count:     int   = 0
    users:           set   = field(default_factory=set)

    @property
    def conversion_rate(self) -> float:
        return self.conversions / max(self.sessions, 1)

    @property
    def ctr(self) -> float:
        return self.clicks / max(self.impressions, 1)

    @property
    def revenue_per_user(self) -> float:
        return self.total_revenue / max(len(self.users), 1)

    @property
    def avg_price(self) -> float:
        return self.price_sum / max(self.price_count, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.latency_samples, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group":            self.group,
            "sessions":         self.sessions,
            "unique_users":     len(self.users),
            "impressions":      self.impressions,
            "clicks":           self.clicks,
            "conversions":      self.conversions,
            "conversion_rate":  round(self.conversion_rate, 4),
            "ctr":              round(self.ctr, 4),
            "total_revenue":    round(self.total_revenue, 2),
            "revenue_per_user": round(self.revenue_per_user, 2),
            "avg_price":        round(self.avg_price, 2),
            "avg_latency_ms":   round(self.avg_latency_ms, 2),
        }


@dataclass
class ABDecision:
    """Output of the Step 21 decision rule."""
    action:           str            # "deploy" | "deploy_revenue" | "adjust_weights" | "continue"
    winner:           Optional[str]  # "A" | "B" | None
    reason:           str
    confidence:       float          # 0–1 (1 = very confident)
    metrics_snapshot: Dict[str, Any]
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action":    self.action,
            "winner":    self.winner,
            "reason":    self.reason,
            "confidence": round(self.confidence, 4),
            "metrics":   self.metrics_snapshot,
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────────
# STEP 19: USER SPLIT
# ─────────────────────────────────────────────────────────────

class UserSplitter:
    """
    Step 19 — Deterministic user-to-group assignment.
    
    Uses MD5 hash of user_id + experiment_id so:
    - Same user → always same group across requests
    - ~50/50 split across large populations
    - No database needed (stateless, O(1))
    """

    def __init__(self, experiment_id: str = "apex_v1", split_pct: float = 0.50):
        """
        split_pct = fraction sent to Group B (ML dynamic).
        Group A (control) gets the rest.
        """
        self.experiment_id = experiment_id
        self.split_pct     = split_pct

    def assign(self, user_id: str) -> str:
        """Return 'A' (control/static) or 'B' (ML/dynamic)."""
        raw = f"{self.experiment_id}:{user_id}".encode("utf-8")
        hv  = int(hashlib.md5(raw).hexdigest(), 16)
        bucket = (hv % 1000) / 1000.0   # 0.000–0.999
        return "B" if bucket < self.split_pct else "A"

    def assign_session(self, session_id: str, user_id: str) -> str:
        """
        Prefer user_id for assignment (consistent across sessions).
        Falls back to session_id hash for anonymous users.
        """
        uid = user_id if user_id and user_id != "anonymous" else session_id
        return self.assign(uid)


# ─────────────────────────────────────────────────────────────
# STEP 20: METRICS TRACKER
# ─────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Step 20 — Real-time metrics accumulation for A/B groups.

    Thread-safe: uses per-group locks.
    All operations are O(1).
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self._groups: Dict[str, GroupMetrics] = {
            "A": GroupMetrics(group="A"),
            "B": GroupMetrics(group="B"),
        }
        # Circular buffer for recent events (for trend analysis)
        self._recent_events: list = []
        self._max_recent = 1000
        self._history: list = []   # snapshot log for time-series charts

    def record_event(
        self,
        group:        str,
        user_id:      str,
        event_type:   str,    # "view" | "click" | "add_to_cart" | "purchase"
        price:        float,
        revenue:      float   = 0.0,
        latency_ms:   float   = 0.0,
    ) -> None:
        """Record one event into the group's running metrics."""
        g_key = group if group in self._groups else "A"
        with self._lock:
            m = self._groups[g_key]
            m.users.add(user_id)
            m.sessions    += 1
            m.impressions += 1
            m.price_sum   += price
            m.price_count += 1

            if event_type in ("click", "add_to_cart"):
                m.clicks += 1
            if event_type == "purchase":
                m.conversions   += 1
                m.total_revenue += revenue or price

            if latency_ms > 0:
                m.total_latency_ms += latency_ms
                m.latency_samples  += 1

            # Rolling buffer
            self._recent_events.append({
                "group": g_key, "event": event_type,
                "price": price, "ts": time.time(),
            })
            if len(self._recent_events) > self._max_recent:
                self._recent_events.pop(0)

    def snapshot(self) -> Dict[str, Any]:
        """Thread-safe snapshot of current metrics for both groups."""
        with self._lock:
            return {
                "A": self._groups["A"].to_dict(),
                "B": self._groups["B"].to_dict(),
            }

    def save_history_point(self) -> None:
        """Save a time-stamped snapshot for the charts endpoint."""
        snap = self.snapshot()
        snap["timestamp"] = time.time()
        with self._lock:
            self._history.append(snap)
            if len(self._history) > 200:
                self._history = self._history[-200:]

    def get_history(self) -> list:
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        """Hard reset — use with caution (e.g. when starting new experiment)."""
        with self._lock:
            self._groups = {
                "A": GroupMetrics(group="A"),
                "B": GroupMetrics(group="B"),
            }
            self._recent_events.clear()
            self._history.clear()
        logger.info("MetricsTracker: reset")

    @property
    def groups(self) -> Dict[str, GroupMetrics]:
        return self._groups


# ─────────────────────────────────────────────────────────────
# STEP 21: DECISION RULE ENGINE
# ─────────────────────────────────────────────────────────────

class DecisionRuleEngine:
    """
    Step 21 — Automated A/B Test Decision Rule.

    Logic (evaluated in priority order):
      1. Not enough data  → "continue"
      2. B conversion > A + threshold  AND  z-test significant → "deploy"
      3. B revenue/user  > A + threshold                       → "deploy_revenue"
      4. A beats B on both metrics                             → "adjust_weights"
      5. Neither clear winner yet                              → "continue"

    "adjust_weights" means: tune the model's pricing formula so demand
    signals are weighted more conservatively (alpha, beta, gamma reduced 10%).
    """

    def evaluate(
        self,
        metrics: Dict[str, GroupMetrics],
        model_registry=None,      # optional: adjust weights if action == "adjust_weights"
    ) -> ABDecision:
        """Run decision rule. Returns ABDecision."""
        a = metrics.get("A", GroupMetrics("A"))
        b = metrics.get("B", GroupMetrics("B"))

        snap = {
            "A": a.to_dict(),
            "B": b.to_dict(),
            "evaluated_at": time.time(),
        }

        # 1. Insufficient data
        if a.sessions < MIN_SAMPLE_SIZE or b.sessions < MIN_SAMPLE_SIZE:
            return ABDecision(
                action="continue", winner=None,
                reason=(f"Insufficient data: A={a.sessions} sessions, "
                        f"B={b.sessions} sessions (need {MIN_SAMPLE_SIZE} each)"),
                confidence=0.0, metrics_snapshot=snap,
            )

        conv_a  = a.conversion_rate
        conv_b  = b.conversion_rate
        rev_a   = a.revenue_per_user
        rev_b   = b.revenue_per_user
        conv_diff = conv_b - conv_a
        rev_diff  = rev_b  - rev_a

        # 2. Z-test for conversion rate difference
        z_score, p_value = self._z_test_proportion(
            conv_a, a.sessions, conv_b, b.sessions
        )
        significant = p_value < SIGNIFICANCE_LEVEL
        confidence  = round(1 - p_value, 4) if p_value < 1 else 0.0

        if conv_diff >= CONV_WIN_THRESHOLD and significant:
            return ABDecision(
                action="deploy", winner="B",
                reason=(f"ML Group B wins: conv_rate A={conv_a:.3%} "
                        f"B={conv_b:.3%} (+{conv_diff:.3%}) "
                        f"z={z_score:.2f} p={p_value:.4f}"),
                confidence=confidence, metrics_snapshot=snap,
            )

        # 3. Revenue win (even without statistical significance on conversion)
        if rev_diff >= rev_a * REV_WIN_THRESHOLD:
            return ABDecision(
                action="deploy_revenue", winner="B",
                reason=(f"ML Group B wins on revenue: "
                        f"A=${rev_a:.2f}/user vs B=${rev_b:.2f}/user "
                        f"(+{rev_diff/max(rev_a,1e-9):.1%})"),
                confidence=min(confidence + 0.1, 1.0), metrics_snapshot=snap,
            )

        # 4. Control group (A) wins → adjust model weights
        if conv_a > conv_b + CONV_WIN_THRESHOLD or rev_a > rev_b + rev_a * REV_WIN_THRESHOLD:
            if model_registry is not None:
                self._adjust_weights(model_registry)
            return ABDecision(
                action="adjust_weights", winner="A",
                reason=(f"Control A outperforms ML B: "
                        f"conv A={conv_a:.3%} B={conv_b:.3%}, "
                        f"rev A=${rev_a:.2f} B=${rev_b:.2f} — "
                        f"model weights reduced 10%"),
                confidence=max(confidence, 0.5), metrics_snapshot=snap,
            )

        # 5. Continue — not enough separation
        return ABDecision(
            action="continue", winner=None,
            reason=(f"No clear winner yet: "
                    f"conv diff={conv_diff:+.3%}, rev diff=${rev_diff:+.2f}/user, "
                    f"z={z_score:.2f} p={p_value:.3f}"),
            confidence=confidence, metrics_snapshot=snap,
        )

    @staticmethod
    def _z_test_proportion(
        p1: float, n1: int, p2: float, n2: int
    ) -> Tuple[float, float]:
        """
        Two-proportion z-test (two-tailed).
        Returns (z_score, p_value).
        """
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        if se < 1e-10:
            return 0.0, 1.0
        z = (p2 - p1) / se
        # Two-tailed p-value: multiply one-tailed area by 2
        one_tail = 1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
        p_val = min(2 * one_tail, 1.0)
        return round(z, 4), round(p_val, 6)

    @staticmethod
    def _adjust_weights(model_registry) -> None:
        """Reduce model pricing formula coefficients by 10% (less aggressive)."""
        try:
            pm = model_registry.pricing
            with pm._lock:
                pm._alpha = max(pm._alpha * 0.90, 0.005)
                pm._beta  = max(pm._beta  * 0.90, 0.002)
                pm._gamma = max(pm._gamma * 0.90, 0.002)
            logger.info(
                "AB: weights adjusted → α=%.4f β=%.4f γ=%.4f",
                pm._alpha, pm._beta, pm._gamma
            )
        except Exception as e:
            logger.warning("AB: weight adjustment failed: %s", e)


# ─────────────────────────────────────────────────────────────
# FACADE — single object imported by server.py
# ─────────────────────────────────────────────────────────────

class ABTestingSystem:
    """
    Singleton facade.
    server.py imports `ab_system` and calls:
      - ab_system.assign(user_id, session_id) → "A" | "B"
      - ab_system.record(group, user_id, event, price, revenue, latency_ms)
      - ab_system.metrics     → current snapshot dict
      - ab_system.evaluate()  → ABDecision
      - ab_system.history     → list of time-series snapshots
    """

    def __init__(self):
        self.splitter  = UserSplitter()
        self.tracker   = MetricsTracker()
        self.rule      = DecisionRuleEngine()
        self._last_decision: Optional[ABDecision] = None
        self._model_registry = None

        # Background snapshot timer (every 30 s)
        self._snapshot_thread = threading.Thread(
            target=self._snapshot_loop, daemon=True, name="ab-snapshot"
        )
        self._snapshot_thread.start()

    def set_model_registry(self, registry) -> None:
        self._model_registry = registry

    def assign(self, user_id: str, session_id: str = "") -> str:
        """Step 19: assign user to A or B group."""
        return self.splitter.assign_session(session_id, user_id)

    def record(
        self,
        group:      str,
        user_id:    str,
        event_type: str,
        price:      float,
        revenue:    float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """Step 20: record one event."""
        self.tracker.record_event(group, user_id, event_type, price, revenue, latency_ms)

    def evaluate(self) -> ABDecision:
        """Step 21: run decision rule and cache result."""
        decision = self.rule.evaluate(
            self.tracker.groups, self._model_registry
        )
        self._last_decision = decision
        return decision

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.tracker.snapshot()

    @property
    def last_decision(self) -> Optional[Dict]:
        return self._last_decision.to_dict() if self._last_decision else None

    @property
    def history(self) -> list:
        return self.tracker.get_history()

    def reset(self) -> None:
        self.tracker.reset()
        self._last_decision = None

    def _snapshot_loop(self) -> None:
        while True:
            time.sleep(30)
            try:
                self.tracker.save_history_point()
            except Exception:
                pass


# Module singleton
ab_system = ABTestingSystem()
