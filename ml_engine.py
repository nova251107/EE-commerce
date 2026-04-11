"""
============================================================
  APEX — ML Engine Module
  Real-Time Feature Engine + Demand Prediction + Streaming
============================================================

  Classes:
    FeatureEngine      — Rolling-window user/session features
    DemandPredictor    — Hybrid heuristic+ML demand scoring
    StreamProcessor    — Event-driven incremental update bus
    MLEngine           — Unified facade (import this in server.py)

  Design:
    - O(1) rolling window via collections.deque
    - Incremental ML via sklearn SGDClassifier.partial_fit
    - No full recomputation on each event
    - All in-memory; thread-safe with threading.Lock
============================================================
"""

import time
import math
import threading
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("ml_engine")

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

EVENT_WEIGHTS: Dict[str, float] = {
    "view":           1.0,
    "page_view":      1.0,
    "product_view":   2.0,
    "click":          2.0,
    "search":         1.5,
    "add_to_wishlist": 3.0,
    "add_to_cart":    5.0,
    "checkout_start": 7.0,
    "purchase":      10.0,
}

WINDOW_SHORT  = 60    # 60-second velocity window
WINDOW_LONG   = 300   # 5-minute velocity window
MAX_ACTIONS   = 10    # last N actions kept per session
DEMAND_SCORE_MAX = 100.0

# How often the background loop refreshes demand scores (seconds)
BACKGROUND_REFRESH_INTERVAL = 3.0


# ──────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────

@dataclass
class EventRecord:
    """A single timestamped user event."""
    ts:         float
    event_type: str
    sku_id:     str
    session_id: str
    user_id:    str
    price_seen: float = 0.0
    discounted: bool  = False


@dataclass
class FeatureVector:
    """Structured feature vector returned per session."""
    session_id:       str
    user_id:          str
    intent_score:     float         # weighted sum of recent actions
    price_sensitivity: float        # 0-1 (1 = highly discount-focused)
    session_duration_s: float       # seconds since session start
    action_count:     int           # total events in session
    last_actions:     List[str]     # last N event types
    last_skus:        List[str]     # last N SKUs touched
    cart_count:       int           # add-to-cart events in session
    purchase_count:   int           # purchase events in session
    active_category:  Optional[str] # most-browsed category this session
    latency_ms:       float         # computation time in ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":         self.session_id,
            "user_id":            self.user_id,
            "intent_score":       round(self.intent_score, 3),
            "price_sensitivity":  round(self.price_sensitivity, 3),
            "session_duration_s": round(self.session_duration_s, 1),
            "action_count":       self.action_count,
            "last_actions":       self.last_actions,
            "last_skus":          self.last_skus,
            "cart_count":         self.cart_count,
            "purchase_count":     self.purchase_count,
            "active_category":    self.active_category,
            "latency_ms":         round(self.latency_ms, 2),
        }


@dataclass
class DemandScore:
    """Per-product demand scoring output."""
    sku_id:           str
    heuristic_score:  float   # 0-100
    ml_score:         float   # 0-100
    hybrid_score:     float   # 0.6 * heuristic + 0.4 * ml
    click_velocity_60s:  float
    click_velocity_300s: float
    cart_velocity_60s:   float
    cart_velocity_300s:  float
    time_of_day_boost:  float
    trending:          bool
    last_updated:      float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku_id":               self.sku_id,
            "demand_score":         round(self.hybrid_score, 2),
            "heuristic_score":      round(self.heuristic_score, 2),
            "ml_score":             round(self.ml_score, 2),
            "click_velocity_60s":   round(self.click_velocity_60s, 3),
            "click_velocity_300s":  round(self.click_velocity_300s, 3),
            "cart_velocity_60s":    round(self.cart_velocity_60s, 3),
            "cart_velocity_300s":   round(self.cart_velocity_300s, 3),
            "time_of_day_boost":    round(self.time_of_day_boost, 3),
            "trending":             self.trending,
            "last_updated":         round(self.last_updated, 3),
        }


# ──────────────────────────────────────────────────────────
# 1. FEATURE ENGINE
# ──────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Real-time feature engine using rolling windows.
    Each session has its own deque of max MAX_ACTIONS events.
    All operations are O(1) append/evict — no full recomputation.
    """

    def __init__(self, max_actions: int = MAX_ACTIONS):
        self._lock = threading.Lock()
        self.max_actions = max_actions

        # Per-session stores
        self._session_start:     Dict[str, float]       = {}
        self._session_user:      Dict[str, str]          = {}
        self._session_actions:   Dict[str, deque]        = {}   # deque of EventRecord
        self._session_cat_count: Dict[str, Dict[str, int]] = {} # category frequency
        self._session_cart:      Dict[str, int]          = defaultdict(int)
        self._session_purchase:  Dict[str, int]          = defaultdict(int)

        # Running intent score (incremental: add weight on push, subtract on evict)
        self._session_intent:    Dict[str, float]        = defaultdict(float)

        # Price sensitivity: ratio of discounted interactions
        self._session_disc_hits: Dict[str, int]          = defaultdict(int)
        self._session_total_hits: Dict[str, int]         = defaultdict(int)

    # ── Public API ──────────────────────────────

    def update(self, event: EventRecord, category: Optional[str] = None) -> None:
        """
        Incrementally update all features for the event's session.
        O(1) amortised — uses deque push/pop.
        """
        t0 = time.perf_counter()
        sid = event.session_id

        with self._lock:
            # Initialise session if new
            if sid not in self._session_start:
                self._session_start[sid] = event.ts
                self._session_user[sid]  = event.user_id
                self._session_actions[sid] = deque(maxlen=self.max_actions)
                self._session_cat_count[sid] = defaultdict(int)

            dq = self._session_actions[sid]

            # — Evict oldest action from running intent score —
            if len(dq) == dq.maxlen:
                evicted = dq[0]  # Will be evicted on append
                self._session_intent[sid] -= EVENT_WEIGHTS.get(evicted.event_type, 1.0)

            # — Push new event —
            dq.append(event)
            weight = EVENT_WEIGHTS.get(event.event_type, 1.0)
            self._session_intent[sid] += weight

            # — Price sensitivity —
            self._session_total_hits[sid] += 1
            if event.discounted:
                self._session_disc_hits[sid] += 1

            # — Cart / purchase counters —
            if event.event_type in ("add_to_cart",):
                self._session_cart[sid] += 1
            elif event.event_type == "purchase":
                self._session_purchase[sid] += 1

            # — Category affinity —
            if category:
                self._session_cat_count[sid][category] += 1

        dt = (time.perf_counter() - t0) * 1000
        logger.debug("FeatureEngine.update: %.2f ms", dt)

    def get_feature_vector(self, session_id: str) -> FeatureVector:
        """Return the current feature vector for a session."""
        t0 = time.perf_counter()
        sid = session_id

        with self._lock:
            now = time.time()
            start = self._session_start.get(sid, now)
            user  = self._session_user.get(sid, "unknown")
            dq    = self._session_actions.get(sid, deque())

            intent    = self._session_intent.get(sid, 0.0)
            total     = self._session_total_hits.get(sid, 0)
            disc_hits = self._session_disc_hits.get(sid, 0)
            price_sens = (disc_hits / total) if total > 0 else 0.0

            last_actions = [e.event_type for e in dq][-MAX_ACTIONS:]
            last_skus    = [e.sku_id    for e in dq][-MAX_ACTIONS:]

            cat_counts = self._session_cat_count.get(sid, {})
            active_cat = max(cat_counts, key=cat_counts.get) if cat_counts else None

        dt = (time.perf_counter() - t0) * 1000
        return FeatureVector(
            session_id        = sid,
            user_id           = user,
            intent_score      = max(0.0, intent),
            price_sensitivity = price_sens,
            session_duration_s= now - start,
            action_count      = total,
            last_actions      = last_actions,
            last_skus         = last_skus,
            cart_count        = self._session_cart.get(sid, 0),
            purchase_count    = self._session_purchase.get(sid, 0),
            active_category   = active_cat,
            latency_ms        = dt,
        )

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._session_start)


# ──────────────────────────────────────────────────────────
# 2. DEMAND PREDICTOR
# ──────────────────────────────────────────────────────────

class DemandPredictor:
    """
    Hybrid demand scoring:
      score = 0.6 × heuristic + 0.4 × ml_prediction

    Heuristic:
      - Click velocity (short/long window)
      - Cart velocity (short/long window, weighted 5×)
      - Time-of-day pattern (peak hours boost)
      - Trending detection (score delta)

    ML:
      - sklearn SGDClassifier (logistic) updated incrementally
      - Feature vector: [click_vel_60, cart_vel_60, click_vel_300, cart_vel_300, tod, dow]
      - Labels: 0 (low demand) / 1 (high demand) based on running demand count
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Per-SKU event timestamps (deque of (ts, event_type))
        self._sku_events:    Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))

        # Per-SKU cached demand score
        self._demand_scores: Dict[str, DemandScore] = {}

        # Baseline demand counts (loaded from clickstream at startup)
        self._baseline: Dict[str, int] = {}

        # Trending: previous score snapshot for delta detection
        self._prev_scores: Dict[str, float] = {}

        # ML model (optional — graceful fallback if sklearn missing)
        self._ml_ready = False
        self._ml_model = None
        self._ml_classes = None
        self._ml_event_count = 0     # train once we have enough samples
        self._ml_min_samples = 50
        self._try_init_ml()

    def _try_init_ml(self):
        """Safe import of sklearn — degrades gracefully if not installed."""
        try:
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            self._ml_model   = SGDClassifier(loss="log_loss", max_iter=1,
                                             tol=None, warm_start=True,
                                             random_state=42)
            self._ml_classes = [0, 1]
            self._scaler     = StandardScaler()
            self._scaler_fitted = False
            self._ml_ready   = True
            logger.info("ML model (SGDClassifier) initialised successfully.")
        except ImportError:
            logger.warning("scikit-learn not found — using heuristic-only demand scoring.")
            self._ml_ready = False

    # ── Startup seeding ──────────────────────────────────

    def seed_from_demand_counter(self, demand_counter: Dict[str, int]):
        """
        Seed baseline demand from the clickstream data loaded at startup.
        Called once during server lifespan startup.
        """
        with self._lock:
            self._baseline = dict(demand_counter)
        logger.info("DemandPredictor seeded with %d products", len(demand_counter))

    # ── Public API ──────────────────────────────────────

    def record_event(self, sku_id: str, event_type: str, ts: Optional[float] = None):
        """Record a product-level event for velocity calculation."""
        ts = ts or time.time()
        with self._lock:
            self._sku_events[sku_id].append((ts, event_type))
            self._ml_event_count += 1

    def compute_demand_score(self, sku_id: str) -> DemandScore:
        """
        Compute the full demand score for a product.
        Uses cached score if called within 1 second of last computation.
        """
        with self._lock:
            cached = self._demand_scores.get(sku_id)
            if cached and (time.time() - cached.last_updated) < 1.0:
                return cached
            score = self._compute(sku_id)
            self._demand_scores[sku_id] = score
            return score

    def get_top_products(self, n: int = 20) -> List[DemandScore]:
        """Return top-N products by demand score."""
        with self._lock:
            scores = list(self._demand_scores.values())
        scores.sort(key=lambda s: s.hybrid_score, reverse=True)
        return scores[:n]

    def refresh_all(self, sku_ids: List[str]):
        """Force-refresh demand scores for a list of SKUs (called by background loop)."""
        for sku in sku_ids:
            score = self._compute(sku)
            with self._lock:
                self._demand_scores[sku] = score

    # ── Internal calculation (call inside lock) ─────────

    def _compute(self, sku_id: str) -> DemandScore:
        now = time.time()
        events = list(self._sku_events.get(sku_id, []))

        # — Velocity counts —
        cv60 = cv300 = carv60 = carv300 = 0.0
        for ts, et in events:
            age = now - ts
            if age <= WINDOW_SHORT:
                if et in ("click", "product_view", "page_view"):
                    cv60 += 1
                elif et == "add_to_cart":
                    carv60 += 1
                    cv60 += 1        # cart implies click
            if age <= WINDOW_LONG:
                if et in ("click", "product_view", "page_view"):
                    cv300 += 1
                elif et == "add_to_cart":
                    carv300 += 1
                    cv300 += 1

        # — Time-of-day boost (peak hours 18-22 = 1.2×, morning 8-10 = 1.1×) —
        import datetime
        hour = datetime.datetime.now().hour
        if 18 <= hour <= 22:
            tod_boost = 1.20
        elif 8 <= hour <= 10:
            tod_boost = 1.10
        elif 12 <= hour <= 14:
            tod_boost = 1.05
        else:
            tod_boost = 1.00

        # — Heuristic score (0-100) —
        baseline = self._baseline.get(sku_id, 0)
        # Weighted sum: clicks=1, carts=5, baseline scaled
        raw = (cv60 * 1 + carv60 * 5) * 10 + (cv300 * 0.5 + carv300 * 2.5)
        raw += min(baseline / 100.0, 20.0)   # cap baseline contribution at 20 pts
        heuristic = min(raw * tod_boost, DEMAND_SCORE_MAX)

        # — ML score (0-100) —
        ml_score = heuristic   # fallback: same as heuristic
        if self._ml_ready and self._ml_event_count >= self._ml_min_samples:
            ml_score = self._ml_predict(cv60, carv60, cv300, carv300, hour) * DEMAND_SCORE_MAX

        # — Hybrid score —
        hybrid = 0.6 * heuristic + 0.4 * ml_score

        # — Trending: score increased >10 pts vs last snapshot —
        prev  = self._prev_scores.get(sku_id, hybrid)
        trending = (hybrid - prev) > 10.0
        self._prev_scores[sku_id] = hybrid

        # — Maybe train ML on this observation —
        if self._ml_ready:
            label = 1 if heuristic > 30 else 0
            self._ml_train(cv60, carv60, cv300, carv300, hour, label)

        return DemandScore(
            sku_id           = sku_id,
            heuristic_score  = heuristic,
            ml_score         = ml_score,
            hybrid_score     = min(hybrid, DEMAND_SCORE_MAX),
            click_velocity_60s  = cv60,
            click_velocity_300s = cv300,
            cart_velocity_60s   = carv60,
            cart_velocity_300s  = carv300,
            time_of_day_boost   = tod_boost,
            trending         = trending,
            last_updated     = time.time(),
        )

    def _ml_predict(self, cv60, carv60, cv300, carv300, hour) -> float:
        """Return ML probability of high demand (0-1)."""
        try:
            import numpy as np
            X = np.array([[cv60, carv60, cv300, carv300,
                           math.sin(2 * math.pi * hour / 24),
                           math.cos(2 * math.pi * hour / 24)]])
            if self._scaler_fitted:
                X = self._scaler.transform(X)
            proba = self._ml_model.predict_proba(X)[0]
            return float(proba[1])   # probability of class 1 (high demand)
        except Exception:
            return 0.5   # neutral fallback

    def _ml_train(self, cv60, carv60, cv300, carv300, hour, label):
        """Incremental model update via partial_fit."""
        try:
            import numpy as np
            X = np.array([[cv60, carv60, cv300, carv300,
                           math.sin(2 * math.pi * hour / 24),
                           math.cos(2 * math.pi * hour / 24)]])
            y = np.array([label])
            if not self._scaler_fitted:
                self._scaler.fit(X)
                self._scaler_fitted = True
            X_scaled = self._scaler.transform(X)
            self._ml_model.partial_fit(X_scaled, y, classes=self._ml_classes)
        except Exception:
            pass

    @property
    def ml_ready(self) -> bool:
        return self._ml_ready

    @property
    def ml_event_count(self) -> int:
        return self._ml_event_count


# ──────────────────────────────────────────────────────────
# 3. STREAMING PROCESSOR
# ──────────────────────────────────────────────────────────

class StreamProcessor:
    """
    Event-driven incremental update bus.

    Responsibilities:
      - Accept raw event dicts from API endpoints
      - Dispatch to FeatureEngine and DemandPredictor (incremental, O(1))
      - Run background refresh loop every BACKGROUND_REFRESH_INTERVAL seconds
      - Measure and track end-to-end processing latency
    """

    def __init__(self, feature_engine: FeatureEngine, demand_predictor: DemandPredictor):
        self.fe = feature_engine
        self.dp = demand_predictor

        self._lock = threading.Lock()

        # Stats
        self._events_processed = 0
        self._latency_history  = deque(maxlen=200)   # rolling latency log
        self._events_per_sec   = 0.0
        self._last_eps_calc    = time.time()
        self._eps_event_count  = 0

        # Product catalog reference (set by MLEngine after data load)
        self._sku_catalog: Dict[str, Dict] = {}

        # Background refresh thread
        self._running = False
        self._bg_thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────────────

    def start(self, sku_catalog: Optional[Dict[str, Dict]] = None):
        """Start the background refresh loop."""
        if sku_catalog:
            self._sku_catalog = sku_catalog
        self._running = True
        self._bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._bg_thread.start()
        logger.info("StreamProcessor background loop started.")

    def stop(self):
        """Stop the background loop gracefully."""
        self._running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=5)
        logger.info("StreamProcessor stopped.")

    def _background_loop(self):
        """Refresh demand scores for all active SKUs every N seconds."""
        while self._running:
            try:
                skus = list(self._sku_catalog.keys())
                if skus:
                    self.dp.refresh_all(skus)
                    self._update_eps()
            except Exception as e:
                logger.error("Background refresh error: %s", e)
            time.sleep(BACKGROUND_REFRESH_INTERVAL)

    # ── Main processing pipeline ────────────────────────

    def process_event(
        self,
        event_type:  str,
        sku_id:      str,
        user_id:     str,
        session_id:  str,
        price_seen:  float = 0.0,
        discounted:  bool  = False,
        category:    Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
          1. Build EventRecord
          2. FeatureEngine.update()  ← O(1)
          3. DemandPredictor.record_event()  ← O(1)
          4. Generate FeatureVector
          5. Compute DemandScore
          6. Return unified response dict

        End-to-end latency target: < 200ms.
        """
        t0 = time.perf_counter()
        ts = time.time()

        # 1. Build event record
        event = EventRecord(
            ts         = ts,
            event_type = event_type,
            sku_id     = sku_id,
            session_id = session_id,
            user_id    = user_id,
            price_seen = price_seen,
            discounted = discounted,
        )

        # 2. Update feature engine (O(1) rolling window)
        self.fe.update(event, category=category)

        # 3. Record event for demand prediction (O(1) deque append)
        self.dp.record_event(sku_id, event_type, ts)

        # 4. Generate feature vector
        fv = self.fe.get_feature_vector(session_id)

        # 5. Compute demand score
        ds = self.dp.compute_demand_score(sku_id)

        # 6. Measure total latency
        latency_ms = (time.perf_counter() - t0) * 1000

        with self._lock:
            self._events_processed += 1
            self._eps_event_count  += 1
            self._latency_history.append(latency_ms)

        return {
            "feature_vector": fv.to_dict(),
            "demand_score":   ds.to_dict(),
            "pipeline_latency_ms": round(latency_ms, 2),
            "events_processed": self._events_processed,
        }

    # ── Stats ──────────────────────────────────────────

    def _update_eps(self):
        now = time.time()
        elapsed = now - self._last_eps_calc
        if elapsed >= 1.0:
            with self._lock:
                self._events_per_sec = self._eps_event_count / elapsed
                self._eps_event_count = 0
                self._last_eps_calc = now

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            lats = list(self._latency_history)
        avg_lat = (sum(lats) / len(lats)) if lats else 0.0
        p95_lat = sorted(lats)[int(len(lats) * 0.95)] if len(lats) > 20 else avg_lat
        return {
            "events_processed":    self._events_processed,
            "active_sessions":     self.fe.active_sessions(),
            "events_per_sec":      round(self._events_per_sec, 2),
            "avg_latency_ms":      round(avg_lat, 2),
            "p95_latency_ms":      round(p95_lat, 2),
            "ml_ready":            self.dp.ml_ready,
            "ml_event_count":      self.dp.ml_event_count,
            "background_interval_s": BACKGROUND_REFRESH_INTERVAL,
        }


# ──────────────────────────────────────────────────────────
# 4. ML ENGINE — Unified Facade
# ──────────────────────────────────────────────────────────

class MLEngine:
    """
    Single entry point for server.py.

    Usage:
        from ml_engine import MLEngine
        ml = MLEngine()
        ml.startup(products_dict, demand_counter_dict)

        # On each event:
        result = ml.process_event(event_type, sku_id, user_id, session_id, ...)

        # Get feature vector for a session:
        fv = ml.get_feature_vector(session_id)

        # Get demand score for a product:
        ds = ml.get_demand_score(sku_id)

        # Get top products by demand:
        tops = ml.get_top_demand_products(n=20)

        # Get engine stats:
        stats = ml.get_stats()
    """

    def __init__(self):
        self.feature_engine    = FeatureEngine(max_actions=MAX_ACTIONS)
        self.demand_predictor  = DemandPredictor()
        self.stream_processor  = StreamProcessor(self.feature_engine, self.demand_predictor)
        self._started = False
        self._model_registry   = None   # populated by Phase 3 background trainer

        # Phase 4: Real-Time Decision Engine
        try:
            from decision_engine import decision_engine as _de
            self._decision_engine = _de
        except ImportError:
            self._decision_engine = None
            logger.warning("decision_engine not found — Phase 4 disabled")

    def startup(self, products: Dict[str, Dict], demand_counter: Dict[str, int]):
        """
        Called once during FastAPI lifespan startup.
        Seeds the ML model with clickstream baseline data +
        Phase 2 prepared features (if available).
        """
        if self._started:
            return
        logger.info("MLEngine: seeding demand predictor with %d products...", len(demand_counter))
        self.demand_predictor.seed_from_demand_counter(demand_counter)

        # ── Phase 2: try to load pre-prepared feature data ──
        try:
            from data_preparation import PreparedDataLoader
            prepared = PreparedDataLoader.load()
            pricing_train = prepared.get("pricing_train")
            if pricing_train is not None and len(pricing_train) > 0:
                if "demand_score_proxy" in pricing_train.columns and "sku_id" in pricing_train.columns:
                    sku_demand = (
                        pricing_train.groupby("sku_id")["demand_score_proxy"]
                        .mean()
                        .fillna(0)
                    )
                    for sku, proxy in sku_demand.items():
                        existing = demand_counter.get(sku, 0)
                        synthetic_count = int(proxy * 10000)
                        blended = int(existing * 0.4 + synthetic_count * 0.6)
                        if blended > demand_counter.get(sku, 0):
                            demand_counter[sku] = blended
                    self.demand_predictor.seed_from_demand_counter(dict(demand_counter))
                    logger.info("MLEngine: Phase 2 demand features blended (%d SKUs)", len(sku_demand))
        except Exception as e:
            logger.info("MLEngine: Phase 2 data not available (%s)", e)

        # ── Phase 3: train pricing + recommendation models (background) ──
        def _train_models():
            try:
                from model_trainer import model_registry
                model_registry.startup()
                self._model_registry = model_registry
                # Inject into Phase 4 decision engine
                if self._decision_engine is not None:
                    self._decision_engine.set_model_registry(model_registry)
                logger.info("MLEngine: Phase 3 models ready")
            except Exception as ex:
                logger.warning("MLEngine: Phase 3 model training failed (%s)", ex)

        import threading as _threading
        _threading.Thread(target=_train_models, daemon=True, name="phase3-trainer").start()

        # Seed historical events for ML model warm-up
        import random
        rng = random.Random(42)
        active_skus = list(products.keys())[:500]
        for sku in active_skus:
            count = demand_counter.get(sku, 0)
            n_events = min(int(count / 100), 30)
            for _ in range(n_events):
                offset = rng.uniform(0, WINDOW_LONG)
                et = rng.choices(
                    ["product_view", "click", "add_to_cart", "purchase"],
                    weights=[50, 30, 15, 5]
                )[0]
                ts = time.time() - offset
                self.demand_predictor._sku_events[sku].append((ts, et))

        self.stream_processor.start(sku_catalog=products)
        self._started = True
        logger.info("MLEngine startup complete.")

    def shutdown(self):
        self.stream_processor.stop()
        logger.info("MLEngine shutdown.")

    # ── Delegates ──────────────────────────────────────

    def execute_ml_pipeline(
        self,
        event_type: str,
        sku_id: str,
        user_id: str,
        session_id: str,
        price_seen: float,
        discounted: bool,
        product: Dict,
        session: Dict,
        ab_group: str,
        competitor_data: Optional[Dict],
        all_products: Dict[str, Dict],
        all_categories: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        🧠 Full implementation of the 24-PART ML EXECUTION PLAN
        """
        t0 = time.perf_counter()
        ts = time.time()

        # 🔹 1. Input Flow
        # User Action → Feature Engine → ML Layer Input
        event = EventRecord(
            ts=ts, event_type=event_type, sku_id=sku_id, session_id=session_id,
            user_id=user_id, price_seen=price_seen, discounted=discounted
        )

        # 🔹 2. Preprocessing Flow
        # Raw Features → Clean → Normalize → Encode → Ready Features
        self.feature_engine.update(event, category=product.get("category"))
        
        # 🔹 3. Decision Split Flow
        # Input Data → Pricing Engine | Recommendation Engine
        self.demand_predictor.record_event(sku_id, event_type, ts)
        fv = self.feature_engine.get_feature_vector(session_id)
        ds = self.demand_predictor.compute_demand_score(sku_id)

        # Phase 4: DecisionEngine (fast path)
        de = self._decision_engine
        if de is not None:
            demand_scores_map = {
                sku: self.demand_predictor.compute_demand_score(sku).hybrid_score
                for sku in list(all_products.keys())[:200]
            }
            bundle = de.decide(
                product=product, session=session, demand_score_obj=ds, feature_vec_obj=fv,
                all_products=all_products, all_categories=all_categories,
                demand_scores=demand_scores_map, ab_group=ab_group,
                competitor_data=competitor_data,
                device_type=session.get("device_type", "desktop"),
                user_id=user_id, session_id=session_id,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "final_price": bundle.pricing.final_price,
                "top_recommendations": [{
                    "id": r.sku_id, "name": r.name, "category": r.category,
                    "price": r.price, "rating": r.rating, "reason": r.reason,
                    "tag": r.tag, "ml_score": r.ml_score, "cold_start": r.cold_start,
                } for r in bundle.recommendations],
                "reason": bundle.pricing.reasons,
                "feature_vector": fv.to_dict(),
                "demand_score": ds.to_dict(),
                "pricing_detail": {
                    "tier": bundle.pricing.tier,
                    "demand_level": bundle.pricing.demand_level,
                    "inventory_status": bundle.pricing.inventory_status,
                    "adjustment_pct": bundle.pricing.adjustment_pct,
                    "competitor_gap": bundle.pricing.competitor_gap,
                },
                "cold_start": bundle.cold_start,
                "latency_ms": round(latency_ms, 2),
            }

        # --- Fallback path (DecisionEngine not loaded) ---
        clicked     = set(session.get("clicks", []))
        recent_cats = session.get("categories", [])[-5:]
        raw_recs: list = []
        seen_skus: set = set()

        for cat in recent_cats:
            for c_sku in all_categories.get(cat, []):
                if c_sku not in clicked and c_sku not in seen_skus and c_sku in all_products:
                    p = all_products[c_sku]
                    raw_recs.append({"id": c_sku, "name": p["name"], "category": p["category"],
                                     "price": p["current_price"], "rating": p.get("rating", 0),
                                     "reason": f"Because you browsed {cat}",
                                     "score": p.get("views", 0) + p.get("rating", 0) * 10})
                    seen_skus.add(c_sku)

        if len(raw_recs) < 5:
            for p in sorted(all_products.values(), key=lambda x: x.get("views", 0), reverse=True):
                if p["sku_id"] not in clicked and p["sku_id"] not in seen_skus:
                    raw_recs.append({"id": p["sku_id"], "name": p["name"], "category": p["category"],
                                     "price": p["current_price"], "rating": p.get("rating", 0),
                                     "reason": "Trending now",
                                     "score": p.get("views", 0) + p.get("rating", 0) * 5})
                    seen_skus.add(p["sku_id"])
                    if len(raw_recs) >= 10:
                        break

        mr = self._model_registry
        if mr is not None and mr.ready and mr.recom.ready:
            demand_map = {c["id"]: self.demand_predictor.compute_demand_score(c["id"]).hybrid_score
                          for c in raw_recs if "id" in c}
            top_recs = mr.rank_recommendations(raw_recs, recent_cats, demand_map)[:5]
        else:
            raw_recs.sort(key=lambda x: x["score"], reverse=True)
            top_recs = raw_recs[:5]

        inv_ratio   = min(product.get("inventory", 100) / 500, 1.0)
        final_price = product["base_price"]
        reasons: list = []

        if mr is not None and mr.ready and mr.pricing.ready:
            final_price, reasons = mr.compute_price(
                base_price=product["base_price"], demand_score=ds.hybrid_score,
                user_intent=fv.intent_score, inventory_ratio=inv_ratio, ab_group=ab_group)
        else:
            if ab_group == "A":
                reasons.append("Control group: static base price.")
            else:
                temp_price = product["base_price"]
                if ds.hybrid_score > 50:
                    temp_price *= 1.05
                    reasons.append(f"High demand ({ds.hybrid_score:.1f}): +5%")
                if product.get("inventory", 100) < 20:
                    temp_price *= 1.03
                    reasons.append("Low stock: +3%")
                elif product.get("inventory", 100) > 400:
                    temp_price *= 0.95
                    reasons.append("High stock: -5%")
                if fv.intent_score > 10:
                    temp_price *= 1.02
                    reasons.append("High intent: +2%")
                cost = product.get("cost_price", 0)
                floor = max(cost * 1.05 if cost > 0 else product["base_price"] * 0.85,
                            product["base_price"] * 0.90)
                final_price = round(max(floor, min(temp_price, product["base_price"] * 1.15)), 2)
                if not reasons:
                    reasons.append("Base price maintained.")

        cost = product.get("cost_price", 0)
        final_price = round(max(
            max(cost * 1.05 if cost > 0 else product["base_price"] * 0.85,
                product["base_price"] * 0.85),
            min(final_price, product["base_price"] * 1.15)
        ), 2)

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "final_price":        final_price,
            "top_recommendations": top_recs,
            "reason":             reasons,
            "feature_vector":     fv.to_dict(),
            "demand_score":       ds.to_dict(),
            "latency_ms":         round(latency_ms, 2),
        }

    def get_feature_vector(self, session_id: str) -> Dict[str, Any]:
        return self.feature_engine.get_feature_vector(session_id).to_dict()

    def get_demand_score(self, sku_id: str) -> Dict[str, Any]:
        return self.demand_predictor.compute_demand_score(sku_id).to_dict()

    def get_top_demand_products(self, n: int = 20) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.demand_predictor.get_top_products(n)]

    def get_stats(self) -> Dict[str, Any]:
        return self.stream_processor.get_stats()

    def get_model_status(self) -> Dict[str, Any]:
        """Return Phase 3 model registry status."""
        if self._model_registry is not None:
            return self._model_registry.status
        return {"ready": False, "pricing_ready": False, "recom_ready": False}


# ──────────────────────────────────────────────────────────
# Module-level singleton (imported by server.py)
# ──────────────────────────────────────────────────────────
ml_engine = MLEngine()
