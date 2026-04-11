"""
============================================================
  APEX — Dynamic Pricing & Recommendation Engine
  FastAPI Backend · Real-Time · Sub-200ms Latency
============================================================

  Run:     py server.py
  Open:    http://localhost:8000

  Data:    Loads from parquet files (all 4 datasets)
  Pricing: Demand-responsive + Scarcity + Competitor-aware
  Recs:    Session-based category affinity
  A/B:     Control vs Dynamic pricing groups
  ML:      Real-time feature engine + demand prediction
============================================================
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from collections import defaultdict, Counter
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import json

try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    # Quick ping to check if Redis is actually up
    redis_client.ping()
    _REDIS_READY = True
except Exception:
    redis_client = None
    _REDIS_READY = False

# ── ML Engine (dedicated module) ──
from ml_engine import ml_engine

# ── Phase 5: A/B Testing System ──
try:
    from ab_testing import ab_system
except ImportError:
    ab_system = None

# ── Phase 6: Explainability + Simulation ──
try:
    from explainability import pricing_explainer, recommendation_explainer, price_simulator
    _EXPL_READY = True
except ImportError:
    _EXPL_READY = False

# ── Unified ML Pipeline ──
from ml_pipeline import MLEngine as MLPipelineEngine
global_pipeline = None

logging.basicConfig(level=logging.INFO)

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(BASE_DIR, "Problem Statement 3 Sample Data")
MAX_PRICE_INCREASE = 0.15   # 15% hard ceiling (fairness)
MAX_PRICE_DECREASE = 0.10   # 10% max discount
DEMAND_THRESHOLD = 50       # ML rule: demand > 50 → +5%
LOW_STOCK_THRESHOLD = 20    # ML rule: stock < 20 → +3%

# ─────────────────────────────────────────────
# Data Stores — In-Memory (fast!)
# ─────────────────────────────────────────────
PRODUCTS = {}
COMPETITOR_DATA = {}
USER_SEGMENTS = {}
CATEGORIES = defaultdict(list)       # category -> [sku_ids]
CLICKSTREAM_STATS = {}               # Pre-computed clickstream analytics
DEMAND_COUNTER = defaultdict(int)    # product:{sku_id} -> demand count
ENGAGEMENT_SCORE = defaultdict(float)  # user:{user_id} -> engagement score
EXPLORE_CACHE = {}                   # Cached exploration results


# ─────────────────────────────────────────────
# 📊 Data Loading — All 4 Datasets
# ─────────────────────────────────────────────
def load_data():
    """Load all 4 datasets from parquet files into memory."""
    global PRODUCTS, COMPETITOR_DATA, USER_SEGMENTS, CATEGORIES, CLICKSTREAM_STATS, DEMAND_COUNTER, ENGAGEMENT_SCORE

    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   📊 Loading All 4 Datasets...          ║")
    print("  ╚══════════════════════════════════════════╝\n")

    # --- 1. Load Product Catalog (5,368 products) ---
    _load_product_catalog()

    # --- 2. Load User Segment Profiles (500K users) ---
    _load_user_segments()

    # --- 3. Load Competitor Pricing Feed (1.69M rows) ---
    _load_competitor_pricing()

    # --- 4. Load Clickstream Events (10.2M rows — sampled) ---
    _load_clickstream_events()

    # --- 5. Pre-compute exploration stats ---
    _compute_explore_cache()

    print(f"\n  ✅ All datasets loaded successfully!")
    print(f"  📦 Products: {len(PRODUCTS)}")
    print(f"  👤 Users:    {len(USER_SEGMENTS)}")
    print(f"  🏪 Competitor records: {len(COMPETITOR_DATA)}")
    print(f"  📊 Clickstream stats ready\n")


def _load_product_catalog():
    """Dataset 1: Product catalog (product_id, category, price, stock)."""
    global PRODUCTS, CATEGORIES
    try:
        import pandas as pd
        catalog = pd.read_parquet(os.path.join(PARQUET_DIR, "product_catalog.parquet"))
        for _, row in catalog.iterrows():
            sku = row["sku_id"]
            PRODUCTS[sku] = {
                "sku_id": sku,
                "name": row["product_name"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "brand": row["brand"],
                "base_price": round(float(row["base_price_usd"]), 2),
                "cost_price": round(float(row["cost_price_usd"]), 2),
                "current_price": round(float(row["current_price_usd"]), 2),
                "min_price": round(float(row["min_price_usd"]), 2),
                "max_price": round(float(row["max_price_usd"]), 2),
                "inventory": int(row["inventory_count"]),
                "rating": round(float(row["avg_rating"]), 1),
                "review_count": int(row["review_count"]),
                "tags": row.get("tags", "[]"),
                "is_active": bool(row.get("is_active", True)),
                "weight_kg": round(float(row.get("weight_kg", 0)), 2),
                # Live tracking fields
                "views": 0,
                "clicks": 0,
                "add_to_carts": 0,
                "purchases": 0,
                "price_history": [round(float(row["current_price_usd"]), 2)],
            }
            CATEGORIES[row["category"]].append(sku)
        print(f"  [OK] Dataset 1 — Product Catalog: {len(PRODUCTS)} products loaded")
        print(f"       Categories: {len(CATEGORIES)} unique")
    except Exception as e:
        print(f"  [WARN] Product catalog load failed: {e}")
        _load_demo_products()


def _load_user_segments():
    """Dataset 2: User segment profiles (500K users)."""
    global USER_SEGMENTS, ENGAGEMENT_SCORE
    try:
        import pandas as pd
        users = pd.read_parquet(os.path.join(PARQUET_DIR, "user_segment_profiles.parquet"))
        for _, row in users.iterrows():
            uid = row["user_id"]
            USER_SEGMENTS[uid] = {
                "segment": row["segment"],
                "country": row.get("country", ""),
                "device_type": row.get("device_type", "desktop"),
                "os": row.get("os", ""),
                "willingness_to_pay": round(float(row["willingness_to_pay_multiplier"]), 3),
                "preferred_categories": row.get("preferred_categories", "[]"),
                "lifetime_value": round(float(row.get("lifetime_value_usd", 0)), 2),
                "avg_order_value": round(float(row.get("avg_order_value_usd", 0)), 2),
                "sessions_per_month": int(row.get("sessions_per_month", 0)),
                "purchase_frequency": round(float(row.get("purchase_frequency", 0)), 3),
                "cart_abandonment_rate": round(float(row.get("cart_abandonment_rate", 0)), 3),
                "age_group": row.get("age_group", ""),
                "gender": row.get("gender", ""),
            }
            # Initialize engagement score from profile data
            ENGAGEMENT_SCORE[uid] = round(
                float(row.get("sessions_per_month", 0)) * 2 +
                float(row.get("purchase_frequency", 0)) * 50 +
                (1 - float(row.get("cart_abandonment_rate", 0.5))) * 20,
                2
            )
        print(f"  [OK] Dataset 2 — User Segments: {len(USER_SEGMENTS)} users loaded")
        segments = Counter(u["segment"] for u in USER_SEGMENTS.values())
        print(f"       Segments: {dict(segments)}")
    except Exception as e:
        print(f"  [WARN] User segments load failed: {e}")


def _load_competitor_pricing():
    """Dataset 3: Competitor pricing feed (1.69M rows)."""
    global COMPETITOR_DATA
    try:
        import pandas as pd
        comp = pd.read_parquet(os.path.join(PARQUET_DIR, "competitor_pricing_feed.parquet"))
        # Keep LATEST competitor price per SKU (most recent date)
        comp_sorted = comp.sort_values("date", ascending=False)
        seen_skus = set()
        for _, row in comp_sorted.iterrows():
            sku = row["sku_id"]
            if sku in seen_skus:
                continue
            if sku in PRODUCTS:
                COMPETITOR_DATA[sku] = {
                    "competitor": row["competitor"],
                    "competitor_price": round(float(row["competitor_price"]), 2),
                    "our_base_price": round(float(row.get("our_base_price", 0)), 2),
                    "price_delta_pct": round(float(row.get("price_delta_pct", 0)), 2),
                    "is_on_promotion": bool(row["is_on_promotion"]),
                    "promo_discount_pct": round(float(row.get("promo_discount_pct", 0)), 1),
                    "in_stock": bool(row.get("in_stock", True)),
                    "date": str(row.get("date", "")),
                }
                seen_skus.add(sku)
        print(f"  [OK] Dataset 3 — Competitor Pricing: {len(COMPETITOR_DATA)} SKUs with competitor data")
    except Exception as e:
        print(f"  [WARN] Competitor data load failed: {e}")


def _load_clickstream_events():
    """
    Dataset 4: Clickstream events (10.2M rows — sampled for memory).
    Columns: user_id, product_id (sku_id), event_type, timestamp, session_id
    + category, device_type, ab_group, price_seen_usd, etc.
    """
    global CLICKSTREAM_STATS, DEMAND_COUNTER, ENGAGEMENT_SCORE
    try:
        import pandas as pd
        # Sample to avoid memory issues on large datasets (424MB parquet)
        print("  [..] Loading clickstream data (sampling 500K of 10.2M rows)...")
        clicks = pd.read_parquet(os.path.join(PARQUET_DIR, "clickstream_events.parquet"),
                                 columns=["user_id", "sku_id", "event_type", "session_id",
                                           "category", "device_type", "ab_group",
                                           "price_seen_usd", "hour_of_day", "day_of_week"])
        # Sample for speed (500K rows from 10.2M)
        if len(clicks) > 500_000:
            clicks_sample = clicks.sample(n=500_000, random_state=42)
        else:
            clicks_sample = clicks

        # ── Compute statistics ──
        total_events = len(clicks)
        total_users = clicks["user_id"].nunique()
        total_sessions = clicks["session_id"].nunique()

        # Event type distribution (from full dataset column)
        event_dist = clicks_sample["event_type"].value_counts().to_dict()

        # Top 10 most viewed products (from sample, scaled)
        view_events = clicks_sample[clicks_sample["event_type"].isin(["page_view", "product_view"])]
        top_viewed = view_events["sku_id"].value_counts().head(10)
        top_viewed_products = []
        for sku, count in top_viewed.items():
            product_name = PRODUCTS[sku]["name"] if sku in PRODUCTS else sku
            category = PRODUCTS[sku]["category"] if sku in PRODUCTS else "Unknown"
            top_viewed_products.append({
                "sku_id": sku,
                "name": product_name,
                "category": category,
                "view_count": int(count),
            })

        # Category distribution
        cat_dist = clicks_sample["category"].value_counts().to_dict()

        # Device distribution
        device_dist = clicks_sample["device_type"].value_counts().to_dict()

        # AB group distribution
        ab_dist = clicks_sample["ab_group"].value_counts().to_dict()

        # Hourly distribution
        hourly_dist = clicks_sample["hour_of_day"].value_counts().sort_index().to_dict()
        # Convert numpy int keys to python int
        hourly_dist = {int(k): int(v) for k, v in hourly_dist.items()}

        # Day-of-week distribution
        dow_dist = clicks_sample["day_of_week"].value_counts().sort_index().to_dict()
        dow_dist = {int(k): int(v) for k, v in dow_dist.items()}

        # ── Build demand counters from clickstream ──
        demand_events = clicks_sample[clicks_sample["event_type"].isin(
            ["product_view", "add_to_cart", "purchase", "add_to_wishlist"]
        )]
        demand_counts = demand_events["sku_id"].value_counts()
        for sku, count in demand_counts.items():
            # Scale back up to approximate full dataset demand
            DEMAND_COUNTER[sku] = int(count * (total_events / len(clicks_sample)))

        # ── Build engagement scores from clickstream ──
        user_event_counts = clicks_sample.groupby("user_id")["event_type"].count()
        for uid, count in user_event_counts.items():
            existing = ENGAGEMENT_SCORE.get(uid, 0)
            ENGAGEMENT_SCORE[uid] = round(existing + count * 0.5, 2)

        # ── Update product view/click counts from clickstream ──
        for sku, count in clicks_sample[clicks_sample["event_type"].isin(["page_view", "product_view"])]["sku_id"].value_counts().items():
            if sku in PRODUCTS:
                PRODUCTS[sku]["views"] = int(count)
        for sku, count in clicks_sample[clicks_sample["event_type"] == "add_to_cart"]["sku_id"].value_counts().items():
            if sku in PRODUCTS:
                PRODUCTS[sku]["add_to_carts"] = int(count)
        for sku, count in clicks_sample[clicks_sample["event_type"] == "purchase"]["sku_id"].value_counts().items():
            if sku in PRODUCTS:
                PRODUCTS[sku]["purchases"] = int(count)

        CLICKSTREAM_STATS = {
            "total_events": int(total_events),
            "sampled_events": int(len(clicks_sample)),
            "total_users": int(total_users),
            "total_sessions": int(total_sessions),
            "event_type_distribution": {str(k): int(v) for k, v in event_dist.items()},
            "top_10_most_viewed": top_viewed_products,
            "category_distribution": {str(k): int(v) for k, v in cat_dist.items()},
            "device_distribution": {str(k): int(v) for k, v in device_dist.items()},
            "ab_group_distribution": {str(k): int(v) for k, v in ab_dist.items()},
            "hourly_distribution": hourly_dist,
            "day_of_week_distribution": dow_dist,
        }

        print(f"  [OK] Dataset 4 — Clickstream: {total_events:,} total events ({len(clicks_sample):,} sampled)")
        print(f"       Users: {total_users:,} | Sessions: {total_sessions:,}")
        print(f"       Event types: {list(event_dist.keys())}")
        print(f"       Demand counters: {len(DEMAND_COUNTER)} products tracked")

    except Exception as e:
        print(f"  [WARN] Clickstream load failed: {e}")
        import traceback
        traceback.print_exc()
        CLICKSTREAM_STATS = {"error": str(e), "total_events": 0}


def _compute_explore_cache():
    """Pre-compute exploration summary for /explore endpoint."""
    global EXPLORE_CACHE

    # Category breakdown from product catalog
    category_stats = {}
    for cat, skus in CATEGORIES.items():
        prices = [PRODUCTS[s]["base_price"] for s in skus if s in PRODUCTS]
        stocks = [PRODUCTS[s]["inventory"] for s in skus if s in PRODUCTS]
        category_stats[cat] = {
            "product_count": len(skus),
            "avg_price": round(sum(prices) / max(1, len(prices)), 2),
            "min_price": round(min(prices) if prices else 0, 2),
            "max_price": round(max(prices) if prices else 0, 2),
            "total_stock": sum(stocks),
            "avg_stock": round(sum(stocks) / max(1, len(stocks)), 0),
        }

    # User segment breakdown
    segment_counts = Counter(u["segment"] for u in USER_SEGMENTS.values())
    country_counts = Counter(u["country"] for u in USER_SEGMENTS.values())
    device_counts = Counter(u["device_type"] for u in USER_SEGMENTS.values())
    age_counts = Counter(u["age_group"] for u in USER_SEGMENTS.values())

    # Top user segments by lifetime value
    segment_ltv = defaultdict(list)
    for u in USER_SEGMENTS.values():
        segment_ltv[u["segment"]].append(u["lifetime_value"])
    segment_avg_ltv = {
        seg: round(sum(vals) / max(1, len(vals)), 2)
        for seg, vals in segment_ltv.items()
    }

    EXPLORE_CACHE = {
        "summary": {
            "total_products": len(PRODUCTS),
            "total_users": len(USER_SEGMENTS),
            "total_categories": len(CATEGORIES),
            "unique_categories": sorted(CATEGORIES.keys()),
            "total_brands": len(set(p["brand"] for p in PRODUCTS.values())),
            "total_competitor_records": len(COMPETITOR_DATA),
            "clickstream_events": CLICKSTREAM_STATS.get("total_events", 0),
        },
        "category_distribution": category_stats,
        "user_segments": {
            "segment_counts": dict(segment_counts),
            "country_top_10": dict(country_counts.most_common(10)),
            "device_distribution": dict(device_counts),
            "age_distribution": dict(age_counts),
            "segment_avg_ltv": segment_avg_ltv,
        },
        "clickstream": CLICKSTREAM_STATS,
        "pricing": {
            "avg_base_price": round(sum(p["base_price"] for p in PRODUCTS.values()) / max(1, len(PRODUCTS)), 2),
            "avg_current_price": round(sum(p["current_price"] for p in PRODUCTS.values()) / max(1, len(PRODUCTS)), 2),
            "products_with_competitor_data": len(COMPETITOR_DATA),
            "products_low_stock": sum(1 for p in PRODUCTS.values() if p["inventory"] < LOW_STOCK_THRESHOLD),
            "products_high_demand": sum(1 for sku in PRODUCTS if DEMAND_COUNTER.get(sku, 0) > DEMAND_THRESHOLD),
        },
    }
    print(f"  [OK] Exploration cache computed")


def _load_demo_products():
    """Emergency fallback with hardcoded demo products."""
    demo = [
        ("SKU001", "Wireless Headphones", "Electronics", "Audio", "SoundMax", 129.99),
        ("SKU002", "Mechanical Keyboard", "Electronics", "Peripherals", "KeyTech", 89.99),
        ("SKU003", "Smart Watch Pro", "Electronics", "Wearables", "TechWear", 299.99),
        ("SKU004", "Running Shoes Elite", "Fashion", "Shoes", "RunFast", 159.99),
        ("SKU005", "Organic Face Cream", "Beauty", "Skincare", "GlowUp", 45.99),
        ("SKU006", "Yoga Mat Premium", "Sports", "Fitness", "ZenFit", 69.99),
        ("SKU007", "LED Desk Lamp", "Home & Kitchen", "Lighting", "BrightLife", 39.99),
        ("SKU008", "Python Programming Book", "Books", "Education", "LearnCo", 34.99),
    ]
    for sku, name, cat, sub, brand, price in demo:
        PRODUCTS[sku] = {
            "sku_id": sku, "name": name, "category": cat, "subcategory": sub,
            "brand": brand, "base_price": price, "cost_price": round(price * 0.5, 2),
            "current_price": price, "min_price": round(price * 0.70, 2),
            "max_price": round(price * 1.30, 2), "inventory": 100,
            "rating": 4.0, "review_count": 500, "tags": "[]",
            "is_active": True, "weight_kg": 1.0,
            "views": 0, "clicks": 0, "add_to_carts": 0, "purchases": 0,
            "price_history": [price],
        }
        CATEGORIES[cat].append(sku)


# ─────────────────────────────────────────────
# In-Memory Session & Event Tracking
# ─────────────────────────────────────────────
SESSION_STORE = defaultdict(lambda: {
    "clicks": [],          # list of sku_ids clicked
    "categories": [],      # categories browsed
    "total_events": 0,
    "start_time": time.time(),
    "ab_group": None,
})

EVENT_LOG = []             # Global event log
ANALYTICS = {
    "total_events": 0,
    "total_revenue_static": 0.0,
    "total_revenue_dynamic": 0.0,
    "group_a_conversions": 0,
    "group_b_conversions": 0,
    "group_a_sessions": 0,
    "group_b_sessions": 0,
}


# ─────────────────────────────────────────────
# 🧠 Feature Logic — Engagement & Demand
# ─────────────────────────────────────────────
def get_engagement_score(user_id: str) -> float:
    """
    Engagement score for a user.
    Combines: profile data (sessions, purchase freq, abandonment)
    + real-time event counts.
    """
    return ENGAGEMENT_SCORE.get(user_id, 0.0)


def get_demand_count(sku_id: str) -> int:
    """
    Demand counter for a product.
    Aggregated from clickstream (views + carts + purchases + wishlists)
    + real-time increments from /event calls.
    """
    return DEMAND_COUNTER.get(sku_id, 0)


def update_engagement(user_id: str, event_type: str):
    """Update user engagement score based on event type."""
    weights = {
        "view": 1,
        "click": 2,
        "add_to_cart": 5,
        "purchase": 10,
        "page_view": 1,
        "product_view": 2,
        "add_to_wishlist": 3,
        "checkout_start": 7,
        "search": 1,
    }
    ENGAGEMENT_SCORE[user_id] = round(
        ENGAGEMENT_SCORE.get(user_id, 0) + weights.get(event_type, 1),
        2
    )


def update_demand(sku_id: str, event_type: str):
    """Update product demand counter based on event type."""
    weights = {
        "view": 1,
        "click": 2,
        "add_to_cart": 5,
        "purchase": 10,
        "product_view": 2,
        "add_to_wishlist": 3,
    }
    DEMAND_COUNTER[sku_id] = DEMAND_COUNTER.get(sku_id, 0) + weights.get(event_type, 1)


# ─────────────────────────────────────────────
# 🤖 ML PIPELINE LOGIC MOVED TO ML_ENGINE.PY
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    global _start_time, global_pipeline
    _start_time = time.time()
    load_data()
    # ── Start ML Engine (seed + background loop) ──
    try:
        global_pipeline = MLPipelineEngine.load_pipeline("trained_models/apex_pipeline_v3.pkl")
        if global_pipeline:
            print("\n  ✅ Unified ML Pipeline (v3) Loaded!  \n")
    except Exception as e:
        print(f" [WARN] ML Pipeline Load Error: {e}")
        
    ml_engine.startup(products=PRODUCTS, demand_counter=DEMAND_COUNTER)
    yield
    # ── Graceful shutdown ──
    ml_engine.shutdown()

app = FastAPI(
    title="APEX Dynamic Pricing Engine",
    description="Real-time dynamic pricing, recommendations & A/B testing — powered by 4 real datasets + ML feature engine",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Serve Frontend ──
@app.get("/")
def serve_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "APEX API running. Place index.html in same folder."}


# ── 📊 GET /explore — Data Exploration Dashboard ──
@app.get("/explore")
def explore_data():
    """
    📊 Data Exploration endpoint.
    Returns comprehensive stats from all 4 datasets:
    - Total users, products, unique categories
    - Event type distribution
    - Top 10 most viewed products
    - Category distribution
    - User segment breakdown
    - Demand & engagement metrics
    """
    return {
        **EXPLORE_CACHE,
        "demand_stats": {
            "total_products_tracked": len(DEMAND_COUNTER),
            "high_demand_products": sum(1 for v in DEMAND_COUNTER.values() if v > DEMAND_THRESHOLD),
            "avg_demand": round(sum(DEMAND_COUNTER.values()) / max(1, len(DEMAND_COUNTER)), 1),
            "top_demand_products": sorted(
                [
                    {"sku_id": sku, "demand": count,
                     "name": PRODUCTS[sku]["name"] if sku in PRODUCTS else sku}
                    for sku, count in DEMAND_COUNTER.items()
                ],
                key=lambda x: x["demand"], reverse=True
            )[:10],
        },
        "engagement_stats": {
            "total_users_tracked": len(ENGAGEMENT_SCORE),
            "avg_engagement": round(sum(ENGAGEMENT_SCORE.values()) / max(1, len(ENGAGEMENT_SCORE)), 2),
            "top_engaged_users": sorted(
                [
                    {"user_id": uid, "engagement_score": score}
                    for uid, score in ENGAGEMENT_SCORE.items()
                ],
                key=lambda x: x["engagement_score"], reverse=True
            )[:10],
        },
    }


# ── GET /products ──
@app.get("/products")
def list_products(
    limit: int = Query(20, description="Max products to return"),
    category: str = Query(None, description="Filter by category"),
    sort_by: str = Query("views", description="Sort: views, price, demand, rating"),
):
    """Return products with current dynamic prices, filterable and sortable."""
    products = list(PRODUCTS.values())

    # Filter by category
    if category:
        products = [p for p in products if p["category"].lower() == category.lower()]

    # Sort
    if sort_by == "price":
        products.sort(key=lambda x: x["current_price"], reverse=True)
    elif sort_by == "demand":
        products.sort(key=lambda x: DEMAND_COUNTER.get(x["sku_id"], 0), reverse=True)
    elif sort_by == "rating":
        products.sort(key=lambda x: x["rating"], reverse=True)
    else:  # views (default)
        products.sort(key=lambda x: x["views"], reverse=True)

    products = products[:limit]

    return {
        "products": [
            {
                "sku_id": p["sku_id"],
                "name": p["name"],
                "category": p["category"],
                "subcategory": p["subcategory"],
                "brand": p["brand"],
                "base_price": p["base_price"],
                "current_price": p["current_price"],
                "inventory": p["inventory"],
                "rating": p["rating"],
                "review_count": p["review_count"],
                "views": p["views"],
                "clicks": p["clicks"],
                "demand": DEMAND_COUNTER.get(p["sku_id"], 0),
            }
            for p in products
        ],
        "total": len(PRODUCTS),
        "categories": sorted(CATEGORIES.keys()),
    }


# ── POST /event ──
@app.post("/event")
def record_event(
    sku_id: str = Query(..., description="Product SKU"),
    event_type: str = Query("click", description="click|view|add_to_cart|purchase"),
    user_id: str = Query("U001", description="User ID"),
    session_id: str = Query("S001", description="Session ID"),
    price_seen: float = Query(0.0, description="Price displayed to user"),
    discounted: bool = Query(False, description="Was a discounted price shown?"),
):
    """
    Record a user event and trigger the full ML pipeline.
    """
    start = time.time()

    if sku_id not in PRODUCTS:
        return {"error": f"Product {sku_id} not found"}

    # ─ Publish to Redis Stream (Phase 7 - Streaming Update) ─
    if _REDIS_READY and redis_client is not None:
        try:
            payload = {
                "sku_id": sku_id,
                "event_type": event_type,
                "user_id": user_id,
                "session_id": session_id,
                "price_seen": price_seen,
                "timestamp": time.time()
            }
            redis_client.xadd("ecommerce:events", {"event": json.dumps(payload)})
        except Exception as e:
            logging.warning(f"Redis Streams fallback triggered. Running in-memory mode. Error: {e}")

    product = PRODUCTS[sku_id]
    session = SESSION_STORE[session_id]
    old_price = product["current_price"]

    # ─ Track event on product ─
    product["views"] += 1
    if event_type == "click":
        product["clicks"] += 1
    elif event_type == "add_to_cart":
        product["add_to_carts"] += 1
        product["clicks"] += 1
    elif event_type == "purchase":
        product["purchases"] += 1
        product["clicks"] += 2
        product["inventory"] = max(0, product["inventory"] - 1)

    update_engagement(user_id, event_type)
    update_demand(sku_id, event_type)
    
    # ─ A/B Group Assignment (Phase 5 — deterministic hash split) ─
    if session["ab_group"] is None:
        if ab_system is not None:
            session["ab_group"] = ab_system.assign(user_id, session_id)
        else:
            session["ab_group"] = "B" if hash(session_id) % 2 == 0 else "A"

    # ─ 🤖 ML PIPELINE ─
    ml_result = ml_engine.execute_ml_pipeline(
        event_type=event_type,
        sku_id=sku_id,
        user_id=user_id,
        session_id=session_id,
        price_seen=price_seen or product["current_price"],
        discounted=discounted or bool(product["current_price"] < product["base_price"]),
        product=product,
        session=session,
        ab_group=session["ab_group"],
        competitor_data=COMPETITOR_DATA.get(sku_id),
        all_products=PRODUCTS,
        all_categories=CATEGORIES
    )

    try:
        if global_pipeline is not None and session["ab_group"] != "A":
            import pandas as pd
            # Create a simple df for the newly trained model
            df_in = pd.DataFrame([{
                'price_seen_usd': price_seen or product["current_price"],
                'base_price_usd': product["base_price"],
                'session_intensity': session["total_events"],
                'click_frequency': product["clicks"],
                'category': product["category"],
                'device_type': session.get("device_type", "desktop")
            }])
            # Ask the global unified model for the conversion probability
            pred_conv_arr = global_pipeline.predict_pricing(df_in)
            if pred_conv_arr is not None and len(pred_conv_arr) > 0:
                pred_conv = float(pred_conv_arr[0])
                # If model predicts high likelihood of conversion, do +10%
                if pred_conv > 0.6:
                    ml_result["final_price"] = round(product["base_price"] * 1.10, 2)
                    ml_result.setdefault("reason", []).append(
                        f"Model v3 Output High Intent ({pred_conv:.2f}): +10%"
                    )
                # If very low likelihood, do a -5% discount to retain
                elif pred_conv < 0.2:
                    ml_result["final_price"] = round(product["base_price"] * 0.95, 2)
                    ml_result.setdefault("reason", []).append(
                        f"Model v3 Output Low Intent ({pred_conv:.2f}): -5%"
                    )
    except Exception:
        # Fallback to decision_engine output silently
        pass

    new_price  = ml_result["final_price"]
    ab_group   = session["ab_group"]
    revenue    = new_price if event_type == "purchase" else 0.0
    latency_ms = round((time.time() - start) * 1000, 1)

    # ─ Phase 5: record event into A/B tracker ─
    if ab_system is not None:
        ab_system.record(
            group=ab_group, user_id=user_id, event_type=event_type,
            price=new_price, revenue=revenue, latency_ms=latency_ms,
        )

    # ─ Phase 6: generate structured explanation ─
    explanation = None
    if _EXPL_READY:
        try:
            ds_dict  = ml_result.get("demand_score", {})
            pd_dict  = ml_result.get("pricing_detail", {})
            fv_dict  = ml_result.get("feature_vector", {})
            comp     = COMPETITOR_DATA.get(sku_id, {})
            expl_obj = pricing_explainer.explain(
                base_price       = product["base_price"],
                final_price      = new_price,
                demand_score     = ds_dict.get("demand_score", 50),
                demand_level     = pd_dict.get("demand_level", "medium"),
                inventory        = product["inventory"],
                inventory_status = pd_dict.get("inventory_status", "normal"),
                user_intent      = fv_dict.get("intent_score", 0),
                ab_group         = ab_group,
                competitor_gap   = pd_dict.get("competitor_gap"),
                raw_reasons      = ml_result.get("reason", []),
                latency_ms       = ml_result.get("latency_ms", 0),
            )
            explanation = expl_obj.to_dict()
        except Exception:
            pass

    # Update legacy analytics + product state
    if ab_group == "A":
        ANALYTICS["group_a_sessions"] += 1
        if event_type == "purchase":
            ANALYTICS["group_a_conversions"] += 1
            ANALYTICS["total_revenue_static"] += new_price
    else:
        product["current_price"] = new_price
        product["price_history"].append(new_price)
        if len(product["price_history"]) > 50:
            product["price_history"] = product["price_history"][-50:]
        ANALYTICS["group_b_sessions"] += 1
        if event_type == "purchase":
            ANALYTICS["group_b_conversions"] += 1
            ANALYTICS["total_revenue_dynamic"] += new_price

    session["clicks"].append(sku_id)
    session["categories"].append(product["category"])
    session["total_events"] += 1
    ANALYTICS["total_events"] += 1

    EVENT_LOG.append({
        "event_type": event_type, "sku_id": sku_id,
        "user_id": user_id, "session_id": session_id,
        "old_price": old_price, "new_price": new_price,
        "ab_group": ab_group, "latency_ms": latency_ms,
        "timestamp": time.time(),
    })
    if len(EVENT_LOG) > 1000:
        del EVENT_LOG[:500]

    return {
        "status":              "ok",
        "sku_id":             sku_id,
        "product_name":       product["name"],
        "category":           product["category"],
        "old_price":          old_price,
        "new_price":          new_price,
        "price_change":       round(new_price - old_price, 2),
        "direction":          "up" if new_price > old_price else ("down" if new_price < old_price else "same"),
        "rationale":          ml_result["reason"],
        "explanation":        explanation,                         # Phase 6
        "ab_group":           ab_group,
        "recommendations":    ml_result["top_recommendations"],
        "demand_score":       ml_result["demand_score"],
        "feature_vector":     ml_result["feature_vector"],
        "pricing_detail":     ml_result.get("pricing_detail", {}),
        "cold_start":         ml_result.get("cold_start", False),
        "pipeline_latency_ms": ml_result["latency_ms"],
        "inventory":          product["inventory"],
        "demand_count":       get_demand_count(sku_id),
        "engagement_score":   get_engagement_score(user_id),
    }


# ── GET /price ──
@app.get("/price")
def get_price(
    product_id: str = Query(..., description="Product SKU ID"),
    readonly: bool = Query(True, description="If false, increment view/click counters"),
):
    """
    Get current dynamic price for a product.
    Read-only by default. Use POST /event to record actual user interactions.
    """
    # Support both exact SKU IDs and simple numeric IDs (1-based)
    sku = product_id if product_id in PRODUCTS else None
    if not sku:
        skus = list(PRODUCTS.keys())
        try:
            idx = int(product_id) - 1
            if 0 <= idx < len(skus):
                sku = skus[idx]
        except (ValueError, IndexError):
            pass

    if not sku or sku not in PRODUCTS:
        return {"error": f"Product {product_id} not found", "price": 0}

    product = PRODUCTS[sku]
    old_price = product["current_price"]

    # Compute price using the ML pipeline demand score
    demand_score = ml_engine.get_demand_score(sku)
    ds_hybrid = demand_score.get("demand_score", 0)

    temp_price = product["base_price"]
    rationale = []

    if ds_hybrid > 50:
        temp_price *= 1.05
        rationale.append(f"📈 High ML demand ({ds_hybrid:.1f}): +5%")
    if product["inventory"] < LOW_STOCK_THRESHOLD:
        temp_price *= 1.03
        rationale.append(f"🔥 Low stock ({product['inventory']} left): +3%")
    elif product["inventory"] > 400:
        surplus = min(0.05, (product["inventory"] - 400) / 2000)
        temp_price *= (1 - surplus)
        rationale.append(f"📦 High stock: -{surplus*100:.1f}%")

    # Competitor awareness
    if sku in COMPETITOR_DATA:
        comp = COMPETITOR_DATA[sku]
        comp_price = comp["competitor_price"]
        if comp_price < temp_price * 0.95:
            adj = min(0.03, (temp_price - comp_price) / temp_price * 0.5)
            temp_price *= (1 - adj)
            rationale.append(f"🏪 Competitor match ({comp['competitor']}): -{adj*100:.1f}%")

    # Guardrails
    ceiling = product["base_price"] * (1 + MAX_PRICE_INCREASE)
    cost = product.get("cost_price", 0)
    floor = max(cost * 1.05 if cost > 0 else 0, product["base_price"] * (1 - MAX_PRICE_DECREASE))
    temp_price = max(floor, min(temp_price, ceiling))
    new_price = round(temp_price, 2)

    if not rationale:
        rationale.append("📊 Base price — stable demand")

    if not readonly:
        product["views"] += 1
        product["clicks"] += 1
        update_demand(sku, "click")
        product["current_price"] = new_price

    return {
        "product_id": sku,
        "name": product["name"],
        "price": new_price,
        "old_price": old_price,
        "base_price": product["base_price"],
        "direction": "up" if new_price > old_price else ("down" if new_price < old_price else "same"),
        "rationale": rationale,
        "inventory": product["inventory"],
        "views": product["views"],
        "clicks": product["clicks"],
        "demand": get_demand_count(sku),
        "demand_score": demand_score,
    }


# ── GET /recommendations ──
@app.get("/recommendations")
def get_recs(
    user_id: str = Query("U001"),
    session_id: str = Query("S001"),
):
    """Get personalized recommendations for a session using ML engine."""
    session = SESSION_STORE[session_id]
    clicked = set(session.get("clicks", []))
    recent_cats = session.get("categories", [])[-5:]
    recs = []
    seen_skus: set = set()

    # Category-affinity recommendations
    for cat in recent_cats:
        for c_sku in CATEGORIES.get(cat, []):
            if c_sku not in clicked and c_sku not in seen_skus and c_sku in PRODUCTS:
                p = PRODUCTS[c_sku]
                recs.append({
                    "sku_id": c_sku, "name": p["name"], "category": p["category"],
                    "price": p["current_price"], "rating": p["rating"],
                    "reason": f"Because you browsed {cat}",
                    "score": p["views"] + p["rating"] * 10,
                })
                seen_skus.add(c_sku)

    # Cold-start / trending fill
    if len(recs) < 5:
        trending = sorted(PRODUCTS.values(), key=lambda x: x["views"], reverse=True)
        for p in trending:
            if p["sku_id"] not in clicked and p["sku_id"] not in seen_skus:
                recs.append({
                    "sku_id": p["sku_id"], "name": p["name"], "category": p["category"],
                    "price": p["current_price"], "rating": p["rating"],
                    "reason": "Trending now",
                    "score": p["views"] + p["rating"] * 5,
                })
                seen_skus.add(p["sku_id"])
            if len(recs) >= 10:
                break

    recs.sort(key=lambda x: x["score"], reverse=True)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "engagement_score": get_engagement_score(user_id),
        "recommendations": recs[:5],
    }


# ── GET /analytics ──
@app.get("/analytics")
def analytics():
    """Dashboard analytics: A/B test results, revenue, engagement."""
    conv_a = (ANALYTICS["group_a_conversions"] / max(1, ANALYTICS["group_a_sessions"])) * 100
    conv_b = (ANALYTICS["group_b_conversions"] / max(1, ANALYTICS["group_b_sessions"])) * 100
    lift = round(conv_b - conv_a, 2) if conv_a > 0 else 0

    top_products = sorted(
        PRODUCTS.values(),
        key=lambda x: x["views"],
        reverse=True
    )[:5]

    recent_events = EVENT_LOG[-10:]  # Last 10 events

    return {
        "total_events": ANALYTICS["total_events"],
        "products_tracked": len(PRODUCTS),
        "avg_latency_ms": round(
            sum(e["latency_ms"] for e in EVENT_LOG[-50:]) / max(1, len(EVENT_LOG[-50:])),
            1
        ) if EVENT_LOG else 0,
        "ab_test": {
            "group_a_static": {
                "sessions": ANALYTICS["group_a_sessions"],
                "conversions": ANALYTICS["group_a_conversions"],
                "conversion_rate": round(conv_a, 2),
                "revenue": round(ANALYTICS["total_revenue_static"], 2),
            },
            "group_b_dynamic": {
                "sessions": ANALYTICS["group_b_sessions"],
                "conversions": ANALYTICS["group_b_conversions"],
                "conversion_rate": round(conv_b, 2),
                "revenue": round(ANALYTICS["total_revenue_dynamic"], 2),
            },
            "lift_pct": lift,
            "summary": f"Dynamic pricing {'increased' if lift > 0 else 'decreased'} conversion by {abs(lift):.1f}%",
        },
        "top_products": [
            {"name": p["name"], "views": p["views"], "clicks": p["clicks"],
             "price": p["current_price"], "demand": DEMAND_COUNTER.get(p["sku_id"], 0)}
            for p in top_products
        ],
        "recent_events": recent_events,
        "feature_summary": {
            "demand_counter_products": len(DEMAND_COUNTER),
            "engagement_tracked_users": len(ENGAGEMENT_SCORE),
            "high_demand_products": sum(1 for v in DEMAND_COUNTER.values() if v > DEMAND_THRESHOLD),
            "low_stock_products": sum(1 for p in PRODUCTS.values() if p["inventory"] < LOW_STOCK_THRESHOLD),
        },
    }


# ── GET /user/{user_id} — User Profile ──
@app.get("/user/{user_id}")
def get_user(user_id: str):
    """Get user segment profile and engagement score."""
    if user_id not in USER_SEGMENTS:
        return {"error": f"User {user_id} not found"}

    profile = USER_SEGMENTS[user_id]
    return {
        "user_id": user_id,
        "profile": profile,
        "engagement_score": get_engagement_score(user_id),
    }


# ── GET /features/{session_id} ──
@app.get("/features/{session_id}")
def get_feature_vector(session_id: str):
    """
    🧠 Real-time feature vector for a session.
    Returns intent score, price sensitivity, session duration,
    action history, cart/purchase counts, active category.
    """
    return ml_engine.get_feature_vector(session_id)


# ── GET /demand ──
@app.get("/demand")
def get_demand_scores(
    limit: int = Query(20, description="Max products to return"),
):
    """
    📈 Real-time demand scores for top products.
    Returns hybrid ML+heuristic demand score (0-100), click/cart velocities,
    trending status, and time-of-day boost factor.
    """
    top = ml_engine.get_top_demand_products(n=limit)
    # Enrich with product name
    for ds in top:
        sku = ds.get("sku_id", "")
        if sku in PRODUCTS:
            ds["product_name"] = PRODUCTS[sku]["name"]
            ds["category"] = PRODUCTS[sku]["category"]
    return {
        "demand_scores": top,
        "total": len(top),
        "ml_stats": ml_engine.get_stats(),
    }


# ── GET /demand/{sku_id} ──
@app.get("/demand/{sku_id}")
def get_product_demand(sku_id: str):
    """
    📊 Demand score for a specific product.
    Includes heuristic score, ML score, hybrid score,
    click/cart velocities, trending flag.
    """
    if sku_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Product {sku_id} not found")
    ds = ml_engine.get_demand_score(sku_id)
    product = PRODUCTS[sku_id]
    return {
        **ds,
        "product_name": product["name"],
        "category":     product["category"],
        "inventory":    product["inventory"],
        "base_price":   product["base_price"],
    }


# ── GET /ml/status ──
@app.get("/ml/status")
def ml_status():
    """
    🤖 ML engine health + Phase 3 model training status.
    Reports feature engine stats, demand predictor health,
    Phase 3 pricing/recommendation model readiness, AUC, and weights.
    """
    stats = ml_engine.get_stats()
    model_status = ml_engine.get_model_status()
    return {
        "status":             "healthy",
        "ml_engine_version":  "4.1.0",
        **stats,
        "demand_products_tracked": len(DEMAND_COUNTER),
        "feature_sessions":        stats.get("active_sessions", 0),
        "phase3_models":           model_status,
    }


# ── GET /model/status ──
@app.get("/model/status")
def model_status():
    """
    🧠 Phase 3 model status — pricing + recommendation models.
    Returns training metrics (AUC, accuracy), learned weights/coefficients,
    and model readiness flag.
    """
    return {
        "status": "ok",
        **ml_engine.get_model_status(),
    }


# ── POST /model/retrain ──
@app.post("/model/retrain")
def model_retrain(force: bool = Query(False, description="Force rebuild prepared data too")):
    """
    🔄 Trigger Phase 3 model retraining.
    Runs in background thread — returns immediately with job ID.
    Use GET /model/status to poll for completion.
    """
    import threading

    def _retrain():
        try:
            if force:
                from data_preparation import run_pipeline
                run_pipeline()
            from model_trainer import model_registry
            result = model_registry.startup(force_retrain=True)
            ml_engine._model_registry = model_registry
            logging.getLogger("server").info("Model retrain complete: %s", result)
        except Exception as e:
            logging.getLogger("server").error("Model retrain failed: %s", e)

    t = threading.Thread(target=_retrain, daemon=True, name="retrain-job")
    t.start()

    return {
        "status":  "retraining_started",
        "message": "Phase 3 models are retraining in background. Poll /model/status for completion.",
        "force_data_rebuild": force,
    }


# ── GET /stream/stats ──
@app.get("/stream/stats")
def stream_stats():
    """⚡ Streaming processor statistics: events/sec, latency, buffer status."""
    return ml_engine.get_stats()


# ── GET /health ──
@app.get("/health")
def health():
    ml_stats = ml_engine.get_stats()
    return {
        "status": "healthy",
        "version": "4.0.0",
        "products_loaded": len(PRODUCTS),
        "competitors_loaded": len(COMPETITOR_DATA),
        "user_segments_loaded": len(USER_SEGMENTS),
        "clickstream_events": CLICKSTREAM_STATS.get("total_events", 0),
        "demand_counter_size": len(DEMAND_COUNTER),
        "engagement_score_size": len(ENGAGEMENT_SCORE),
        "uptime_seconds": round(time.time() - _start_time, 1) if _start_time else 0,
        "ml_engine": {
            "ready":              ml_stats.get("ml_ready", False),
            "active_sessions":    ml_stats.get("active_sessions", 0),
            "events_processed":   ml_stats.get("events_processed", 0),
            "avg_latency_ms":     ml_stats.get("avg_latency_ms", 0),
        },
    }


# ── POST /decide ── (Phase 4 direct decision endpoint)
@app.post("/decide")
def decide(
    sku_id:      str   = Query(..., description="Product SKU ID"),
    user_id:     str   = Query("U001"),
    session_id:  str   = Query("S001"),
    device_type: str   = Query("desktop", description="mobile|tablet|desktop"),
    event_type:  str   = Query("view"),
):
    """
    Phase 4 — Real-Time Decision Engine.
    Returns instant pricing + top-5 recommendations + cold-start detection.
    Does NOT update product state (read-mostly). Sub-50ms target latency.
    """
    if sku_id not in PRODUCTS:
        raise HTTPException(404, f"SKU {sku_id} not found")

    product  = PRODUCTS[sku_id]
    session  = SESSION_STORE[session_id]

    if session["ab_group"] is None:
        session["ab_group"] = ab_system.assign(user_id, session_id) if ab_system else "B"

    # Build demand snapshot (top 100 only for speed)
    demand_map = {
        sku: ml_engine.demand_predictor.compute_demand_score(sku).hybrid_score
        for sku in list(PRODUCTS.keys())[:100]
    }

    try:
        from decision_engine import decision_engine as de
        de.set_model_registry(ml_engine._model_registry)
        ds  = ml_engine.demand_predictor.compute_demand_score(sku_id)
        fv  = ml_engine.feature_engine.get_feature_vector(session_id)
        bundle = de.decide(
            product=product, session=session,
            demand_score_obj=ds, feature_vec_obj=fv,
            all_products=PRODUCTS, all_categories=CATEGORIES,
            demand_scores=demand_map,
            ab_group=session["ab_group"],
            competitor_data=COMPETITOR_DATA.get(sku_id),
            device_type=device_type,
            user_id=user_id, session_id=session_id,
        )
        return bundle.to_dict()
    except ImportError:
        raise HTTPException(503, "DecisionEngine not available")


# ── GET /ab/metrics ── (Phase 5)
@app.get("/ab/metrics")
def ab_metrics():
    """
    Phase 5 — Live A/B test metrics.
    Returns conversion rate, CTR, revenue/user, avg price for both groups.
    """
    if ab_system is None:
        raise HTTPException(503, "A/B Testing System not loaded")
    return {"status": "ok", "metrics": ab_system.metrics}


# ── GET /ab/decision ── (Phase 5)
@app.get("/ab/decision")
def ab_decision():
    """
    Phase 5 — Run the A/B decision rule.
    Returns: deploy | deploy_revenue | adjust_weights | continue
    Includes statistical confidence and reason.
    """
    if ab_system is None:
        raise HTTPException(503, "A/B Testing System not loaded")
    decision = ab_system.evaluate()
    return {"status": "ok", "decision": decision.to_dict()}


# ── GET /ab/history ── (Phase 5)
@app.get("/ab/history")
def ab_history():
    """
    Phase 5 — Time-series snapshots of A/B metrics (every 30s).
    Use for rendering charts in the frontend dashboard.
    """
    if ab_system is None:
        raise HTTPException(503, "A/B Testing System not loaded")
    return {"status": "ok", "history": ab_system.history}


# ── POST /ab/reset ── (Phase 5)
@app.post("/ab/reset")
def ab_reset():
    """Reset A/B test metrics (start a new experiment)."""
    if ab_system is None:
        raise HTTPException(503, "A/B Testing System not loaded")
    ab_system.reset()
    return {"status": "ok", "message": "A/B test metrics reset. New experiment started."}


# _start_time is set in lifespan() at server startup; fallback to module load time
_start_time = time.time()


# ─────────────────────────────────────────────
# Phase 6 — Explainability + Simulation (Steps 22–24)
# ─────────────────────────────────────────────

# ── GET /simulate ── (Phase 6 — Step 23)
@app.get("/simulate")
def simulate_pricing(
    base_price: float = Query(1000.0, description="Base product price in USD"),
    scenario:   str   = Query("all",  description="'all' or one of: surge|boost|clearance|value_hold|stable|max"),
):
    """
    Phase 6 Step 23 — Price Simulation.
    Runs all demand×inventory scenarios and shows how the ML engine responds.
    
    Matrix:
      High demand + Low stock  → surge pricing  ↑
      High demand + High stock → moderate boost ↑
      Low  demand + High stock → clearance       ↓
      Low  demand + Low stock  → value hold (margin protected)
    """
    if not _EXPL_READY:
        raise HTTPException(503, "Explainability module not loaded")

    coefs = None
    if ml_engine._model_registry is not None:
        try:
            coefs = ml_engine._model_registry.pricing.coefficients
        except Exception:
            pass

    results = price_simulator.run_matrix(base_price=base_price, model_coefs=coefs)
    return {
        "status":     "ok",
        "base_price": base_price,
        "currency":   "USD",
        "model_coefs": coefs,
        "scenarios":  results,
        "summary": (
            "Simulation shows system intelligence: high demand → price up, "
            "low demand + surplus → clearance, low stock → urgency boost. "
            "All prices bounded by ±15% guardrail (Challenge 5 solution)."
        ),
    }


# ── POST /simulate/custom ── (Phase 6)
@app.post("/simulate/custom")
def simulate_custom(
    base_price:   float = Query(1000.0),
    demand_score: float = Query(50.0, ge=0, le=100),
    inventory:    int   = Query(100, ge=0),
    user_intent:  float = Query(5.0, ge=0),
    competitor_price: float = Query(0.0),
    ab_group:     str   = Query("B"),
):
    """
    Step 23 — Custom scenario simulation.
    Set any demand/inventory/intent combination and see the ML pricing decision.
    """
    if not _EXPL_READY:
        raise HTTPException(503, "Explainability module not loaded")

    coefs = None
    if ml_engine._model_registry is not None:
        try:
            coefs = ml_engine._model_registry.pricing.coefficients
        except Exception:
            pass

    result = price_simulator.run_scenario(
        base_price       = base_price,
        demand_score     = demand_score,
        inventory        = inventory,
        user_intent      = user_intent,
        competitor_price = competitor_price if competitor_price > 0 else None,
        ab_group         = ab_group,
        model_coefs      = coefs,
    )
    return {"status": "ok", **result}


# ── GET /explain/price ── (Phase 6 — Step 22)
@app.get("/explain/price")
def explain_price(
    sku_id:    str   = Query(..., description="Product SKU ID"),
    session_id: str  = Query("S001"),
    user_id:   str   = Query("U001"),
):
    """
    Step 22 — Price Explainability.
    Returns structured, human-readable explanation of why the current price was set.
    Includes: headline, badge, per-factor breakdown, summary paragraph, challenge notes.
    """
    if sku_id not in PRODUCTS:
        raise HTTPException(404, f"SKU {sku_id} not found")
    if not _EXPL_READY:
        raise HTTPException(503, "Explainability module not loaded")

    product = PRODUCTS[sku_id]
    ds      = ml_engine.demand_predictor.compute_demand_score(sku_id)
    fv      = ml_engine.feature_engine.get_feature_vector(session_id)
    comp    = COMPETITOR_DATA.get(sku_id, {})

    # Compute current dynamic price using DecisionEngine
    coefs = None
    if ml_engine._model_registry is not None:
        try:
            coefs = ml_engine._model_registry.pricing.coefficients
        except Exception:
            pass

    result = price_simulator.run_scenario(
        base_price   = product["base_price"],
        demand_score = ds.hybrid_score,
        inventory    = product["inventory"],
        user_intent  = fv.intent_score,
        competitor_price = comp.get("competitor_price") if comp else None,
        model_coefs  = coefs,
    )

    return {
        "status":        "ok",
        "sku_id":        sku_id,
        "product_name":  product["name"],
        "base_price":    product["base_price"],
        "current_price": result["final_price"],
        "adjustment_pct": result["adjustment_pct"],
        "explanation":   result["explanation"],
        "demand_score":  ds.to_dict(),
        "feature_vector": fv.to_dict(),
    }


# ── GET /explain/recommendations ── (Phase 6 — Step 22)
@app.get("/explain/recommendations")
def explain_recommendations(
    session_id:  str = Query("S001"),
    device_type: str = Query("desktop"),
    limit:       int = Query(5, ge=1, le=10),
):
    """
    Step 22 — Recommendation Explainability.
    Returns top-N recommendations with rich per-factor explanations:
    why each product was recommended (interest, similarity, trending).
    """
    if not _EXPL_READY:
        raise HTTPException(503, "Explainability module not loaded")

    session     = SESSION_STORE[session_id]
    is_cold     = len(session.get("clicks", [])) == 0
    recent_cats = session.get("categories", [])

    # Get demand scores for top 100 products
    demand_map = {
        sku: ml_engine.demand_predictor.compute_demand_score(sku).hybrid_score
        for sku in list(PRODUCTS.keys())[:100]
    }

    # Run decision engine to get ranked recs
    try:
        from decision_engine import decision_engine as de
        de.set_model_registry(ml_engine._model_registry)
        fv = ml_engine.feature_engine.get_feature_vector(session_id)
        if is_cold:
            raw_recs = de.step18_cold_start(PRODUCTS, demand_map, device_type, limit)
        else:
            raw_recs = de.step17_recommend(session, PRODUCTS, CATEGORIES, demand_map, limit)
    except Exception:
        raw_recs = []

    # Attach explainability to each
    explained = []
    for r in raw_recs:
        expl = recommendation_explainer.explain_item(
            name         = r.name,
            category     = r.category,
            interest     = r.interest,
            similarity   = r.similarity,
            trending     = r.trending,
            ml_score     = r.ml_score,
            cold_start   = r.cold_start,
            session_cats = recent_cats,
            device_type  = device_type,
        )
        explained.append({
            "rank":        r.rank,
            "sku_id":      r.sku_id,
            "name":        r.name,
            "category":    r.category,
            "price":       r.price,
            "rating":      r.rating,
            "ml_score":    r.ml_score,
            "tag":         r.tag,
            "explanation": expl.to_dict(),
        })

    return {
        "status":      "ok",
        "session_id":  session_id,
        "cold_start":  is_cold,
        "device_type": device_type,
        "recommendations": explained,
    }

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 62)
    print("  APEX -- Dynamic Pricing Engine v5.0")
    print("  Phases 1-5 · A/B Testing · Real-Time Decision Engine")
    print("=" * 62)
    print(f"  Data dir    : {PARQUET_DIR}")
    print(f"  API         : http://localhost:8000")
    print(f"  Explore     : http://localhost:8000/explore")
    print(f"  ML Status   : http://localhost:8000/ml/status")
    print(f"  Model Status: http://localhost:8000/model/status")
    print(f"  A/B Metrics : http://localhost:8000/ab/metrics")
    print(f"  A/B Decision: http://localhost:8000/ab/decision")
    print(f"  Demand API  : http://localhost:8000/demand")
    print(f"  Decide API  : http://localhost:8000/decide")
    print(f"  Docs        : http://localhost:8000/docs")
    print("=" * 62)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
