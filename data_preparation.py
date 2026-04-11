"""
============================================================
  APEX — Phase 2: Data Preparation Pipeline
  Steps 5-8 of the 24-Part ML Execution Plan
============================================================

  Step 5 — Feature Selection
    User features  : session_intensity, click_frequency,
                     price_sensitivity, engagement_score
    Product features: demand_score, trend_velocity,
                      inventory_ratio, rating_score

  Step 6 — Feature Transformation
    - Normalize demand → [0, 1]
    - Encode categories (LabelEncoder)
    - Time-based features (hour_sin, hour_cos, day_sin, day_cos)

  Step 7 — Label Creation
    Pricing model   : label = 1 if purchase happened, else 0
    Recommendation  : label = 1 if clicked, else 0

  Step 8 — Train-Test Split
    - 80% training / 20% testing
    - Temporal split: most-recent data goes to test set

  Output files (saved to ./prepared_data/):
    pricing_features_train.parquet
    pricing_features_test.parquet
    recommendation_features_train.parquet
    recommendation_features_test.parquet
    feature_metadata.json   ← scaler params + encoder maps
============================================================
"""

import os
import math
import json
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_prep")

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "Problem Statement 3 Sample Data")
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")   # output of data_cleaning.py
OUTPUT_DIR  = os.path.join(BASE_DIR, "prepared_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If cleaned_data/ exists and has the required files, prefer it over raw data
USE_CLEANED_DIR = os.path.isdir(CLEANED_DIR) and os.path.exists(
    os.path.join(CLEANED_DIR, "clickstream_clean.parquet")
)

CLICKSTREAM_SAMPLE  = 500_000   # rows to sample from 10.2M
RANDOM_STATE        = 42
TRAIN_RATIO         = 0.80      # 80/20 split

# ─────────────────────────────────────────────────────────
# Step 1: Load Datasets
# ─────────────────────────────────────────────────────────

def load_all() -> dict:
    """Load all 4 parquet datasets into memory.

    Prefers cleaned_data/ (output of data_cleaning.py) when available,
    falls back to raw Problem Statement 3 data.
    """
    t0 = time.time()
    if USE_CLEANED_DIR:
        log.info("Loading cleaned datasets from %s", CLEANED_DIR)
        source_dir = CLEANED_DIR
        prod_file   = "product_catalog_clean.parquet"
        user_file   = "user_segments_clean.parquet"
        comp_file   = "competitor_pricing_clean.parquet"
        click_file  = "clickstream_clean.parquet"
    else:
        log.info("cleaned_data/ not found — loading raw datasets from %s", DATA_DIR)
        source_dir = DATA_DIR
        prod_file   = "product_catalog.parquet"
        user_file   = "user_segment_profiles.parquet"
        comp_file   = "competitor_pricing_feed.parquet"
        click_file  = "clickstream_events.parquet"

    # 1. Product catalog
    products = pd.read_parquet(os.path.join(source_dir, prod_file))
    log.info("  Product catalog: %d rows", len(products))

    # 2. User segment profiles
    users = pd.read_parquet(os.path.join(source_dir, user_file))
    log.info("  User segments:   %d rows", len(users))

    # 3. Competitor pricing
    competitor = pd.read_parquet(os.path.join(source_dir, comp_file))
    log.info("  Competitor feed: %d rows", len(competitor))

    # 4. Clickstream events — sampled for memory efficiency
    log.info("  Clickstream:     sampling %s rows…", f"{CLICKSTREAM_SAMPLE:,}")
    clicks_full = pd.read_parquet(
        os.path.join(source_dir, click_file),
        columns=[
            "user_id", "sku_id", "event_type", "session_id",
            "category", "device_type", "ab_group",
            "price_seen_usd", "hour_of_day", "day_of_week",
        ],
    )
    if len(clicks_full) > CLICKSTREAM_SAMPLE:
        # Sample by unique session to preserve event sequences
        unique_sessions = clicks_full["session_id"].unique()
        frac = CLICKSTREAM_SAMPLE / len(clicks_full)
        sample_size = int(len(unique_sessions) * frac)
        np.random.seed(RANDOM_STATE)
        sampled_sessions = np.random.choice(unique_sessions, size=sample_size, replace=False)
        clicks = clicks_full[clicks_full["session_id"].isin(sampled_sessions)].copy()
    else:
        clicks = clicks_full.copy()
    log.info("  Clickstream:     %d sampled (full: %d)", len(clicks), len(clicks_full))

    log.info("All datasets loaded in %.1f s", time.time() - t0)
    return {
        "products":   products,
        "users":      users,
        "competitor": competitor,
        "clicks":     clicks,
        "clicks_full": clicks_full,
    }


# ─────────────────────────────────────────────────────────
# Step 5: Feature Selection
# Return clean DataFrames with only selected columns
# ─────────────────────────────────────────────────────────

def select_product_features(products: pd.DataFrame) -> pd.DataFrame:
    """
    Product features selected:
    - base_price_usd, cost_price_usd, current_price_usd
    - inventory_count  → inventory_ratio (normalized)
    - avg_rating       → rating_score
    - review_count
    - category, subcategory
    """
    cols = [
        "sku_id", "category", "subcategory",
        "base_price_usd", "cost_price_usd", "current_price_usd",
        "min_price_usd", "max_price_usd",
        "inventory_count", "avg_rating", "review_count",
    ]
    available = [c for c in cols if c in products.columns]
    df = products[available].copy()

    # Derived features
    max_inv = df["inventory_count"].max() if "inventory_count" in df else 1
    df["inventory_ratio"] = df["inventory_count"] / max(max_inv, 1)   # 0-1

    max_rev = df["review_count"].max() if "review_count" in df else 1
    df["review_popularity"] = df["review_count"] / max(max_rev, 1)     # 0-1

    if "avg_rating" in df:
        df["rating_score"] = df["avg_rating"] / 5.0   # 0-1

    log.info("  Product features selected: %d rows × %d cols", *df.shape)
    return df


def select_user_features(users: pd.DataFrame) -> pd.DataFrame:
    """
    User features selected:
    - segment, device_type
    - sessions_per_month        → session_intensity (normalized)
    - purchase_frequency        → click_frequency proxy
    - cart_abandonment_rate
    - willingness_to_pay_multiplier
    - lifetime_value_usd
    """
    cols = [
        "user_id", "segment", "device_type",
        "sessions_per_month", "purchase_frequency",
        "cart_abandonment_rate", "willingness_to_pay_multiplier",
        "lifetime_value_usd", "avg_order_value_usd",
    ]
    available = [c for c in cols if c in users.columns]
    df = users[available].copy()

    # Derived — session intensity (normalized sessions per month)
    if "sessions_per_month" in df:
        smax = df["sessions_per_month"].max()
        df["session_intensity"] = df["sessions_per_month"] / max(smax, 1)

    # click_frequency = normalized purchase frequency as proxy
    if "purchase_frequency" in df:
        pmax = df["purchase_frequency"].max()
        df["click_frequency"] = df["purchase_frequency"] / max(pmax, 1)

    log.info("  User features selected:    %d rows × %d cols", *df.shape)
    return df


def select_clickstream_features(clicks: pd.DataFrame) -> pd.DataFrame:
    """
    Clickstream features selected:
    - user_id, sku_id, session_id, event_type
    - category, device_type, ab_group
    - price_seen_usd
    - hour_of_day, day_of_week   → time-context features
    """
    cols = [
        "user_id", "sku_id", "session_id", "event_type",
        "category", "device_type", "ab_group",
        "price_seen_usd", "hour_of_day", "day_of_week",
    ]
    available = [c for c in cols if c in clicks.columns]
    df = clicks[available].copy()
    df = df.dropna(subset=["user_id", "sku_id", "event_type"])
    log.info("  Clickstream features:      %d rows × %d cols", *df.shape)
    return df


# ─────────────────────────────────────────────────────────
# Step 6: Feature Transformation
# Normalize → Encode → Time-based features
# ─────────────────────────────────────────────────────────

def _minmax(series: pd.Series, lo: float = None, hi: float = None) -> pd.Series:
    """Min-max normalization to [0, 1]."""
    lo = lo if lo is not None else series.min()
    hi = hi if hi is not None else series.max()
    rng = hi - lo
    return ((series - lo) / rng).clip(0, 1) if rng > 0 else series * 0.0


def add_time_features(df: pd.DataFrame, hour_col: str = "hour_of_day",
                      dow_col: str = "day_of_week") -> pd.DataFrame:
    """
    Cyclical encoding of hour and day-of-week using sine/cosine.
    This preserves the circular nature (hour 23 is close to hour 0).
    """
    if hour_col in df.columns:
        df["hour_sin"] = df[hour_col].apply(lambda h: math.sin(2 * math.pi * h / 24))
        df["hour_cos"] = df[hour_col].apply(lambda h: math.cos(2 * math.pi * h / 24))
        # Peak hour flag (18-22 = evening peak)
        df["is_peak_hour"] = df[hour_col].between(18, 22).astype(int)
    if dow_col in df.columns:
        df["dow_sin"] = df[dow_col].apply(lambda d: math.sin(2 * math.pi * d / 7))
        df["dow_cos"] = df[dow_col].apply(lambda d: math.cos(2 * math.pi * d / 7))
        df["is_weekend"] = df[dow_col].isin([5, 6]).astype(int)
    return df


def encode_categories(df: pd.DataFrame, cat_cols: list) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns (transformed_df, encoding_map_dict).
    """
    encoding_map: dict = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.lower().fillna("unknown")
        unique_vals = sorted(df[col].unique())
        code_map = {v: i for i, v in enumerate(unique_vals)}
        df[col + "_encoded"] = df[col].map(code_map).fillna(-1).astype(int)
        encoding_map[col] = code_map
    return df, encoding_map


def transform_product_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize numeric product features, encode categories."""
    meta: dict = {}

    # Normalize demand proxy: base price relative range
    for col in ["base_price_usd", "current_price_usd", "cost_price_usd"]:
        if col in df.columns:
            lo, hi = df[col].min(), df[col].max()
            df[col + "_norm"] = _minmax(df[col], lo, hi)
            meta[f"{col}_range"] = {"min": float(lo), "max": float(hi)}

    # Demand score proxy from inventory + reviews
    if "inventory_ratio" in df.columns and "review_popularity" in df.columns:
        df["demand_score_proxy"] = (
            0.5 * (1 - df["inventory_ratio"]) +   # scarcity signal
            0.3 * df["review_popularity"] +
            0.2 * df.get("rating_score", 0)
        ).clip(0, 1)

    # Trend velocity proxy: review_count decile rank
    if "review_count" in df.columns:
        df["trend_velocity"] = pd.qcut(
            df["review_count"].rank(method="first"), 10,
            labels=False, duplicates="drop"
        ).fillna(0) / 9.0    # normalize to 0-1

    # Encode categories
    df, enc = encode_categories(df, ["category", "subcategory"])
    meta["category_encoding"] = enc

    log.info("  Product features transformed")
    return df, meta


def transform_user_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize user numeric features, encode segment & device."""
    meta: dict = {}

    for col in ["lifetime_value_usd", "avg_order_value_usd"]:
        if col in df.columns:
            lo, hi = df[col].min(), df[col].max()
            df[col + "_norm"] = _minmax(df[col], lo, hi)
            meta[f"{col}_range"] = {"min": float(lo), "max": float(hi)}

    # price_sensitivity proxy: high cart_abandonment = price sensitive
    if "cart_abandonment_rate" in df.columns:
        df["price_sensitivity"] = df["cart_abandonment_rate"].clip(0, 1)

    df, enc = encode_categories(df, ["segment", "device_type"])
    meta["user_encoding"] = enc

    log.info("  User features transformed")
    return df, meta


def transform_clickstream_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add time features, encode event type, device, ab_group."""
    meta: dict = {}

    # Normalize price seen
    if "price_seen_usd" in df.columns:
        lo, hi = df["price_seen_usd"].min(), df["price_seen_usd"].max()
        df["price_seen_norm"] = _minmax(df["price_seen_usd"], lo, hi)
        meta["price_seen_range"] = {"min": float(lo), "max": float(hi)}

    # Time features
    df = add_time_features(df)

    # Encode categoricals
    df, enc = encode_categories(df, ["event_type", "device_type", "ab_group", "category"])
    meta["click_encoding"] = enc

    log.info("  Clickstream features transformed")
    return df, meta


# ─────────────────────────────────────────────────────────
# Step 7: Label Creation
# ─────────────────────────────────────────────────────────

def create_pricing_labels(clicks: pd.DataFrame, products: pd.DataFrame,
                          users: pd.DataFrame) -> pd.DataFrame:
    """
    Pricing label: 1 = conversion (purchase happened), 0 = no conversion.

    Strategy:
    - Join clickstream events with product and user features.
    - Each row = one user-product interaction.
    - label_conversion = 1 if event_type == 'purchase' else 0.
    """
    log.info("Creating pricing labels…")

    # Compute per-session conversion: session had ANY purchase
    session_purchases = (
        clicks[clicks["event_type"] == "purchase"]
        [["user_id", "sku_id", "session_id"]]
        .drop_duplicates()
        .assign(label_conversion=1)
    )

    # All distinct user-product-session interactions
    interactions = (
        clicks[["user_id", "sku_id", "session_id", "price_seen_usd",
                "hour_of_day", "day_of_week", "device_type", "ab_group", "category"]]
        .drop_duplicates(subset=["user_id", "sku_id", "session_id"])
    )

    # Merge label
    labeled = interactions.merge(
        session_purchases[["user_id", "sku_id", "session_id", "label_conversion"]],
        on=["user_id", "sku_id", "session_id"],
        how="left",
    )
    labeled["label_conversion"] = labeled["label_conversion"].fillna(0).astype(int)

    # Merge product features
    prod_cols = ["sku_id", "base_price_usd", "inventory_count", "avg_rating",
                 "category", "inventory_ratio", "demand_score_proxy", "trend_velocity"]
    prod_available = [c for c in prod_cols if c in products.columns]
    labeled = labeled.merge(products[prod_available].drop_duplicates("sku_id"),
                            on="sku_id", how="left", suffixes=("", "_prod"))

    # Merge user features
    user_cols = ["user_id", "session_intensity", "click_frequency", "price_sensitivity",
                 "willingness_to_pay_multiplier", "cart_abandonment_rate"]
    user_available = [c for c in user_cols if c in users.columns]
    labeled = labeled.merge(users[user_available].drop_duplicates("user_id"),
                            on="user_id", how="left")

    # Add time features
    labeled = add_time_features(labeled)

    log.info("  Pricing dataset: %d rows  |  conversion rate: %.2f%%",
             len(labeled), labeled["label_conversion"].mean() * 100)
    return labeled


def create_recommendation_labels(clicks: pd.DataFrame) -> pd.DataFrame:
    """
    Recommendation label: 1 = clicked or added to cart, 0 = only viewed.

    Strategy:
    - Each row = one user-product impression (page_view / product_view).
    - label_clicked = 1 if user later clicked or added that SKU in same session.
    """
    log.info("Creating recommendation labels…")

    CLICK_EVENTS = {"click", "product_view", "add_to_cart", "add_to_wishlist",
                    "checkout_start", "purchase"}

    # View impressions (what was shown to the user)
    views = clicks[clicks["event_type"].isin(["page_view", "product_view"])][
        ["user_id", "sku_id", "session_id", "hour_of_day", "day_of_week",
         "device_type", "category", "price_seen_usd"]
    ].copy()

    # Engagement events (user responded)
    engaged = (
        clicks[clicks["event_type"].isin(CLICK_EVENTS)][["user_id", "sku_id", "session_id"]]
        .drop_duplicates()
        .assign(label_clicked=1)
    )

    # Join
    labeled = views.merge(
        engaged, on=["user_id", "sku_id", "session_id"], how="left"
    )
    labeled["label_clicked"] = labeled["label_clicked"].fillna(0).astype(int)
    labeled = add_time_features(labeled)

    log.info("  Recommendation dataset: %d rows  |  CTR: %.2f%%",
             len(labeled), labeled["label_clicked"].mean() * 100)
    return labeled


# ─────────────────────────────────────────────────────────
# Step 8: Train-Test Split (temporal — recent = test)
# ─────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO,
                   time_col: str = "hour_of_day") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split.
    - Sort by time proxy (hour_of_day + day_of_week or row index).
    - First 80% → train, last 20% → test.
    This keeps realistic evaluation: model trained on past, tested on future.
    """
    if time_col in df.columns and "day_of_week" in df.columns:
        df = df.sort_values(["day_of_week", time_col]).reset_index(drop=True)
    # else: use natural row order (already sorted by parquet index)

    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    return train, test


# ─────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """Run full Phase 2 pipeline. Returns metadata dict."""
    t_total = time.time()
    print()
    print("=" * 60)
    print("  APEX — Phase 2: Data Preparation Pipeline")
    print("  Steps 5→8 of the 24-Part ML Execution Plan")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────
    data = load_all()
    products_raw  = data["products"]
    users_raw     = data["users"]
    clicks_raw    = data["clicks"]

    # ── Step 5: Feature Selection ─────────────────────────
    print("\n[Step 5] Feature Selection…")
    prod_df = select_product_features(products_raw)
    user_df = select_user_features(users_raw)
    click_df = select_clickstream_features(clicks_raw)

    # ── Step 6: Feature Transformation ───────────────────
    print("\n[Step 6] Feature Transformation…")
    prod_df,  prod_meta  = transform_product_features(prod_df)
    user_df,  user_meta  = transform_user_features(user_df)
    click_df, click_meta = transform_clickstream_features(click_df)

    all_meta = {**prod_meta, **user_meta, **click_meta,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ── Step 7: Label Creation ────────────────────────────
    print("\n[Step 7] Label Creation…")
    pricing_df      = create_pricing_labels(click_df, prod_df, user_df)
    recom_df        = create_recommendation_labels(click_df)

    # ── Step 8: Train-Test Split ──────────────────────────
    print("\n[Step 8] Train-Test Split (80/20 temporal)…")
    p_train, p_test = temporal_split(pricing_df)
    r_train, r_test = temporal_split(recom_df)

    log.info("  Pricing    — train: %d  |  test: %d", len(p_train), len(p_test))
    log.info("  Recomm.   — train: %d  |  test: %d", len(r_train), len(r_test))

    # ── Save to parquet ───────────────────────────────────
    print("\n[Save] Writing prepared datasets…")
    p_train.to_parquet(os.path.join(OUTPUT_DIR, "pricing_features_train.parquet"), index=False)
    p_test.to_parquet( os.path.join(OUTPUT_DIR, "pricing_features_test.parquet"),  index=False)
    r_train.to_parquet(os.path.join(OUTPUT_DIR, "recommendation_features_train.parquet"), index=False)
    r_test.to_parquet( os.path.join(OUTPUT_DIR, "recommendation_features_test.parquet"),  index=False)

    # Persist feature metadata (scaler ranges + encoding maps)
    meta_path = os.path.join(OUTPUT_DIR, "feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2, default=str)

    elapsed = time.time() - t_total
    print()
    print("=" * 60)
    print("  ✅ Phase 2 Complete!")
    print(f"  ⏱  Total time: {elapsed:.1f} s")
    print(f"  📁 Output dir: {OUTPUT_DIR}")
    print(f"  📋 Pricing train:  {len(p_train):,} rows × {p_train.shape[1]} features")
    print(f"  📋 Pricing test:   {len(p_test):,} rows × {p_test.shape[1]} features")
    print(f"  📋 Recom. train:   {len(r_train):,} rows × {r_train.shape[1]} features")
    print(f"  📋 Recom. test:    {len(r_test):,} rows × {r_test.shape[1]} features")
    print("=" * 60)
    print()

    return {
        "pricing_train_shape":  p_train.shape,
        "pricing_test_shape":   p_test.shape,
        "recom_train_shape":    r_train.shape,
        "recom_test_shape":     r_test.shape,
        "feature_metadata_path": meta_path,
        "elapsed_s": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────
# Integrate with ml_engine: expose prepared features
# ─────────────────────────────────────────────────────────

class PreparedDataLoader:
    """
    Thin wrapper so ml_engine (and server.py) can load  
    the prepared datasets at startup without re-running the full pipeline.
    """
    _cache: dict = {}

    @classmethod
    def load(cls, force_rebuild: bool = False) -> dict:
        """Load prepared data; rebuild if missing or force_rebuild=True."""
        needed = [
            "pricing_features_train.parquet",
            "pricing_features_test.parquet",
            "recommendation_features_train.parquet",
            "recommendation_features_test.parquet",
            "feature_metadata.json",
        ]
        all_exist = all(
            os.path.exists(os.path.join(OUTPUT_DIR, n)) for n in needed
        )

        if not all_exist or force_rebuild:
            log.info("Prepared data not found — running pipeline…")
            run_pipeline()

        if not cls._cache or force_rebuild:
            cls._cache = {
                "pricing_train":  pd.read_parquet(os.path.join(OUTPUT_DIR, "pricing_features_train.parquet")),
                "pricing_test":   pd.read_parquet(os.path.join(OUTPUT_DIR, "pricing_features_test.parquet")),
                "recom_train":    pd.read_parquet(os.path.join(OUTPUT_DIR, "recommendation_features_train.parquet")),
                "recom_test":     pd.read_parquet(os.path.join(OUTPUT_DIR, "recommendation_features_test.parquet")),
            }
            # Use context manager to ensure file is properly closed
            meta_path = os.path.join(OUTPUT_DIR, "feature_metadata.json")
            with open(meta_path) as f:
                cls._cache["metadata"] = json.load(f)
            log.info("Prepared datasets loaded into memory.")
        return cls._cache


if __name__ == "__main__":
    run_pipeline()
