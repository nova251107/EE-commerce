"""
============================================================
  APEX — Data Cleaning Pipeline
  Pre-processing step before model training
============================================================

  Reads the 4 raw parquet datasets from Problem Statement 3:
    1. clickstream_events.parquet       (10.2M rows)
    2. product_catalog.parquet          (5,368 rows)
    3. user_segment_profiles.parquet    (500K rows)
    4. competitor_pricing_feed.parquet  (1.69M rows)

  Cleaning steps applied:
    Step 1 — Detect & report missing values
    Step 2 — Handle duplicate rows
    Step 3 — Standardize event type names
    Step 4 — Validate ID columns (sku_id, user_id, session_id)
    Step 5 — Parse & validate timestamps
    Step 6 — Outlier detection & capping (price, inventory, clicks)
    Step 7 — Data leakage prevention audit

  Output:
    ./cleaned_data/clickstream_clean.parquet
    ./cleaned_data/product_catalog_clean.parquet
    ./cleaned_data/user_segments_clean.parquet
    ./cleaned_data/competitor_pricing_clean.parquet
    ./cleaned_data/cleaning_report.json
============================================================
"""

import os
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
log = logging.getLogger("data_cleaner")

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Problem Statement 3 Sample Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "cleaned_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLICKSTREAM_SAMPLE = 500_000   # Sample for memory safety
RANDOM_STATE = 42

# Canonical event type mapping (standardize all aliases)
EVENT_TYPE_MAP = {
    # Already canonical
    "page_view":         "page_view",
    "product_view":      "product_view",
    "search":            "search",
    "add_to_cart":       "add_to_cart",
    "add_to_wishlist":   "add_to_wishlist",
    "checkout_start":    "checkout_start",
    "purchase":          "purchase",
    "remove_from_cart":  "remove_from_cart",
    "page_exit":         "page_exit",
    # Common aliases / typos to catch
    "view":              "page_view",
    "click":             "product_view",
    "cart":              "add_to_cart",
    "buy":               "purchase",
    "bought":            "purchase",
    "wishlist":          "add_to_wishlist",
    "checkout":          "checkout_start",
    "exit":              "page_exit",
}

# Outlier caps (IQR-based or domain-driven)
PRICE_CAP_LOW    = 1.0       # No product costs less than $1
PRICE_CAP_HIGH   = 5000.0    # No single item above $5000
INVENTORY_CAP    = 1000      # Warehouse max per SKU
CLICK_PRICE_LOW  = 0.50      # Minimum realistic price seen

report = {}   # Collects stats for the cleaning report


# ─────────────────────────────────────────────────────────
# Step 1: Detect & Report Missing Values
# ─────────────────────────────────────────────────────────

def detect_missing(df: pd.DataFrame, dataset_name: str) -> dict:
    """Log and return missing value counts per column."""
    nulls = df.isnull().sum()
    total = len(df)
    missing = {col: int(n) for col, n in nulls.items() if n > 0}

    if missing:
        log.warning("  [%s] Missing values found:", dataset_name)
        for col, n in missing.items():
            pct = n / total * 100
            log.warning("    %-30s  %d (%.2f%%)", col, n, pct)
    else:
        log.info("  [%s] No missing values ✓", dataset_name)

    return {"dataset": dataset_name, "total_rows": total, "missing_columns": missing}


# ─────────────────────────────────────────────────────────
# Step 2: Handle Duplicate Rows
# ─────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, dataset_name: str,
                      subset: list = None) -> pd.DataFrame:
    """Remove exact duplicates; log count."""
    before = len(df)
    # Only use columns that actually exist in the dataframe
    if subset is not None:
        subset = [c for c in subset if c in df.columns] or None
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        log.info("  [%s] Removed %d duplicate rows", dataset_name, removed)
    else:
        log.info("  [%s] No duplicates ✓", dataset_name)
    report[f"{dataset_name}_duplicates_removed"] = removed
    return df


# ─────────────────────────────────────────────────────────
# Step 3: Standardize Event Type Names
# ─────────────────────────────────────────────────────────

def standardize_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """Map all event_type values to canonical names."""
    if "event_type" not in df.columns:
        return df

    original_types = set(df["event_type"].unique())
    # Lowercase + strip whitespace first
    df["event_type"] = df["event_type"].str.strip().str.lower()
    df["event_type"] = df["event_type"].map(EVENT_TYPE_MAP).fillna(df["event_type"])

    new_types = set(df["event_type"].unique())
    unknown = new_types - set(EVENT_TYPE_MAP.values())
    if unknown:
        log.warning("  Unknown event types after mapping: %s", unknown)
    else:
        log.info("  Event types standardized ✓  (%d unique)", len(new_types))

    report["event_types_before"] = sorted(original_types)
    report["event_types_after"]  = sorted(new_types)
    return df


# ─────────────────────────────────────────────────────────
# Step 4: Validate ID Columns
# ─────────────────────────────────────────────────────────

def validate_ids(df: pd.DataFrame, dataset_name: str,
                 id_cols: list) -> pd.DataFrame:
    """
    Ensure ID columns are non-null, non-empty strings.
    Drop rows with invalid IDs.
    """
    before = len(df)
    for col in id_cols:
        if col not in df.columns:
            continue
        # Convert to string and strip
        df[col] = df[col].astype(str).str.strip()
        # Flag empties and 'nan'
        invalid_mask = df[col].isin(["", "nan", "None", "null", "NaN"])
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            log.warning("  [%s] %s has %d invalid IDs — dropping", dataset_name, col, n_invalid)
            df = df[~invalid_mask].reset_index(drop=True)

    removed = before - len(df)
    report[f"{dataset_name}_invalid_ids_removed"] = removed
    if removed == 0:
        log.info("  [%s] All IDs valid ✓", dataset_name)
    return df


# ─────────────────────────────────────────────────────────
# Step 5: Parse & Validate Timestamps
# ─────────────────────────────────────────────────────────

def parse_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Parse ISO-8601 string timestamps to proper datetime,
    then extract useful time components.
    """
    if ts_col not in df.columns:
        log.info("  No '%s' column — skipping timestamp parsing", ts_col)
        return df

    original_dtype = str(df[ts_col].dtype)

    # Parse to datetime (coerce errors to NaT)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    nat_count = df[ts_col].isna().sum()
    if nat_count > 0:
        log.warning("  %d timestamps could not be parsed — dropping rows", nat_count)
        df = df.dropna(subset=[ts_col]).reset_index(drop=True)

    # Validate date range (should be 2023-2026 for this dataset)
    min_ts = df[ts_col].min()
    max_ts = df[ts_col].max()
    log.info("  Timestamp range: %s to %s (parsed from %s)", min_ts, max_ts, original_dtype)

    # Flag future timestamps
    now = pd.Timestamp.now(tz="UTC")
    future_count = (df[ts_col] > now).sum()
    if future_count > 0:
        log.warning("  %d future timestamps found — clamping to now", future_count)
        df.loc[df[ts_col] > now, ts_col] = now

    report["timestamp_range"] = {"min": str(min_ts), "max": str(max_ts),
                                  "nat_dropped": int(nat_count),
                                  "future_clamped": int(future_count)}
    return df


# ─────────────────────────────────────────────────────────
# Step 6: Outlier Detection & Capping
# ─────────────────────────────────────────────────────────

def cap_outliers(df: pd.DataFrame, col: str,
                 low: float, high: float, dataset_name: str) -> pd.DataFrame:
    """Winsorize / cap a numeric column to [low, high]."""
    if col not in df.columns:
        return df

    below = (df[col] < low).sum()
    above = (df[col] > high).sum()
    total = len(df)

    df[col] = df[col].clip(lower=low, upper=high)

    if below + above > 0:
        log.info("  [%s] %s capped: %d below %.1f, %d above %.1f (%.2f%% affected)",
                 dataset_name, col, below, low, above, high,
                 (below + above) / total * 100)
    else:
        log.info("  [%s] %s within bounds [%.1f, %.1f] ✓", dataset_name, col, low, high)

    report[f"{dataset_name}_{col}_outliers"] = {"below": int(below), "above": int(above)}
    return df


def detect_price_anomalies(products: pd.DataFrame) -> pd.DataFrame:
    """Flag products where cost > base or min > max."""
    # Initialize counters before conditional blocks to prevent NameError
    cost_above_base = 0
    min_above_max = 0

    if "cost_price_usd" in products.columns and "base_price_usd" in products.columns:
        cost_above_base = int((products["cost_price_usd"] > products["base_price_usd"]).sum())
        if cost_above_base > 0:
            log.warning("  [Products] %d products have cost_price > base_price — clamping cost", cost_above_base)
            mask = products["cost_price_usd"] > products["base_price_usd"]
            products.loc[mask, "cost_price_usd"] = products.loc[mask, "base_price_usd"] * 0.6

    if "min_price_usd" in products.columns and "max_price_usd" in products.columns:
        min_above_max = int((products["min_price_usd"] > products["max_price_usd"]).sum())
        if min_above_max > 0:
            log.warning("  [Products] %d products have min_price > max_price — swapping", min_above_max)
            mask = products["min_price_usd"] > products["max_price_usd"]
            products.loc[mask, ["min_price_usd", "max_price_usd"]] = (
                products.loc[mask, ["max_price_usd", "min_price_usd"]].values
            )

    report["price_anomalies"] = {"cost_above_base": cost_above_base,
                                  "min_above_max": min_above_max}
    return products


# ─────────────────────────────────────────────────────────
# Step 7: Data Leakage Prevention
# ─────────────────────────────────────────────────────────

def leakage_audit(clicks: pd.DataFrame, products: pd.DataFrame,
                  users: pd.DataFrame) -> dict:
    """
    Check for potential data leakage risks and log warnings.
    Returns a summary dict of findings.
    """
    findings = []

    # 1. Check: clickstream should NOT contain 'label' or 'target' columns
    leak_cols = [c for c in clicks.columns if "label" in c.lower() or "target" in c.lower()]
    if leak_cols:
        findings.append(f"Clickstream has pre-computed label columns: {leak_cols}")
        log.warning("  ⚠ LEAKAGE RISK: %s", findings[-1])

    # 2. Check: product current_price should NOT be a feature when predicting price
    if "current_price_usd" in products.columns:
        findings.append(
            "product_catalog contains 'current_price_usd' — must be EXCLUDED from "
            "pricing model features (it IS the target proxy)"
        )
        log.info("  ℹ Note: current_price_usd will be excluded from pricing features")

    # 3. Check: user lifetime_value should not predict conversion
    if "lifetime_value_usd" in users.columns:
        findings.append(
            "user lifetime_value_usd encodes future information — should be excluded "
            "or used only in segment analysis, not as a real-time feature"
        )
        log.info("  ℹ Note: lifetime_value_usd flagged as potential future-leakage")

    # 4. Check: purchase events feeding back into demand score for same window
    #    (this is a design-level check — log the reminder)
    findings.append(
        "Ensure temporal split is applied before label creation so test labels "
        "never influence train-set demand counters"
    )

    # 5. Fairness: demographic columns must NOT be pricing features
    demographic_cols = {"age_group", "gender", "country", "os"}
    present_demo = demographic_cols & set(users.columns)
    if present_demo:
        findings.append(
            f"Demographic columns present: {sorted(present_demo)} — MUST be excluded "
            f"from pricing model per fairness policy"
        )
        log.info("  ℹ Fairness: demographics %s flagged for exclusion", sorted(present_demo))

    if not findings:
        log.info("  Data leakage audit: all clear ✓")
    else:
        log.info("  Data leakage audit: %d findings logged", len(findings))

    report["leakage_audit"] = findings
    return {"findings": findings}


# ─────────────────────────────────────────────────────────
# Handle Missing Values (fills)
# ─────────────────────────────────────────────────────────

def fill_missing_values(products: pd.DataFrame, users: pd.DataFrame,
                        competitor: pd.DataFrame) -> tuple:
    """Apply domain-appropriate missing value strategies."""

    # Products: restock_days has 3658 nulls (68% of catalog)
    # Strategy: fill with median (conservative estimate)
    if "restock_days" in products.columns:
        median_restock = products["restock_days"].median()
        n_fill = products["restock_days"].isna().sum()
        products["restock_days"] = products["restock_days"].fillna(median_restock)
        log.info("  [Products] restock_days: filled %d nulls with median=%.0f", n_fill, median_restock)

    # Competitor: no nulls found, but guard against future ones
    for col in ["competitor_price", "our_base_price"]:
        if col in competitor.columns:
            n_null = competitor[col].isna().sum()
            if n_null > 0:
                competitor = competitor.dropna(subset=[col])
                log.info("  [Competitor] Dropped %d rows with null %s", n_null, col)

    return products, users, competitor


# ─────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────

def run_cleaning():
    """Execute the full 7-step data cleaning pipeline."""
    t_total = time.time()

    print()
    print("=" * 60)
    print("  APEX — Data Cleaning Pipeline")
    print("  7-Step Pre-processing for Model Training")
    print("=" * 60)

    # ── Load raw datasets ─────────────────────────────────
    print("\n[Load] Reading raw datasets...")
    products = pd.read_parquet(os.path.join(DATA_DIR, "product_catalog.parquet"))
    users = pd.read_parquet(os.path.join(DATA_DIR, "user_segment_profiles.parquet"))
    competitor = pd.read_parquet(os.path.join(DATA_DIR, "competitor_pricing_feed.parquet"))

    log.info("  Loading clickstream (sampling %d sessions)...", CLICKSTREAM_SAMPLE)
    clicks_full = pd.read_parquet(
        os.path.join(DATA_DIR, "clickstream_events.parquet"),
    )
    # Session-aware sampling (preserves complete session sequences)
    if len(clicks_full) > CLICKSTREAM_SAMPLE:
        unique_sessions = clicks_full["session_id"].unique()
        frac = CLICKSTREAM_SAMPLE / len(clicks_full)
        sample_size = int(len(unique_sessions) * frac)
        np.random.seed(RANDOM_STATE)
        sampled_sessions = np.random.choice(unique_sessions, size=sample_size, replace=False)
        clicks = clicks_full[clicks_full["session_id"].isin(set(sampled_sessions))].copy()
    else:
        clicks = clicks_full.copy()
    del clicks_full  # free memory
    log.info("  Clickstream: %d rows loaded (session-sampled)", len(clicks))

    report["raw_shapes"] = {
        "clickstream": clicks.shape[0],
        "products": products.shape[0],
        "users": users.shape[0],
        "competitor": competitor.shape[0],
    }

    # ── Step 1: Missing values ────────────────────────────
    print("\n[Step 1] Detecting missing values...")
    report["missing_clickstream"] = detect_missing(clicks, "Clickstream")
    report["missing_products"]    = detect_missing(products, "Products")
    report["missing_users"]       = detect_missing(users, "Users")
    report["missing_competitor"]  = detect_missing(competitor, "Competitor")

    # Fill missing values with domain-appropriate strategies
    products, users, competitor = fill_missing_values(products, users, competitor)

    # ── Step 2: Duplicates ────────────────────────────────
    print("\n[Step 2] Handling duplicates...")
    clicks     = remove_duplicates(clicks, "Clickstream", subset=["event_id"])
    products   = remove_duplicates(products, "Products", subset=["sku_id"])
    users      = remove_duplicates(users, "Users", subset=["user_id"])
    competitor = remove_duplicates(competitor, "Competitor",
                                   subset=["date", "sku_id", "competitor"])

    # ── Step 3: Standardize event names ───────────────────
    print("\n[Step 3] Standardizing event types...")
    clicks = standardize_event_types(clicks)

    # ── Step 4: Validate IDs ──────────────────────────────
    print("\n[Step 4] Validating ID columns...")
    clicks   = validate_ids(clicks, "Clickstream", ["user_id", "sku_id", "session_id"])
    products = validate_ids(products, "Products", ["sku_id"])
    users    = validate_ids(users, "Users", ["user_id"])

    # Cross-reference: clickstream sku_ids should exist in product catalog
    valid_skus = set(products["sku_id"].unique())
    orphan_mask = ~clicks["sku_id"].isin(valid_skus)
    n_orphan = orphan_mask.sum()
    if n_orphan > 0:
        log.warning("  [Clickstream] %d events reference non-existent SKUs — dropping", n_orphan)
        clicks = clicks[~orphan_mask].reset_index(drop=True)
    else:
        log.info("  [Clickstream] All SKU references valid ✓")
    report["orphan_sku_events"] = int(n_orphan)

    # ── Step 5: Timestamps ────────────────────────────────
    print("\n[Step 5] Parsing & validating timestamps...")
    clicks = parse_timestamps(clicks, "timestamp")

    # ── Step 6: Outlier capping ───────────────────────────
    print("\n[Step 6] Detecting & capping outliers...")
    # Product prices
    for pcol in ["base_price_usd", "cost_price_usd", "current_price_usd",
                 "min_price_usd", "max_price_usd"]:
        products = cap_outliers(products, pcol, PRICE_CAP_LOW, PRICE_CAP_HIGH, "Products")

    products = cap_outliers(products, "inventory_count", 0, INVENTORY_CAP, "Products")
    products = detect_price_anomalies(products)

    # Clickstream price_seen
    clicks = cap_outliers(clicks, "price_seen_usd", CLICK_PRICE_LOW, PRICE_CAP_HIGH, "Clickstream")

    # Competitor prices
    competitor = cap_outliers(competitor, "competitor_price",
                              PRICE_CAP_LOW, PRICE_CAP_HIGH, "Competitor")

    # ── Step 7: Data leakage audit ────────────────────────
    print("\n[Step 7] Running data leakage prevention audit...")
    leakage_audit(clicks, products, users)

    # ── Save cleaned datasets ─────────────────────────────
    print("\n[Save] Writing cleaned datasets...")
    clicks.to_parquet(os.path.join(OUTPUT_DIR, "clickstream_clean.parquet"), index=False)
    products.to_parquet(os.path.join(OUTPUT_DIR, "product_catalog_clean.parquet"), index=False)
    users.to_parquet(os.path.join(OUTPUT_DIR, "user_segments_clean.parquet"), index=False)
    competitor.to_parquet(os.path.join(OUTPUT_DIR, "competitor_pricing_clean.parquet"), index=False)

    report["clean_shapes"] = {
        "clickstream": clicks.shape[0],
        "products": products.shape[0],
        "users": users.shape[0],
        "competitor": competitor.shape[0],
    }

    # Save cleaning report
    report_path = os.path.join(OUTPUT_DIR, "cleaning_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t_total

    print()
    print("=" * 60)
    print("  ✅ Data Cleaning Complete!")
    print(f"  ⏱  Total time: {elapsed:.1f} s")
    print(f"  📁 Output dir:  {OUTPUT_DIR}")
    print(f"  📋 Clickstream: {clicks.shape[0]:,} rows × {clicks.shape[1]} cols")
    print(f"  📋 Products:    {products.shape[0]:,} rows × {products.shape[1]} cols")
    print(f"  📋 Users:       {users.shape[0]:,} rows × {users.shape[1]} cols")
    print(f"  📋 Competitor:  {competitor.shape[0]:,} rows × {competitor.shape[1]} cols")
    print(f"  📄 Report:      {report_path}")
    print("=" * 60)
    print()

    return report


if __name__ == "__main__":
    run_cleaning()
