"""
============================================================
  APEX — Model Training Pipeline (v2)
  Pricing + Recommendation + Online Incremental
============================================================

  Handles synthetic hackathon datasets where raw feature
  distributions are identical across labels. Adds engineered
  interaction features to create discriminative power.

  Run:  py train_models.py
============================================================
"""

import os
import time
import json
import logging
import warnings
import pickle

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "prepared_data")
MODEL_DIR  = os.path.join(BASE_DIR, "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

PRICING_LABEL = "label_conversion"
RECOM_LABEL   = "label_clicked"

# Categorical columns to encode
CAT_COLS_PRICING = ["category", "device_type", "ab_group"]
CAT_COLS_RECOM   = ["category", "device_type"]


# ─────────────────────────────────────────────────────────
# Feature Engineering (interaction features for signal)
# ─────────────────────────────────────────────────────────

def engineer_pricing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features that capture non-linear
    relationships between price, inventory, and user behavior.
    These generate discriminative signal even when raw features
    are identically distributed across labels.
    """
    df = df.copy()

    # Price ratio: how far is seen price from base? (discount or surge indicator)
    if "price_seen_usd" in df.columns and "base_price_usd" in df.columns:
        df["price_ratio"] = (df["price_seen_usd"] / df["base_price_usd"].clip(lower=1)).clip(0.1, 5.0)
        df["price_deviation"] = (df["price_seen_usd"] - df["base_price_usd"]).abs()

    # Scarcity × demand interaction
    if "inventory_ratio" in df.columns and "demand_score_proxy" in df.columns:
        df["scarcity_urgency"] = (1 - df["inventory_ratio"]) * df["demand_score_proxy"]

    # User engagement composite
    if "session_intensity" in df.columns and "click_frequency" in df.columns:
        df["engagement_index"] = df["session_intensity"] * df["click_frequency"]

    # WTP vs price gap (is this user willing to pay this price?)
    if "willingness_to_pay_multiplier" in df.columns and "price_ratio" in df.columns:
        df["wtp_price_gap"] = df["willingness_to_pay_multiplier"] - df["price_ratio"]

    # Abandonment risk (high sensitivity + high price = low conversion)
    if "price_sensitivity" in df.columns and "price_ratio" in df.columns:
        df["abandon_risk"] = df["price_sensitivity"] * df["price_ratio"]

    # Rating × demand interaction
    if "avg_rating" in df.columns and "demand_score_proxy" in df.columns:
        df["quality_demand"] = df["avg_rating"] * df["demand_score_proxy"]

    return df


def engineer_recom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features for recommendation model."""
    df = df.copy()

    if "price_seen_usd" in df.columns:
        # Price bucket (budget / mid / premium)
        df["price_bucket"] = pd.cut(
            df["price_seen_usd"],
            bins=[0, 300, 1000, 10000],
            labels=[0, 1, 2],
        ).astype(float).fillna(1)

    # Time interaction
    if "is_peak_hour" in df.columns and "is_weekend" in df.columns:
        df["peak_weekend"] = df["is_peak_hour"] * df["is_weekend"]

    return df


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def encode_categoricals(train_df, test_df, cat_cols):
    """Label-encode categorical columns. Fit on train, transform both."""
    encoders = {}
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        train_df[col + "_enc"] = le.fit_transform(train_df[col].astype(str).fillna("unknown"))
        test_vals = test_df[col].astype(str).fillna("unknown")
        test_df[col + "_enc"] = test_vals.map(
            lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_ else -1
        )
        encoders[col] = le
    return train_df, test_df, encoders


def auto_select_features(df, label_col, cat_cols):
    """Auto-select all numeric + encoded categorical columns, excluding IDs and labels."""
    drop_cols = {"user_id", "sku_id", "session_id", label_col,
                 "category", "device_type", "ab_group", "category_prod"}
    encoded = {c + "_enc" for c in cat_cols if c + "_enc" in df.columns}
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    features = sorted((numeric | encoded) - drop_cols)
    return features


def prepare_Xy(df, feature_cols, label_col):
    """Extract X matrix and y vector."""
    X = df[feature_cols].fillna(0).astype(float)
    y = df[label_col].values
    return X, y


def print_metrics(name, y_true, y_pred, y_proba):
    """Print classification metrics."""
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr  = average_precision_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"  {name} — Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  AUC-ROC:    {auc_roc:.4f}")
    print(f"  AUC-PR:     {auc_pr:.4f}")
    print(f"  Precision:  {report['1']['precision']:.4f}")
    print(f"  Recall:     {report['1']['recall']:.4f}")
    print(f"  F1-Score:   {report['1']['f1-score']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print(f"{'=' * 50}\n")

    return {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────
# Model 1: Pricing (Conversion Prediction)
# ─────────────────────────────────────────────────────────

def train_pricing_model():
    t0 = time.time()

    train_path = os.path.join(DATA_DIR, "pricing_features_train.parquet")
    test_path  = os.path.join(DATA_DIR, "pricing_features_test.parquet")
    if not os.path.exists(train_path):
        log.warning("prepared_data/ not found — running data_preparation pipeline first...")
        try:
            from data_preparation import run_pipeline
            run_pipeline()
        except Exception as prep_err:
            raise FileNotFoundError(
                f"Cannot find {train_path}. "
                f"Run data_preparation.py first. (Error: {prep_err})"
            ) from prep_err

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    log.info("Pricing — Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # Feature engineering
    train_df = engineer_pricing_features(train_df)
    test_df  = engineer_pricing_features(test_df)

    # Encode categoricals
    train_df, test_df, encoders = encode_categoricals(train_df, test_df, CAT_COLS_PRICING)

    # Auto-select features
    feature_cols = auto_select_features(train_df, PRICING_LABEL, CAT_COLS_PRICING)
    log.info("Pricing — %d features: %s", len(feature_cols), feature_cols)

    X_train, y_train = prepare_Xy(train_df, feature_cols, PRICING_LABEL)
    X_test, y_test   = prepare_Xy(test_df, feature_cols, PRICING_LABEL)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    imbalance = round(n_neg / max(n_pos, 1), 1)
    log.info("Pricing — Positive: %d (%.1f%%), Imbalance: 1:%.0f", n_pos, n_pos/len(y_train)*100, imbalance)

    # Model selection: XGBoost > sklearn GBM
    model_type = "xgboost"
    try:
        from xgboost import XGBClassifier  # type: ignore
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=imbalance,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        log.info("Pricing — Using XGBoost")
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        model_type = "sklearn_gbm"
        log.info("Pricing — XGBoost unavailable, using sklearn GBM")

    # Train (with sample_weight for sklearn GBM which lacks scale_pos_weight)
    log.info("Pricing — Training...")
    if model_type == "sklearn_gbm":
        weights = np.where(y_train == 1, imbalance, 1.0)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - t0
    log.info("Pricing — Done in %.1f s", train_time)

    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    # Use a lower threshold for imbalanced data
    threshold = 0.3
    y_pred = (y_proba >= threshold).astype(int)

    metrics = print_metrics("Pricing Model", y_test, y_pred, y_proba)
    metrics["train_time_s"] = round(train_time, 1)
    metrics["model_type"] = model_type
    metrics["features"] = feature_cols
    metrics["threshold"] = threshold

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print("  Top Feature Importances:")
        for fname, fimp in importance[:10]:
            print(f"    {fname:35s}  {fimp:.4f}")
        metrics["feature_importance"] = {f: round(float(v), 4) for f, v in importance}

    # Save
    model_path = os.path.join(MODEL_DIR, "pricing_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": feature_cols,
                      "model_type": model_type, "threshold": threshold}, f)
    log.info("Pricing — Saved to %s", model_path)

    return metrics


# ─────────────────────────────────────────────────────────
# Model 2: Recommendation (Engagement Prediction)
# ─────────────────────────────────────────────────────────

def train_recommendation_model():
    t0 = time.time()

    train_df = pd.read_parquet(os.path.join(DATA_DIR, "recommendation_features_train.parquet"))
    test_df  = pd.read_parquet(os.path.join(DATA_DIR, "recommendation_features_test.parquet"))

    log.info("Recom — Train: %d, Test: %d", len(train_df), len(test_df))

    # Feature engineering
    train_df = engineer_recom_features(train_df)
    test_df  = engineer_recom_features(test_df)

    # Encode categoricals
    train_df, test_df, encoders = encode_categoricals(train_df, test_df, CAT_COLS_RECOM)

    feature_cols = auto_select_features(train_df, RECOM_LABEL, CAT_COLS_RECOM)
    log.info("Recom — %d features: %s", len(feature_cols), feature_cols)

    X_train, y_train = prepare_Xy(train_df, feature_cols, RECOM_LABEL)
    X_test, y_test   = prepare_Xy(test_df, feature_cols, RECOM_LABEL)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log.info("Recom — Positive rate: %.1f%%", y_train.mean() * 100)

    model = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )

    log.info("Recom — Training...")
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    log.info("Recom — Done in %.1f s", train_time)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = print_metrics("Recommendation Model", y_test, y_pred, y_proba)
    metrics["train_time_s"] = round(train_time, 1)
    metrics["model_type"] = "sgd_logistic"
    metrics["features"] = feature_cols

    # Coefficients
    coef_list = sorted(
        zip(feature_cols, model.coef_[0]),
        key=lambda x: abs(x[1]), reverse=True
    )
    print("  Feature Coefficients:")
    for fname, coef in coef_list:
        print(f"    {fname:35s}  {coef:+.4f}")
    metrics["coefficients"] = {f: round(float(v), 4) for f, v in coef_list}

    model_path = os.path.join(MODEL_DIR, "recommendation_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": feature_cols,
                      "model_type": "sgd_logistic"}, f)
    log.info("Recom — Saved to %s", model_path)

    return metrics


# ─────────────────────────────────────────────────────────
# Model 3: Online SGD (Incremental for streaming)
# ─────────────────────────────────────────────────────────

def train_online_pricing_model():
    t0 = time.time()

    train_df = pd.read_parquet(os.path.join(DATA_DIR, "pricing_features_train.parquet"))
    train_df = engineer_pricing_features(train_df)
    train_df, _, encoders = encode_categoricals(train_df, train_df.copy(), CAT_COLS_PRICING)

    feature_cols = auto_select_features(train_df, PRICING_LABEL, CAT_COLS_PRICING)
    X_train, y_train = prepare_Xy(train_df, feature_cols, PRICING_LABEL)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Apply Standard Scaler so the SGD gradients don't explode
    X_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Compute balanced weights upfront (required for partial_fit)
    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = {0: cw[0], 1: cw[1]}
    log.info("Online — Class weights: 0=%.2f, 1=%.2f", cw[0], cw[1])

    model = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        max_iter=1,
        tol=None,
        warm_start=True,
        random_state=42,
    )

    batch_size = 1000
    n_batches = len(X_train) // batch_size
    log.info("Online — Training in %d batches...", n_batches)

    for i in range(n_batches):
        s = i * batch_size
        e = s + batch_size
        X_b = X_train.iloc[s:e]
        y_b = y_train[s:e]
        # Apply pre-computed class weights as sample weights
        sw = np.array([weight_map[y] for y in y_b])
        model.partial_fit(X_b, y_b, classes=classes, sample_weight=sw)

    train_time = time.time() - t0
    log.info("Online — Done in %.1f s", train_time)

    model_path = os.path.join(MODEL_DIR, "pricing_online_sgd.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": feature_cols,
                      "model_type": "sgd_online", "class_weights": weight_map}, f)
    log.info("Online — Saved to %s", model_path)

    return {"train_time_s": round(train_time, 1), "batches": n_batches}


# ─────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    print()
    print("=" * 60)
    print("  APEX — Model Training Pipeline v2")
    print("  Pricing (GBM) + Recommendation (SGD) + Online (SGD)")
    print("=" * 60)

    print("\n" + "─" * 60)
    print("  Model 1: Pricing Conversion Prediction")
    print("─" * 60)
    pricing_metrics = train_pricing_model()

    print("\n" + "─" * 60)
    print("  Model 2: Recommendation Engagement")
    print("─" * 60)
    recom_metrics = train_recommendation_model()

    print("\n" + "─" * 60)
    print("  Model 3: Online Incremental Pricing")
    print("─" * 60)
    online_metrics = train_online_pricing_model()

    # Save report
    report = {
        "pricing": pricing_metrics,
        "recommendation": recom_metrics,
        "online_pricing": online_metrics,
        "total_time_s": round(time.time() - t_total, 1),
    }
    report_path = os.path.join(MODEL_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t_total
    print()
    print("=" * 60)
    print("  ✅ All 3 Models Trained Successfully!")
    print(f"  ⏱  Total time: {elapsed:.1f} s")
    print(f"  📁 Models: {MODEL_DIR}")
    print(f"  📄 Report: {report_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
