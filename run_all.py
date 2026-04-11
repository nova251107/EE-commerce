"""
============================================================
  APEX — Full Pipeline Runner
  Runs all stages in order: Clean → Prepare → Train → Serve
============================================================

  Usage:
    py run_all.py              # run everything (skip clean/prep if data exists)
    py run_all.py --clean      # force re-run data cleaning
    py run_all.py --prepare    # force re-run feature preparation
    py run_all.py --train      # force re-train all models
    py run_all.py --all        # force re-run every stage
    py run_all.py --server     # just start the FastAPI server
============================================================
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR  = os.path.join(BASE_DIR, "cleaned_data")
PREPARED_DIR = os.path.join(BASE_DIR, "prepared_data")
MODELS_DIR   = os.path.join(BASE_DIR, "trained_models")


def _header(title: str):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def _cleaned_data_exists() -> bool:
    return os.path.exists(os.path.join(CLEANED_DIR, "clickstream_clean.parquet"))


def _prepared_data_exists() -> bool:
    return all(
        os.path.exists(os.path.join(PREPARED_DIR, f))
        for f in [
            "pricing_features_train.parquet",
            "pricing_features_test.parquet",
            "recommendation_features_train.parquet",
            "recommendation_features_test.parquet",
        ]
    )


def _models_exist() -> bool:
    return all(
        os.path.exists(os.path.join(MODELS_DIR, f))
        for f in [
            "pricing_model.pkl",
            "recommendation_model.pkl",
        ]
    )


# ── Stage 1: Data Cleaning ──────────────────────────────

def run_clean(force: bool = False):
    if not force and _cleaned_data_exists():
        log.info("Stage 1: cleaned_data/ already exists \u2014 skipping (use --clean to force)")
        return

    _header("Stage 1: Data Cleaning")
    t0 = time.time()
    try:
        from data_cleaning import run_cleaning
        run_cleaning()
        log.info("Stage 1 complete in %.1f s", time.time() - t0)
    except Exception as e:
        log.error("Stage 1 FAILED: %s", e)
        log.warning("Continuing without clean data \u2014 preparation will use raw files")


# ── Stage 2: Feature Preparation ────────────────────────

def run_prepare(force: bool = False):
    if not force and _prepared_data_exists():
        log.info("Stage 2: prepared_data/ already exists \u2014 skipping (use --prepare to force)")
        return

    _header("Stage 2: Feature Preparation")
    t0 = time.time()
    try:
        from data_preparation import PreparedDataLoader
        PreparedDataLoader.load(force_rebuild=force)
        log.info("Stage 2 complete in %.1f s", time.time() - t0)
    except Exception as e:
        log.error("Stage 2 FAILED: %s", e)
        raise


# ── Stage 3: Model Training ──────────────────────────────

def run_train(force: bool = False):
    if not force and _models_exist():
        log.info("Stage 3: trained_models/ already exists \u2014 skipping (use --train to force)")
        return

    _header("Stage 3: Model Training")
    t0 = time.time()
    try:
        # Train via model_trainer.py (Phase 3 logistic models)
        from model_trainer import model_registry
        metrics = model_registry.startup(force_retrain=force)
        log.info("Phase 3 models: pricing_ready=%s  recom_ready=%s",
                 metrics.get("pricing_ready"), metrics.get("recom_ready"))

        # Also train heavier GBM models via train_models.py (if data ready)
        if _prepared_data_exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("train_models",
                    os.path.join(BASE_DIR, "train_models.py"))
            tm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tm)
            tm.main()

        log.info("Stage 3 complete in %.1f s", time.time() - t0)
    except Exception as e:
        log.error("Stage 3 FAILED: %s", e)
        log.warning("Server will start in heuristic-only mode")


# ── Stage 4: FastAPI Server ──────────────────────────────

def run_server():
    _header("Stage 4: Starting APEX FastAPI Server")
    try:
        import uvicorn
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
        )
    except ImportError:
        log.error("uvicorn not installed: pip install uvicorn")
        sys.exit(1)


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="APEX Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clean",   action="store_true", help="Force re-run data cleaning")
    parser.add_argument("--prepare", action="store_true", help="Force re-run feature preparation")
    parser.add_argument("--train",   action="store_true", help="Force re-train all models")
    parser.add_argument("--all",     action="store_true", help="Force all stages (clean + prepare + train)")
    parser.add_argument("--server",  action="store_true", help="Only start the server (skip pipeline)")
    args = parser.parse_args()

    if args.server:
        run_server()
        return

    force_all = args.all

    t_start = time.time()
    print()
    print("=" * 62)
    print("  APEX \u2014 Full Pipeline Runner")
    print("  clean \u2192 prepare \u2192 train \u2192 serve")
    print("=" * 62)

    run_clean(force=force_all or args.clean)
    run_prepare(force=force_all or args.prepare)
    run_train(force=force_all or args.train)

    print()
    print("=" * 62)
    print(f"  \u2705 Pipeline ready in {time.time() - t_start:.1f}s")
    print("  Starting server at http://localhost:8000")
    print("=" * 62)
    run_server()


if __name__ == "__main__":
    main()
