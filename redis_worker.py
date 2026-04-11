"""
============================================================
  APEX — Redis Streams Consumer Worker
  Real-time feature updater via Consumer Group
============================================================

  Run:   py redis_worker.py
  Deps:  pip install redis

  Reads from stream 'ecommerce:events', computes rolling
  demand velocity / engagement / intent, writes to Redis Hashes.
  Falls back gracefully if Redis is unavailable.
============================================================
"""

import time
import json
import logging
import sys
from collections import defaultdict
from typing import Dict, List

# ── Configure logging FIRST (before any logging calls) ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STREAM WORKER] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("redis_worker")

# ── Redis Connection (with clean error handling) ──
try:
    import redis
except ImportError:
    log.error("Redis package not installed. Install using: pip install redis")
    sys.exit(1)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

STREAM_KEY = "ecommerce:events"
GROUP_NAME = "apex_engine_group"
CONSUMER_NAME = "worker_1"

# Stream trimming: keep last N entries to prevent unbounded growth
STREAM_MAX_LEN = 10_000


def _connect_redis() -> redis.Redis:
    """Create and validate a Redis connection."""
    client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )
    client.ping()
    return client


def _ensure_consumer_group(client: redis.Redis):
    """Create consumer group idempotently."""
    try:
        client.xgroup_create(STREAM_KEY, GROUP_NAME, id="$", mkstream=True)
        log.info("Created consumer group '%s' on stream '%s'", GROUP_NAME, STREAM_KEY)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            log.info("Consumer group '%s' already exists — resuming.", GROUP_NAME)
        else:
            raise


# In-memory feature caches (mirrored to Redis Hashes)
engagement_scores = defaultdict(float)
demand_velocity = defaultdict(int)
category_affinity: Dict[str, List[str]] = defaultdict(list)

# Weight map matching server.py EVENT_WEIGHTS
DEMAND_WEIGHTS = {
    "view": 1,
    "page_view": 1,
    "product_view": 2,
    "click": 2,
    "search": 1,
    "add_to_wishlist": 3,
    "add_to_cart": 5,
    "checkout_start": 7,
    "purchase": 10,
}

MAX_CATEGORY_HISTORY = 10  # rolling window for affinity


def update_real_time_features(client: redis.Redis, event_data: dict):
    """
    Compute and persist all 4 real-time features from one event:
      1. Product Demand Velocity
      2. Session Engagement Score
      3. Category Affinity
      4. Purchase Intent Score
    """
    user_id = event_data.get("user_id")
    sku_id = event_data.get("sku_id")
    event_type = event_data.get("event_type", "click")
    category = event_data.get("category", "")

    # Guard against malformed payloads
    if not user_id or not sku_id:
        log.warning("Skipping event with missing user_id or sku_id: %s", event_data)
        return

    weight = DEMAND_WEIGHTS.get(event_type, 1)

    # 1. Product Demand Velocity
    demand_velocity[sku_id] += weight
    client.hincrby(f"product:{sku_id}:features", "demand_velocity", weight)
    client.hset(f"product:{sku_id}:features", "last_event_ts", str(time.time()))

    # 2. Session Engagement Score
    engagement_scores[user_id] += weight * 0.5
    client.hincrbyfloat(f"user:{user_id}:features", "engagement_score", weight * 0.5)

    # 3. Category Affinity (rolling window of recent categories)
    if category:
        history = category_affinity[user_id]
        history.append(category)
        if len(history) > MAX_CATEGORY_HISTORY:
            history.pop(0)
        # Store top category in Redis
        from collections import Counter
        top_cat = Counter(history).most_common(1)[0][0]
        client.hset(f"user:{user_id}:features", "top_category", top_cat)
        client.hset(f"user:{user_id}:features", "category_depth", len(set(history)))

    # 4. Purchase Intent Score (normalized engagement → 0-1)
    current_intent = min(engagement_scores[user_id] / 50.0, 1.0)
    client.hset(f"user:{user_id}:features", "intent_score", round(current_intent, 4))

    log.info(
        "✓ %s | sku=%s user=%s | demand=%d engagement=%.1f intent=%.2f",
        event_type, sku_id, user_id,
        demand_velocity[sku_id], engagement_scores[user_id], current_intent,
    )


def consume_stream():
    """Main consumer loop with reconnection logic."""
    client = None
    consecutive_errors = 0

    while True:
        # ── Establish / re-establish connection ──
        if client is None:
            try:
                client = _connect_redis()
                _ensure_consumer_group(client)
                consecutive_errors = 0
                log.info("Connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
                log.info("Listening on stream: %s", STREAM_KEY)
            except Exception as e:
                consecutive_errors += 1
                backoff = min(30, 2 ** consecutive_errors)
                log.error("Redis connection failed (%s). Retrying in %ds...", e, backoff)
                time.sleep(backoff)
                continue

        # ── Read & process events ──
        try:
            messages = client.xreadgroup(
                GROUP_NAME, CONSUMER_NAME,
                {STREAM_KEY: ">"},
                count=10,
                block=2000,
            )

            if not messages:
                continue  # no new messages, loop back to block again

            for _stream_name, events in messages:
                for event_id, event_dict in events:
                    try:
                        payload = json.loads(event_dict.get("event", "{}"))
                        update_real_time_features(client, payload)
                        client.xack(STREAM_KEY, GROUP_NAME, event_id)
                    except json.JSONDecodeError:
                        log.error("Malformed JSON in event %s — skipping", event_id)
                        client.xack(STREAM_KEY, GROUP_NAME, event_id)
                    except Exception as e:
                        log.error("Error processing event %s: %s", event_id, e)
                        # Still acknowledge to prevent infinite re-delivery of bad events
                        try:
                            client.xack(STREAM_KEY, GROUP_NAME, event_id)
                        except Exception:
                            pass

            # Periodic trim to prevent unbounded stream growth
            try:
                client.xtrim(STREAM_KEY, maxlen=STREAM_MAX_LEN, approximate=True)
            except Exception:
                pass

        except (redis.ConnectionError, redis.TimeoutError) as e:
            log.error("Lost Redis connection: %s — reconnecting...", e)
            client = None  # force reconnect on next loop
            time.sleep(2)
        except Exception as e:
            log.error("Unexpected error: %s", e)
            time.sleep(1)


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("  APEX Redis Streams Worker")
    log.info("=" * 50)
    consume_stream()
