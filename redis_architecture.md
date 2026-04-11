# Redis Streams Real-Time Architecture Upgrade

This document outlines the lightweight, hackathon-friendly architecture upgrade to introduce a dedicated **Redis Streams** event-processing layer to the APEX engine. This transforms the current in-memory simulation into a true event-driven, microservice-like real-time streaming prototype appropriate for final judging.

## 1. Updated Architecture Flow
We are moving from a synchronous monolithic flow to an asynchronous producer-consumer model for event ingest and feature generation.

**Before:**
`Frontend Event -> API FastAPI -> In-Memory State Updates -> ML Pipeline execution (sync) -> Response`

**After (Redis Streams Upgrade):**
1. **Frontend Event** -> `API FastAPI POST /event`
2. **API (Producer)** extracts immediate rules, responds instantly, and **PUBLISHES** event to `Redis Stream` (`XADD ecommerce:events`).
3. **Redis Stream** acts as point-in-time messaging queue.
4. **Python Background Worker (Consumer)** continuously reads (`XREADGROUP`) from stream.
5. **Worker** computes rolling features (Demand Velocity, Engagement Score, Intent Score) and writes them to Redis caching layer (or shared memory).
6. Future API calls pull warmed, pre-computed real-time features.

## 2. Component Responsibilities

* **FastAPI Backend (Producer):** 
  * Ingests REST requests.
  * Writes user events into Redis Streams (`ecommerce:events`).
  * Immediately retrieves the most recent ML features and dynamic prices computed by the worker to send back to the frontend.
* **Redis Instance:**
  * **Redis Streams:** Distributed append-only log for event streaming.
  * **Redis Hashes / Dicts:** Low-latency caching layer for computed features.
* **Stream Worker (`redis_worker.py`):**
  * Subscribes to the Redis Stream (Consumer Group).
  * Executes the feature extraction logic incrementally.
  * Updates the state arrays (Demand Velocity, Category Affinity).
  * Periodically trims the stream.

## 3. API Flow (Minimal Extensibility)

* **`POST /event` Endpoint:**
  1. Validates the event quickly.
  2. Constructs JSON payload of the event.
  3. `redis_client.xadd("ecommerce:events", {"event": json.dumps(payload)})`
  4. Fetches instant dynamic price from the fastest cache/memory (instead of blocking on the full ML pipeline computation).
  5. Returns `status: published` with instant pricing.

## 4. Worker Flow

The Python worker runs as an independent detached process.
1. Connects to Redis and creates Consumer Group `apex_engine`.
2. `XREADGROUP` blocks and waits for new event IDs (e.g., `>`).
3. Parses the event payload.
4. **State updates:** Updates intermediate historical states.
5. `XACK` the message to confirm successful processing.

## 5. Data Flow (Ingestion to Feature Update)

1. **Ingest:** `<User_123, click, SKU_99, category: shoes, timestamp>`
2. **Stream Append:** Redis Stream assigns ID `16712399-0`.
3. **Worker Pickup:** Worker pulls event in a batch of 10.
4. **Feature Logic Updates:**
   * **Session Engagement Score:** Increments `user_123_engagement` based on event weight (click = +2).
   * **Category Affinity:** Pushes `shoes` to user's recent categories list (max 5 items).
   * **Product Demand Velocity:** Increments `product_99_demand_count` and records timestamp for exponential decay.
   * **Purchase Intent Score:** Combines Engagement and Category rules to yield immediate intent > 0.8.
5. **Cache Refresh:** Worker executes `HSET user:123 intent_score 0.8 engagement 15.0`.
6. **Closing:** Stream message is acknowledged and removed from Pending entries.

## 6. Fallback Plan (If Redis Fails)

To ensure smooth judge presentations, we build a seamless fallback loop natively.
- **Circuit Breaker:** Wrap the `xadd` call in a `try...except` block.
- **Failover:** If Redis connection is refused (ConnectionError), the system gracefully catches it and logs a warning: `"Redis unreachable. Falling back to in-memory streaming."`
- **Degraded Mode:** Continues routing the event directly to the in-memory `update_engagement()` and `update_demand()` functions as it currently operates. No functionality drops for the end user.

## 7. Demo Story for Judges

**"Scaling APEX for Real-Time"**
1. **The Problem:** "Initially, our ML pipeline processed events synchronously in-memory. This worked for a prototype but wouldn't scale under massive Black Friday traffic."
2. **The Solution:** "We upgraded our architecture by introducing Redis Streams. Our FastAPI server now instantly acknowledges events, pushing them to an append-only log."
3. **Showcase:** "Here is our independent Python worker processing the stream live. As I click on the frontend, watch the worker console ingest the event, compute the *Purchase Intent* and *Demand Velocity*, and update the caches asynchronously in real-time."
4. **Resilience:** "And if our cache layer drops? Our API catches it instantly and falls back to our local memory state—ensuring high availability and 0% downtime during peak demand."
