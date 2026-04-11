"""
APEX Phase 4+5 Demo — offline prototype test.
Runs without the server: directly imports and exercises the full stack.
"""
import time, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("\n" + "="*62)
print("  APEX Phase 4+5 Demo — Real-Time Decision + A/B Testing")
print("="*62)

# ── 1. Phase 5: A/B System ──────────────────────────────────
print("\n[1] Phase 5 — A/B Testing System")
from ab_testing import ABTestingSystem, UserSplitter, MetricsTracker, DecisionRuleEngine

ab = ABTestingSystem()

# Step 19: user split
for uid in [f"U{i:04d}" for i in range(50)]:
    g = ab.assign(uid, f"S{uid}")
groups = [ab.assign(f"U{i:04d}", f"SU{i:04d}") for i in range(200)]
a_count = groups.count("A")
b_count = groups.count("B")
print(f"   Split (200 users): A={a_count}  B={b_count}  ratio={b_count/200:.1%}")
assert 40 <= b_count <= 160, "Split is too imbalanced"

# Step 20: simulate events
import random, math
rng = random.Random(42)

for i in range(200):
    grp   = groups[i]
    uid   = f"U{i:04d}"
    event = rng.choices(["view","click","purchase"], weights=[60,30,10])[0]
    price = rng.uniform(20, 200)
    rev   = price if event == "purchase" else 0
    ab.record(grp, uid, event, price, rev, latency_ms=rng.uniform(10,80))

snap = ab.metrics
print(f"   Group A: sessions={snap['A']['sessions']}  conv={snap['A']['conversion_rate']:.2%}  rev/user=${snap['A']['revenue_per_user']:.2f}")
print(f"   Group B: sessions={snap['B']['sessions']}  conv={snap['B']['conversion_rate']:.2%}  rev/user=${snap['B']['revenue_per_user']:.2f}")

# Step 21: decision rule
dec = ab.evaluate()
print(f"   Decision: action='{dec.action}'  winner={dec.winner}  confidence={dec.confidence:.3f}")
print(f"   Reason: {dec.reason}")

# ── 2. Phase 4: Decision Engine ─────────────────────────────
print("\n[2] Phase 4 — Real-Time Decision Engine")
from decision_engine import DecisionEngine, ColdStartHandler, DynamicAdjustmentCalculator

# Step 16: Dynamic Adjustment
print("\n   Step 16 — Dynamic Adjustment Logic:")
calc = DynamicAdjustmentCalculator()
scenarios = [
    ("High demand + scarce", 80, 10, 15.0, None),
    ("Low demand + surplus", 15, 600, 5.0, None),
    ("Normal + competitor cheaper", 50, 100, 8.0, 90.0),
    ("Surge + critical stock", 90, 5, 20.0, None),
]
for label, demand, inv, intent, comp in scenarios:
    final, adj, tier, reasons, dlevel, inv_status, gap = calc.calculate(
        base_price=100.0, demand_score=demand, inventory=inv,
        user_intent=intent, competitor_price=comp, ab_group="B"
    )
    print(f"   [{label}] base=$100  final=${final}  adj={adj:+.1f}%  tier={tier}")
    for r in reasons[:2]:
        print(f"     → {r}")

# Step 18: Cold Start
print("\n   Step 18 — Cold Start Handler:")
from decision_engine import ColdStartHandler
cs = ColdStartHandler()
# Build fake product catalog
fake_products = {
    f"SKU{i:03d}": {
        "sku_id": f"SKU{i:03d}",
        "name": f"Product {i}",
        "category": rng.choice(["Electronics","Fashion","Books","Sports","Home & Kitchen"]),
        "current_price": round(rng.uniform(20, 300), 2),
        "rating": round(rng.uniform(3, 5), 1),
        "views": rng.randint(0, 10000),
    }
    for i in range(100)
}
demand_scores = {sku: rng.uniform(0, 100) for sku in fake_products}

for device in ["mobile", "desktop", "tablet"]:
    recs = cs.generate(fake_products, demand_scores, device_type=device, n=3)
    print(f"   {device.upper()} cold-start:")
    for r in recs:
        print(f"     #{r['ml_score']:.3f} {r['name']} [{r['category']}] — {r['reason'][:60]}")

# Step 17: Warm Recommendation
print("\n   Step 17 — Warm Recommendation (session-aware):")
de = DecisionEngine()
session = {
    "clicks": [f"SKU{i:03d}" for i in range(5)],
    "categories": ["Electronics", "Electronics", "Fashion", "Books", "Electronics"],
    "ab_group": "B",
}
all_cats = {}
for sku, p in fake_products.items():
    cat = p["category"]
    if cat not in all_cats:
        all_cats[cat] = []
    all_cats[cat].append(sku)

recs17 = de.step17_recommend(
    session=session, all_products=fake_products,
    all_categories=all_cats, demand_scores=demand_scores, n=5
)
print(f"   Top 5 recommendations:")
for r in recs17:
    print(f"     #{r.rank} {r.tag} {r.name} [{r.category}] ml_score={r.ml_score:.3f}")

# Step 15: Live Pricing
print("\n   Step 15 — Live Pricing Decision:")
from dataclasses import dataclass
@dataclass
class MockDemand:
    hybrid_score: float = 72.0
@dataclass
class MockFV:
    intent_score: float = 18.0

product = {
    "sku_id": "SKU001", "base_price": 199.99,
    "inventory": 12, "cost_price": 80.0,
    "min_price": 150.0, "max_price": 250.0,
}
pricing_decision = de.step15_pricing(product, MockDemand(72), MockFV(18), ab_group="B",
                        competitor_data={"competitor_price": 195.0})
print(f"   Price: ${pricing_decision.base_price} → ${pricing_decision.final_price}  ({pricing_decision.adjustment_pct:+.2f}%)")
print(f"   Tier: {pricing_decision.tier}  Demand: {pricing_decision.demand_level}  Inventory: {pricing_decision.inventory_status}")
for r in pricing_decision.reasons:
    print(f"   → {r}")

# ── 3. Full Bundle ───────────────────────────────────────────
print("\n[3] Full Decision Bundle (pricing + recs combined):")
bundle = de.decide(
    product=product, session=session,
    demand_score_obj=MockDemand(72), feature_vec_obj=MockFV(18),
    all_products=fake_products, all_categories=all_cats,
    demand_scores=demand_scores, ab_group="B",
    device_type="mobile", user_id="U0001", session_id="S0001",
)
print(f"   Final price: ${bundle.pricing.final_price}  ({bundle.pricing.tier} tier)")
print(f"   Cold start:  {bundle.cold_start}")
print(f"   Recommendations: {len(bundle.recommendations)} items")
for r in bundle.recommendations:
    print(f"     #{r.rank} {r.tag}  {r.name}  ml={r.ml_score:.3f}")
print(f"   Total latency: {bundle.total_latency_ms:.2f}ms")

# ── 4. Phase 3 A/B Z-test demo ──────────────────────────────
print("\n[4] Statistical Test (Z-test):")
rule = DecisionRuleEngine()
z, p = rule._z_test_proportion(0.05, 1000, 0.08, 1000)
print(f"   A=5% conv (n=1000) vs B=8% conv (n=1000)")
print(f"   z={z:.3f}  p={p:.6f}  significant={'YES' if p<0.05 else 'NO'}")

z2, p2 = rule._z_test_proportion(0.05, 20, 0.08, 20)
print(f"   A=5% conv (n=20)   vs B=8% conv (n=20)")
print(f"   z={z2:.3f}  p={p2:.6f}  significant={'YES' if p2<0.05 else 'NO'} (small sample)")

print("\n" + "="*62)
print("  ✅ APEX Phase 4+5 Demo complete — all systems operational")
print("="*62 + "\n")
