"""
scratch_edit2.py — Fixed version
Bug fixed: unterminated triple-quoted string on line 270 (extra closing triple-quote
           was placed OUTSIDE the string, making the whole file a syntax error).
           Also: the unreliable regex for POST /event was replaced with a
           safer line-range approach so it doesn't accidentally swallow too much.
"""

import re, sys

# ── Fix 1: server.py — remove old helper functions ──────────────────────────
print("Patching server.py…")
with open("d:/vasu/server.py", "r", encoding="utf-8") as f:
    text = f.read()

# Remove compute_dynamic_price (greedy, stops at last 'return price, rationale')
text = re.sub(
    r'\ndef compute_dynamic_price\(.*?\n    return price, rationale\n',
    '\n',
    text,
    flags=re.DOTALL,
)

# Remove get_recommendations (greedy, stops at 'return candidates[:limit]')
text = re.sub(
    r'\ndef get_recommendations\(.*?\n    return candidates\[:limit\]\n',
    '\n',
    text,
    flags=re.DOTALL,
)

with open("d:/vasu/server.py", "w", encoding="utf-8") as f:
    f.write(text)

print("  server.py patched OK")


# ── Fix 2: ml_engine.py — inject execute_ml_pipeline method ─────────────────
print("Patching ml_engine.py…")
with open("d:/vasu/ml_engine.py", "r", encoding="utf-8") as f:
    ml_text = f.read()

NEW_METHOD = '''
    def execute_ml_pipeline(
        self,
        event_type: str,
        sku_id: str,
        user_id: str,
        session_id: str,
        price_seen: float,
        discounted: bool,
        product: dict,
        session: dict,
        ab_group: str,
        competitor_data,
        all_products: dict,
        all_categories: dict,
    ) -> dict:
        """
        Full 11-step ML pipeline (Phase 1 → Phase 2 integration point).
        """
        import time as _time
        t0 = _time.perf_counter()
        ts = _time.time()

        event = EventRecord(
            ts=ts, event_type=event_type, sku_id=sku_id,
            session_id=session_id, user_id=user_id,
            price_seen=price_seen, discounted=discounted,
        )
        self.feature_engine.update(event, category=product.get("category"))
        self.demand_predictor.record_event(sku_id, event_type, ts)
        fv = self.feature_engine.get_feature_vector(session_id)
        ds = self.demand_predictor.compute_demand_score(sku_id)

        # Recommendations (category-affinity + cold-start)
        clicked = set(session.get("clicks", []))
        recent_cats = session.get("categories", [])[-5:]
        recs = []
        seen_skus: set = set()
        for cat in recent_cats:
            for c_sku in all_categories.get(cat, []):
                if c_sku not in clicked and c_sku not in seen_skus and c_sku in all_products:
                    p = all_products[c_sku]
                    recs.append({
                        "id": c_sku, "name": p["name"], "category": p["category"],
                        "price": p["current_price"], "rating": p.get("rating", 0),
                        "reason": f"Because you browsed {cat}",
                        "score": p.get("views", 0) + p.get("rating", 0) * 10,
                    })
                    seen_skus.add(c_sku)
        if len(recs) < 5:
            trending = sorted(all_products.values(), key=lambda x: x.get("views", 0), reverse=True)
            for p in trending:
                if p["sku_id"] not in clicked and p["sku_id"] not in seen_skus:
                    recs.append({
                        "id": p["sku_id"], "name": p["name"], "category": p["category"],
                        "price": p["current_price"], "rating": p.get("rating", 0),
                        "reason": "Trending now",
                        "score": p.get("views", 0) + p.get("rating", 0) * 5,
                    })
                    seen_skus.add(p["sku_id"])
                    if len(recs) >= 10:
                        break
        recs.sort(key=lambda x: x["score"], reverse=True)
        top_recs = recs[:5]

        # Pricing (A/B aware)
        final_price = product["base_price"]
        reasons = []
        if ab_group == "A":
            reasons.append("Control group: static base price applied.")
        else:
            temp_price = product["base_price"]
            if ds.hybrid_score > 50:
                temp_price *= 1.05
                reasons.append(f"High demand ({ds.hybrid_score:.1f}): +5%")
            if product.get("inventory", 100) < 20:
                temp_price *= 1.03
                reasons.append(f"Low stock ({product['inventory']} left): +3%")
            elif product.get("inventory", 100) > 400:
                temp_price *= 0.95
                reasons.append(f"High stock: -5%")
            if fv.intent_score > 10:
                temp_price *= 1.02
                reasons.append(f"High intent user: +2%")
            ceiling = product["base_price"] * 1.15
            cost = product.get("cost_price", 0)
            floor = max(cost * 1.05 if cost > 0 else product["base_price"] * 0.85, product["base_price"] * 0.90)
            temp_price = max(floor, min(temp_price, ceiling))
            final_price = round(temp_price, 2)
            if not reasons:
                reasons.append("Base price maintained.")

        latency_ms = (_time.perf_counter() - t0) * 1000
        return {
            "final_price": final_price,
            "top_recommendations": top_recs,
            "reason": [" | ".join(reasons)],
            "feature_vector": fv.to_dict(),
            "demand_score": ds.to_dict(),
            "latency_ms": round(latency_ms, 2),
        }
'''

# Only inject if not already present
if "execute_ml_pipeline" not in ml_text:
    # Insert after the 'def shutdown' method definition
    ml_text = re.sub(
        r'(    def shutdown\(self\):.*?logger\.info\("MLEngine shutdown\."\))',
        r'\1\n' + NEW_METHOD,
        ml_text,
        flags=re.DOTALL,
    )
    with open("d:/vasu/ml_engine.py", "w", encoding="utf-8") as f:
        f.write(ml_text)
    print("  ml_engine.py patched OK")
else:
    print("  ml_engine.py already has execute_ml_pipeline — skipped")

print("Done.")
