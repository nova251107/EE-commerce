"""Surgical fix for ml_engine.py — replaces lines 793-958 to fix the uninitialized variable bug."""
with open('ml_engine.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original lines: {len(lines)}")

NEW_SECTION = """\
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
"""

# Replace lines 793-958 (0-indexed 792-957)
new_lines = lines[:792] + [NEW_SECTION] + lines[958:]
with open('ml_engine.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

import ast
with open('ml_engine.py', 'r', encoding='utf-8') as f:
    src = f.read()
try:
    ast.parse(src)
    print(f"ml_engine.py OK — {len(new_lines)} lines")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
