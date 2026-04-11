# Fairness & Explainability Layer Design

This document details the Ethical AI architecture and Explainability Module designed for the APEX dynamic pricing engine. This layer mitigates algorithmic bias and ensures complete transparency for both consumers and business auditors (judges).

## 1. Allowed vs. Excluded Features Table

To guarantee fairness, our ML feature extraction pipeline strictly enforces an "Ethical Firewall" that strips highly sensitive or discriminatory variables before pricing inference occurs.

| Feature Category | Features | Status | Rationale for Pricing Decisions |
| :--- | :--- | :--- | :--- |
| **Market Constraints** | Competitor Price, Restock ETA | ✅ **ALLOWED** | Dictates external market realities and supply chain logic. |
| **Supply Constraints** | Inventory Levels | ✅ **ALLOWED** | Objective metric of scarcity and business risk. |
| **Behavioral Intent** | Session Clicks, Cart Adds, Velocity | ✅ **ALLOWED** | Shows product popularity; treats all users making the same clicks equally. |
| **Demographics** | Age, Gender, Ethnicity | ❌ **EXCLUDED** | High risk of protected-class discrimination. |
| **Socioeconomic** | Zip code, Income Bracket | ❌ **EXCLUDED** | High risk of wealth-based price gouging. |
| **Hardware / OS** | Device Type (e.g., Mac vs PC) | ❌ **EXCLUDED** | Known unethical pattern (charging iOS users more). |

## 2. Fairness Policy Wording

**APEX Zero-Discrimination Pricing Policy:**
> *"APEX dynamically adjusts pricing based exclusively on macroeconomic supply constraints, market benchmarks, and aggregated behavioral demand. We employ a strict Zero-Demographic firewall. Who you are, what device you use, and where you live inherently cannot influence your price. Our algorithm prices the market, never the individual."*

## 3. Fairness Audit Checklist
During development and CI/CD pipelines, the following criteria must be validated:
*   [ ] **Feature Pruning:** Confirm `age_group`, `gender`, and `device_type` columns are natively dropped in the `data_preparation.py` pipeline.
*   [ ] **Parity Testing:** Run simulated A/A pricing tests segmented by device/age. Verify that the median predicted price difference across these segments is $0.00$.
*   [ ] **Surge Cap Integrity:** Verify that the Maximum Surge Guardrail (+15%) cannot be bypassed under any algorithmic circumstance, preventing predatory gouging.
*   [ ] **Explainability Trace:** Ensure every dynamic price modification generates a traceable log explaining perfectly *why* it changed based on valid allowed features.

## 4. User-Facing Explanation Examples

Instead of a black box changing prices silently, the UI proactively displays tooltips explaining the economic reasoning.

**Scenario A (High Demand, Scarcity, Competitor Clamped):**
> *"Why did this price change? This item is currently trending with high browsing demand. Combined with low inventory (only 4 left) and a 3-week restock delay, market prices have surged. However, our fairness cap capped the surge, and our Competitor Match ensures you are still getting a better deal than [Competitor X]."*

**Scenario B (Low Demand Discount):**
> *"Why the discount? We have healthy inventory and fresh restocks arriving tomorrow. Because demand is currently stable, we've dynamically lowered the price by 5% below market average to give you the best deal possible!"*

## 5. Judge-Facing Explanation Script

*To the Judges:*
"One of the biggest pitfalls in modern dynamic pricing is algorithmic discrimination—big tech companies have faced massive backlash for accidentally charging Mac users or specific zip codes higher prices. 

We circumvented this by building an **Ethical Firewall** and an **Explainability Layer**. First, our ML engine is structurally blind to demographics, hardware, and location. It only trades on raw supply-side metrics—Inventory, Restock ETA, and Competitor Benchmarks—and demand-side velocity. Second, we don't just change prices; we explain *why*. Our Explainability API intercepts the ML output and generates human-readable causality strings detailing exactly which guardrails and supply constraints triggered the price shift. We believe enterprise AI must be auditable, transparent, and undeniably fair."

## 6. Where This Sits in the Architecture

1.  **Ingestion & Firewall (Data Layer):** The Fairness Filter explicitly drops excluded features before they enter the rolling session state.
2.  **Pricing Inference (ML Layer):** Calculates optimal price using only approved Demand/Supply tensors.
3.  **Guardrail Constraints (Business Layer):** Caps the ML output (Parity, Margin, Max Surge).
4.  **Explainability Generator (API/Presentation Layer):** A final post-processing function that reads the weights of the ML inference (e.g., "Was demand velocity > 0.8?") and the Guardrail limits hit (e.g., "Was Parity Cap triggered?"), mapping them into the final JSON explanation string sent to the frontend UI.
