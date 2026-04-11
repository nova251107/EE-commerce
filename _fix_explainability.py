"""Fix explainability.py bugs."""
with open('explainability.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Fix 1: 3-value unpack → 2-value (BADGES are already (full_text, color) 2-tuples)
src = src.replace(
    '''        badge_label, badge_emoji, badge_color = self.BADGES.get(
            badge_key, ("⚖️ Balanced", "#2ed573")
        )''',
    '''        badge_text, badge_color = self.BADGES.get(
            badge_key, ("⚖️ Balanced", "#2ed573")
        )'''
)

# Fix 2: Use badge_text directly instead of f"{badge_emoji} {badge_label}"
src = src.replace(
    '            badge          = f"{badge_emoji} {badge_label}",',
    '            badge          = badge_text,'
)

# Fix 3: Remove circular import inside PriceSimulator.__init__
src = src.replace(
    '''        try:
            from explainability import PricingExplainer
            self._explainer = PricingExplainer()
        except ImportError:
            self._explainer = None''',
    '        self._explainer = PricingExplainer()'
)

with open('explainability.py', 'w', encoding='utf-8') as f:
    f.write(src)

import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SyntaxError: {e}")

# Quick functional test
from explainability import price_simulator
results = price_simulator.run_matrix(1000.0)
for r in results:
    e = r.get('explanation', {})
    print(f"  {r['scenario_label']:30s} → ${r['final_price']:.2f}  {r['tier']:10s}  badge={e.get('badge','?')[:25] if e else 'None'}")
print("All OK")
