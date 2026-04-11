import urllib.request, json, sys
try:
    r = urllib.request.urlopen('http://localhost:8000/simulate?base_price=1000', timeout=6)
    data = json.loads(r.read())
    print("✅ /simulate API OK")
    for s in data['scenarios']:
        b = s.get('explanation', {}) or {}
        print(f"  {s['scenario_label']:30s}  ${s['final_price']:.2f}  {s['tier']:10s}  {b.get('badge','')}")
except Exception as e:
    print(f"❌ Server not ready: {e}")
    sys.exit(1)
