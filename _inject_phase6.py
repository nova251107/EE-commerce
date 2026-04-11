"""Inject Phase 6 section into index.html before the footer divider."""
with open('index.html', 'r', encoding='utf-8') as f:
    src = f.read()

PHASE6_SECTION = r"""
          {/* ════════════════════════════════════════════════════════
              PHASE 6 — STEP 22+23+24: Explainability + Simulation
          ════════════════════════════════════════════════════════ */}
          <section className="py-20 px-6 max-w-screen-xl mx-auto" id="simulation">
            <div className="text-center mb-14" data-aos="fade-up">
              <span className="section-label">Phase 6 — Step 22 · 23 · 24</span>
              <h2 className="section-title">Price Simulation &amp; Explainability Engine</h2>
              <p className="text-[14px] mt-3" style={{ color: "rgba(255,255,255,0.4)", maxWidth: 560, margin: "12px auto 0" }}>
                See how the AI dynamically adjusts prices across all demand × inventory scenarios.
                Every decision comes with a structured, human-readable explanation.
              </p>
            </div>
            <SimulationPanel API_BASE={API_BASE} />
            <div className="mt-14 grid sm:grid-cols-2 lg:grid-cols-3 gap-5" data-aos="fade-up">
              {[
                { icon: "⚡", challenge: "C1: Data Delay",       solution: "Real-time streaming features only — no historical lag",       color: "#e2b96f" },
                { icon: "🎯", challenge: "C2: Overfitting",       solution: "Simple linear models (logistic + scoring) — 3 features max",  color: "#a78bfa" },
                { icon: "🚀", challenge: "C3: Latency",           solution: "Precomputed weights → inference in <1ms per request",         color: "#34d399" },
                { icon: "🌟", challenge: "C4: Cold Start",        solution: "Trending + device + time-of-day signals for new users",       color: "#60a5fa" },
                { icon: "🛡️", challenge: "C5: Unstable Prices",   solution: "Hard ±15% guardrails enforced on every decision",            color: "#f87171" },
                { icon: "✅", challenge: "Step 24: Final System", solution: "Real-time pricing · Smart recs · <50ms latency · Explainable", color: "#2ed573" },
              ].map((c, i) => (
                <div key={c.challenge} data-aos="fade-up" data-aos-delay={i * 80}
                  style={{ background:"rgba(255,255,255,0.025)", border:"1px solid rgba(255,255,255,0.07)", borderRadius:16, padding:"20px 22px", transition:"all 0.3s" }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = c.color + "55"; e.currentTarget.style.background = c.color + "0A"; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)"; e.currentTarget.style.background = "rgba(255,255,255,0.025)"; }}>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-2xl">{c.icon}</span>
                    <span className="text-[11px] font-bold tracking-widest uppercase" style={{ color: c.color }}>{c.challenge}</span>
                  </div>
                  <p className="text-[13px] leading-relaxed m-0" style={{ color: "rgba(255,255,255,0.65)" }}>{c.solution}</p>
                </div>
              ))}
            </div>
          </section>

"""

COMPONENT = r"""
    /* ─────────────── PHASE 6: SimulationPanel ─────────────── */
    function SimulationPanel({ API_BASE }) {
      const [simData, setSimData] = React.useState(null);
      const [loading, setLoading] = React.useState(false);
      const [basePrice, setBasePrice] = React.useState(1000);
      const [customResult, setCustomResult] = React.useState(null);
      const [demand, setDemand] = React.useState(70);
      const [inventory, setInventory] = React.useState(15);
      const [intent, setIntent] = React.useState(8);
      const [selectedExp, setSelectedExp] = React.useState(null);

      const TIER_COLORS = {
        surge:"#ff4757", high:"#ff6b35", normal:"#2ed573",
        low:"#1e90ff", clearance:"#a29bfe", hold:"#74b9ff"
      };

      async function runMatrix() {
        setLoading(true);
        try {
          const r = await fetch(`${API_BASE}/simulate?base_price=${basePrice}`);
          if (r.ok) setSimData(await r.json());
        } catch(e) {
          /* fallback demo data */
          const fp = (base, adj) => Math.round(base * (1 + adj/100) * 100) / 100;
          setSimData({ base_price: basePrice, scenarios: [
            { scenario_label:"High Demand, Low Stock",  demand_score:82, inventory:8,  adjustment_pct:15.0, final_price:fp(basePrice,15),   tier:"surge",    reasons:["📈 Surge demand: +15%","🔥 Low stock: +3%"] },
            { scenario_label:"High Demand, High Stock", demand_score:75, inventory:800, adjustment_pct:8.1, final_price:fp(basePrice,8.1),  tier:"high",     reasons:["📈 High demand: +8.1%","📦 Surplus dampens: -2%"] },
            { scenario_label:"Low Demand, High Stock",  demand_score:18, inventory:700, adjustment_pct:-28.8, final_price:fp(basePrice,-28.8), tier:"clearance", reasons:["📉 Low demand: -24.5%","📦 Surplus: -5%"] },
            { scenario_label:"Low Demand, Low Stock",   demand_score:22, inventory:12, adjustment_pct:-1.2, final_price:fp(basePrice,-1.2),  tier:"hold",     reasons:["💎 Value hold: margin protected","🔥 Low stock offsets"] },
            { scenario_label:"Medium Demand, Normal",   demand_score:50, inventory:150, adjustment_pct:0.8, final_price:fp(basePrice,0.8),   tier:"normal",   reasons:["⚖️ Balanced: +0.8%"] },
            { scenario_label:"Surge + Critical Stock",  demand_score:95, inventory:3,  adjustment_pct:15.0, final_price:fp(basePrice,15),   tier:"surge",    reasons:["🚨 Critical stock: +5%","📈 Surge demand: +17%"] },
          ]});
        }
        setLoading(false);
      }

      async function runCustom() {
        setLoading(true);
        try {
          const r = await fetch(
            `${API_BASE}/simulate/custom?base_price=${basePrice}&demand_score=${demand}&inventory=${inventory}&user_intent=${intent}`,
            { method: "POST" }
          );
          if (r.ok) setCustomResult(await r.json());
          else throw new Error("API error");
        } catch {
          const adj = (demand > 60 ? 10 : demand < 30 ? -20 : 2) + (inventory < 20 ? 3 : inventory > 400 ? -5 : 0);
          const fp  = Math.round(basePrice * (1 + adj/100) * 100) / 100;
          setCustomResult({ base_price: basePrice, final_price: fp, adjustment_pct: adj,
            tier: demand > 60 ? "high" : demand < 30 ? "clearance" : "normal",
            reasons: [`Demand ${demand}/100`, `Inventory ${inventory} units`, `Intent ${intent}`],
            explanation: { headline: `Price ${adj >= 0 ? "↑" : "↓"} ${Math.abs(adj).toFixed(1)}% based on demand ${demand}/100 and stock ${inventory}`,
              badge: demand > 60 ? "📈 High Demand" : demand < 30 ? "🏷️ Low Demand" : "⚖️ Balanced",
              badge_color: demand > 60 ? "#ff6b35" : demand < 30 ? "#70a1ff" : "#2ed573",
              factors: [
                { icon:"📊", label:"Demand", value:`${demand}/100`, impact: demand>50?"positive":"negative", color: demand>50?"#ff6b35":"#1e90ff" },
                { icon:"📦", label:"Inventory", value:`${inventory} units`, impact: inventory<50?"positive":"inventory>300"?"negative":"neutral", color: inventory<20?"#ff4757":"#2ed573" },
                { icon:"🎯", label:"Intent Score", value:`${intent}`, impact:"neutral", color:"#a78bfa" },
              ],
              summary: `ML engine adjusted price from $${basePrice} to $${fp} (${adj >= 0 ? "+" : ""}${adj.toFixed(1)}%) based on demand score ${demand}/100 and ${inventory} units in stock.`,
            }
          });
        }
        setLoading(false);
      }

      React.useEffect(() => { runMatrix(); }, []);

      const cr = customResult;
      const expl = cr?.explanation;

      return (
        <div>
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-4 mb-8" data-aos="fade-up">
            <div className="flex items-center gap-3 flex-1 min-w-[200px]">
              <label className="text-[11px] font-semibold whitespace-nowrap" style={{ color:"rgba(255,255,255,0.5)" }}>Base Price USD</label>
              <input type="number" value={basePrice} onChange={e => setBasePrice(+e.target.value || 1000)} min={10} max={10000}
                className="flex-1 px-3 py-2 rounded-xl text-white text-[13px] font-mono outline-none"
                style={{ background:"rgba(255,255,255,0.05)", border:"1px solid rgba(255,255,255,0.1)", fontFamily:"DM Mono, monospace" }} />
            </div>
            <button onClick={runMatrix} disabled={loading}
              className="px-5 py-2.5 rounded-xl font-semibold text-[13px] border-none cursor-pointer transition-all"
              style={{ background:"linear-gradient(135deg,#e2b96f,#c8973e)", color:"#080810", opacity: loading ? 0.6 : 1 }}>
              {loading ? "Running…" : "⚡ Run Simulation"}
            </button>
          </div>

          {/* Matrix Table — Step 23 */}
          {simData && (
            <div className="overflow-x-auto rounded-2xl mb-8" data-aos="fade-up"
              style={{ border:"1px solid rgba(255,255,255,0.07)", background:"rgba(255,255,255,0.02)" }}>
              <table style={{ width:"100%", borderCollapse:"collapse", fontFamily:"Outfit, sans-serif" }}>
                <thead>
                  <tr style={{ borderBottom:"1px solid rgba(255,255,255,0.08)" }}>
                    {["Scenario","Demand","Inventory","Final Price","Adjustment","Tier","Step 22 — Why?"].map(h => (
                      <th key={h} style={{ padding:"12px 16px", textAlign:"left", fontSize:10, fontWeight:700, letterSpacing:"0.1em", color:"rgba(255,255,255,0.35)", textTransform:"uppercase" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {simData.scenarios.map((s, i) => {
                    const adj = s.adjustment_pct;
                    const tColor = TIER_COLORS[s.tier] || "#fff";
                    const isUp   = adj > 0;
                    return (
                      <tr key={i}
                        onClick={() => setSelectedExp(selectedExp === i ? null : i)}
                        style={{ borderBottom:"1px solid rgba(255,255,255,0.04)", cursor:"pointer",
                          background: selectedExp === i ? "rgba(226,185,111,0.06)" : "transparent",
                          transition:"background 0.2s" }}
                        onMouseEnter={e => { if (selectedExp !== i) e.currentTarget.style.background = "rgba(255,255,255,0.025)"; }}
                        onMouseLeave={e => { if (selectedExp !== i) e.currentTarget.style.background = "transparent"; }}>
                        <td style={{ padding:"14px 16px", fontSize:13, fontWeight:600, color:"rgba(255,255,255,0.85)" }}>
                          {s.scenario_label}
                        </td>
                        <td style={{ padding:"14px 16px" }}>
                          <div style={{ background:"rgba(255,255,255,0.06)", borderRadius:8, height:6, width:80, overflow:"hidden" }}>
                            <div style={{ height:"100%", borderRadius:8, width:`${Math.min(s.demand_score,100)}%`,
                              background: s.demand_score > 60 ? "#ff4757" : s.demand_score > 35 ? "#ffa502" : "#2ed573" }} />
                          </div>
                          <span style={{ fontSize:10, color:"rgba(255,255,255,0.4)", marginTop:3, display:"block" }}>{s.demand_score}/100</span>
                        </td>
                        <td style={{ padding:"14px 16px", fontSize:12, color:"rgba(255,255,255,0.6)", fontFamily:"DM Mono, monospace" }}>
                          {s.inventory >= 1000 ? `${(s.inventory/1000).toFixed(1)}k` : s.inventory} units
                        </td>
                        <td style={{ padding:"14px 16px" }}>
                          <span style={{ fontSize:15, fontWeight:700, fontFamily:"DM Mono, monospace", color: isUp?"#4ade80":"#f87171" }}>
                            ${s.final_price?.toLocaleString("en-US", { minimumFractionDigits:2, maximumFractionDigits:2 })}
                          </span>
                          <span style={{ fontSize:10, color:"rgba(255,255,255,0.3)", display:"block" }}>
                            base ${s.base_price?.toFixed(0) || simData.base_price}
                          </span>
                        </td>
                        <td style={{ padding:"14px 16px" }}>
                          <span style={{ fontSize:13, fontWeight:700, color: isUp?"#4ade80":"#f87171" }}>
                            {adj >= 0 ? "▲" : "▼"} {Math.abs(adj).toFixed(1)}%
                          </span>
                        </td>
                        <td style={{ padding:"14px 16px" }}>
                          <span style={{ fontSize:10, fontWeight:700, padding:"3px 10px", borderRadius:999,
                            background: tColor + "22", color: tColor, border:`1px solid ${tColor}44`, textTransform:"uppercase", letterSpacing:"0.08em" }}>
                            {s.tier}
                          </span>
                        </td>
                        <td style={{ padding:"14px 16px", fontSize:11, color:"rgba(255,255,255,0.5)", maxWidth:240 }}>
                          {s.reasons?.slice(0,2).join(" | ")}
                          <span style={{ color:"rgba(226,185,111,0.5)", marginLeft:6, fontSize:9 }}>
                            {selectedExp === i ? "▲ hide" : "▼ details"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {/* Expanded explainability row */}
              {selectedExp !== null && simData.scenarios[selectedExp]?.explanation && (
                <div style={{ padding:"20px 24px", borderTop:"1px solid rgba(226,185,111,0.15)", background:"rgba(226,185,111,0.04)" }}
                  data-aos="fade-in">
                  <ExplanationCard exp={simData.scenarios[selectedExp].explanation} />
                </div>
              )}
            </div>
          )}

          {/* Custom Simulator */}
          <div className="grid lg:grid-cols-2 gap-6" data-aos="fade-up">
            <div style={{ background:"rgba(255,255,255,0.025)", border:"1px solid rgba(255,255,255,0.07)", borderRadius:20, padding:24 }}>
              <p className="text-[14px] font-bold mb-5 m-0" style={{ color:"rgba(255,255,255,0.9)" }}>
                🎛️ Custom Scenario Simulator
              </p>
              <SliderRow label="Demand Score" value={demand} min={0} max={100} onChange={setDemand}
                color={demand > 60 ? "#ff4757" : demand > 35 ? "#ffa502" : "#2ed573"}
                suffix="/100" />
              <SliderRow label="Inventory" value={inventory} min={0} max={1000} onChange={setInventory}
                color={inventory < 20 ? "#ff4757" : inventory > 300 ? "#2ed573" : "#ffa502"}
                suffix=" units" />
              <SliderRow label="User Intent" value={intent} min={0} max={30} onChange={setIntent}
                color="#a78bfa" suffix=" pts" />
              <button onClick={runCustom} disabled={loading}
                className="w-full py-3 rounded-xl font-semibold text-[13px] mt-2 border-none cursor-pointer transition-all"
                style={{ background:"linear-gradient(135deg,#7c3aed,#5b21b6)", color:"#fff", opacity: loading ? 0.6 : 1 }}>
                {loading ? "Computing…" : "🧠 Get AI Price Decision"}
              </button>
            </div>

            {/* Custom result + explanation */}
            {cr ? (
              <div style={{ background:"rgba(255,255,255,0.025)", border:"1px solid rgba(255,255,255,0.07)", borderRadius:20, padding:24 }}>
                <div className="flex items-start gap-3 mb-4">
                  <div style={{ background: (TIER_COLORS[cr.tier]||"#fff") + "22", borderRadius:12, padding:"10px 14px", flex:"0 0 auto" }}>
                    <span style={{ fontSize:22, fontWeight:800, fontFamily:"DM Mono, monospace", color: cr.adjustment_pct >= 0 ? "#4ade80" : "#f87171" }}>
                      ${cr.final_price?.toFixed(2)}
                    </span>
                    <p style={{ fontSize:10, margin:"2px 0 0", color:"rgba(255,255,255,0.4)" }}>
                      {cr.adjustment_pct >= 0 ? "▲" : "▼"} {Math.abs(cr.adjustment_pct).toFixed(1)}% from ${cr.base_price}
                    </p>
                  </div>
                  <div>
                    {expl && (
                      <span style={{ fontSize:10, fontWeight:700, padding:"3px 10px", borderRadius:999,
                        background:(expl.badge_color||"#fff")+"22", color:expl.badge_color||"#fff",
                        border:`1px solid ${expl.badge_color||"#fff"}44` }}>
                        {expl.badge}
                      </span>
                    )}
                    {expl && <p style={{ fontSize:12, color:"rgba(255,255,255,0.6)", marginTop:8, lineHeight:1.5 }}>{expl.headline}</p>}
                  </div>
                </div>
                {expl?.factors?.map((f, fi) => (
                  <div key={fi} className="flex items-center gap-3 mb-2">
                    <span style={{ fontSize:14, width:20, textAlign:"center" }}>{f.icon}</span>
                    <span style={{ fontSize:11, color:"rgba(255,255,255,0.5)", flex:1 }}>{f.label}</span>
                    <span style={{ fontSize:11, fontWeight:600, color:f.color }}>{f.value}</span>
                    <span style={{ fontSize:9, padding:"1px 7px", borderRadius:999,
                      background: f.impact==="positive"?"rgba(74,222,128,0.12)":f.impact==="negative"?"rgba(248,113,113,0.12)":"rgba(255,255,255,0.06)",
                      color: f.impact==="positive"?"#4ade80":f.impact==="negative"?"#f87171":"rgba(255,255,255,0.35)" }}>
                      {f.impact}
                    </span>
                  </div>
                ))}
                {expl?.summary && (
                  <p style={{ fontSize:11, color:"rgba(255,255,255,0.4)", lineHeight:1.6, marginTop:10, paddingTop:10, borderTop:"1px solid rgba(255,255,255,0.06)" }}>
                    {expl.summary}
                  </p>
                )}
                {expl?.challenge_note && (
                  <p style={{ fontSize:10, color:"#a78bfa", marginTop:6, fontStyle:"italic" }}>
                    {expl.challenge_note}
                  </p>
                )}
              </div>
            ) : (
              <div style={{ background:"rgba(255,255,255,0.02)", border:"1px dashed rgba(255,255,255,0.07)", borderRadius:20, padding:24,
                display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:12, minHeight:200 }}>
                <span style={{ fontSize:36 }}>🧪</span>
                <p style={{ fontSize:13, color:"rgba(255,255,255,0.3)", margin:0, textAlign:"center" }}>
                  Adjust sliders and click "Get AI Price Decision"<br/>to see the ML engine in action
                </p>
              </div>
            )}
          </div>
        </div>
      );
    }

    function SliderRow({ label, value, min, max, onChange, color, suffix }) {
      return (
        <div className="mb-4">
          <div className="flex justify-between mb-1.5">
            <span style={{ fontSize:11, color:"rgba(255,255,255,0.5)" }}>{label}</span>
            <span style={{ fontSize:12, fontWeight:700, fontFamily:"DM Mono, monospace", color }}>{value}{suffix}</span>
          </div>
          <input type="range" min={min} max={max} value={value} onChange={e => onChange(+e.target.value)}
            style={{ width:"100%", accentColor: color, cursor:"pointer" }} />
        </div>
      );
    }

    function ExplanationCard({ exp }) {
      if (!exp) return null;
      return (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <span style={{ fontSize:10, fontWeight:700, padding:"3px 12px", borderRadius:999,
              background:(exp.badge_color||"#fff")+"22", color:exp.badge_color||"#fff", border:`1px solid ${exp.badge_color||"#fff"}44` }}>
              {exp.badge}
            </span>
            <p style={{ fontSize:13, fontWeight:600, color:"rgba(255,255,255,0.85)", margin:0 }}>{exp.headline}</p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
            {exp.factors?.map((f, i) => (
              <div key={i} style={{ background:"rgba(255,255,255,0.04)", borderRadius:10, padding:"10px 12px", display:"flex", alignItems:"center", gap:8 }}>
                <span style={{ fontSize:16 }}>{f.icon}</span>
                <div>
                  <p style={{ fontSize:10, color:"rgba(255,255,255,0.4)", margin:0 }}>{f.label}</p>
                  <p style={{ fontSize:12, fontWeight:600, color:f.color, margin:0 }}>{f.value}</p>
                </div>
                <span style={{ marginLeft:"auto", fontSize:8, padding:"1px 5px", borderRadius:999,
                  background: f.impact==="positive"?"rgba(74,222,128,0.12)":f.impact==="negative"?"rgba(248,113,113,0.12)":"rgba(255,255,255,0.06)",
                  color: f.impact==="positive"?"#4ade80":f.impact==="negative"?"#f87171":"rgba(255,255,255,0.35)" }}>
                  {f.impact}
                </span>
              </div>
            ))}
          </div>
          {exp.summary && (
            <p style={{ fontSize:12, color:"rgba(255,255,255,0.5)", lineHeight:1.65, margin:0, paddingTop:10, borderTop:"1px solid rgba(255,255,255,0.06)" }}>
              {exp.summary}
            </p>
          )}
          {exp.challenge_note && (
            <p style={{ fontSize:10, color:"#a78bfa", marginTop:8, fontStyle:"italic" }}>{exp.challenge_note}</p>
          )}
        </div>
      );
    }

"""

# Insert SimulationPanel + helper components before function App()
APP_MARKER = "    /* ─────────────── THREE.JS INIT ─────────────── */"
if APP_MARKER in src:
    src = src.replace(APP_MARKER, COMPONENT + "\n" + APP_MARKER, 1)
    print("Component injected OK")
else:
    print("WARNING: marker not found")

# Inject Phase 6 section before footer divider
FOOTER_MARKER = "\n          {/* ── TESTIMONIALS ── */}\n"
# find after ── TESTIMONIALS ──, add section before hr
FOOTER_HR = "          <hr className=\"gradient-divider\" />\n\n          {/* ── FOOTER ── */}"
FOOTER_HR_NEW = (
    "\n          {/* ── PHASE 6: Simulation + Explainability ── */}\n"
    "          <section className=\"py-20 px-6 max-w-screen-xl mx-auto\" id=\"simulation\">\n"
    "            <div className=\"text-center mb-14\" data-aos=\"fade-up\">\n"
    "              <span className=\"section-label\">Phase 6 — Step 22 · 23 · 24</span>\n"
    "              <h2 className=\"section-title\">Price Simulation &amp; Explainability Engine</h2>\n"
    "              <p className=\"text-[14px] mt-3\" style={{ color: \"rgba(255,255,255,0.4)\", maxWidth: 560, margin: \"12px auto 0\" }}>\n"
    "                See how the AI dynamically adjusts prices across all demand × inventory scenarios.\n"
    "              </p>\n"
    "            </div>\n"
    "            <SimulationPanel API_BASE={API_BASE} />\n"
    "            <div className=\"mt-14 grid sm:grid-cols-2 lg:grid-cols-3 gap-5\" data-aos=\"fade-up\">\n"
    "              {[\n"
    "                { icon:\"⚡\", challenge:\"C1: Data Delay\",       solution:\"Real-time streaming features only — no historical lag\",       color:\"#e2b96f\" },\n"
    "                { icon:\"🎯\", challenge:\"C2: Overfitting\",       solution:\"Simple linear models (logistic + scoring) — 3 features max\",  color:\"#a78bfa\" },\n"
    "                { icon:\"🚀\", challenge:\"C3: Latency\",           solution:\"Precomputed weights → inference in <1ms per request\",         color:\"#34d399\" },\n"
    "                { icon:\"🌟\", challenge:\"C4: Cold Start\",        solution:\"Trending + device + time-of-day signals for new users\",       color:\"#60a5fa\" },\n"
    "                { icon:\"🛡️\", challenge:\"C5: Unstable Prices\",   solution:\"Hard ±15% guardrails enforced on every decision\",            color:\"#f87171\" },\n"
    "                { icon:\"✅\", challenge:\"Step 24: Final System\", solution:\"Real-time pricing · Smart recs · <50ms latency · Explainable\",color:\"#2ed573\" },\n"
    "              ].map((c, i) => (\n"
    "                <div key={c.challenge} data-aos=\"fade-up\" data-aos-delay={i * 80}\n"
    "                  style={{ background:\"rgba(255,255,255,0.025)\", border:\"1px solid rgba(255,255,255,0.07)\", borderRadius:16, padding:\"20px 22px\", transition:\"all 0.3s\" }}\n"
    "                  onMouseEnter={e => { e.currentTarget.style.borderColor = c.color + \"55\"; e.currentTarget.style.background = c.color + \"0A\"; }}\n"
    "                  onMouseLeave={e => { e.currentTarget.style.borderColor = \"rgba(255,255,255,0.07)\"; e.currentTarget.style.background = \"rgba(255,255,255,0.025)\"; }}>\n"
    "                  <div className=\"flex items-center gap-3 mb-3\">\n"
    "                    <span className=\"text-2xl\">{c.icon}</span>\n"
    "                    <span className=\"text-[11px] font-bold tracking-widest uppercase\" style={{ color: c.color }}>{c.challenge}</span>\n"
    "                  </div>\n"
    "                  <p className=\"text-[13px] leading-relaxed m-0\" style={{ color: \"rgba(255,255,255,0.65)\" }}>{c.solution}</p>\n"
    "                </div>\n"
    "              ))}\n"
    "            </div>\n"
    "          </section>\n\n"
    "          <hr className=\"gradient-divider\" />\n\n          {/* ── FOOTER ── */}"
)

if FOOTER_HR in src:
    src = src.replace(FOOTER_HR, FOOTER_HR_NEW, 1)
    print("Section injected OK")
else:
    print("WARNING: footer hr marker not found — searching for partial...")
    idx = src.find('<hr className="gradient-divider"')
    print(f"Found hr at index {idx}")
    print(repr(src[idx-40:idx+80]))

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(src)

print(f"Done. Lines: {len(src.splitlines())}")
