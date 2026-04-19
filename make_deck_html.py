"""Build HTML presentation deck and convert to PDF via weasyprint."""

import base64, os

PLOTS = "outputs/plots"

def img_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return ""

cum     = img_b64(f"{PLOTS}/cumulative_returns.png")
weights = img_b64(f"{PLOTS}/portfolio_weights.png")
value   = img_b64(f"{PLOTS}/regime_value.png")
size_   = img_b64(f"{PLOTS}/regime_size.png")
quality = img_b64(f"{PLOTS}/regime_quality.png")
growth  = img_b64(f"{PLOTS}/regime_growth.png")
mom     = img_b64(f"{PLOTS}/regime_momentum.png")

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: Arial, Helvetica, sans-serif;
  background: #ffffff;
}

.slide {
  width: 960px;
  height: 540px;
  background: #ffffff;
  position: relative;
  overflow: hidden;
  page-break-after: always;
  break-after: page;
}

/* ── Footer ── */
.footer {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 33px;
  background: #F2F2F2;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 18px;
}
.pgnum { font-size: 11px; color: #888; }

/* ── Content slide header ── */
.hdr { padding: 15px 26px 0 26px; }
.hdr-title  { font-size: 27px; font-weight: 700; color: #000; line-height: 1.1; }
.hdr-sub    { font-size: 12px; color: #666; margin-top: 4px; }
.hrule      { height: 1.5px; background: #1a1a1a; margin: 7px 26px 0 26px; }

/* ── Title slide ── */
.title-left-bar {
  position: absolute; left: 0; top: 0; bottom: 0;
  width: 8px; background: #A5A5A5;
}
.title-block {
  position: absolute; left: 28px; top: 145px;
}
.title-block h1 { font-size: 36px; font-weight: 700; color: #000; line-height: 1.15; }
.title-block .t-phase { font-size: 19px; color: #666; margin-top: 14px; }
.title-block .t-names { font-size: 14px; color: #555; margin-top: 22px; line-height: 1.8; }

.glance {
  position: absolute; right: 30px; top: 100px;
  width: 268px;
  border-left: 1.5px solid #1a1a1a;
  padding-left: 16px;
}
.glance h3 {
  font-size: 12px; font-weight: 700; color: #000;
  margin-bottom: 8px; padding-bottom: 5px;
  border-bottom: 1.5px solid #1a1a1a;
}
.grow { display: flex; gap: 8px; margin-bottom: 9px; font-size: 11px; line-height: 1.3; }
.gkey { font-weight: 700; color: #000; min-width: 82px; }
.gval { color: #444; }

/* ── Flex cards row ── */
.cards-row {
  display: flex; gap: 9px;
  padding: 12px 26px 0 26px;
}
.card {
  background: #F2F2F2;
  flex: 1; padding: 11px 11px 10px 11px;
}
.card-title {
  font-size: 11px; font-weight: 700; color: #000;
  padding-bottom: 5px; border-bottom: 1.5px solid #1a1a1a;
  margin-bottom: 7px; line-height: 1.2;
}
.card-body { font-size: 9.5px; color: #555; line-height: 1.55; }
.card-body li {
  list-style: none; padding-left: 10px;
  position: relative; margin-bottom: 1px;
}
.card-body li::before { content: "–"; position: absolute; left: 0; color: #999; }

/* ── Two-column ── */
.two-col {
  display: flex; gap: 0;
  padding: 11px 26px 0 26px;
}
.col-l { flex: 1; padding-right: 20px; border-right: 1.5px solid #1a1a1a; }
.col-r { flex: 1; padding-left: 20px; }
.col-label {
  font-size: 11px; font-weight: 700; color: #000;
  margin-bottom: 5px; padding-bottom: 4px;
  border-bottom: 1.5px solid #1a1a1a;
}
.col-note { font-size: 9px; color: #888; font-style: italic; margin-top: 4px; }

/* ── KPI blocks ── */
.kpi-block {
  padding: 7px 0 6px 0;
  border-bottom: 1px solid #e0e0e0;
}
.kpi-block:last-child { border-bottom: none; }
.kpi-lbl { font-size: 10px; font-weight: 700; color: #000; }
.kpi-big { font-size: 28px; font-weight: 700; color: #000; line-height: 1.05; margin: 1px 0; }
.kpi-sub { font-size: 9px; color: #777; }

/* ── Table ── */
table { width: 100%; border-collapse: collapse; font-size: 10px; }
thead tr { background: #1a1a1a; color: #fff; }
thead th { padding: 6px 7px; text-align: center; font-weight: 700; }
thead th:first-child { text-align: left; }
tbody tr:nth-child(odd)  { background: #F2F2F2; }
tbody tr:nth-child(even) { background: #fff; }
tbody tr.best { font-weight: 700; }
tbody td { padding: 5px 7px; text-align: center; color: #222; }
tbody td:first-child { text-align: left; }
.tnote { font-size: 9px; color: #888; font-style: italic; margin-top: 4px; }

/* ── Regime grid ── */
.reg-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 9px;
  padding: 9px 26px 42px 26px;
  height: calc(100% - 82px);
}
.reg-cell { display: flex; flex-direction: column; }
.reg-lbl {
  font-size: 9.5px; font-weight: 700; color: #000;
  margin-bottom: 2px; border-bottom: 1px solid #ccc;
  padding-bottom: 2px;
}
.reg-cell img { width: 100%; flex: 1; object-fit: contain; object-position: top; }
.reg-bar { height: 4px; display: flex; margin-top: 2px; border-radius: 2px; overflow: hidden; }
.bull { background: #2a9d5c; }
.bear { background: #c62828; }
.reg-insight {
  display: flex; flex-direction: column;
  justify-content: center; padding: 10px;
}
.reg-insight h4 { font-size: 11px; font-weight: 700; color: #000; margin-bottom: 7px; }
.reg-insight p  { font-size: 10px; color: #555; line-height: 1.55; }

/* ── 2×2 card grid ── */
.grid2x2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 9px;
  padding: 11px 26px 42px 26px;
  height: calc(100% - 82px);
}
.tcard { background: #F2F2F2; padding: 11px 13px; display: flex; flex-direction: column; }
.tcard-title {
  font-size: 12px; font-weight: 700; color: #000;
  padding-bottom: 5px; border-bottom: 1.5px solid #1a1a1a;
  margin-bottom: 6px;
}
.tcard-body { font-size: 10px; color: #555; line-height: 1.5; flex: 1; }
.tcard-body li {
  list-style: none; padding-left: 11px;
  position: relative; margin-bottom: 2px;
}
.tcard-body li::before { content: "–"; position: absolute; left: 0; color: #999; }

@media print {
  .slide { margin: 0; page-break-after: always; }
}
"""

GLANCE_ROWS = [
    ("Universe",    "6 factor portfolios (FF5 + Mom)"),
    ("Period",      "Jan 2000 – Jan 2026 (6,511 days)"),
    ("Regime",      "Sparse Jump Model, monthly refit"),
    ("Portfolio",   "Black-Litterman + SLSQP"),
    ("Best Sharpe", "0.57 at TE = 2%  (vs 0.55 EW)"),
    ("Paper",       "Shu &amp; Mulvey (2025), JPM"),
]

def glance_rows():
    return "\n".join(
        f'<div class="grow"><span class="gkey">{k}</span><span class="gval">{v}</span></div>'
        for k, v in GLANCE_ROWS
    )

def regime_cell(label, b64img, bull_pct):
    bear_pct = 100 - bull_pct
    return f"""
<div class="reg-cell">
  <div class="reg-lbl">{label} &nbsp;|&nbsp; {bull_pct}% Bull / {bear_pct}% Bear</div>
  <img src="{b64img}">
  <div class="reg-bar">
    <div class="bull" style="width:{bull_pct}%"></div>
    <div class="bear" style="width:{bear_pct}%"></div>
  </div>
</div>"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>{CSS}</style>
</head>
<body>

<!-- ===== SLIDE 1: Title ===== -->
<div class="slide">
  <div class="title-left-bar"></div>
  <div class="title-block">
    <h1>QWIM – Market Regimes</h1>
    <div class="t-phase">Phase 1: Dynamic Factor Allocation via SJM + Black-Litterman</div>
    <div class="t-names">Aditya Modi<br>Tanmay Kadam<br>Ishan Kakodkar</div>
  </div>
  <div class="glance">
    <h3>At a Glance</h3>
    {glance_rows()}
  </div>
  <div class="footer"></div>
</div>

<!-- ===== SLIDE 2: Methodology ===== -->
<div class="slide">
  <div class="hdr">
    <div class="hdr-title">Methodology</div>
    <div class="hdr-sub">Sparse Jump Model regime detection + Black-Litterman portfolio construction</div>
  </div>
  <div class="hrule"></div>
  <div class="cards-row" style="height:calc(100% - 155px);padding-top:13px;">
    <div class="card">
      <div class="card-title">1. Feature Engineering</div>
      <ul class="card-body">
        <li>17 features per factor</li>
        <li>EWMA returns (8/21/63d)</li>
        <li>RSI, Stochastic %K, MACD</li>
        <li>Log downside deviation</li>
        <li>Active beta vs market</li>
        <li>Macro: VIX, 2Y yield, 10Y–2Y</li>
      </ul>
    </div>
    <div class="card">
      <div class="card-title">2. Clip + Scale</div>
      <ul class="card-body">
        <li>DataClipperStd (3-sigma)</li>
        <li>StandardScalerPD</li>
        <li>Fit on train window only</li>
        <li>No lookahead leakage</li>
      </ul>
    </div>
    <div class="card">
      <div class="card-title">3. SJM Fit</div>
      <ul class="card-body">
        <li>Expanding window (8–12 yr)</li>
        <li>Monthly refit dates</li>
        <li>&#955; = 50, &#954;&#178; = 9.5</li>
        <li>10 random restarts</li>
      </ul>
    </div>
    <div class="card">
      <div class="card-title">4. Online Inference</div>
      <ul class="card-body">
        <li>predict_online() method</li>
        <li>1-day prediction delay</li>
        <li>State 0 = Bull</li>
        <li>No lookahead bias</li>
      </ul>
    </div>
    <div class="card">
      <div class="card-title">5. BL Portfolio</div>
      <ul class="card-body">
        <li>5&#xD7;6 view matrix P</li>
        <li>&#937; via Brent's method</li>
        <li>Target TE: 1/2/3/4%</li>
        <li>SLSQP long-only optimizer</li>
      </ul>
    </div>
  </div>
  <div style="padding:7px 26px 0 26px;border-top:1px solid #ddd;margin-top:5px;">
    <span style="font-size:9px;color:#888;font-style:italic;">
      State 0 = Bull &nbsp;|&nbsp; 1-day prediction delay &nbsp;|&nbsp; &#937; calibrated via Brent's method &nbsp;|&nbsp; Long-only SLSQP rebalance
    </span>
  </div>
  <div class="footer"><span class="pgnum">2</span></div>
</div>

<!-- ===== SLIDE 3: Regime Detection ===== -->
<div class="slide">
  <div class="hdr">
    <div class="hdr-title">Regime Detection Results</div>
    <div class="hdr-sub">Online inferred bull/bear regimes per factor &nbsp;|&nbsp; Test period 2008–2026 &nbsp;|&nbsp; 4,514 trading days</div>
  </div>
  <div class="hrule"></div>
  <div class="reg-grid">
    {regime_cell("Value",    value,   43.5)}
    {regime_cell("Size",     size_,   44.0)}
    {regime_cell("Quality",  quality, 44.5)}
    {regime_cell("Growth",   growth,  60.1)}
    {regime_cell("Momentum", mom,     60.4)}
    <div class="reg-insight">
      <h4>Insight</h4>
      <p>Growth &amp; Momentum spend &gt;60% in bull regime, consistent with persistent risk-on character.<br><br>
      Value, Size, and Quality are more bear-heavy, reflecting factor cyclicality documented in the literature.</p>
    </div>
  </div>
  <div class="footer"><span class="pgnum">3</span></div>
</div>

<!-- ===== SLIDE 4: Performance Results ===== -->
<div class="slide">
  <div class="hdr">
    <div class="hdr-title">Empirical Results</div>
    <div class="hdr-sub">Walk-forward backtest 2008–2026 &nbsp;|&nbsp; After 5 bps/side transaction costs</div>
  </div>
  <div class="hrule"></div>
  <div class="two-col" style="height:calc(100% - 118px);">
    <div class="col-l">
      <div class="kpi-block">
        <div class="kpi-lbl">Best Sharpe Ratio</div>
        <div class="kpi-big">0.57</div>
        <div class="kpi-sub">Dynamic TE=2% &nbsp;(vs 0.55 EW benchmark)</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-lbl">Ann. Excess Return</div>
        <div class="kpi-big">9.70%</div>
        <div class="kpi-sub">Dynamic TE=2% &nbsp;(vs 9.51% EW)</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-lbl">Max Drawdown</div>
        <div class="kpi-big">&#8722;51.9%</div>
        <div class="kpi-sub">Dynamic TE=2% &nbsp;(vs &#8722;53.0% EW)</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-lbl">Info Ratio vs EW</div>
        <div class="kpi-big">+0.04</div>
        <div class="kpi-sub">TE=1% and TE=2% targets</div>
      </div>
    </div>
    <div class="col-r">
      <table>
        <thead>
          <tr>
            <th>Strategy</th><th>Ann. Ret</th><th>Sharpe</th>
            <th>Max DD</th><th>IR vs EW</th><th>Turnover</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>EW Benchmark</td><td>9.51%</td><td>0.55</td><td>&#8722;53.0%</td><td>&#8212;</td><td>&#8212;</td></tr>
          <tr><td>Dynamic TE=1%</td><td>9.60%</td><td>0.56</td><td>&#8722;53.0%</td><td>+0.04</td><td>2.7x/yr</td></tr>
          <tr class="best"><td>Dynamic TE=2% &#9733;</td><td>9.70%</td><td>0.57</td><td>&#8722;51.9%</td><td>+0.04</td><td>5.7x/yr</td></tr>
          <tr><td>Dynamic TE=3%</td><td>9.54%</td><td>0.56</td><td>&#8722;52.4%</td><td>&#8722;0.02</td><td>9.2x/yr</td></tr>
          <tr><td>Dynamic TE=4%</td><td>9.54%</td><td>0.56</td><td>&#8722;52.8%</td><td>&#8722;0.02</td><td>11.8x/yr</td></tr>
        </tbody>
      </table>
      <div class="tnote">&#9733; Best risk-adjusted strategy &nbsp;|&nbsp; All results out-of-sample</div>
      <div class="tnote" style="margin-top:2px;">Training: expanding window (8 yr min, 12 yr max) &nbsp;|&nbsp; SJM refitted monthly</div>
    </div>
  </div>
  <div class="footer"><span class="pgnum">4</span></div>
</div>

<!-- ===== SLIDE 5: Portfolio Analysis ===== -->
<div class="slide">
  <div class="hdr">
    <div class="hdr-title">Portfolio Analysis</div>
    <div class="hdr-sub">Cumulative excess returns and dynamic weight allocation over time</div>
  </div>
  <div class="hrule"></div>
  <div class="two-col" style="height:calc(100% - 118px);">
    <div class="col-l" style="padding-right:16px;">
      <div class="col-label">Cumulative Excess Returns</div>
      <img src="{cum}" style="width:100%;height:calc(100% - 50px);object-fit:contain;object-position:top;">
      <div class="col-note">Solid = Dynamic strategies &nbsp;|&nbsp; Dashed = EW benchmark &nbsp;|&nbsp; Dotted = Market</div>
    </div>
    <div class="col-r" style="padding-left:16px;">
      <div class="col-label">Dynamic Portfolio Weights (TE=3%)</div>
      <img src="{weights}" style="width:100%;height:calc(100% - 50px);object-fit:contain;object-position:top;">
      <div class="col-note">Regime-driven tilts shift weight across factors over time</div>
    </div>
  </div>
  <div class="footer"><span class="pgnum">5</span></div>
</div>

<!-- ===== SLIDE 6: Key Takeaways ===== -->
<div class="slide">
  <div class="hdr">
    <div class="hdr-title">Key Takeaways</div>
    <div class="hdr-sub">Summary of findings from Phase 1 replication</div>
  </div>
  <div class="hrule"></div>
  <div class="grid2x2">
    <div class="tcard">
      <div class="tcard-title">Regime detection is reliable</div>
      <div class="tcard-body">SJM identifies persistent bull/bear states per factor. All 5 factors show clear regime structure. Growth and Momentum spend &gt;60% in bull regimes; Value/Size/Quality are more bear-heavy over the 2008–2026 test period.</div>
    </div>
    <div class="tcard">
      <div class="tcard-title">Low-TE tilts deliver value</div>
      <div class="tcard-body">TE=2% achieves the best Sharpe (0.57 vs 0.55 EW) and positive IR (+0.04) with manageable turnover (5.7&#xD7;/year). The BL framework correctly translates regime signals into modest, cost-aware tilts.</div>
    </div>
    <div class="tcard">
      <div class="tcard-title">Aggressive tilts hurt after costs</div>
      <div class="tcard-body">TE=3–4% targets increase turnover to 9–12&#xD7;/year without commensurate return improvement. Transaction costs erode alpha from stronger tilts. Sweet spot is TE = 1–2%.</div>
    </div>
    <div class="tcard">
      <div class="tcard-title">Signal quality is the bottleneck</div>
      <div class="tcard-body">The regime signal is informative but not strong enough for large active bets. Improvements should focus on richer features, better hyperparameter tuning, and an expanded factor universe.</div>
    </div>
  </div>
  <div class="footer"><span class="pgnum">6</span></div>
</div>

<!-- ===== SLIDE 7: Phase 2 Roadmap ===== -->
<div class="slide">
  <div class="hdr">
    <div class="hdr-title">Phase 2 Roadmap</div>
    <div class="hdr-sub">Contributions and enhancements beyond Phase 1 replication</div>
  </div>
  <div class="hrule"></div>
  <div class="grid2x2">
    <div class="tcard">
      <div class="tcard-title">01 — Hyperparameter Optimization</div>
      <ul class="tcard-body">
        <li>Grid/Bayesian search over SJM &#955; and &#954;&#178;</li>
        <li>Cross-validate TE targets on held-out validation set</li>
        <li>Per-factor tuning — value vs momentum differ</li>
        <li>Sensitivity analysis: stability to perturbations</li>
      </ul>
    </div>
    <div class="tcard">
      <div class="tcard-title">02 — Expanded Factor Universe</div>
      <ul class="tcard-body">
        <li>Add Low Volatility (BAB / IVOL)</li>
        <li>Quality decomposition: profitability vs investment vs accruals</li>
        <li>International factors: EM and DM versions of FF5</li>
        <li>Sector momentum and industry-level factors</li>
      </ul>
    </div>
    <div class="tcard">
      <div class="tcard-title">03 — Model Enhancements</div>
      <ul class="tcard-body">
        <li>Compare SJM vs HMM-MLE and Gaussian Mixture</li>
        <li>Explore 3-state models (Bull / Neutral / Bear)</li>
        <li>Ensemble regimes: combine per-factor signals</li>
        <li>Incremental SJM updates to reduce refit latency</li>
      </ul>
    </div>
    <div class="tcard">
      <div class="tcard-title">04 — Portfolio &amp; Infrastructure</div>
      <ul class="tcard-body">
        <li>Turnover constraints directly in SLSQP</li>
        <li>Risk-parity and factor-risk-parity as priors</li>
        <li>Streamlit dashboard: live regime monitor</li>
        <li>Stress-test: GFC (2008), COVID (2020), 2022 hikes</li>
      </ul>
    </div>
  </div>
  <div class="footer"><span class="pgnum">7</span></div>
</div>

</body>
</html>"""

with open("outputs/deck.html", "w", encoding="utf-8") as f:
    f.write(HTML)
print("HTML written to outputs/deck.html")

# Convert to PDF via weasyprint
import weasyprint
print("Converting to PDF...")
weasyprint.HTML(filename="outputs/deck.html").write_pdf(
    "outputs/Dynamic_Factor_Allocation.pdf",
    stylesheets=[weasyprint.CSS(string="@page { size: 960px 540px; margin: 0; }")]
)
print("PDF saved to outputs/Dynamic_Factor_Allocation.pdf")
