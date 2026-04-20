# App Improvements Design
**Date:** 2026-04-20
**Context:** QWIM Market Regimes dashboard — pre-presentation polish + config panel feature

---

## Background

The dashboard has 6 model tabs (Comparison, SJM+BL, SC-HMM, Simple-HMM, HMM, MS-GARCH, HSMM). Three issues motivate this work:

1. HMM, HSMM, and MS-GARCH tabs are visually inconsistent with Model 1 and SC-HMM (no sidebar, no formatted metrics)
2. Simple-HMM has no outputs and shows a dead placeholder tab
3. Model 1 has no way to adjust parameters without editing `config.yaml` manually

---

## Section 1 — Cleanup

**Goal:** Remove dead code and fix a startup warning.

- Remove `simple_hmm` entry from `shiny_app/app_config.yaml`
- Delete `shiny_app/modules/simple_hmm.py`
- Delete orphaned `shiny_app/modules/model2.py`
- Fix `page_navbar(bg=..., inverse=...)` → `navbar_options=NavbarOptions(bg=..., inverse=...)` in `shiny_app/app.py` to resolve the Shiny v1.3 deprecation warning on startup

---

## Section 2 — Shared Generic Module for HMM / HSMM / MS-GARCH

**Goal:** Replace three near-identical flat modules with one reusable module that matches the visual quality of Model 1 and SC-HMM.

### New file: `shiny_app/modules/generic_model_tab.py`

Exports the standard `model_tab_ui(cfg)` / `model_tab_server(input, output, session, cfg, project_root)` pair.

**UI layout:**
- Sticky sidebar (190px):
  - Anchor navigation links: Metrics, Cumulative Returns, Regime Timeline, Regime Characteristics, Transition Matrix
  - Model overview card (name + description from `cfg`)
- Scrollable main area with five labelled sections (using existing `section()` helper from `components/layout.py`):
  1. **Metrics** — formatted HTML table, color-coded: Sharpe green if ≥ 0.5, Max DD always red, others neutral
  2. **Cumulative Returns** — `plots/cumulative_returns.png`
  3. **Regime Timeline** — `plots/regime_timeline.png`
  4. **Regime Characteristics** — `plots/regime_characteristics.png`
  5. **Transition Matrix** — `plots/transition_matrix.png`
  - Each plot uses `img_tag()` from `components/charts.py` with graceful fallback

**Config entries updated** — `hmm`, `hsmm`, `msgarch` in `app_config.yaml` all point to `shiny_app.modules.generic_model_tab`. An optional `description` field can be added per model for the overview card.

**Files deleted:** `shiny_app/modules/hmm.py`, `shiny_app/modules/hsmm.py`, `shiny_app/modules/msgarch.py`

### Metrics table formatting

`results.csv` columns for new models: `target_te`, `sharpe`, `ir_vs_market`, `max_drawdown`, `volatility`, `active_ret_vs_market` (after rename), `turnover`.

Render as an HTML table (not a raw DataFrame) with:
- Sharpe: green if ≥ 0.5, red if < 0
- Max DD: always red
- IR vs Market: green if > 0, red if < 0
- Other columns: no color

---

## Section 3 — Comparison Tab Fixes

**Goal:** Fix the TE selector being silently misleading, and add a market reference line for models without an EW column.

### TE selector note
Add a small italic note below the dropdown:
> *TE target applies to Model 1 only. Other models display their single pre-computed series.*

The dropdown stays functional — it still controls which Model 1 series is loaded.

### Benchmark line for new models
In `comparison_server.overlay_chart()`:
- After all model lines are plotted, plot a single dashed grey "Market" line using the `market` column from the first successfully loaded model's returns DataFrame, labeled "Market (S&P 500)"
- This replaces the existing per-model market line logic and gives all models a shared reference

---

## Section 4 — Config Panel in Model 1 Sidebar

**Goal:** Allow the user to tune SJM and Black-Litterman parameters from the browser and re-run the pipeline.

### UI changes in `shiny_app/modules/model1.py`

Add a "⚙ Parameters" section to the existing sticky sidebar, above the Run button, with four numeric inputs (always visible, not collapsible):

| Label | Config key | Default | Step |
|---|---|---|---|
| Jump Penalty (λ) | `sjm.jump_penalty` | 50.0 | 1 |
| Feature Sparsity (κ²) | `sjm.max_feats` | 9.5 | 0.5 |
| Risk Aversion (δ) | `black_litterman.risk_aversion` | 2.5 | 0.1 |
| Txn Cost (bps) | `black_litterman.transaction_cost_bps` | 5 | 1 |

Inputs are pre-populated from `config.yaml` on page load.

### Run logic changes in `shiny_app/modules/model1.py`

When "Run Model" is clicked:
1. Read current `config.yaml` using PyYAML
2. Override the four keys with the current input values
3. Write the modified config to `outputs/tmp_config.yaml`
4. Execute: `python main.py --config outputs/tmp_config.yaml` (instead of `python main.py`)
5. Existing async progress streaming and error log remain unchanged
6. On completion, delete `outputs/tmp_config.yaml`

`config.yaml` is **never modified**. If the user wants to persist new defaults, they edit `config.yaml` directly.

---

## Files Changed

| File | Change |
|---|---|
| `shiny_app/app.py` | Fix `page_navbar` deprecation |
| `shiny_app/app_config.yaml` | Remove `simple_hmm`; update `hmm`/`hsmm`/`msgarch` module path; add optional `description` field |
| `shiny_app/modules/model1.py` | Add config inputs + temp-config run logic |
| `shiny_app/modules/generic_model_tab.py` | New shared module (create) |
| `shiny_app/modules/simple_hmm.py` | Delete |
| `shiny_app/modules/model2.py` | Delete |
| `shiny_app/modules/hmm.py` | Delete |
| `shiny_app/modules/hsmm.py` | Delete |
| `shiny_app/modules/msgarch.py` | Delete |
| `shiny_app/components/comparison.py` | Add TE note; add market benchmark line for new models |

---

## Out of Scope

- Config panels for HMM/HSMM/MS-GARCH (no training scripts in repo)
- Simple-HMM outputs (no model code available)
- Changes to `src/` pipeline code
- New chart types beyond the four standard plots
