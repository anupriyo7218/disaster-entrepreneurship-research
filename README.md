


## 🚀 Live Dashboard

**Interactive Streamlit App:** https://disaster-entrepreneurship-research-5rtxhyy6kwhrb879nfg5fb.streamlit.app/

# Do Natural Disasters Affect New Business Creation in the U.S.?

A fully reproducible empirical research pipeline combining U.S. Census
Business Formation Statistics with FEMA Disaster Declarations into a
state-month panel, analyzed with a two-way fixed effects regression, and
presented through an interactive dashboard. Built as a portfolio piece for
a PhD application in Sustainable Entrepreneurship and Innovation.

**Status:** all 5 modules complete. See [Results](#results) for the headline finding.

---

## Table of Contents

- [Research Question & Motivation](#research-question--motivation)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Dashboard](#running-the-dashboard)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Testing](#testing)
- [Key Design Decisions](#key-design-decisions)

---

## Research Question & Motivation

**Research question:** When a natural disaster is officially declared in a
U.S. state, is that state-month associated with a different level of new
business applications, compared to state-months without a declared disaster?

**Motivation:** Disasters are usually studied for their destructive costs —
property damage, displacement, fatalities. Less studied is what happens to
*entrepreneurial activity* in their aftermath: does disruption suppress new
business formation, or does rebuilding and recovery spending create a burst
of new business activity (construction, contracting, insurance-adjacent
services)? This sits at the intersection of sustainability, economic
resilience, and entrepreneurship — asking whether disaster exposure is a
headwind or, counterintuitively, a catalyst for local business creation.

This project treats the question as an **observational, correlational**
empirical exercise, not a causal claim — see [Limitations](#limitations).

---

## Data Sources

| Dataset | Source | Granularity | Coverage | License / Access |
|---|---|---|---|---|
| **Business Formation Statistics (BFS)** | U.S. Census Bureau | State, monthly | Jul 2004 – present | Public domain, `census.gov/econ/bfs` |
| **Disaster Declarations Summaries** | FEMA / OpenFEMA v2 | County (aggregated to state), event-based | 1953 – present | Public domain, `fema.gov/openfema-data-page` |

**Series used from BFS:** `BA_BA` (core Business Applications), not
seasonally adjusted, `naics_sector = TOTAL`, filtered to the 50 states + DC.

**Measure used from FEMA:** distinct disaster declarations
(`disasterNumber`) per state per month, using `incidentBeginDate` as the
timing anchor (not the administratively-lagged `declarationDate`).

Raw files are expected in `data/raw/` (gitignored — not committed):
- `bfs_monthly.csv`
- `DisasterDeclarationsSummaries.csv`
- `month_date_table.csv`

---

## Methodology

1. **Load & validate** both raw sources without modification (Module 1) —
   confirms shape, schema, and structural integrity before any cleaning.
2. **Clean, reshape, and merge** (Module 2) — BFS is filtered and melted
   from wide to long; FEMA's county-level records are deduplicated to
   distinct state-level events; both are merged into a balanced
   state × month panel, with `disaster_count = 0` filled in for
   no-disaster months (a real zero, not missing data).
3. **Explore descriptively** (Module 3) — distributions, trends, and a raw
   disaster-vs-no-disaster comparison, explicitly before any model is fit.
4. **Estimate a two-way fixed effects panel regression** (Module 4):

   ```
   business_applications_it = β · disaster_count_it + state_i + month_year_t + ε_it
   ```

   State fixed effects absorb permanent differences between states (size,
   industry mix); month-year fixed effects absorb nationwide shocks common
   to a given month (recessions, seasonality, policy). Standard errors are
   clustered by state. Estimated via `linearmodels.PanelOLS`.
5. **Present interactively** (Module 5) — a Streamlit dashboard covering
   the full pipeline from raw data to regression output, written for both
   a technical and non-technical audience.

---

## Repository Structure

```
disaster-entrepreneurship-research/
├── data/
│   ├── raw/                    # original source files (gitignored)
│   └── processed/               # cleaned/merged panel data (gitignored)
├── src/                         # production pipeline code
│   ├── config_loader.py
│   ├── logging_setup.py
│   ├── data_loader.py           # Module 1
│   ├── preprocessing.py         # Module 2
│   ├── eda.py                   # Module 3
│   └── modeling.py              # Module 4
├── streamlit_app/                # Module 5 — interactive dashboard
│   ├── Home.py                   # Project Overview page
│   ├── utils.py                  # shared cached data loaders
│   └── pages/
│       ├── 1_Dataset_Explorer.py
│       ├── 2_Exploratory_Data_Analysis.py
│       ├── 3_Regression_Results.py
│       └── 4_Conclusions.py
├── outputs/
│   ├── figures/                  # generated plots (EDA + regression diagnostics)
│   ├── tables/                   # generated CSV tables
│   └── reports/                  # regression summaries & interpretation text
├── notebooks/                     # exploratory notebooks (ad hoc analysis)
├── paper/                         # arXiv-ready manuscript (future work)
├── tests/                         # pytest suite (54 tests)
├── config/
│   ├── config.yaml                # dataset schema, filters, paths, model spec
│   └── logging_config.yaml
├── logs/                           # runtime logs (gitignored)
├── models/                         # reserved for saved model artifacts
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone <repository-url>
cd disaster-entrepreneurship-research
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the three raw CSV files (see [Data Sources](#data-sources)) into
`data/raw/`.

---

## Running the Pipeline

Run modules in order — each depends on the previous one's output.

### Module 1 — Raw data load & validation
```bash
python -m src.data_loader
```
Loads both raw files, logs shape/columns/dtypes/missing values, and
validates structural integrity. **No cleaning happens here** — this is
intentionally a side-effect-free sanity check of the raw inputs.

### Module 2 — Preprocessing & merge
```bash
python -m src.preprocessing
```
Cleans both datasets, builds the FEMA state-month disaster panel, merges
with BFS, and validates the result. Writes to `data/processed/`:
`bfs_clean.csv`, `fema_clean.csv`, `disaster_panel.csv`,
`final_analysis_dataset.csv`, `preprocessing_summary.json`.

### Module 3 — Exploratory data analysis
```bash
python -m src.eda
```
Purely descriptive — no model fitting. Writes 13 figures to
`outputs/figures/` and 7 tables to `outputs/tables/` covering dataset
overview, business application distributions/trends, FEMA disaster
patterns, and a raw disaster-vs-no-disaster comparison.

### Module 4 — Econometric analysis
```bash
python -m src.modeling
```
Fits the two-way fixed effects regression described in
[Methodology](#methodology). Writes regression/diagnostics tables to
`outputs/tables/`, the full summary and auto-generated plain-language
interpretation to `outputs/reports/`, and 5 diagnostic figures to
`outputs/figures/`.

---

## Running the Dashboard

```bash
streamlit run streamlit_app/Home.py
```

Opens an interactive dashboard at `http://localhost:8501` with 5 pages
(sidebar navigation): **Project Overview**, **Dataset Explorer**,
**Exploratory Data Analysis** (interactive Plotly charts with state and
year-range filters), **Regression Results** (plain-language explanation of
the coefficient, p-value, and confidence interval), and **Conclusions**.

The dashboard reads directly from `data/processed/` and `outputs/` — run
Modules 2–4 first, or each page will show a friendly message pointing to
the exact command needed to generate its data.

---

## Results

On the full panel (**51 states, 264 months, 13,464 observations**, Jul
2004 – Jun 2026):

| Statistic | Value |
|---|---|
| Coefficient (`disaster_count`) | **-110.97** |
| Clustered std. error (by state) | 70.14 |
| t-statistic | -1.58 |
| p-value | 0.1136 |
| 95% confidence interval | [-248.47, 26.52] |
| Statistically significant at 5%? | **No** |

**Interpretation:** holding each state's baseline characteristics and each
month's nationwide conditions fixed, one additional disaster declaration in
a state-month is associated with roughly 111 fewer business applications
that state-month — but the confidence interval crosses zero, so this
analysis **cannot rule out no association at all**. This null result is
consistent with the weak raw correlation (Pearson r ≈ 0.11) already
observed in the exploratory analysis, before controlling for confounds.

**This is an association, not evidence of a causal effect.** See
[Limitations](#limitations).

---

## Limitations

- **Observational design** — no random assignment of disasters to states;
  fixed effects rule out a specific class of confounds (time-invariant
  state traits, nationwide monthly shocks) but not all of them.
- **FEMA's disaster count is administrative, not purely physical** — it
  reflects formal federal declarations, which involve a request/review
  process on top of physical severity.
- **State-month granularity** — a disaster affecting one county is treated
  the same as one affecting many counties within the same state.
- **No lag structure** — the model only tests same-month association; a
  delayed effect (e.g., rebuilding-driven business formation months later)
  would not be captured.
- **Coverage boundary** — FEMA history goes back to 1953, but BFS only
  starts mid-2004, so the analysis is limited to that overlap.
- **Single regressor** — no controls for other time-varying state-level
  factors (state policy shifts, local economic shocks) beyond what month
  fixed effects absorb nationally.

---

## Future Improvements

- Add a lagged/distributed-lag specification to detect delayed effects.
- Incorporate disaster severity (declared damage cost, declaration type)
  rather than a simple count.
- Disaggregate by NAICS industry sector — construction and retail likely
  respond differently than professional services.
- Move to county-level analysis if a compatible county-level business
  formation measure becomes available.
- Explore an event-study design plotting pre/post-disaster trends rather
  than a single pooled coefficient.
- Investigate instruments based on purely physical hazard measures (e.g.,
  NOAA wind speed, rainfall totals) to move toward causal identification.
- Publish the accompanying paper (see `paper/`, planned).

---

## Testing

```bash
pytest tests/ -v
```

**54 tests** across all 5 modules:
- Module 1: raw load & structural validation
- Module 2: cleaning, deduplication, merge logic (synthetic + real-data assumption checks)
- Module 3: EDA computation functions (stats, grouping, correlation)
- Module 4: TWFE model fit against a synthetic panel with a *known engineered
  coefficient* (confirms the fixed-effects logic actually recovers the true
  effect, not just "runs without error")
- Module 5: dashboard data-loading utilities

The Module 4 regression coefficient has additionally been independently
cross-checked against a completely different estimation method (explicit
dummy-variable OLS via `statsmodels`, rather than `linearmodels`' effect
absorption) — both methods agree to 11 decimal places.

---

## Key Design Decisions

- **BFS filters:** `naics_sector = TOTAL`, `series = BA_BA`,
  `sa = U` (not seasonally adjusted, to preserve monthly shock timing).
- **FEMA date field:** `incidentBeginDate`, not `declarationDate`, to avoid
  the administrative lag between an event and its federal declaration.
- **FEMA aggregation:** deduplicated on `(disasterNumber, state)` — each
  `disasterNumber` maps to exactly one state (verified empirically), with
  the row-level fan-out in the raw file occurring purely within-state,
  across counties.
- **Geography reconciliation:** FEMA includes 7 U.S. territories/freely-
  associated-states (`AS, FM, GU, MH, MP, PW, VI`) absent from the BFS
  panel; excluded during merge. Analysis is scoped to 50 states + DC.
- **Standard errors clustered by state**, not heteroskedasticity-robust
  alone, to account for serial correlation within a state's residuals
  over time.

See `config/config.yaml` for the full, versioned record of these decisions.

---

## License

TBD.
