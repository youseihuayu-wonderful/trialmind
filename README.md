# TrialMind — An Agentic AI Pipeline for Clinical Trial Risk Modeling

> Built a modular Python-based agentic AI pipeline using synthetic clinical-operations
> data to process inputs, engineer risk features, train predictive models, generate
> explanations, and produce human-reviewable recommendations.

TrialMind ingests synthetic clinical-operations data, engineers operational risk
features, trains predictive models for **site risk** and **patient dropout**, explains
them, and produces human-reviewable recommendations and an executive summary.

It is **agentic by design**: deterministic data/ML work runs as plain Python; the
open-ended, natural-language steps (explanation, recommendation, executive summary) run
as LLM agents on **Claude Opus 4.8** (`claude-opus-4-8`), with a human-in-the-loop
review gate before any recommendation is acted on.

## Why this exists

Clinical trials are the most expensive, failure-prone stage of drug development. A single
day of delay can cost a sponsor \$0.6M–\$8M, and most delays trace back to the same
operational signals: enrollment shortfalls, under-performing sites, patient dropout,
protocol deviations, and data-quality backlogs. TrialMind predicts those risks *before*
they surface and translates the model output into language a clinical-operations team can
act on.

> All data is **synthetic** — no PHI, fully HIPAA-safe. This is a portfolio / research
> codebase, not a clinical product.

## Pipeline

```
raw clinical-operations data
   -> data-quality agent       validate completeness / anomalies      (Python)
   -> feature-engineering      derive operational risk features       (Python)
   -> site-risk model          classify per-site risk                 (Python / sklearn)
   -> dropout model            score per-patient dropout probability  (Python / sklearn)
   -> explainability agent     SHAP + LLM: "why is this site risky?"  (Claude Opus 4.8)
   -> recommendation agent     rule + model -> actionable advice      (Claude Opus 4.8)
   -> exec-summary agent       one-page stakeholder summary           (Claude Opus 4.8)
human review -> decision
```

## Engineered operational features

`enrollment_attainment`, `query_backlog`, `protocol_deviation_rate`, `site_capacity`,
`patient_travel_burden`, `prior_no_show_behavior`, `digital_engagement`, and more —
all computed from the synthetic relational data, never hard-coded.

## Data model

PostgreSQL-compatible relational schema (defaults to SQLite for zero-setup local runs;
set `DATABASE_URL` to a Postgres DSN for production-style runs). Core tables:

- `sites` — research sites (capacity, staffing, enrollment target)
- `patients` — enrolled patients (travel burden, no-show history, engagement; dropout label)
- `visits` — visit/event stream (attendance, protocol deviations, data queries)
- `site_features` — site-level aggregated features + risk label (derived)

See `trialmind/db/models.py`.

## Project layout

```
trialmind/
├── trialmind/
│   ├── config.py            # settings (DB URL, model id, paths)
│   ├── db/
│   │   ├── base.py          # engine + session factory
│   │   └── models.py        # SQLAlchemy ORM schema
│   ├── agents/
│   │   └── base.py          # BaseAgent abstract interface
│   ├── models/              # ML model code (site-risk, dropout)
│   └── orchestrator.py      # runs the agent pipeline end to end
├── scripts/
│   └── init_db.py           # create tables
├── tests/
└── data/                    # generated synthetic data / artifacts (gitignored)
```

## Setup

```bash
cd trialmind
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set ANTHROPIC_API_KEY for the LLM agents
python scripts/init_db.py     # create the database schema
```

## Dashboard

A Streamlit dashboard visualizes the risk posture — headline metrics, the ranked
high-risk sites with their SHAP drivers, and the patient-dropout outlook. It runs
without an API key (deterministic analysis); with a key configured, the LLM
agents can be triggered from the UI.

```bash
streamlit run trialmind/dashboard.py
```

## Tech

Python · SQLAlchemy · pandas · scikit-learn · SHAP · Streamlit · Anthropic Claude Opus 4.8

## Models & results

Two scikit-learn classifiers, each evaluated with ROC-AUC, PR-AUC, confusion
matrix, precision/recall/F1, and SHAP feature attributions (shared code in
`trialmind/models/evaluation.py`). Plots and `metrics.json` are written to
`data/artifacts/` for dashboarding.

| Model | Unit | ROC-AUC | PR-AUC | Validation |
|-------|------|--------:|-------:|------------|
| Site risk (`trialmind/models/site_risk.py`) | site | **0.93** | 0.88 | stratified 5-fold CV (150 sites) |
| Patient dropout (`trialmind/models/dropout.py`) | patient | **0.75** | 0.47 | held-out test split (~3k patients) |

```bash
python scripts/train_site_risk.py   # trains + evaluates the site-risk model
python scripts/train_dropout.py     # trains + evaluates the dropout model
```

Leakage is controlled deliberately: the site label comes from a hidden
`latent_quality` factor the model never sees, and the dropout model uses only
enrollment-time patient features — so the metrics reflect genuine signal, not
circular definitions.

## Status

End-to-end working: synthetic data generation → feature engineering → site-risk
and patient-dropout models (evaluation + SHAP) → three Claude Opus 4.8 agents
(explainability / recommendation / executive summary) → Streamlit dashboard.
The LLM agents require an Anthropic API key; everything else runs without one.
