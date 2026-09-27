# 🌱  Snap Greenhouse

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-batch-orange)
![River](https://img.shields.io/badge/River-online-red)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

<p align="center">
  <img src="plan_greenhouse.jpg" alt="Greenhouse U5 layout" width="40%">
  &nbsp;&nbsp;
  <img src="viz_sensor.jpg" alt="Sensor data visualization" width="40%">
</p>

🚧 todo:
. data origin expl: sensor collection

The goal is to optimize energy management (heating, ventilation) by anticipating thermal drift, by forecasting the indoor temperature (`tempint`) of a connected greenhouse U5 from sensor data. To do this, we compare different ways of predicting temperature and humidity — this first version focuses on temperature only. The project compares two inference paradigms: Batch, lot-based forecasting, and Online, learning on the fly on real data.

## Overview

**Batch vs Online** — each test window = <=24 h :

![Batch vs online comparison](results/figures/batch_vs_river.gif)

**Adaptation to a new change** — a regime shift injected mid-stream: the batch model (frozen) stays off, the online model re-learns and returns to its pre-shock error level:

![Online vs batch facing a sudden change](results/figures/online_adapts_drift.gif)

> Reproducible: `python results/make_comparison_gif.py` · `python results/make_drift_gif.py`


## 1. Context

A greenhouse is a non-stationary environment: seasons, outdoor weather, structural
ageing, crops, sensor failures. A model trained once and frozen drifts out of date — which is where online learning earns its place.

Objective: compare the two approaches for forecasting `tempint`:

- Batch — trained once on history, then frozen, lot-based prediction
- Online — updated on each new sensor reading, predict → learn

The trade-off: online is generally less accurate than batch on a stable regime, but more robust over the long run in a living environment.

### Methodology

**Validation**
- 2 test windows of 24h per month (week-2 and week-4);
- train = all history prior to the window.
- script: `src/cv.py`

**Feature engineering**
- calendar: hour, day, month (sine/cosine encoding), weekend
- lags: `tempint` shifted by 24 h, 48 h…
- rolling: moving mean / std (24 h, 48 h…)
- trend: differences between consecutive lags
- script: `src/features.py`, driven by `config.yaml`

**Evaluated models**
- baseline: `naive_24h`, ŷ(t) = temperature 24 h earlier
- batch: LinearRegression, Ridge, Lasso, Ridge_Poly2, RandomForest, LightGBM
- online: River (incremental LinearRegression, SGD / Adam / RMSProp).


## 2. Usage

### Architecture

The fundamental difference is the model update flow.

#### Batch, frozen between scheduled retrainings

```mermaid
flowchart LR
    S1[IoT sensors] --> H[(History<br/>CSV / TSDB)]
    H -->|cron: every 6 h| FE1[Feature engineering<br/>lags · rolling · calendar]
    FE1 --> TR[Training<br/>sklearn / LightGBM]
    TR --> MF[Frozen model]
    MF --> P1[Day-ahead forecast<br/>by lot]
```

#### Online, learns continuously, sample by sample

```mermaid
flowchart LR
    S2[IoT sensors] -->|stream| FE2[Feature engineering<br/>on the fly]
    FE2 --> PR[predict_one<br/>→ instant forecast]
    PR --> LR[learn_one<br/>→ immediate update]
    LR -->|updated state| PR
```

### Project layout

```
snap_greenhouse/
├── config/config.yaml         # spec
├── datas/                     # hidden : sensor + external weather data (CSV)
├── notebooks/                 # xps + viz
├── src/                       # source code
│   ├── config.py              # loads config.yaml
│   ├── preprocess.py          # prepare_raw, GreenhousePipeline
│   ├── features.py            # create features
│   ├── cv.py                  # time cross validation
│   ├── model_factory.py       # builds models from the YAML
│   ├── river_model.py         # online model River
│   ├── evaluation.py          # evaluate_folds engine + 3 evaluators
│   ├── metrics.py             # compute_metrics (MAE/RMSE/R²/MAPE)
│   ├── journal.py             # experiment journal (JSON)
│   ├── results.py             # comparison table
│   ├── plots.py               # viz
│   └── runner.py              # orchestration of all models
├── outputs/                   # hidden : run journal + per-fold detail
├── results/                   # gifs + scripts
├── requirements-notebook.txt
└── README.md
```

### Stack

- Python ≥ 3.10
- scikit-learn, LightGBM, batch models
- River, online learning 
- Pandas, Matplotlib, visualizations & GIF, PillowWriter
- PyYAML config

### Installation & configuration

🚧 todo

```yaml
target:    tempint
... 
```

## 3. Results

### Data
🚧 todo: size, granularity, number of seasons covered

### Batch vs Online

| Criterion | Batch `Ridge_Poly2` | Online `River` (`river_sgd`) |
|---|---|---|
| Global MAE (°C) | **≈ 1.9** | **≈ 2.3** |
| Drift adaptation | ❌ frozen between retrainings | ✅ continuous |
| Inference latency | per lot | - |
| Model update | full retraining | incremental 1 sample |
| Compute cost | - | - |
| Memory footprint | history required | no history |

### Leaderboard

| Model | Type | MAE (°C) | RMSE (°C) | R² | vs baseline |
|---|---|---:|---:|---:|---:|
| Ridge_Poly2 | batch | **1.89** | 2.59 | **0.958** | −32.3 % |
| LightGBM | batch | 2.17 | 3.13 | 0.938 | −22.2 % |
| river_sgd | online | 2.32 | 3.10 | 0.939 | −17.1 % |
| RandomForest | batch | 2.36 | 3.41 | 0.926 | −15.5 % |
| river | online | 2.45 | 3.24 | 0.933 | −12.4 % |
| naive_24h | baseline | 2.79 | 4.51 | 0.870 | — |
| Lasso | batch | 2.90 | 3.98 | 0.899 | +3.7 % |
| Ridge | batch | 2.97 | 3.98 | 0.900 | +6.5 % |
| LinearRegression | batch | 2.98 | 3.98 | 0.899 | +6.6 % |

⚠️ **R² pitfall**: do not average the R² per fold, low variance over 24 h → unstable R², sometimes negative. The table above recomputes each metric on concatenated predictions across all folds <br>
script: `src/results.py > global_metrics`

🌊 **River fix**: the model never learned during the test window (frozen, like a batch model). Fixed to keep learning (`predict_one` → `learn_one`) throughout — MAE dropped from ≈ 3.44 to ≈ 2.32.

## 4. Roadmap

- [x] EDA, feature engineering, temporal CV
- [x] Batch models (sklearn / LightGBM) + baseline
- [x] Online model (River) + comparison
- [x] Experiment journal + global metrics (R²-pitfall-proof)
- [x] Visualizations (batch/online GIF, drift adaptation)
- [ ] Latency & compute-cost benchmark (fill the table with real measurements)
- [ ] Exogenous variables (external weather, solar radiation) — major accuracy lever
- [ ] Tune LightGBM (num_leaves, max_depth, min_child_samples, early stopping)
- [ ] Predict a target variant (residual vs baseline `tempint(t) − tempint(t−24h)`) — easier to learn, beats the naive baseline by design
- [ ] CLI packaging (`src/cli.py`) and online API (real-time MQTT ingestion)
- [ ] authors & license description
- [ ] usage description for this library
- [ ] predict humidity
---
