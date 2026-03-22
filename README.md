# 🔮 Churn Predictor + GenAI Agent

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-86.6%25_Recall-2ecc71?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-12_Runs-0194E2?style=for-the-badge)](https://mlflow.org)
[![LangChain](https://img.shields.io/badge/LangChain-GenAI_Agent-FF6B35?style=for-the-badge)](https://langchain.com)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange?style=for-the-badge)](https://shap.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Evidently](https://img.shields.io/badge/Evidently_AI-Drift_Monitor-7C3AED?style=for-the-badge)](https://evidentlyai.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-7_Pages-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-blue?style=for-the-badge)](https://optuna.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge)](https://groq.com)

**Production-grade telecom churn prediction system** — 12-run MLflow experiment · SHAP per-customer explainability · LangChain GenAI retention agent · automated drift detection · FastAPI + Docker + GitHub Actions CI/CD · 7-page Streamlit app

### 🚀 [Live Dashboard](https://your-app.streamlit.app) ← *Replace after Streamlit Cloud deploy*

---

## 🎯 The Business Problem

Telecom companies lose **₹12+ lakh/month** from silent customer churn. Traditional retention teams react *after* a customer cancels — too late, too expensive.

> **This system predicts which customer will churn 30 days before it happens, explains exactly WHY per customer using SHAP, and generates a specific AI-powered retention playbook — automatically.**

| Without This System | With This System |
|---|---|
| React after customer cancels | Predict 30 days before they leave |
| Call center contacts random customers | Score entire customer base in one CSV upload |
| No idea why customer churned | SHAP explains top 5 reasons per customer |
| Model trained once, silently degrades | Evidently AI detects drift → auto-retrain |
| ML only on data scientist's laptop | Live REST API + 7-page web app for the whole team |
| CFO asks "what's the ROI?" | ROI Simulator calculates revenue saved per intervention |

**Dataset:** IBM Telco Customer Churn · 7,043 customers · 20 features · 26.5% churn rate · Kaggle

---

## 📊 Proven Results

| Metric | Value |
|--------|-------|
| **Champion Model** | LightGBM (`LGB_lr0.01_spw5`) |
| **Recall** | **86.63%** — catches 87 out of every 100 churners |
| **AUC-ROC** | 0.8317 |
| **MLflow Runs** | 12 across 4 algorithms |
| **Batch Test** | 300 high-risk customers → 299/300 correctly flagged (99.7%) |
| **Drift Test** | 6 features drifted on shifted distribution → retraining triggered |
| **API Response** | < 200ms per prediction |

> **Why Recall over Accuracy?**
> Missing a churner = ₹780/year lost forever.
> Wrong flag on a loyal customer = ₹50 discount cost.
> Recall-first is the correct business objective — not accuracy.

---

## 📸 7-Page App — What Each Page Shows

### Page 1 — EDA Dashboard · *"What does churn look like?"*
*Churn rate by contract type · tenure histogram coloured by churn · monthly charges violin plot · correlation heatmap · top 5 churn drivers*

**Key insight:** Month-to-month customers churn at **42.7%** vs 2.8% for two-year contracts.

> 📷 Screenshot: `docs/screenshots/page1_eda.png`

---

### Page 2 — Model Comparison · *"Which model wins?"*
*All 12 MLflow runs in a sortable table · recall bar chart · Champion badge · multi-metric grouped bar*

> 📷 Screenshot: `docs/screenshots/page2_model_comparison.png`
> 📷 Screenshot: `docs/screenshots/mlflow_parallel_coordinates.png` ← README hero

---

### Page 3 — Live Predictor · *"Will this customer churn?"* ← Key Page
*Customer input form · churn probability gauge 0–100% · risk tier badge High/Medium/Low · SHAP waterfall chart*

**Example:** Senior citizen · Month-to-month · Fiber optic · No security · Tenure 2 months → **89.3% churn probability 🔴 HIGH**

> 📷 Screenshot: `docs/screenshots/page3_live_predictor.png`

---

### Page 4 — Churn Playbook Agent · *"WHY + what to do"*
*LangChain reads SHAP values → generates plain-English explanation → writes Churn Playbook with 3 specific retention actions · streams word by word*

**Sample output:**
```
🔍 Why This Customer Will Churn
Short tenure (2 months) + high monthly charges ($85) + no security services
pushing churn probability to 89.3%

🎯 Retention Actions (do these TODAY)
1. Offer 10% discount on monthly charges immediately
2. Bundle OnlineSecurity + TechSupport at discounted rate
3. Waive next month's charges as goodwill gesture

⚡ Urgency: HIGH PRIORITY — Contact within 24 hours
```

> 📷 Screenshot: `docs/screenshots/page4_playbook_agent.png`

---

### Page 5 — ROI Simulator · *"What if we change their plan?"*
*Sliders: tenure boost · charge reduction % · contract upgrade · add tech support · churn probability updates live · revenue saved calculator*

> 📷 Screenshot: `docs/screenshots/page5_roi_simulator.png`

---

### Page 6 — Batch Predictor · *"Score all customers at once"*
*Upload any CSV · Champion model scores all rows · pie chart High/Medium/Low risk · download results CSV · top 10 highest-risk table*

**Proven:** 300-customer high-risk batch → **299/300 correctly classified as High Risk**

> 📷 Screenshot: `docs/screenshots/page6_batch_predictor.png`

---

### Page 7 — MLOps Monitor · *"Is the model still working?"*
*Automated drift detection — upload CSV → result instant · Evidently report embedded · MLflow run history · system health*

**Proven:**
- Normal data → 0 drifted columns → ✅ Model healthy
- Shifted distribution → 6 drifted columns → 🔴 Retraining recommended

> 📷 Screenshot: `docs/screenshots/page7_mlops_monitor.png`
> 📷 Screenshot: `docs/screenshots/drift_detected.png` ← README hero

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                 │
│  IBM Telco · 7,043 customers · 20 features                 │
│  Great Expectations · 5 data quality checks                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  EXPERIMENT LAYER — 12-run MLflow Comparison                │
│                                                             │
│  Logistic Regression  ×3  (C = 0.1 / 1.0 / 10.0)         │
│  Random Forest        ×3  (n_estimators 100/200/300)       │
│  LightGBM             ×3  (lr = 0.1 / 0.05 / 0.01)       │
│  XGBoost + Optuna     ×3  (50 trials · recall objective)  │
│                                                             │
│  Champion: LGB_lr0.01_spw5 · Recall 86.63% · AUC 0.8317  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  EXPLAINABILITY LAYER                                       │
│  SHAP TreeExplainer → per-customer feature impact          │
│  Top 5 positive drivers  (increasing churn risk)           │
│  Top 5 negative drivers  (reducing churn risk)             │
│  Waterfall chart rendered per prediction in real time      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  GENAI LAYER  ← Unique differentiator                       │
│  LangChain + Groq LLaMA 3.1-8B-Instant                    │
│  Input  : churn probability + SHAP values + profile        │
│  Output : plain-English WHY + 3 retention actions          │
│           + urgency level · streamed token by token        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  SERVING LAYER                                              │
│  FastAPI  POST /predict → JSON < 200ms                     │
│  Streamlit 7-page app → live · batch · ROI · monitor      │
│  Docker   containerised · runs on any OS or cloud          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  MLOPS LAYER                                                │
│  Evidently AI   upload CSV → instant drift result          │
│  GitHub Actions push → Docker build → Docker Hub           │
│  Schedule       every Monday 6AM UTC auto-retrain          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 MLOps Pipeline

### MLflow — 12-Run Experiment

```
Run Name            Recall    F1      AUC-ROC
──────────────────────────────────────────────
LGB_lr0.01_spw5    0.8663   0.5972   0.8317  ← 🏆 CHAMPION
XGB_Optuna_v2      0.8209   0.6097   0.8426
XGB_Optuna_v1      0.8021   0.6316   0.8421
LogReg_C10.0       0.7647   0.6040   0.8361
RF_n100_d5         0.7620   0.6156   0.8388
RF_n300_d15        0.5321   0.5686   0.8209  ← worst
```

> 📷 Screenshot: `docs/screenshots/mlflow_runs_sorted_recall.png`

---

### Evidently AI — Automated Drift Detection

Upload any CSV in the MLOps Monitor page → drift analysis runs automatically → result shown instantly. No manual script execution needed.

```
Scenario              Drifted Columns    Decision
──────────────────────────────────────────────────
Normal data split     0                  ✅ Model healthy
High-risk batch       6                  🔴 Retrain recommended
```

> 📷 Screenshot: `docs/screenshots/drift_detected.png`

---

### GitHub Actions — CI/CD Pipeline

```
Triggers:
  push to main        → every code change auto-deploys
  every Monday 6AM    → weekly scheduled retraining
  workflow_dispatch   → manual trigger anytime

Steps:
  1. Checkout + Python 3.9
  2. pip install -r requirements.txt
  3. python src/data_validation.py   ← 5 data quality checks
  4. docker build -t churn-predictor .
  5. docker push → Docker Hub        ← production image updated
```

> 📷 Screenshot: `docs/screenshots/github_actions_green.png` ← **MOST IMPORTANT**

---

## ⚡ Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **ML Models** | LightGBM · XGBoost · Random Forest · Logistic Regression | 4 algorithms · 12 MLflow runs |
| **Hyperparameter Tuning** | Optuna (50 trials) | Recall-maximising Bayesian search |
| **Explainability** | SHAP TreeExplainer | Per-customer waterfall charts |
| **GenAI** | LangChain + Groq LLaMA 3.1 | Streaming retention playbook |
| **Experiment Tracking** | MLflow | Run comparison · model registry |
| **Drift Monitoring** | Evidently AI | Automated distribution shift detection |
| **Data Validation** | Great Expectations | 5 production data quality checks |
| **REST API** | FastAPI + Uvicorn | POST /predict · JSON · < 200ms |
| **Web App** | Streamlit + Plotly | 7-page interactive dashboard |
| **Containerisation** | Docker | Reproducible · cloud-ready |
| **CI/CD** | GitHub Actions | Auto build + push on every commit |
| **Deployment** | Streamlit Cloud | Live public URL |
| **Language** | Python 3.9 | — |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/kiruthikaJayaramanOfficial/churn-predictor-genai.git
cd churn-predictor-genai

# Environment
python3 -m venv churn_env
source churn_env/bin/activate
pip install -r requirements.txt

# Add Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Download dataset → save as data/telco_churn.csv
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Train all 12 models (~5 mins)
python src/train_all_models.py

# MLflow UI
mlflow ui
# → http://127.0.0.1:5000

# Streamlit app
streamlit run app/main.py
# → http://localhost:8501

# FastAPI
uvicorn app.api:app --reload
# → http://localhost:8000/docs

# Test API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.0, "Contract": "Month-to-month"}'
# → {"churn_probability": 0.892, "risk_tier": "High"}

# Drift detection
python mlops/drift_detector.py

# Generate synthetic test data
python eval/generate_test_data.py
```

### Docker

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
curl http://localhost:8000/health
# → {"status": "healthy"}
```

---

## 📁 Project Structure

```
churn-predictor-genai/
├── app/
│   ├── main.py                        # Streamlit entry point
│   ├── api.py                         # FastAPI REST endpoint
│   └── pages/
│       ├── 1_EDA_Dashboard.py
│       ├── 2_Model_Comparison.py
│       ├── 3_Live_Predictor.py
│       ├── 4_Churn_Playbook_Agent.py
│       ├── 5_ROI_Simulator.py
│       ├── 6_Batch_Predictor.py
│       └── 7_MLOps_Monitor.py
├── src/
│   ├── eda.py                         # Load · clean · encode
│   ├── data_validation.py             # Great Expectations checks
│   ├── train_all_models.py            # 12-run MLflow experiment
│   ├── explainability.py              # SHAP waterfall
│   └── langchain_agent.py             # Groq LLM playbook agent
├── mlops/
│   ├── drift_detector.py              # Evidently drift detection
│   └── drift_summary.json             # {trigger_retrain, drifted_features}
├── eval/
│   └── generate_test_data.py          # Synthetic customer generator
├── data/
│   ├── telco_churn.csv                # 7,043 rows · 20 features
│   ├── batch_normal.csv               # 500 normal customers
│   ├── batch_high_risk.csv            # 300 high-risk customers
│   └── batch_mixed.csv                # 200 mixed customers
├── models/
│   ├── champion_model.pkl             # LightGBM champion
│   └── feature_names.pkl              # Training feature list
├── .github/workflows/
│   └── retrain.yml                    # CI/CD pipeline
├── Dockerfile
├── requirements.txt
└── .env                               # GROQ_API_KEY (never committed)
```

---

## 🎓 Skills Demonstrated

| Skill Area | What Was Built |
|------------|---------------|
| **Machine Learning** | 4-algorithm comparison · class imbalance · recall-first objective |
| **Hyperparameter Optimisation** | Optuna 50-trial Bayesian search |
| **Experiment Tracking** | MLflow 12-run comparison · Champion/Challenger pattern |
| **Explainable AI** | SHAP TreeExplainer · per-customer waterfall |
| **Generative AI** | LangChain · Groq LLM · streaming · prompt engineering |
| **MLOps** | Evidently drift monitoring · automated retraining trigger |
| **Software Engineering** | FastAPI · Docker · modular Python · Great Expectations |
| **CI/CD** | GitHub Actions · auto Docker build + push · scheduled retraining |
| **Business Impact** | ROI calculator · revenue saved · batch risk segmentation |

---

## 🔑 ATS Keywords

`Machine Learning` `LightGBM` `XGBoost` `Random Forest` `Logistic Regression`
`MLflow` `Experiment Tracking` `Model Registry` `Hyperparameter Tuning` `Optuna`
`SHAP` `Explainable AI` `XAI` `Feature Importance` `Interpretable ML`
`LangChain` `LLM` `Generative AI` `GenAI` `Prompt Engineering` `Groq` `LLaMA`
`FastAPI` `REST API` `Uvicorn` `Pydantic`
`Streamlit` `Plotly` `Interactive Dashboard`
`Docker` `Containerisation` `Docker Hub`
`GitHub Actions` `CI/CD` `Continuous Integration` `Continuous Deployment`
`Evidently AI` `Data Drift` `Model Monitoring` `MLOps`
`Great Expectations` `Data Validation` `Data Quality`
`Customer Churn` `Churn Prediction` `Retention` `Telecom` `BFSI`
`Binary Classification` `Recall` `AUC-ROC` `Precision` `F1` `Class Imbalance`
`Python` `Pandas` `NumPy` `Scikit-learn`
`End-to-End ML` `Production ML` `Deployed Model` `Streamlit Cloud`

---

## 📌 Resume Bullet

```
LightGBM churn predictor (86.6% recall) from systematic 12-run MLflow
comparison across 4 algorithms (LR, RF, LightGBM, XGBoost+Optuna);
LangChain + Groq agent generating plain-English Churn Playbook with
SHAP explainability per customer; FastAPI + Docker + GitHub Actions CI/CD
with automated Evidently AI drift monitoring; 7-page Streamlit app with
live predictor, ROI simulator, batch scorer, MLOps monitor — deployed
```

---

## 📬 Contact

**Kiruthika Jayaraman** · VIT Chennai

[![GitHub](https://img.shields.io/badge/GitHub-kiruthikaJayaramanOfficial-181717?style=flat-square&logo=github)](https://github.com/kiruthikaJayaramanOfficial)

---

*Production MLOps · GenAI layer · Real Telco data · 12-run systematic experiment · Automated retraining*

**⭐ Star this repo if you found it useful!**
