import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import optuna
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eda import load_and_clean_data, encode_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load + prepare data ───────────────────────────────────────
def prepare_data():
    df = load_and_clean_data()
    df = encode_features(df)
    df = df.select_dtypes(include=[np.number])
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def log_metrics(y_test, y_pred, y_prob):
    return {
        "recall":    recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "auc_roc":   roc_auc_score(y_test, y_prob)
    }

# ── 1. Logistic Regression (3 runs) ──────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    for C in [0.1, 1.0, 10.0]:
        with mlflow.start_run(run_name=f"LogReg_C{C}"):
            model = LogisticRegression(C=C, class_weight='balanced',
                                       max_iter=1000, random_state=42)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]
            metrics = log_metrics(y_test, y_pred, y_prob)
            mlflow.log_params({"model": "LogisticRegression", "C": C})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"  LogReg C={C} → Recall: {metrics['recall']:.3f}")

# ── 2. Random Forest (3 runs) ─────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test):
    configs = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": 15},
    ]
    for cfg in configs:
        with mlflow.start_run(run_name=f"RF_n{cfg['n_estimators']}_d{cfg['max_depth']}"):
            model = RandomForestClassifier(**cfg, class_weight='balanced', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = log_metrics(y_test, y_pred, y_prob)
            mlflow.log_params({**cfg, "model": "RandomForest"})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"  RF n={cfg['n_estimators']} d={cfg['max_depth']} → Recall: {metrics['recall']:.3f}")

# ── 3. LightGBM (3 runs) ──────────────────────────────────────
def train_lightgbm(X_train, X_test, y_train, y_test):
    configs = [
        {"learning_rate": 0.1,  "scale_pos_weight": 3},
        {"learning_rate": 0.05, "scale_pos_weight": 3},
        {"learning_rate": 0.01, "scale_pos_weight": 5},
    ]
    for cfg in configs:
        with mlflow.start_run(run_name=f"LGB_lr{cfg['learning_rate']}_spw{cfg['scale_pos_weight']}"):
            model = lgb.LGBMClassifier(**cfg, n_estimators=300,
                                        random_state=42, verbose=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = log_metrics(y_test, y_pred, y_prob)
            mlflow.log_params({**cfg, "model": "LightGBM"})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"  LGB lr={cfg['learning_rate']} spw={cfg['scale_pos_weight']} → Recall: {metrics['recall']:.3f}")

# ── 4. XGBoost + Optuna (3 runs) ─────────────────────────────
def train_xgboost_optuna(X_train, X_test, y_train, y_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state":     42,
            "use_label_encoder": False,
            "eval_metric":      "logloss"
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        return recall_score(y_test, y_pred)

    print("  Running Optuna (50 trials) ...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    best = study.best_params

    # Log 3 runs: best params + 2 slight variations
    for i, spw_mult in enumerate([1.0, 1.2, 0.8]):
        with mlflow.start_run(run_name=f"XGB_Optuna_v{i+1}"):
            params = {**best,
                      "scale_pos_weight": scale_pos_weight * spw_mult,
                      "use_label_encoder": False,
                      "eval_metric": "logloss",
                      "random_state": 42}
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = log_metrics(y_test, y_pred, y_prob)
            mlflow.log_params({**params, "model": "XGBoost_Optuna", "optuna_trial": i+1})
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, "model")
            print(f"  XGB Optuna v{i+1} spw_mult={spw_mult} → Recall: {metrics['recall']:.3f}")

            # Save best model (v1)
            if i == 0:
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, "models/champion_model.pkl")
                joblib.dump(list(X_train.columns), "models/feature_names.pkl")
                print("  ✅ Champion model saved to models/champion_model.pkl")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    mlflow.set_experiment("churn_predictor")

    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    print("\n[1/4] Training Logistic Regression...")
    train_logistic_regression(X_train, X_test, y_train, y_test)

    print("\n[2/4] Training Random Forest...")
    train_random_forest(X_train, X_test, y_train, y_test)

    print("\n[3/4] Training LightGBM...")
    train_lightgbm(X_train, X_test, y_train, y_test)

    print("\n[4/4] Training XGBoost + Optuna...")
    train_xgboost_optuna(X_train, X_test, y_train, y_test)

    print("\n✅ All 12 runs complete! Run: mlflow ui")