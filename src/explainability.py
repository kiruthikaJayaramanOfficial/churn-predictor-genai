import joblib
import numpy as np
import pandas as pd
import shap
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eda import load_and_clean_data, encode_features

def load_champion():
    model = joblib.load("models/champion_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

def get_shap_explainer(model):
    return shap.TreeExplainer(model)

def get_shap_for_customer(customer_df, model, feature_names):
    """Returns top 5 positive + top 5 negative SHAP features for one customer"""
    explainer = shap.TreeExplainer(model)
    customer_df = customer_df[feature_names]
    shap_values = explainer.shap_values(customer_df)

    # LightGBM binary: shap_values is array of shape (1, n_features)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # class 1 = churn
    else:
        sv = shap_values[0]

    feature_impacts = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sv,
        'feature_value': customer_df.values[0]
    }).sort_values('shap_value', key=abs, ascending=False)

    top_positive = feature_impacts[feature_impacts['shap_value'] > 0].head(5)
    top_negative = feature_impacts[feature_impacts['shap_value'] < 0].head(5)

    return {
        'top_positive': top_positive.to_dict('records'),
        'top_negative': top_negative.to_dict('records'),
        'all_impacts': feature_impacts
    }

def prepare_customer_input(input_dict, feature_names):
    """Convert raw form input → encoded DataFrame matching training features"""
    df_raw = pd.DataFrame([input_dict])

    # Binary encode
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in binary_cols:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].map({'Yes': 1, 'No': 0,
                                           'No phone service': 0,
                                           'No internet service': 0})

    df_raw['gender'] = df_raw['gender'].map({'Male': 1, 'Female': 0})

    # One-hot encode
    df_raw = pd.get_dummies(df_raw, columns=['InternetService', 'Contract', 'PaymentMethod'])

    # Add missing columns with 0
    for col in feature_names:
        if col not in df_raw.columns:
            df_raw[col] = 0

    return df_raw[feature_names]

if __name__ == "__main__":
    model, feature_names = load_champion()
    print("Features:", feature_names)
    print("Model loaded:", type(model).__name__)