import pandas as pd
import numpy as np

def load_and_clean_data(path="data/telco_churn.csv"):
    df = pd.read_csv(path)
    
    # Fix TotalCharges blank strings → 0, cast to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Encode Churn as 0/1
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def compute_churn_by_contract(df):
    return df.groupby('Contract')['Churn'].mean().reset_index()

def compute_class_imbalance(df):
    churn_pct = df['Churn'].mean() * 100
    print(f"Churn rate: {churn_pct:.1f}%")
    return churn_pct

def encode_features(df):
    # Binary Yes/No columns → 0/1
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})
    
    # Gender → 0/1
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=False)
    
    # Drop customerID
    df = df.drop(columns=['customerID'], errors='ignore')
    
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    compute_class_imbalance(df)
    print(compute_churn_by_contract(df))