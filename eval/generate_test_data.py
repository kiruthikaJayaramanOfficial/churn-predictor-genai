import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

def generate_customers(n):
    data = {
        'customerID': [f'CUST-{i:04d}' for i in range(n)],
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'tenure': np.random.randint(0, 72, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n, p=[0.44, 0.34, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n),
        'MonthlyCharges': np.round(np.random.uniform(18, 120, n), 2),
    }
    df = pd.DataFrame(data)
    df['TotalCharges'] = np.round(df['MonthlyCharges'] * df['tenure'] * np.random.uniform(0.9, 1.1, n), 2)
    return df

# ── Generate 3 datasets ───────────────────────────────────────

# 1. Normal batch (similar to training data)
normal = generate_customers(500)
normal.to_csv('data/batch_normal.csv', index=False)
print("✅ data/batch_normal.csv — 500 normal customers")

# 2. High-risk batch (drift simulation — shifted distribution)
high_risk = generate_customers(300)
# Inject drift: more month-to-month, higher charges, shorter tenure
high_risk['Contract'] = np.random.choice(
    ['Month-to-month', 'One year', 'Two year'], 300, p=[0.85, 0.10, 0.05])
high_risk['MonthlyCharges'] = np.round(np.random.uniform(75, 120, 300), 2)
high_risk['tenure'] = np.random.randint(0, 12, 300)
high_risk['InternetService'] = np.random.choice(
    ['Fiber optic', 'DSL', 'No'], 300, p=[0.80, 0.15, 0.05])
high_risk['OnlineSecurity'] = np.random.choice(
    ['Yes', 'No', 'No internet service'], 300, p=[0.05, 0.90, 0.05])
high_risk['TechSupport'] = np.random.choice(
    ['Yes', 'No', 'No internet service'], 300, p=[0.05, 0.90, 0.05])
high_risk['TotalCharges'] = np.round(high_risk['MonthlyCharges'] * high_risk['tenure'], 2)
high_risk.to_csv('data/batch_high_risk.csv', index=False)
print("✅ data/batch_high_risk.csv — 300 high-risk customers (drifted)")

# 3. Mixed batch (200 customers)
mixed = generate_customers(200)
mixed.to_csv('data/batch_mixed.csv', index=False)
print("✅ data/batch_mixed.csv — 200 mixed customers")

print("\nFiles ready for testing:")
print("  • Upload batch_normal.csv to Batch Predictor")
print("  • Upload batch_high_risk.csv to Batch Predictor (expect 70%+ high risk)")
print("  • Use batch_high_risk.csv for drift detection test")