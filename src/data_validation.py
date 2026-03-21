import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eda import load_and_clean_data

def run_validations(df):
    results = []

    def check(name, condition, passed):
        status = "✅ PASS" if passed else "❌ FAIL"
        results.append({"check": name, "status": status})
        print(f"{status} — {name}")

    # 1. tenure >= 0
    check("tenure >= 0", None, (df['tenure'] >= 0).all())

    # 2. MonthlyCharges > 0
    check("MonthlyCharges > 0", None, (df['MonthlyCharges'] > 0).all())

    # 3. Churn is 0 or 1 (already encoded)
    check("Churn in [0, 1]", None, df['Churn'].isin([0, 1]).all())

    # 4. No nulls in customerID
    check("No nulls in customerID", None, df['customerID'].isnull().sum() == 0)

    # 5. TotalCharges >= 0
    check("TotalCharges >= 0", None, (df['TotalCharges'] >= 0).all())

    print(f"\n{sum('PASS' in r['status'] for r in results)}/5 checks passed")
    return results

if __name__ == "__main__":
    df = load_and_clean_data()
    run_validations(df)