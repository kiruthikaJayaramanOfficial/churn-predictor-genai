import pandas as pd
import numpy as np
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eda import load_and_clean_data, encode_features

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric

def run_drift_detection():
    print("Loading data...")
    df = load_and_clean_data()
    df = encode_features(df)
    df = df.select_dtypes(include=[np.number])

    # Split: reference = first 5000, current = last 2043
    reference = df.iloc[:5000]
    current   = df.iloc[5000:]

    print(f"Reference: {reference.shape} | Current: {current.shape}")

    # ── Generate Evidently report ─────────────────────────────
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=reference, current_data=current)

    # Save HTML report
    os.makedirs("mlops", exist_ok=True)
    report.save_html("mlops/drift_report.html")
    print("✅ drift_report.html saved")

    # ── Save JSON summary ─────────────────────────────────────
    report_dict = report.as_dict()

    # Extract drift info
    drifted_features = []
    trigger_retrain = False

    try:
        metrics = report_dict.get('metrics', [])
        for metric in metrics:
            if metric.get('metric') == 'DatasetDriftMetric':
                result = metric.get('result', {})
                trigger_retrain = result.get('dataset_drift', False)
                drifted_count = result.get('number_of_drifted_columns', 0)
                print(f"Dataset drift detected: {trigger_retrain}")
                print(f"Drifted columns: {drifted_count}")

            if metric.get('metric') == 'ColumnDriftMetric':
                result = metric.get('result', {})
                if result.get('drift_detected', False):
                    drifted_features.append({
                        'feature': result.get('column_name', ''),
                        'drift_score': round(result.get('drift_score', 0), 4)
                    })
    except Exception as e:
        print(f"Warning parsing drift results: {e}")
        trigger_retrain = False

    summary = {
        "trigger_retrain": trigger_retrain,
        "drifted_features": drifted_features,
        "reference_size": len(reference),
        "current_size": len(current),
        "report_path": "mlops/drift_report.html"
    }

    with open("mlops/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ drift_summary.json saved")
    print(f"Trigger retrain: {trigger_retrain}")
    print(f"Drifted features: {[d['feature'] for d in drifted_features]}")
    return summary
def run_drift_on_file(current_csv_path):
    """Test drift against a custom CSV file"""
    import pandas as pd
    from src.eda import load_and_clean_data, encode_features

    print(f"Running drift detection on {current_csv_path}...")

    # Reference = original training data
    df_ref = load_and_clean_data()
    df_ref = encode_features(df_ref)
    df_ref = df_ref.select_dtypes(include=[np.number])
    reference = df_ref.iloc[:5000]

    # Current = uploaded file
    df_cur = pd.read_csv(current_csv_path)
    # Add dummy Churn column if missing
    if 'Churn' not in df_cur.columns:
        df_cur['Churn'] = 0
    from src.eda import encode_features as ef
    df_cur['Churn'] = df_cur['Churn'].map({'Yes': 1, 'No': 0}).fillna(0)
    df_cur = ef(df_cur)
    df_cur = df_cur.select_dtypes(include=[np.number])

    # Align columns
    common_cols = [c for c in reference.columns if c in df_cur.columns]
    reference = reference[common_cols]
    df_cur = df_cur[common_cols]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=df_cur)
    report.save_html("mlops/drift_report_custom.html")

    result = report.as_dict()
    for metric in result.get('metrics', []):
        if metric.get('metric') == 'DatasetDriftMetric':
            drift = metric['result'].get('dataset_drift', False)
            n_drifted = metric['result'].get('number_of_drifted_columns', 0)
            print(f"Drift detected: {drift}")
            print(f"Drifted columns: {n_drifted}")
            return drift, n_drifted
    return False, 0

if __name__ == "__main__":
    # Default run
    summary = run_drift_detection()

    # Also test with high-risk data if it exists
    if os.path.exists("data/batch_high_risk.csv"):
        print("\n--- Testing with HIGH RISK batch ---")
        run_drift_on_file("data/batch_high_risk.csv")
