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

if __name__ == "__main__":
    run_drift_detection()