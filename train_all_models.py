"""
Master ML Training Pipeline for RailBlock AI.
Executes end-to-end dataset profiling, cleaning, feature engineering, model training,
evaluation, SHAP explainability, and artifact persistence.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from ml.data.dataset_manager import DatasetManager
from ml.preprocessing.cleaner import DataCleaner
from ml.training.train_priority import train_priority_model
from ml.training.train_failure import train_failure_model
from ml.training.train_severity import train_severity_model
from ml.evaluation.evaluate import ModelEvaluator


def run_full_training_pipeline() -> Dict[str, Any]:
    t0 = time.time()
    base = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = base / "ml" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("  RAILBLOCK AI — MASTER ML MODEL TRAINING (ACTUAL UPLOADED DATASETS)")
    print("=" * 75)

    # 1. Dataset loading & profiling
    print("\n[Step 1/6] Loading and inspecting uploaded datasets ...")
    dm = DatasetManager(reports_dir=reports_dir)
    data_pack = dm.load_and_profile()
    df_raw = data_pack["df"]
    report = data_pack["report"]

    total_rows = report["summary"]["total_rows"]
    total_cols = report["summary"]["total_columns"]
    print(f"  -> Discovered {len(report['load_meta']['files'])} dataset(s): {total_rows:,} rows, {total_cols} columns")

    # 2. Data Cleaning
    print("\n[Step 2/6] Cleaning and imputing data ...")
    cleaner = DataCleaner()
    df_clean, clean_meta = cleaner.clean(df_raw, is_training=True)
    print(f"  -> Deduplicated dataset: {len(df_clean):,} rows (removed {clean_meta['duplicates_removed']:,} duplicates)")

    # 3. Model 1: Maintenance Priority / Urgency (risk_score)
    print("\n[Step 3/6] Training Model 1: Maintenance Priority (XGBoost Regressor -> risk_score) ...")
    m1_res = train_priority_model(df_clean)

    # 4. Model 2: Maintenance Duration (Check if present in data)
    print("\n[Step 4/6] Checking Model 2: Maintenance Duration ...")
    if not report["summary"]["m2_duration_available"]:
        print("  -> MAINTENANCE DURATION: DATA NOT AVAILABLE")
        print("     Neither uploaded dataset contains recorded maintenance duration/repair time.")
        print("     Architecture is preserved for future work-order data; no fake labels created.")
        m2_status = {
            "status": "DATA_NOT_AVAILABLE",
            "reason": (
                "Neither uploaded dataset contains actual recorded maintenance duration. "
                "Provide work-order data with duration_hours/minutes to enable M2."
            ),
        }
    else:
        m2_status = {"status": "AVAILABLE"}

    # 5. Model 3: Failure Risk (maintenance_required)
    print("\n[Step 5/6] Training Model 3: Failure Risk (XGBoost Classifier -> maintenance_required) ...")
    m3_res = train_failure_model(df_clean)

    # 6. Model 4: Failure Severity (failure_severity)
    print("\n[Step 6/6] Training Model 4: Failure Severity (XGBoost Classifier -> failure_severity) ...")
    m3_sev_res = train_severity_model(df_clean)

    # 7. Reload and verify saved models from disk
    print("\n[Step 7/7] Unloading and testing reloaded models from disk ...")
    from ml.evaluation.test_reloaded_models import test_reloaded_models
    test_reloaded_models()

    elapsed = round(time.time() - t0, 1)

    # Aggregate statuses and metrics
    all_metrics = {
        "status": "TRAINED",
        "dataset_mode": "REAL_DATA",
        "timestamp": pd.Timestamp.now().isoformat(),
        "training_duration_seconds": elapsed,
        "m1_priority": m1_res.get("metrics", {}),
        "m3_failure": m3_res.get("metrics", {}),
        "m3_severity": m3_sev_res.get("metrics", {}),
    }
    evaluator = ModelEvaluator(artifacts_dir)
    evaluator.save_global_metrics(all_metrics)

    status_doc = {
        "status": "TRAINED",
        "is_trained": True,
        "trained_at": pd.Timestamp.now().isoformat(),
        "training_duration_seconds": elapsed,
        "dataset_source": "Actual Uploaded Railway Datasets",
        "total_rows_used": int(len(df_clean)),
        "models": {
            "m1_priority": {
                "ready": m1_res.get("status") == "TRAINED",
                "status": m1_res.get("status"),
                "target": m1_res.get("target"),
                "metrics": m1_res.get("metrics"),
            },
            "m2_duration": m2_status,
            "m3_failure": {
                "ready": m3_res.get("status") == "TRAINED",
                "status": m3_res.get("status"),
                "target": m3_res.get("target"),
                "metrics": m3_res.get("metrics"),
            },
            "m3_severity": {
                "ready": m3_sev_res.get("status") == "TRAINED",
                "status": m3_sev_res.get("status"),
                "target": m3_sev_res.get("target"),
                "metrics": m3_sev_res.get("metrics"),
            },
        },
    }

    with open(artifacts_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status_doc, f, indent=2)

    # Final summary banner
    print("\n" + "=" * 75)
    print("  TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print(f"  Total Processed Rows:   {len(df_clean):,}")
    print(f"  Training Duration:      {elapsed}s")
    print(f"  M1 (Priority / Risk):   {m1_res.get('status')} | MAE={m1_res.get('metrics', {}).get('mae')} | R2={m1_res.get('metrics', {}).get('r2')}")
    print(f"  M2 (Duration):          {m2_status.get('status')} (Architecture preserved, no fake data)")
    print(f"  M3 (Failure Risk):      {m3_res.get('status')} | Accuracy={m3_res.get('metrics', {}).get('accuracy')} | ROC-AUC={m3_res.get('metrics', {}).get('roc_auc')}")
    print(f"  M4 (Failure Severity):  {m3_sev_res.get('status')} | Accuracy={m3_sev_res.get('metrics', {}).get('accuracy')} | Macro-F1={m3_sev_res.get('metrics', {}).get('macro_f1')}")
    print("=" * 75 + "\n")

    return status_doc


if __name__ == "__main__":
    run_full_training_pipeline()
