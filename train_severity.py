"""
Independent training script for Model 3: Failure Severity (XGBoost Multiclass -> failure_severity).
"""
import json
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from ml.data.dataset_manager import DatasetManager
from ml.preprocessing.cleaner import DataCleaner
from ml.preprocessing.encoder import CategoricalEncoder, TargetLabelEncoder
from ml.features.feature_engineering import FeatureEngineer
from ml.models.failure_severity import FailureSeverityModel
from ml.evaluation.evaluate import ModelEvaluator


def train_severity_model(df_clean: pd.DataFrame = None) -> Dict[str, Any]:
    base = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = base / "ml" / "artifacts"
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if df_clean is None:
        dm = DatasetManager()
        pack = dm.load_and_profile()
        cleaner = DataCleaner()
        df_clean, _ = cleaner.clean(pack["df"])

    target_col = "failure_severity"
    if target_col not in df_clean.columns:
        print(f"  [Severity] Target '{target_col}' not available in dataset.")
        return {"status": "DATA_NOT_AVAILABLE", "reason": f"Target column '{target_col}' not found"}

    # Filter to instances where failure/severity is present
    df_sev = df_clean.dropna(subset=[target_col]).copy()
    if "maintenance_required" in df_sev.columns:
        # Prefer cases where maintenance is needed
        maint_cases = df_sev[df_sev["maintenance_required"] == 1]
        if len(maint_cases) >= 500:
            df_sev = maint_cases

    if len(df_sev) < 100:
        return {"status": "DATA_NOT_AVAILABLE", "reason": f"Insufficient records ({len(df_sev)}) for severity"}

    print(f"  [Severity] Preparing {len(df_sev):,} records for severity classification ...")
    fe = FeatureEngineer()
    X_raw, num_cols, cat_cols = fe.select_features(df_sev, "m3_severity", target_col)

    # Encode target labels
    target_encoder = TargetLabelEncoder(["Low", "Medium", "High", "Critical"])
    target_encoder.fit(df_sev[target_col])
    y_encoded = target_encoder.transform(df_sev[target_col])

    # Encode categoricals
    encoder = CategoricalEncoder(cat_cols)
    X = encoder.fit_transform(X_raw)

    # 70% Train / 15% Validation / 15% Test (Stratified)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    )

    # Calculate balanced sample weights to prevent majority-class collapse
    sample_weights_train = compute_sample_weight("balanced", y_train)

    print(f"  [Severity] Training XGBoost Multiclass Classifier ({len(target_encoder.classes_)} classes) on {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test rows ...")

    model = FailureSeverityModel(num_classes=len(target_encoder.classes_))
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        class_labels=target_encoder.classes_,
        sample_weight=sample_weights_train,
    )

    # Evaluate on strictly unseen Test set
    test_preds = model.predict(X_test)
    evaluator = ModelEvaluator(artifacts_dir)
    test_metrics = evaluator.evaluate_severity(y_test, test_preds, label_names=target_encoder.classes_)
    print(f"  [Severity] Unseen Test Evaluation: Accuracy={test_metrics['accuracy']:.4f}, Macro-F1={test_metrics['macro_f1']:.4f}, Macro-Precision={test_metrics['macro_precision']:.4f}, Macro-Recall={test_metrics['macro_recall']:.4f}")

    # Save artifact
    payload = {
        "model": model.model,
        "features": X.columns.tolist(),
        "cat_categories": encoder.categories_,
        "label_encoder_classes": target_encoder.classes_,
        "target": target_col,
        "metrics": test_metrics,
        "test_samples_count": len(X_test),
    }

    joblib.dump(payload, models_dir / "failure_severity_model.joblib")
    joblib.dump(payload, artifacts_dir / "m3_severity.joblib")

    # Feature schema
    schema = {
        "model": "m3_severity",
        "target_column": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "all_features": X.columns.tolist(),
        "cat_categories": encoder.categories_,
        "classes": target_encoder.classes_,
    }
    with open(artifacts_dir / "feature_schema_m3_severity.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return {
        "status": "TRAINED",
        "model_name": "failure_severity_model",
        "target": target_col,
        "metrics": test_metrics,
        "classes": target_encoder.classes_,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
    }


if __name__ == "__main__":
    train_severity_model()
