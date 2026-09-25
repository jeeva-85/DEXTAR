"""
Independent training script for Model 2: Failure Risk (XGBoost Classifier -> maintenance_required).
"""
import json
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data.dataset_manager import DatasetManager
from ml.preprocessing.cleaner import DataCleaner
from ml.preprocessing.encoder import CategoricalEncoder
from ml.features.feature_engineering import FeatureEngineer
from ml.models.failure_risk import FailureRiskModel
from ml.evaluation.evaluate import ModelEvaluator
from ml.explainability.shap_explainer import RailwayShapExplainer


def train_failure_model(df_clean: pd.DataFrame = None) -> Dict[str, Any]:
    base = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = base / "ml" / "artifacts"
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if df_clean is None:
        dm = DatasetManager()
        pack = dm.load_and_profile()
        cleaner = DataCleaner()
        df_clean, _ = cleaner.clean(pack["df"])

    target_col = "maintenance_required"
    if target_col not in df_clean.columns:
        print(f"  [Failure Risk] Target '{target_col}' not available in dataset.")
        return {"status": "DATA_NOT_AVAILABLE", "reason": f"Target column '{target_col}' not found"}

    print("  [Failure Risk] Engineering features and preparing data ...")
    fe = FeatureEngineer()
    X_raw, num_cols, cat_cols = fe.select_features(df_clean, "m3_failure", target_col)
    y = df_clean[target_col].astype(int)

    # Encode categoricals
    encoder = CategoricalEncoder(cat_cols)
    X = encoder.fit_transform(X_raw)

    # 70% Train / 15% Validation / 15% Test (Stratified)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    )

    print(f"  [Failure Risk] Training XGBoost Classifier on {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test rows ...")

    model = FailureRiskModel()
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate on strictly unseen Test set
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    evaluator = ModelEvaluator(artifacts_dir)
    test_metrics = evaluator.evaluate_failure_risk(y_test.to_numpy(), test_preds, test_probs)
    print(f"  [Failure Risk] Unseen Test Evaluation: Accuracy={test_metrics['accuracy']:.4f}, Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}, PR-AUC={test_metrics['pr_auc']:.4f}")

    # SHAP Explanations on Test sample
    shap_exp = RailwayShapExplainer(artifacts_dir / "reports")
    shap_report = shap_exp.generate_global_summary(model.model, X_test.head(500), "m3_failure")

    # Save artifact
    payload = {
        "model": model.model,
        "features": X.columns.tolist(),
        "cat_categories": encoder.categories_,
        "target": target_col,
        "metrics": test_metrics,
        "test_samples_count": len(X_test),
    }

    joblib.dump(payload, models_dir / "failure_risk_model.joblib")
    joblib.dump(payload, artifacts_dir / "m3_failure.joblib")

    # Feature schema
    schema = {
        "model": "m3_failure",
        "target_column": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "all_features": X.columns.tolist(),
        "cat_categories": encoder.categories_,
    }
    with open(artifacts_dir / "feature_schema_m3_failure.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return {
        "status": "TRAINED",
        "model_name": "failure_risk_model",
        "target": target_col,
        "metrics": test_metrics,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
    }


if __name__ == "__main__":
    train_failure_model()

