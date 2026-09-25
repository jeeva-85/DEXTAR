"""
=============================================================================
RAILBLOCK AI — REAL DATA TRAINING PIPELINE
=============================================================================
Trains ML models using the actual uploaded Indian Railway maintenance datasets.

Models trained:
  M1 — Maintenance Priority (XGBoost Regressor → risk_score)
  M2 — Maintenance Duration (SKIPPED — target column absent)
  M3 — Failure Risk (XGBoost Classifier → maintenance_required)
  M3-Severity — Failure Severity (XGBoost Classifier → failure_severity,
                                  trained on failure-positive subset)

Run:
    python ml/training/train_real_data.py

Or from project root:
    python train_all_models.py
=============================================================================
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── Project paths ───────────────────────────────────────────────────────────
# __file__ = ml/training/train_real_data.py  →  .parent x3 = project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
RAW_DIR = BASE_DIR / "ml" / "data" / "raw"
ARTIFACT_DIR = BASE_DIR / "ml" / "artifacts"
REPORT_DIR = ARTIFACT_DIR / "reports"

for d in [ARTIFACT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Column metadata ─────────────────────────────────────────────────────────

# Sensor / operational feature columns used as model inputs
# These are present in at least one of the two datasets.
# The pipeline inspects actual columns and only uses those that exist.
CANDIDATE_FEATURE_COLS = [
    "train_age_years",
    "average_speed_kmph",
    "distance_travelled_km",
    "track_temperature_c",
    "ambient_temperature_c",
    "humidity_percent",
    "rainfall_mm",
    "wind_speed_kmph",
    "wheel_wear_percent",
    "track_vibration_level",
    "rail_wear_mm",
    "ballast_condition",
    "track_curvature_degree",
    "bearing_temperature_c",
    "axle_temperature_c",
    "brake_pad_wear_percent",
    "brake_pressure_psi",
    "battery_voltage",
    "traction_motor_temp_c",
    "signal_system_status",
    "power_consumption_kw",
    "load_factor_percent",
    "daily_trips",
    "last_maintenance_days",
    "sensor_health_index",
    "inspection_score",
    "delay_minutes",
    "region",
    "season",
    "train_type",
]

# Categorical columns (encoded as pd.Categorical for XGBoost native support)
CATEGORICAL_COLS = [
    "region", "season", "train_type",
    "ballast_condition", "signal_system_status",
]

# Target columns and their models
TARGET_MAP = {
    "m1_priority":   "risk_score",
    "m3_failure":    "maintenance_required",
    "m3_severity":   "failure_severity",
}

# Leakage — columns that must NEVER be features for a given model
LEAKAGE_MAP = {
    "m1_priority":   {"failure_type", "failure_severity", "maintenance_required"},
    "m3_failure":    {"failure_type", "failure_severity", "risk_score"},
    "m3_severity":   {"failure_type", "maintenance_required", "risk_score"},
}

# M2 is not trainable with current data
M2_STATUS = {
    "status": "DATA_NOT_AVAILABLE",
    "reason": (
        "Neither uploaded dataset contains a maintenance duration column "
        "(e.g., actual_maintenance_duration, repair_time). "
        "M2 architecture is ready — provide a work-order history file with "
        "actual recorded maintenance times to enable training."
    ),
}

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING & MERGING
# ═════════════════════════════════════════════════════════════════════════════

def load_and_merge_datasets() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load all CSV files from ml/data/raw/, merge on common columns.
    Returns the merged DataFrame and a metadata dict.
    """
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}. "
            "Please place your datasets in ml/data/raw/ and re-run."
        )

    print(f"  Found {len(csv_files)} CSV file(s):")
    dfs = []
    file_meta = []
    for fp in csv_files:
        print(f"    Loading: {fp.name} ...", end=" ")
        df = pd.read_csv(fp, low_memory=False)
        print(f"({len(df):,} rows × {len(df.columns)} cols)")
        dfs.append(df)
        file_meta.append({
            "filename": fp.name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
        })

    if len(dfs) == 1:
        merged = dfs[0]
        strategy = "SINGLE_FILE"
    else:
        # Find common columns
        col_sets = [set(df.columns) for df in dfs]
        common = set.intersection(*col_sets)
        print(f"\n  Merge strategy: CONCATENATE — {len(common)} common columns")
        print(f"  Common columns: {sorted(common)}")
        # Concat with union of columns (non-common → NaN for the other file)
        merged = pd.concat(dfs, ignore_index=True, sort=False)
        strategy = "CONCATENATE_UNION"

    print(f"\n  Merged dataset: {len(merged):,} rows × {len(merged.columns)} cols")
    meta = {
        "files": file_meta,
        "merge_strategy": strategy,
        "total_rows": int(len(merged)),
        "total_columns": int(len(merged.columns)),
    }
    return merged, meta


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and impute the merged DataFrame.
    - Drop duplicate rows
    - Impute numeric NaN with column median
    - Impute categorical NaN with column mode
    """
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} duplicate rows.")

    # Numeric columns — median imputation
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # String/object columns — mode imputation
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in str_cols:
        if df[col].isna().any():
            mode_vals = df[col].mode(dropna=True)
            fill = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            df[col] = df[col].fillna(fill)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def build_features(
    df: pd.DataFrame,
    model_key: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Select features for a given model, applying leakage exclusions.
    Returns (X_df, numeric_feature_cols, categorical_feature_cols).
    """
    # Start with candidate features that actually exist in this DataFrame
    existing_candidates = [c for c in CANDIDATE_FEATURE_COLS if c in df.columns]

    # Remove leakage columns
    leakage = LEAKAGE_MAP.get(model_key, set())
    # Also remove the target itself and all other targets
    all_targets = set(TARGET_MAP.values()) | {"train_id", "task_id", "id"}
    exclude = leakage | all_targets

    feature_cols = [c for c in existing_candidates if c not in exclude]

    # Identify numeric vs categorical among selected features
    num_feats = []
    cat_feats = []
    for col in feature_cols:
        if col in CATEGORICAL_COLS and col in df.columns:
            cat_feats.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_feats.append(col)
        else:
            cat_feats.append(col)

    X = df[feature_cols].copy()

    # Encode categoricals as pd.Categorical (XGBoost native)
    for col in cat_feats:
        if col in X.columns:
            X[col] = X[col].astype(str)
            uniq = sorted(X[col].unique().tolist())
            X[col] = pd.Categorical(X[col], categories=uniq)

    return X, num_feats, cat_feats


def save_feature_schema(
    model_key: str,
    num_feats: List[str],
    cat_feats: List[str],
    target_col: str,
    cat_categories: Dict[str, List[str]],
):
    schema = {
        "model": model_key,
        "target_column": target_col,
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "categorical_categories": cat_categories,
        "all_features": num_feats + cat_feats,
        "leakage_excluded": sorted(LEAKAGE_MAP.get(model_key, set())),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    path = ARTIFACT_DIR / f"feature_schema_{model_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Also write combined schema.json (M1 schema used as primary for inference)
    if model_key == "m1_priority":
        combined_path = ARTIFACT_DIR / "feature_schema.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

    return path


# ═════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def train_xgb_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    r2 = float(r2_score(y_val, preds))
    return model, {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def train_xgb_binary_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    # Handle class imbalance with scale_pos_weight
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, preds)), 4),
        "precision": round(float(precision_score(y_val, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_val, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_val, probs)), 4),
        "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
        "class_distribution_train": {
            "0": int(neg), "1": int(pos)
        },
    }
    return model, metrics


def train_xgb_multiclass_classifier(
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    X_val: pd.DataFrame,
    y_val_encoded: np.ndarray,
    num_classes: int,
    label_names: List[str],
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=num_classes,
        objective="multi:softprob",
        enable_categorical=True,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        verbose=False,
    )
    preds = model.predict(X_val)
    report = classification_report(
        y_val_encoded, preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": round(float(accuracy_score(y_val_encoded, preds)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
            for label in label_names if label in report
        },
    }
    return model, metrics


# ═════════════════════════════════════════════════════════════════════════════
# SHAP EXPLANATIONS
# ═════════════════════════════════════════════════════════════════════════════

def generate_shap_summary(
    model: Any,
    X_sample: pd.DataFrame,
    model_key: str,
) -> Dict[str, Any]:
    """Generate SHAP feature importance from the trained model."""
    try:
        import shap  # type: ignore

        # Encode categoricals as codes for SHAP (TreeExplainer needs numeric)
        X_enc = X_sample.copy()
        for col in X_enc.select_dtypes(include="category").columns:
            X_enc[col] = X_enc[col].cat.codes

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_enc)

        # For multi-class, take abs mean across classes
        if isinstance(shap_values, list):
            sv = np.abs(np.mean(np.stack(shap_values, axis=0), axis=0))
        else:
            sv = np.abs(shap_values)

        mean_abs_shap = np.abs(sv).mean(axis=0)
        feature_names = X_sample.columns.tolist()
        importance = sorted(
            zip(feature_names, mean_abs_shap.tolist()),
            key=lambda x: x[1], reverse=True
        )
        top10 = importance[:10]

        shap_report = {
            "top_features": [
                {"feature": f, "mean_abs_shap": round(v, 4)}
                for f, v in top10
            ],
            "model": model_key,
        }
        shap_path = ARTIFACT_DIR / "reports" / f"shap_{model_key}.json"
        with open(shap_path, "w", encoding="utf-8") as f:
            json.dump(shap_report, f, indent=2)
        return shap_report
    except Exception as e:
        return {"error": str(e), "model": model_key}


# ═════════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ═════════════════════════════════════════════════════════════════════════════

def save_model_artifact(
    model: Any,
    feature_names: List[str],
    extra: Dict[str, Any],
    filename: str,
):
    path = ARTIFACT_DIR / filename
    payload = {
        "model": model,
        "features": feature_names,
        **extra,
    }
    joblib.dump(payload, path)
    print(f"    Saved → {path.name} ({path.stat().st_size / 1024:.0f} KB)")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_training_pipeline() -> Dict[str, Any]:
    t0 = time.time()
    print("\n" + "=" * 70)
    print("  RAILBLOCK AI — REAL DATA ML TRAINING PIPELINE")
    print("=" * 70)

    # ── Step 1: Load datasets ───────────────────────────────────────────────
    print("\n[1/7] Loading datasets from ml/data/raw/ ...")
    df_raw, load_meta = load_and_merge_datasets()

    # ── Step 2: Inspect & report ────────────────────────────────────────────
    print("\n[2/7] Running dataset inspection ...")
    from ml.training.real_data.dataset_inspector import DatasetInspector
    inspector = DatasetInspector(reports_dir=str(REPORT_DIR))
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    dataset_report = inspector.inspect_multiple(csv_files)

    # ── Step 3: Preprocess ──────────────────────────────────────────────────
    print("\n[3/7] Cleaning & imputing data ...")
    df = preprocess(df_raw.copy())
    print(f"  Clean dataset: {len(df):,} rows")

    # Track model statuses
    model_statuses: Dict[str, Any] = {}
    all_metrics: Dict[str, Any] = {}
    feature_schemas: Dict[str, Any] = {}

    # ── Step 4: Train M1 — Priority (risk_score) ────────────────────────────
    print("\n[4/7] Training M1 — Maintenance Priority Model (target: risk_score) ...")

    target_m1 = TARGET_MAP["m1_priority"]
    if target_m1 not in df.columns:
        print(f"  [SKIP] '{target_m1}' column not found. M1 cannot be trained.")
        model_statuses["m1_priority"] = {
            "status": "DATA_NOT_AVAILABLE",
            "reason": f"Target column '{target_m1}' not found",
        }
    else:
        X_m1, num_m1, cat_m1 = build_features(df, "m1_priority")
        y_m1 = df[target_m1].astype(float)

        print(f"  Features ({len(X_m1.columns)}): {num_m1[:5]}... + categoricals {cat_m1}")
        print(f"  Target: {target_m1} | range=[{y_m1.min():.1f}, {y_m1.max():.1f}] | mean={y_m1.mean():.1f}")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_m1, y_m1, test_size=0.2, random_state=42
        )
        print(f"  Train: {len(X_tr):,}  |  Val: {len(X_val):,}")

        m1_model, m1_metrics = train_xgb_regressor(X_tr, y_tr, X_val, y_val)
        print(f"  [OK] M1 Complete | MAE={m1_metrics['mae']:.4f}  RMSE={m1_metrics['rmse']:.4f}  R²={m1_metrics['r2']:.4f}")

        # SHAP
        shap_m1 = generate_shap_summary(m1_model, X_val.head(500), "m1_priority")
        if "top_features" in shap_m1:
            print("  Top SHAP features:", [f["feature"] for f in shap_m1["top_features"][:5]])

        # Save
        cat_cats_m1 = {
            col: X_m1[col].cat.categories.tolist()
            for col in cat_m1 if col in X_m1.columns
        }
        save_model_artifact(
            m1_model,
            X_m1.columns.tolist(),
            {"target": target_m1, "std_err": float(np.std(y_val - m1_model.predict(X_val))),
             "cat_categories": cat_cats_m1},
            "m1_priority.joblib",
        )
        schema_path = save_feature_schema(
            "m1_priority", num_m1, cat_m1, target_m1, cat_cats_m1
        )
        model_statuses["m1_priority"] = {
            "status": "TRAINED",
            "target_column": target_m1,
            "train_rows": int(len(X_tr)),
            "val_rows": int(len(X_val)),
        }
        all_metrics["m1_priority"] = m1_metrics
        feature_schemas["m1_priority"] = {"numeric": num_m1, "categorical": cat_m1}

    # ── Step 5: M2 — Duration (NOT AVAILABLE) ───────────────────────────────
    print("\n[5/7] M2 — Maintenance Duration Model ...")
    print("  [SKIP] No duration target column found in uploaded datasets.")
    print("  → M2 marked as DATA_NOT_AVAILABLE (architecture preserved for future data)")
    model_statuses["m2_duration"] = M2_STATUS

    # ── Step 6: Train M3 — Failure Risk (maintenance_required) ───────────────
    print("\n[6/7] Training M3 — Failure Risk Model (target: maintenance_required) ...")

    target_m3 = TARGET_MAP["m3_failure"]
    if target_m3 not in df.columns:
        print(f"  [SKIP] '{target_m3}' not found. M3 cannot be trained.")
        model_statuses["m3_failure"] = {
            "status": "DATA_NOT_AVAILABLE",
            "reason": f"Target column '{target_m3}' not found",
        }
    else:
        X_m3, num_m3, cat_m3 = build_features(df, "m3_failure")
        y_m3 = df[target_m3].astype(int)

        class_counts = y_m3.value_counts().to_dict()
        print(f"  Features ({len(X_m3.columns)}): {num_m3[:5]}... + categoricals {cat_m3}")
        print(f"  Target: {target_m3} | class distribution: {class_counts}")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_m3, y_m3, test_size=0.2, random_state=42, stratify=y_m3
        )
        print(f"  Train: {len(X_tr):,}  |  Val: {len(X_val):,}")

        m3_model, m3_metrics = train_xgb_binary_classifier(X_tr, y_tr, X_val, y_val)
        print(f"  [OK] M3 Complete | Accuracy={m3_metrics['accuracy']:.4f}  "
              f"F1={m3_metrics['f1']:.4f}  ROC-AUC={m3_metrics['roc_auc']:.4f}  "
              f"PR-AUC={m3_metrics['pr_auc']:.4f}")

        # SHAP
        shap_m3 = generate_shap_summary(m3_model, X_val.head(500), "m3_failure")
        if "top_features" in shap_m3:
            print("  Top SHAP features:", [f["feature"] for f in shap_m3["top_features"][:5]])

        # Save
        cat_cats_m3 = {
            col: X_m3[col].cat.categories.tolist()
            for col in cat_m3 if col in X_m3.columns
        }
        save_model_artifact(
            m3_model,
            X_m3.columns.tolist(),
            {"target": target_m3, "cat_categories": cat_cats_m3},
            "m3_failure.joblib",
        )
        save_feature_schema("m3_failure", num_m3, cat_m3, target_m3, cat_cats_m3)
        model_statuses["m3_failure"] = {
            "status": "TRAINED",
            "target_column": target_m3,
            "train_rows": int(len(X_tr)),
            "val_rows": int(len(X_val)),
        }
        all_metrics["m3_failure"] = m3_metrics

        # ── Step 6b: M3-Severity — failure_severity (failure subset) ────────
        target_sev = TARGET_MAP["m3_severity"]
        if target_sev in df.columns:
            df_failures = df[df["maintenance_required"] == 1].copy()
            df_failures = df_failures.dropna(subset=[target_sev])
            print(f"\n     M3-Severity | failure subset: {len(df_failures):,} rows")

            if len(df_failures) >= 500:
                X_sev, num_sev, cat_sev = build_features(df_failures, "m3_severity")
                y_sev_raw = df_failures[target_sev].astype(str)

                # Ordinal encoding preserving Low→Medium→High→Critical order
                known_labels = [l for l in SEVERITY_ORDER if l in y_sev_raw.unique()]
                le = LabelEncoder()
                le.fit(known_labels)
                y_sev = le.transform(y_sev_raw[y_sev_raw.isin(known_labels)])
                X_sev = X_sev[y_sev_raw.isin(known_labels)]

                X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(
                    X_sev, y_sev, test_size=0.2, random_state=42, stratify=y_sev
                )
                sev_model, sev_metrics = train_xgb_multiclass_classifier(
                    X_tr_s, y_tr_s, X_val_s, y_val_s,
                    num_classes=len(known_labels),
                    label_names=known_labels,
                )
                print(f"     [OK] M3-Severity | Accuracy={sev_metrics['accuracy']:.4f}  "
                      f"Macro-F1={sev_metrics['macro_f1']:.4f}")

                cat_cats_sev = {
                    col: X_sev[col].cat.categories.tolist()
                    for col in cat_sev if col in X_sev.columns
                }
                save_model_artifact(
                    sev_model,
                    X_sev.columns.tolist(),
                    {"target": target_sev, "label_encoder_classes": le.classes_.tolist(),
                     "cat_categories": cat_cats_sev},
                    "m3_severity.joblib",
                )
                save_feature_schema("m3_severity", num_sev, cat_sev, target_sev, cat_cats_sev)
                model_statuses["m3_severity"] = {
                    "status": "TRAINED",
                    "target_column": target_sev,
                    "training_note": "Trained on maintenance_required=1 subset (failure cases only)",
                    "train_rows": int(len(X_tr_s)),
                    "val_rows": int(len(X_val_s)),
                }
                all_metrics["m3_severity"] = sev_metrics
            else:
                print(f"     [SKIP] Insufficient failure rows ({len(df_failures)}) for severity training")
                model_statuses["m3_severity"] = {
                    "status": "DATA_NOT_AVAILABLE",
                    "reason": f"Only {len(df_failures)} failure rows — need at least 500",
                }
        else:
            model_statuses["m3_severity"] = {
                "status": "DATA_NOT_AVAILABLE",
                "reason": f"Column '{target_sev}' not found in dataset",
            }

    # ── Step 7: Save global artifacts ───────────────────────────────────────
    print("\n[7/7] Saving global artifacts ...")

    elapsed = round(time.time() - t0, 1)
    status_doc = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "training_duration_seconds": elapsed,
        "dataset_source": "ml/data/raw/",
        "total_rows_used": int(len(df)),
        "models": model_statuses,
    }
    with open(ARTIFACT_DIR / "status.json", "w", encoding="utf-8") as f:
        json.dump(status_doc, f, indent=2)
    print(f"  status.json written.")

    metrics_doc = {
        "status": "TRAINED",
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_mode": "REAL_DATA",
        "training_duration_seconds": elapsed,
        **{k: v for k, v in all_metrics.items()},
        "model_statuses": model_statuses,
    }
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_doc, f, indent=2)
    print(f"  metrics.json written.")

    # Print final summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    for model_key, status in model_statuses.items():
        icon = "[OK]" if status["status"] == "TRAINED" else "[SKIP]"
        print(f"  {icon} {model_key:25s} → {status['status']}")
        if status["status"] == "TRAINED":
            m = all_metrics.get(model_key, {})
            metric_str = "  ".join(f"{k}={v}" for k, v in list(m.items())[:3] if isinstance(v, float))
            if metric_str:
                print(f"      Metrics: {metric_str}")
    print(f"\n  Total training time: {elapsed}s")
    print("=" * 70 + "\n")

    return {
        "status": "completed",
        "training_duration_seconds": elapsed,
        "models": model_statuses,
        "metrics": all_metrics,
    }


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure project root is on sys.path
    sys.path.insert(0, str(BASE_DIR))
    result = run_training_pipeline()
    sys.exit(0 if result["status"] == "completed" else 1)
