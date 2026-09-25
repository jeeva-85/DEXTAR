"""
=============================================================================
RAILBLOCK AI — DATASET INSPECTOR
=============================================================================
Inspects the uploaded railway maintenance CSV files.
- Detects columns, dtypes, missing values, unique counts
- Identifies viable target columns for M1, M2, M3
- Reports leakage risks
- Outputs a JSON report to ml/artifacts/reports/dataset_report.json
=============================================================================
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Known target columns and their model roles
# ─────────────────────────────────────────────────────────────────────────────
TARGET_CANDIDATES = {
    "m1_priority": {
        "columns": ["risk_score", "priority_score", "urgency_score"],
        "type": "regression",
        "leakage_sources": ["failure_type", "failure_severity", "maintenance_required"],
    },
    "m2_duration": {
        "columns": [
            "actual_maintenance_duration",
            "maintenance_duration_hours",
            "duration_minutes",
            "repair_time",
            "actual_duration",
            "maintenance_time",
        ],
        "type": "regression",
        "leakage_sources": [],
    },
    "m3_failure": {
        "columns": ["maintenance_required", "failure_occurred", "defect_escalated"],
        "type": "binary_classification",
        "leakage_sources": ["failure_type", "failure_severity", "risk_score"],
    },
    "m3_severity": {
        "columns": ["failure_severity", "severity_level"],
        "type": "multiclass_classification",
        "leakage_sources": ["failure_type", "maintenance_required", "risk_score"],
    },
}

# Columns that are identifiers (not features)
ID_COLUMNS = ["train_id", "task_id", "record_id", "id", "index"]


class DatasetInspector:
    """
    Inspects one or more CSV datasets, determines their viability for ML
    training, and writes a structured JSON report.
    """

    def __init__(self, reports_dir: str = "ml/artifacts/reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def inspect_file(self, filepath: str | Path) -> Dict[str, Any]:
        """Inspect a single CSV/Parquet file and return a column-level report."""
        filepath = Path(filepath)
        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}

        print(f"  Inspecting: {filepath.name} ...")
        df = self._load_file(filepath)
        if df is None:
            return {"error": f"Could not load {filepath.name}"}

        report = {
            "filename": filepath.name,
            "filepath": str(filepath),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_profiles": self._profile_columns(df),
            "target_viability": self._assess_targets(df),
            "dataset_type": self._infer_dataset_type(df),
        }
        return report

    def inspect_multiple(self, filepaths: List[str | Path]) -> Dict[str, Any]:
        """Inspect multiple files and return a combined report with merge feasibility."""
        individual = {}
        dfs = {}
        for fp in filepaths:
            fp = Path(fp)
            report = self.inspect_file(fp)
            individual[fp.name] = report
            if "error" not in report:
                dfs[fp.name] = self._load_file(fp)

        combined_cols = set()
        for df in dfs.values():
            if df is not None:
                combined_cols.update(df.columns.tolist())

        # Find common columns across all files
        if dfs:
            all_col_sets = [set(df.columns) for df in dfs.values() if df is not None]
            common_cols = set.intersection(*all_col_sets) if all_col_sets else set()
        else:
            common_cols = set()

        combined_report = {
            "files_inspected": len(filepaths),
            "total_rows": sum(r.get("rows", 0) for r in individual.values()),
            "common_columns": sorted(common_cols),
            "all_columns": sorted(combined_cols),
            "extra_columns_per_file": {
                fname: sorted(set(dfs[fname].columns) - common_cols)
                for fname in dfs
            },
            "merge_strategy": self._recommend_merge_strategy(dfs, common_cols),
            "individual_files": individual,
            "training_targets": self._combined_target_summary(individual),
        }

        # Save report
        report_path = self.reports_dir / "dataset_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(combined_report, f, indent=2, default=str)
        print(f"\n  Dataset report saved → {report_path}")
        return combined_report

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _load_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        try:
            suffix = filepath.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(filepath, low_memory=False)
            elif suffix in (".parquet", ".pq"):
                return pd.read_parquet(filepath)
            elif suffix == ".json":
                return pd.read_json(filepath)
            elif suffix in (".xlsx", ".xls"):
                return pd.read_excel(filepath)
            else:
                return None
        except Exception as e:
            print(f"    ERROR loading {filepath.name}: {e}")
            return None

    def _profile_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        profiles = []
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            missing = int(series.isna().sum())
            missing_pct = round(missing / len(df) * 100, 2)
            nunique = int(series.nunique(dropna=True))

            profile: Dict[str, Any] = {
                "column": col,
                "dtype": dtype,
                "missing": missing,
                "missing_pct": missing_pct,
                "unique_values": nunique,
                "is_id_column": col.lower() in [c.lower() for c in ID_COLUMNS],
            }

            # For categoricals / low-cardinality
            if nunique <= 20 or dtype in ("object", "string", "category"):
                top_vals = series.dropna().value_counts().head(10).to_dict()
                profile["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

            # For numerics
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe()
                profile["min"] = _safe_float(desc.get("min"))
                profile["max"] = _safe_float(desc.get("max"))
                profile["mean"] = _safe_float(desc.get("mean"))
                profile["std"] = _safe_float(desc.get("std"))

            profiles.append(profile)
        return profiles

    def _assess_targets(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check which target columns are viable."""
        viability: Dict[str, Any] = {}
        cols_lower = {c.lower(): c for c in df.columns}

        for model_key, meta in TARGET_CANDIDATES.items():
            found_col = None
            for candidate in meta["columns"]:
                if candidate.lower() in cols_lower:
                    found_col = cols_lower[candidate.lower()]
                    break

            if found_col is None:
                viability[model_key] = {
                    "status": "DATA_NOT_AVAILABLE",
                    "reason": f"None of {meta['columns']} found in dataset",
                    "target_column": None,
                }
                continue

            series = df[found_col]
            missing = int(series.isna().sum())
            missing_pct = round(missing / len(df) * 100, 2)
            nunique = int(series.nunique(dropna=True))
            non_null = len(df) - missing

            assessment: Dict[str, Any] = {
                "target_column": found_col,
                "missing": missing,
                "missing_pct": missing_pct,
                "non_null_rows": non_null,
                "unique_values": nunique,
                "leakage_columns_to_exclude": [
                    c for c in meta["leakage_sources"] if c in df.columns
                ],
            }

            # Viability check
            if missing_pct > 95:
                assessment["status"] = "TOO_MANY_MISSING"
            elif non_null < 500:
                assessment["status"] = "TOO_FEW_RECORDS"
            elif meta["type"] == "binary_classification" and nunique != 2:
                assessment["status"] = "INVALID_BINARY_TARGET"
            elif meta["type"] == "multiclass_classification" and nunique < 2:
                assessment["status"] = "INSUFFICIENT_CLASSES"
            else:
                assessment["status"] = "VIABLE"
                # Extra stats for viable targets
                if meta["type"] == "binary_classification":
                    vc = series.value_counts(dropna=True).to_dict()
                    assessment["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
                elif meta["type"] == "multiclass_classification":
                    vc = series.value_counts(dropna=True).to_dict()
                    assessment["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
                elif meta["type"] == "regression":
                    assessment["range"] = [
                        _safe_float(series.min()), _safe_float(series.max())
                    ]

            viability[model_key] = assessment

        return viability

    def _infer_dataset_type(self, df: pd.DataFrame) -> str:
        """Heuristically classify dataset as maintenance, schedule, or unknown."""
        cols_lower = set(c.lower() for c in df.columns)
        maint_signals = {"failure_type", "maintenance_required", "risk_score", "wheel_wear_percent", "rail_wear_mm"}
        sched_signals = {"arrival", "departure", "station_code", "train_number", "headway"}

        maint_score = len(cols_lower & maint_signals)
        sched_score = len(cols_lower & sched_signals)

        if maint_score > sched_score:
            return "MAINTENANCE_SENSOR_DATA"
        elif sched_score > 0:
            return "TRAIN_SCHEDULE_DATA"
        else:
            return "UNKNOWN"

    def _recommend_merge_strategy(
        self, dfs: Dict[str, pd.DataFrame], common_cols: set
    ) -> str:
        if len(dfs) <= 1:
            return "SINGLE_FILE_NO_MERGE_NEEDED"
        if len(common_cols) >= 5:
            return "CONCATENATE_ON_COMMON_COLUMNS"
        return "USE_LARGEST_DATASET_ONLY"

    def _combined_target_summary(
        self, individual_reports: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate target viability across all files."""
        summary: Dict[str, Any] = {}
        for model_key in TARGET_CANDIDATES:
            statuses = []
            for fname, report in individual_reports.items():
                tv = report.get("target_viability", {}).get(model_key, {})
                statuses.append(
                    {"file": fname, "status": tv.get("status"), "target_column": tv.get("target_column")}
                )
            # If any file has a VIABLE target, mark as viable
            best = next((s for s in statuses if s["status"] == "VIABLE"), None)
            summary[model_key] = {
                "overall_status": "VIABLE" if best else "DATA_NOT_AVAILABLE",
                "best_source": best,
                "all_sources": statuses,
            }
        return summary


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None and not np.isnan(float(val)) else None
    except Exception:
        return None


# ── Standalone run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent.parent.parent  # project root
    RAW_DIR = BASE / "ml" / "data" / "raw"

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in ml/data/raw/")
    else:
        inspector = DatasetInspector(reports_dir=str(BASE / "ml" / "artifacts" / "reports"))
        report = inspector.inspect_multiple(csv_files)
        print("\n── DATASET INSPECTION SUMMARY ──────────────────────────")
        print(f"  Files  : {report['files_inspected']}")
        print(f"  Total rows: {report['total_rows']:,}")
        print(f"  Common cols: {len(report['common_columns'])}")
        print("\n── TARGET VIABILITY ─────────────────────────────────────")
        for model, info in report["training_targets"].items():
            status = info["overall_status"]
            best = info.get("best_source") or {}
            col = best.get("target_column", "—")
            print(f"  {model:20s}  {status:20s}  target={col}")
