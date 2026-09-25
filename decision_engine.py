"""
AIDecisionEngine: Evaluates maintenance urgency, failure criticality,
and candidate maintenance opportunities prior to OR-Tools CP-SAT scheduling.
"""
from typing import Any, Dict, List, Optional
import numpy as np


class AIDecisionEngine:
    def __init__(self, urgency_threshold_high: float = 65.0, urgency_threshold_critical: float = 80.0):
        self.urgency_threshold_high = urgency_threshold_high
        self.urgency_threshold_critical = urgency_threshold_critical

    def evaluate_task_urgency(
        self,
        ml_prediction: Dict[str, Any],
        nominal_duration_minutes: int = 90,
        department: str = "ENGINEERING",
    ) -> Dict[str, Any]:
        """
        Evaluates task urgency by fusing ML Priority score, Failure probability, and Severity.
        """
        p_data = ml_prediction.get("priority", {})
        f_data = ml_prediction.get("failure_risk", {})
        s_data = ml_prediction.get("failure_severity", {})

        priority_score = float(p_data.get("score", 50.0))
        fail_prob = float(f_data.get("probability", 0.3))
        severity = s_data.get("predicted_severity", "Medium")

        # Severity multiplier
        sev_multiplier = {
            "Critical": 1.35,
            "High": 1.20,
            "Medium": 1.0,
            "Low": 0.85,
        }.get(severity, 1.0)

        # Fused urgency index (0 - 100)
        base_urgency = (priority_score * 0.6) + ((fail_prob * 100.0) * 0.4)
        fused_urgency = round(float(np.clip(base_urgency * sev_multiplier, 0.0, 100.0)), 1)

        if fused_urgency >= self.urgency_threshold_critical:
            urgency_level = "CRITICAL"
            scheduling_recommendation = "IMMEDIATE_BLOCK_REQUIRED"
        elif fused_urgency >= self.urgency_threshold_high:
            urgency_level = "HIGH"
            scheduling_recommendation = "PRIORITIZE_WITHIN_48H"
        elif fused_urgency >= 40.0:
            urgency_level = "MEDIUM"
            scheduling_recommendation = "SCHEDULE_REGULAR_WINDOW"
        else:
            urgency_level = "LOW"
            scheduling_recommendation = "OPPORTUNISTIC_OR_DEFER"

        return {
            "fused_urgency_score": fused_urgency,
            "urgency_level": urgency_level,
            "scheduling_recommendation": scheduling_recommendation,
            "nominal_duration_minutes": nominal_duration_minutes,
            "department": department,
            "ml_inputs": {
                "priority_score": priority_score,
                "failure_probability": fail_prob,
                "failure_severity": severity,
                "shap_reasons": ml_prediction.get("shap_reasons", []),
            },
        }

    def score_candidate_opportunity(
        self,
        task_evaluation: Dict[str, Any],
        window: Dict[str, Any],
        is_coordinated: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates the synergy between an urgent task and a candidate track window.
        Produces candidate match score used by CP-SAT solver objective.
        """
        urgency = task_evaluation["fused_urgency_score"]
        win_dur = window.get("duration_minutes", 120)
        req_dur = task_evaluation.get("nominal_duration_minutes", 90)

        # Duration fit penalty if window is too tight or excessively long
        if req_dur > win_dur:
            fit_score = 0.0  # Infeasible
        else:
            utilization = (req_dur / max(win_dur, 1)) * 100.0
            fit_score = 100.0 - abs(utilization - 85.0)  # Optimal utilization around 85%

        window_quality = window.get("window_quality", 80.0)
        traffic_suitability = 95.0 if window.get("traffic_level") == "LOW" else 75.0

        coord_bonus = 25.0 if is_coordinated else 0.0

        total_score = (
            (urgency * 0.40)
            + (window_quality * 0.25)
            + (traffic_suitability * 0.20)
            + (fit_score * 0.15)
            + coord_bonus
        )

        return {
            "match_score": round(float(np.clip(total_score, 0.0, 100.0)), 1),
            "feasible": req_dur <= win_dur,
            "coordination_advantage": is_coordinated,
        }
