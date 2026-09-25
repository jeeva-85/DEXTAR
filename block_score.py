from typing import Dict, Any

class BlockScoreEngine:
    """
    Transparent Candidate Block Scoring Engine.
    Combines machine learning predictions (M1 priority, M3 failure risk) with
    operational traffic factors, window quality, and multi-department synergy.
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        # Configurable scoring weights
        self.w_priority = cfg.get("weight_priority", 0.35)
        self.w_failure_risk = cfg.get("weight_failure_risk", 0.25)
        self.w_window_quality = cfg.get("weight_window_quality", 0.15)
        self.w_traffic_suitability = cfg.get("weight_traffic_suitability", 0.10)
        self.w_coordination_bonus = cfg.get("weight_coordination_bonus", 0.15)

    def calculate_score(
        self,
        m1_priority: float,
        m3_failure_risk: float,
        window_quality: float,
        traffic_suitability: float,
        is_coordinated: bool,
        conflict_risk: float = 0.0
    ) -> Dict[str, Any]:
        """Calculates final candidate block score and returns exact mathematical breakdown."""
        # Normalize terms to 0-100 scale
        p_term = m1_priority * self.w_priority
        f_term = (m3_failure_risk * 100.0) * self.w_failure_risk
        w_term = window_quality * self.w_window_quality
        t_term = traffic_suitability * self.w_traffic_suitability
        c_term = (100.0 if is_coordinated else 0.0) * self.w_coordination_bonus
        conf_penalty = conflict_risk * 25.0

        raw_score = p_term + f_term + w_term + t_term + c_term - conf_penalty
        final_score = max(0.0, min(100.0, round(raw_score, 1)))

        return {
            "block_score": final_score,
            "components": {
                "priority_component": round(p_term, 1),
                "failure_risk_component": round(f_term, 1),
                "window_quality_component": round(w_term, 1),
                "traffic_suitability_component": round(t_term, 1),
                "department_coordination_component": round(c_term, 1),
                "conflict_penalty": round(conf_penalty, 1)
            },
            "formula": f"({self.w_priority}*Priority) + ({self.w_failure_risk}*FailureRisk) + ({self.w_window_quality}*WindowQuality) + ({self.w_traffic_suitability}*Traffic) + ({self.w_coordination_bonus}*Coordination) - Penalty"
        }
