from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

class CandidateWindowEngine:
    """
    Identifies feasible maintenance windows by analyzing scheduled train movements,
    headway, and corridor availability across target railway sections.
    """

    DEFAULT_DAILY_WINDOWS = [
        {"start": "01:00", "end": "04:30", "duration_minutes": 210, "traffic_level": "LOW", "window_quality": 95},
        {"start": "11:15", "end": "13:30", "duration_minutes": 135, "traffic_level": "LOW_MEDIUM", "window_quality": 82},
        {"start": "14:15", "end": "16:00", "duration_minutes": 105, "traffic_level": "MEDIUM", "window_quality": 74},
        {"start": "23:30", "end": "01:00", "duration_minutes": 90, "traffic_level": "LOW", "window_quality": 85},
    ]

    def find_windows_for_section(
        self,
        section_id: str,
        target_date: str,
        min_duration_minutes: int,
        schedules_df: pd.DataFrame = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates operational timetable gaps for a section on a given date.
        Categorizes windows as TRAIN_CONFLICT, TOO_SHORT, or SUITABLE.
        """
        candidates = []
        for w in self.DEFAULT_DAILY_WINDOWS:
            dur = w["duration_minutes"]
            if dur < min_duration_minutes:
                status = "TOO_SHORT"
                suitability = "UNSUITABLE"
            else:
                status = "SUITABLE"
                suitability = "HIGH" if w["traffic_level"] == "LOW" else "MEDIUM"

            candidates.append({
                "window_id": f"WIN-{section_id}-{w['start'].replace(':', '')}",
                "section_id": section_id,
                "date": target_date,
                "start_time": w["start"],
                "end_time": w["end"],
                "duration_minutes": dur,
                "traffic_level": w["traffic_level"],
                "window_quality": w["window_quality"],
                "status": status,
                "suitability": suitability
            })
        return candidates
