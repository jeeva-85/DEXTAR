from typing import List, Dict, Any, Tuple

class DepartmentCoordinator:
    """
    Evaluates multi-department compatibility between Engineering, S&T, and TRD
    to combine compatible tasks into a single coordinated block.
    """

    # Compatible maintenance task pairings that can safely share track possessions
    COMPATIBLE_PAIRS = {
        ("ENGINEERING", "SNT"): True,
        ("ENGINEERING", "TRD"): True,
        ("SNT", "TRD"): True,
    }

    # Certain hazardous operations cannot be done concurrently with work crews directly beneath
    INCOMPATIBLE_OPERATIONS = {
        ("DEEP_SCREENING", "CONTACT_WIRE_MEASUREMENT"),
        ("BALLAST_SCREENING", "SIGNAL_CALIBRATION")
    }

    def are_compatible(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks if two tasks can safely share the same block window."""
        dept1, dept2 = task1["department"], task2["department"]
        type1, type2 = task1.get("maintenance_type", ""), task2.get("maintenance_type", "")

        if (type1, type2) in self.INCOMPATIBLE_OPERATIONS or (type2, type1) in self.INCOMPATIBLE_OPERATIONS:
            return False, f"Safety restriction: {type1} and {type2} cannot occur concurrently."

        pair = (min(dept1, dept2), max(dept1, dept2))
        if dept1 == dept2:
            return True, "Same department coordination."
        if pair in self.COMPATIBLE_PAIRS:
            return True, "Multi-department integrated window allowed."

        return False, "Department combinations require separate safety clearances."

    def group_tasks_for_coordination(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups compatible tasks on the same section into coordinated block proposals.
        """
        by_section: Dict[str, List[Dict[str, Any]]] = {}
        for t in tasks:
            sec = t.get("section", "UNKNOWN")
            by_section.setdefault(sec, []).append(t)

        coordinated_groups = []
        for sec, sec_tasks in by_section.items():
            depts = set(t["department"] for t in sec_tasks)
            if len(depts) > 1:
                coordinated_groups.append({
                    "section_id": sec,
                    "is_coordinated": True,
                    "departments": sorted(list(depts)),
                    "tasks": sec_tasks,
                    "max_duration_minutes": max(t.get("duration", 90) for t in sec_tasks),
                    "synergy_type": "MULTI_DEPARTMENT_INTEGRATED_BLOCK"
                })
            else:
                coordinated_groups.append({
                    "section_id": sec,
                    "is_coordinated": False,
                    "departments": list(depts),
                    "tasks": sec_tasks,
                    "max_duration_minutes": max(t.get("duration", 90) for t in sec_tasks),
                    "synergy_type": "SINGLE_DEPARTMENT_BLOCK"
                })
        return coordinated_groups
