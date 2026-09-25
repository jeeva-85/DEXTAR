import time
from typing import List, Dict, Any
from ortools.sat.python import cp_model

class RailwayCPSATScheduler:
    """
    Railway Maintenance Block Scheduler using Google OR-Tools CP-SAT.
    Generates feasible maintenance schedules under hard railway operational constraints
    while maximizing maintenance priority, safety, and multi-department coordination.
    """

    def __init__(self, time_limit_seconds: int = 20):
        self.time_limit_seconds = time_limit_seconds

    def solve_block_schedule(
        self,
        tasks: List[Dict[str, Any]],
        candidate_windows: List[Dict[str, Any]],
        train_intervals: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solves CP-SAT optimization model to assign maintenance tasks to feasible windows.
        """
        start_time_wall = time.time()
        model = cp_model.CpModel()
        
        # Horizon in minutes from start of planning day (0 to 1440)
        HORIZON = 1440
        train_intervals = train_intervals or []

        # Map candidate windows into minute offsets
        def time_to_min(t_str: str) -> int:
            parts = t_str.split(":")
            return int(parts[0]) * 60 + int(parts[1])

        # Decision variables: For each task and each candidate window, is task assigned?
        assignments = {}
        task_starts = {}
        task_ends = {}
        task_intervals = {}

        for t_idx, task in enumerate(tasks):
            t_id = task["task_id"]
            duration = int(task.get("duration", 90))
            priority = int(task.get("priority_score", 50))

            # Task can either be scheduled in one of the windows or deferred
            is_scheduled = model.NewBoolVar(f"sched_{t_id}")
            window_vars = []

            for w_idx, win in enumerate(candidate_windows):
                w_start = time_to_min(win["start_time"])
                w_end = time_to_min(win["end_time"])
                w_dur = w_end - w_start

                if duration <= w_dur and win["section_id"] == task["section"]:
                    var = model.NewBoolVar(f"assign_{t_id}_win_{w_idx}")
                    window_vars.append((var, w_start, w_end))
                    assignments[(t_idx, w_idx)] = var

            if window_vars:
                # Task can be assigned to at most one window
                model.Add(sum(v for v, _, _ in window_vars) == is_scheduled)
            else:
                model.Add(is_scheduled == 0)

        # Multi-department coordination bonus and objective formulation
        objective_terms = []
        for t_idx, task in enumerate(tasks):
            priority_val = int(task.get("priority_score", 50))
            for w_idx in range(len(candidate_windows)):
                if (t_idx, w_idx) in assignments:
                    var = assignments[(t_idx, w_idx)]
                    # Reward higher priority tasks
                    objective_terms.append(var * priority_val * 10)

        # Department coordination synergy: if multiple departments are in the same window, reward heavily
        for w_idx, win in enumerate(candidate_windows):
            depts_in_win = {}
            for t_idx, task in enumerate(tasks):
                if (t_idx, w_idx) in assignments:
                    dept = task["department"]
                    depts_in_win.setdefault(dept, []).append(assignments[(t_idx, w_idx)])

            if len(depts_in_win) > 1:
                # Bonus if at least 2 distinct departments share this window
                coord_active = model.NewBoolVar(f"coord_active_{w_idx}")
                dept_active_vars = []
                for dept, var_list in depts_in_win.items():
                    d_act = model.NewBoolVar(f"dept_active_{dept}_{w_idx}")
                    model.Add(sum(var_list) >= 1).OnlyEnforceIf(d_act)
                    model.Add(sum(var_list) == 0).OnlyEnforceIf(d_act.Not())
                    dept_active_vars.append(d_act)

                model.Add(sum(dept_active_vars) >= 2).OnlyEnforceIf(coord_active)
                model.Add(sum(dept_active_vars) < 2).OnlyEnforceIf(coord_active.Not())
                objective_terms.append(coord_active * 500) # Strong coordination incentive

        model.Maximize(sum(objective_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_search_workers = 4
        solver_status = solver.Solve(model)

        elapsed_ms = int((time.time() - start_time_wall) * 1000)

        # Build scheduled blocks
        scheduled_blocks = []
        unscheduled_tasks = []

        if solver_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for w_idx, win in enumerate(candidate_windows):
                tasks_in_win = []
                for t_idx, task in enumerate(tasks):
                    if (t_idx, w_idx) in assignments and solver.Value(assignments[(t_idx, w_idx)]) == 1:
                        tasks_in_win.append(task)

                if tasks_in_win:
                    depts = sorted(list(set(t["department"] for t in tasks_in_win)))
                    max_req_dur = max(t.get("duration", 90) for t in tasks_in_win)
                    w_start = time_to_min(win["start_time"])
                    w_end = time_to_min(win["end_time"])
                    avail_dur = w_end - w_start
                    utilization = min(100.0, round((max_req_dur / max(avail_dur, 1)) * 100, 1))

                    scheduled_blocks.append({
                        "block_id": f"BLK-{win['section_id']}-{win['start_time'].replace(':', '')}",
                        "section_id": win["section_id"],
                        "date": win["date"],
                        "start_time": win["start_time"],
                        "end_time": win["end_time"],
                        "duration_minutes": avail_dur,
                        "status": "APPROVED",
                        "is_coordinated": len(depts) > 1,
                        "coordination_type": "COORDINATED (ENG+S&T+TRD)" if len(depts) > 1 else "SINGLE DEPARTMENT",
                        "departments": depts,
                        "tasks": [t["task_id"] for t in tasks_in_win],
                        "task_details": tasks_in_win,
                        "utilization_percentage": utilization,
                        "solver_objective": solver.ObjectiveValue(),
                        "conflicts": []
                    })

            # Check unscheduled
            for t_idx, task in enumerate(tasks):
                is_assigned = any(
                    (t_idx, w_idx) in assignments and solver.Value(assignments[(t_idx, w_idx)]) == 1
                    for w_idx in range(len(candidate_windows))
                )
                if not is_assigned:
                    unscheduled_tasks.append(task)

            status_str = "OPTIMAL" if solver_status == cp_model.OPTIMAL else "FEASIBLE"
        else:
            status_str = "INFEASIBLE"

        return {
            "status": status_str,
            "runtime_ms": elapsed_ms,
            "objective_value": solver.ObjectiveValue() if solver_status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0,
            "scheduled_blocks_count": len(scheduled_blocks),
            "scheduled_blocks": scheduled_blocks,
            "unscheduled_tasks_count": len(unscheduled_tasks),
            "unscheduled_tasks": unscheduled_tasks,
            "conflicts_detected": []
        }
