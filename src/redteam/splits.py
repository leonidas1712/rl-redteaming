"""Read the fixed goal-level split (data/splits/split_v2.json, made by scripts/make_split.py).

The test split is locked: load_split("test") raises unless RL_REDTEAM_FINAL_EVAL=1 is set, so nothing touches
test goals before the final evaluation by accident.
"""
import json
import os

SPLIT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "splits", "split_v2.json")  # v1 superseded


def _doc():
    return json.load(open(SPLIT_PATH))


def load_split(name, cases):
    """name: train | dev | test | canary. cases: list from episode.load_cases(("dh", "ds")). Returns the cases."""
    if name == "test" and os.environ.get("RL_REDTEAM_FINAL_EVAL") != "1":
        raise PermissionError("test split is locked until the final eval (set RL_REDTEAM_FINAL_EVAL=1)")
    d = _doc()
    ids = set(d["canary_task_ids"] if name == "canary" else d["task_ids"][name])
    out = [c for c in cases if c["task_id"] in ids]
    assert len(out) == len(ids), f"{name}: found {len(out)} of {len(ids)} tasks"
    return out


def goal_split(goal_id):
    """Which split a goal belongs to (safe to call for any goal; reveals membership, not data)."""
    for k, goals in _doc()["goals"].items():
        if goal_id in goals:
            return k
    raise KeyError(goal_id)
