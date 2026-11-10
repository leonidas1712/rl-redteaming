"""HUD taskset: 12 DEV tasks, one per dev goal (first scenario alphabetically). Never test tasks (pre-registration F5).
Calling the template mints a task row without running anything."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import CASES, attack  # noqa: E402

from redteam.splits import load_split  # noqa: E402

dev = load_split("dev", list(CASES.values()))
first_per_goal = {}
for c in sorted(dev, key=lambda c: c["task_id"]):
    first_per_goal.setdefault(c["goal_id"], c["task_id"])

tasks = [attack(task_id=t) for t in sorted(first_per_goal.values())]
