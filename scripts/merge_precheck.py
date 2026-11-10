"""Append runs/precheck/local_targets.jsonl to runs/precheck/rollouts.jsonl with new episode ids.
Existing ids never change (viewer links stay valid). Re-running replaces the previously merged local batch."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
main_p = os.path.join(ROOT, "runs", "precheck", "rollouts.jsonl")
new_p = os.path.join(ROOT, "runs", "precheck", "local_targets.jsonl")
base = [json.loads(l) for l in open(main_p)]
base = [r for r in base if r.get("batch") != "local_targets"]
nxt = max(int(r["episode_id"][2:]) for r in base) + 1
new = [json.loads(l) for l in open(new_p)]
for i, r in enumerate(new):
    r["batch"] = "local_targets"
    r["episode_id"] = f"ep{nxt + i:04d}"
with open(main_p, "w") as f:
    for r in base + new:
        f.write(json.dumps(r) + "\n")
print(f"{len(base)} existing + {len(new)} new = {len(base) + len(new)} episodes")
