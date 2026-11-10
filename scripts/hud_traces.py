"""List HUD's local traces (~/.hud/spans) as a table: time, reward, task, start of the injection, trace id.

  .venv/bin/python scripts/hud_traces.py              # all traces, oldest first
  .venv/bin/python scripts/hud_traces.py --last 20    # newest 20
  .venv/bin/python scripts/hud_traces.py --reward 1   # only successes (or 0 for failures)

Then open one in full:  .venv/bin/hud trace get <trace id>
The trace does not record which model answered; runs are told apart by time (see LOG.md 21:54: base ran first, then step80).
"""
import argparse
import ast
import glob
import json
import os


def payload(span):
    p = span.get("attributes", {}).get("hud.payload")
    if isinstance(p, str):
        try:
            return json.loads(p)
        except json.JSONDecodeError:
            return ast.literal_eval(p)
    return p or {}


def reward_of(result):
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        for k in ("reward", "value", "score"):
            if isinstance(result.get(k), (int, float)):
                return float(result[k])
    return None


def read(path):
    row = dict(trace=os.path.basename(path)[:-6], time="", task="", reward=None, injection="")
    for line in open(path):
        span = json.loads(line)
        p = payload(span)
        call = p.get("task_call") or {}
        if call.get("phase") == "setup":
            row["task"] = call.get("arguments", {}).get("task_id", "")
        elif call.get("phase") == "evaluate":
            row["reward"] = reward_of(call.get("result"))
        if span["name"] == "step.agent":
            row["time"] = (p.get("started_at") or "")[11:19]
            text = p.get("content") or ""
            inj = text.split("<prompt>", 1)[-1].replace("\n", " ").strip()
            row["injection"] = inj[:70]
    return row


ap = argparse.ArgumentParser()
ap.add_argument("--last", type=int)
ap.add_argument("--reward", type=float)
a = ap.parse_args()

rows = sorted((read(f) for f in glob.glob(os.path.expanduser("~/.hud/spans/*.jsonl"))), key=lambda r: r["time"])
if a.reward is not None:
    rows = [r for r in rows if r["reward"] == a.reward]
if a.last:
    rows = rows[-a.last:]
for r in rows:
    rw = "?" if r["reward"] is None else f"{r['reward']:.0f}"
    print(f"{r['time']} UTC  r={rw}  {r['task'][:55]:55}  {r['injection']:70}  {r['trace']}")
print(f"\n{len(rows)} traces")
