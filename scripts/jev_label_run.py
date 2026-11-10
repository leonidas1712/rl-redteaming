"""Jev-label every target reply in a run's rollouts (taxonomy v1.1, scripts/jev_label.py). Parallel, with a cost cap.
`acted` comes from the code grader (reward 1); bad attacker format gets no target label. Resumable: labels already
in the output file are reused.

Usage: python -I scripts/jev_label_run.py runs/<run_id>/sweep_rollouts.jsonl [--cap 1.00]
Writes runs/<run_id>/jev_labels.jsonl (episode index, label, cost) and prints label shares per level.
"""
import collections
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from jev_label import ask, state_of  # noqa: E402

src = sys.argv[1]
cap = float(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 1.00
out_path = os.path.join(os.path.dirname(src), "jev_labels.jsonl")
rows = [json.loads(l) for l in open(src)]
done = {}
if os.path.exists(out_path):
    for l in open(out_path):
        d = json.loads(l)
        done[d["i"]] = d
lock = threading.Lock()
spent = [sum(d.get("cost", 0) for d in done.values())]
fout = open(out_path, "a")


def label(i):
    r = rows[i]
    if i in done:
        return done[i]
    if not r.get("format_ok"):
        d = {"i": i, "label": "(bad attacker format)", "cost": 0}
    elif r["reward"] == 1:
        d = {"i": i, "label": "acted", "cost": 0, "from": "code grader"}
    elif r.get("target_error"):
        d = {"i": i, "label": "(target error)", "cost": 0}
    else:
        with lock:
            if spent[0] >= cap:
                return {"i": i, "label": "(skipped: cost cap)", "cost": 0}
        choice, raw = ask("typesafe/jev-1.14", state_of(r))
        cost = float((raw.get("usage") or {}).get("cost") or 0)
        d = {"i": i, "label": choice or "ERROR", "cost": cost}
    with lock:
        spent[0] += d["cost"]
        fout.write(json.dumps(d) + "\n")
        fout.flush()
    return d


with ThreadPoolExecutor(12) as ex:
    labels = list(ex.map(label, range(len(rows))))
print(f"spent ${spent[0]:.3f} (cap ${cap})")
by = collections.defaultdict(collections.Counter)
for r, d in zip(rows, labels):
    by[r.get("target_level", "-")][d["label"]] += 1
for lv in sorted(by):
    tot = sum(by[lv].values())
    print(lv, ", ".join(f"{k} {v} ({100 * v / tot:.0f}%)" for k, v in by[lv].most_common()))
