"""Viewer sample of a sweep: all successes + a seeded random sample of failures, each failure weighted so that
weighted averages (ASR) in the viewer equal the full run's. Adds Jev labels if runs/<run>/jev_labels.jsonl exists.
Usage: python3 scripts/make_sweep_viewer_sample.py runs/<run_id> [n_failures]"""
import json, os, random, sys
run = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 400
R = [json.loads(l) for l in open(os.path.join(run, "sweep_rollouts.jsonl"))]
jev = {}
jp = os.path.join(run, "jev_labels.jsonl")
if os.path.exists(jp):
    for l in open(jp):
        d = json.loads(l); jev[d["i"]] = d["label"]
for i, r in enumerate(R):
    r["episode_id"] = f"sw-{i:05d}"
    if i in jev: r["jev_label"] = jev[i]
succ = [r for r in R if r["reward"] == 1]; fail = [r for r in R if r["reward"] != 1]
# stratify by ladder level so per-level weighted ASR equals the full run's
levels = sorted({r["target_level"] for r in R}); rng = random.Random(0); pick = []
for lv in levels:
    fl = [r for r in fail if r["target_level"] == lv]
    p = rng.sample(fl, min(k // len(levels), len(fl)))
    for r in p: r["sample_weight"] = len(fl) / len(p)
    pick += p
for r in succ: r["sample_weight"] = 1.0
out = sorted(succ + pick, key=lambda r: r["episode_id"])
with open(os.path.join(run, "viewer_sample.jsonl"), "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
full = sum(r["reward"] for r in R) / len(R); w = sum(r["sample_weight"] * r["reward"] for r in out) / sum(r["sample_weight"] for r in out)
for lv in levels:
    a = [r for r in R if r["target_level"] == lv]; b = [r for r in out if r["target_level"] == lv]
    print(lv, f"full {sum(r['reward'] for r in a)/len(a):.4f} weighted {sum(r['sample_weight']*r['reward'] for r in b)/sum(r['sample_weight'] for r in b):.4f}")
print(f"{len(succ)} successes + {len(pick)} failures; full-run ASR {full:.4f}, weighted sample ASR {w:.4f}; jev labels: {len(jev)}")
