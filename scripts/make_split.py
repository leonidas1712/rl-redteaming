"""Make the fixed goal-level split. Run once per version; the output file is the source of truth.

v2 (current, 2026-11-10 16:00, Leo): used goals go to TRAIN (seeing a goal is fine for training; it is not fine
for dev, which drives decisions). DEV = 6 dh + 6 ds fresh untouched goals. TEST identical to v1.
v1 (superseded, kept for the record): used goals went to DEV.


Rules (v2):
  - Split by attacker GOAL (all 17 user cases of a goal stay together), pooling direct-harm (dh) and
    data-stealing stage 1 (ds).
  - Every goal already touched by a pre-GPU check (USED_IN_PRECHECKS: 17 dh, 0 ds, found by scanning every log on
    2026-11-10 15:45) goes to TRAIN.
  - TEST = 6 dh + 6 ds untouched goals (seed 0; identical to v1). Locked: only read with RL_REDTEAM_FINAL_EVAL=1.
  - DEV = the next 6 dh + 6 ds untouched goals (seed 0). Everything else is TRAIN.
  - Canary = 16 fixed dev tasks (seed 0) for the every-25-steps check during training.
"""
import hashlib
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "v2"
OUT = os.path.join(ROOT, f"data/splits/split_{VERSION}.json")
USED_IN_PRECHECKS = sorted([
    "dh:AugustSmartLockUnlockDoor", "dh:BinancePlaceOrder", "dh:CiscoUmbrellaCreatePolicy", "dh:DropboxMoveItem",
    "dh:EpicFHIRManageAppointments", "dh:EpicFHIRManageClinicalDocuments", "dh:EpicFHIRManagePatientRecords",
    "dh:EvernoteManagerManageNoteSharingLevel", "dh:FacebookManagerUpdateUserProfile",
    "dh:NortonIdentitySafeAutoFillPersonalInfo", "dh:NortonIdentitySafeSharePassword", "dh:SlackLeaveChannel",
    "dh:TerminalExecute", "dh:The23andMeShareData", "dh:The23andMeUpdateData",
    "dh:TrafficControlGivePriorityToEmergencyVehicle", "dh:VenmoWithdrawMoney"])

assert not os.path.exists(OUT), f"{OUT} exists: the split is fixed. Make a new version instead of overwriting."
files = {s: os.path.join(ROOT, f"data/injecagent/test_cases_{s}_base.json") for s in ("dh", "ds")}
cases = {s: json.load(open(p)) for s, p in files.items()}
goals = {s: sorted({f"{s}:{c['Attacker Tools'][0]}" for c in cases[s]}) for s in cases}
assert set(USED_IN_PRECHECKS) <= set(goals["dh"])

rng = random.Random(0)
dh_free = [g for g in goals["dh"] if g not in USED_IN_PRECHECKS]
ds_free = list(goals["ds"])
rng.shuffle(dh_free)
rng.shuffle(ds_free)
if VERSION == "v1":
    split = {"test": sorted(dh_free[:6] + ds_free[:6]),
             "dev": sorted(USED_IN_PRECHECKS + ds_free[6:10]),
             "train": sorted(dh_free[6:] + ds_free[10:])}
else:  # v2: same shuffles, so the test draw is identical to v1
    split = {"test": sorted(dh_free[:6] + ds_free[:6]),
             "dev": sorted(dh_free[6:12] + ds_free[6:12]),
             "train": sorted(USED_IN_PRECHECKS + dh_free[12:] + ds_free[12:])}
task_ids = {k: [] for k in split}
goal_to_split = {g: k for k, gs in split.items() for g in gs}
for s in cases:
    for c in cases[s]:
        g = f"{s}:{c['Attacker Tools'][0]}"
        task_ids[goal_to_split[g]].append(f"{g}|{c['User Tool']}")
for k in task_ids:
    task_ids[k].sort()
canary = sorted(random.Random(0).sample(task_ids["dev"], 16))

doc = {
    "version": f"split_{VERSION}", "created": "2026-11-10", "seed": 0,
    "rules": __doc__.strip(),
    "source_sha256": {s: hashlib.sha256(open(p, "rb").read()).hexdigest() for s, p in files.items()},
    "used_in_prechecks": USED_IN_PRECHECKS,
    "counts": {k: {"goals": len(split[k]), "tasks": len(task_ids[k]),
                   "dh_goals": sum(g.startswith("dh:") for g in split[k]),
                   "ds_goals": sum(g.startswith("ds:") for g in split[k])} for k in split},
    "goals": split, "task_ids": task_ids, "canary_task_ids": canary,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(doc, open(OUT, "w"), indent=1)
os.chmod(OUT, 0o444)  # read-only on disk as a reminder
print(json.dumps(doc["counts"], indent=1))
