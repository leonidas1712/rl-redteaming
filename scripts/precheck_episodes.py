"""Collect all pre-GPU check episodes into runs/precheck/rollouts.jsonl (the planned rollout schema).

Sources:
  1. fixed-injection target checks (InjecAgent's own base / enhanced injection; no attacker model)
     - local MPS Qwen runs  (scratchpad mc_target.jsonl)
     - OpenRouter runs      (scratchpad or_check.jsonl)
  2. NEW full episodes: Qwen2.5-3B attacks (sampled locally earlier) -> OpenRouter targets, graded here.
  3. attacker-only samples (format check), no target.

Usage: python -I scripts/precheck_episodes.py <scratchpad_dir>
"""
import collections
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from redteam.episode import (ENHANCED_PREFIX, attacker_messages, build_target_messages, extract_injection,
                             format_ok, grade, load_cases, load_tools, tool_response_with_injection)

SCRATCH = sys.argv[1]
OUT_DIR = os.path.join(ROOT, "runs", "precheck")
os.makedirs(OUT_DIR, exist_ok=True)
KEY = [l.split("=", 1)[1].strip().strip("\"'") for l in open(os.path.join(ROOT, ".env"))
       if l.startswith("OPENROUTER_API_KEY")][0]
NEW_TARGETS = ["google/gemma-3-4b-it", "mistralai/ministral-3b-2512", "meta-llama/llama-3.2-3b-instruct"]

tools = load_tools()
cases = load_cases(("dh",))
by_task = {(c["Attacker Tools"][0], c["User Tool"]): c for c in cases}
by_goal = collections.defaultdict(list)
for c in cases:
    by_goal[c["Attacker Tools"][0]].append(c)


def read_jsonl(name):
    p = os.path.join(SCRATCH, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def chat(model, messages, temperature, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {KEY}",
                                                  "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"].get("content") or "", r.get("provider")
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return None, f"error: {err}"


def record(source, case, target_model, target_source, injection, target_output, provider=None,
           attacker_model=None, attacker_output=None, setting=None):
    g = grade(case, injection, target_output, attacker_output)
    return {
        "run_id": "precheck", "source": source, "setting": setting,
        "task_id": case["task_id"], "goal_id": case["goal_id"], "set": case["set"],
        "attack_type": case["Attack Type"], "attacker_tool": case["Attacker Tools"][0],
        "goal": case["Attacker Instruction"], "expected_achievement": case["Expected Achievements"],
        "user_tool": case["User Tool"], "user_instruction": case["User Instruction"],
        "attacker_model": attacker_model,
        "attacker_messages": attacker_messages(case) if attacker_output is not None else None,
        "attacker_output": attacker_output,
        "format_ok": format_ok(attacker_output) if attacker_output is not None else None,
        "injection": injection,
        "tool_response_with_injection": tool_response_with_injection(case, injection),
        "target_model": target_model, "target_source": target_source, "target_provider": provider,
        "target_temperature": 0.0,
        "target_messages": build_target_messages(case, injection, tools) if target_model else None,
        "target_output": target_output,
        "eval": g["eval"], "invalid_reason": g["invalid_reason"], "parsed": g["info_list"],
        "attacker_tool_args": g["attacker_tool_args"],
        "reward": g["reward"], "reward_reason": g["reason"],
        "completion_sha1": hashlib.sha1((attacker_output or "").encode()).hexdigest()[:12],
    }


records = []

# 1. fixed-injection target checks
for r in read_jsonl("mc_target.jsonl") + [x for x in read_jsonl("or_check.jsonl") if x["kind"] == "target"]:
    case = by_task[(r["attacker_tool"], r["user_tool"])]
    inj = case["Attacker Instruction"] if r["setting"] == "base" else ENHANCED_PREFIX + case["Attacker Instruction"]
    src = "local-mps" if "mode" in r else "openrouter"
    records.append(record("fixed_injection", case, r["model"], src, inj, r["output"],
                          provider=r.get("provider"), setting=r["setting"]))

# 3. attacker-only samples
for r in read_jsonl("mc_attacker.jsonl"):
    case = by_task[(r["attacker_tool"], r["user_tool"])]
    rec = record("attacker_only", case, None, None, extract_injection(r["output"]), None,
                 attacker_model=r["model"], attacker_output=r["output"])
    rec["reward_reason"] = "no_target" if rec["format_ok"] else "bad_format"
    records.append(rec)
for r in read_jsonl("or_check.jsonl"):
    if r["kind"] != "attacker":
        continue
    case = by_goal[[c for c in cases if c["Attacker Instruction"] == r["goal"]][0]["Attacker Tools"][0]][0]
    rec = record("attacker_only", case, None, None, extract_injection(r["output"] or ""), None,
                 attacker_model=r["model"], attacker_output=r["output"] or "")
    rec["reward_reason"] = "no_target" if rec["format_ok"] else "bad_format"
    records.append(rec)

# 2. NEW full episodes: Qwen2.5-3B attacks -> each target, on the sampled case + one more user case of that goal
jobs = []
for r in read_jsonl("mc_attacker.jsonl"):
    if r["model"] != "Qwen/Qwen2.5-3B-Instruct":
        continue
    first = by_task[(r["attacker_tool"], r["user_tool"])]
    second = [c for c in by_goal[r["attacker_tool"]] if c["User Tool"] != r["user_tool"]][0]
    for case in (first, second):
        for t in NEW_TARGETS:
            jobs.append((case, t, r["output"]))


def run(job):
    case, t, attacker_out = job
    inj = extract_injection(attacker_out)
    out, prov = chat(t, build_target_messages(case, inj, tools), 0.0, 512)
    return record("full_episode", case, t, "openrouter", inj, out, provider=prov,
                  attacker_model="Qwen/Qwen2.5-3B-Instruct", attacker_output=attacker_out)


with ThreadPoolExecutor(16) as ex:
    records += list(ex.map(run, jobs))

for i, rec in enumerate(records):
    rec["episode_id"] = f"ep{i:04d}"
with open(os.path.join(OUT_DIR, "rollouts.jsonl"), "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

full = [r for r in records if r["source"] == "full_episode"]
print(len(records), "records;", len(full), "new full episodes")
for t in NEW_TARGETS:
    rs = [r for r in full if r["target_model"] == t]
    print(t, collections.Counter(r["reward_reason"] for r in rs))
