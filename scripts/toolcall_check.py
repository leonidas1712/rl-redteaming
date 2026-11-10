"""Native tool-calling version of the check (OpenRouter models that accept `tools`).
Same 16 fixed-injection cases and the same 30 Qwen2.5-3B injections as the text-ReAct check.
Writes runs/precheck/toolcall_openrouter.jsonl.

Usage: python -I scripts/toolcall_check.py mistralai/ministral-3b-2512,liquid/lfm-3-1.7b
"""
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from redteam.episode import (ENHANCED_PREFIX, attacker_messages, build_toolcall_messages, extract_injection,
                             format_ok, grade_toolcall, load_cases, load_tools, tool_response_with_injection)

MODELS = sys.argv[1].split(",")
LEVELS_RUN = sys.argv[2].split(",") if len(sys.argv) > 2 else ["L2_default"]
OUT = os.path.join(ROOT, "runs", "precheck", "toolcall_openrouter.jsonl")
KEY = [l.split("=", 1)[1].strip().strip("\"'") for l in open(os.path.join(ROOT, ".env"))
       if l.startswith("OPENROUTER_API_KEY")][0]
tools = load_tools()
cases = load_cases(("dh",))
by_id = {c["task_id"]: c for c in cases}
random.seed(0)
fixed = random.sample(cases, 16)
full = {}
for l in open(os.path.join(ROOT, "runs", "precheck", "rollouts.jsonl")):
    r = json.loads(l)
    if r["source"] == "full_episode":
        full[(r["task_id"], r["attacker_output"])] = 1
jobs = [(m, lv, "fixed_injection", c, s, None) for m in MODELS for lv in LEVELS_RUN for c in fixed for s in ("base", "enhanced")]
jobs += [(m, lv, "full_episode", by_id[t], None, a) for m in MODELS for lv in LEVELS_RUN for (t, a) in full]


def call(model, messages, schemas):
    body = json.dumps({"model": model, "messages": messages, "tools": schemas, "temperature": 0.0,
                       "max_tokens": 512}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            msg = r["choices"][0]["message"]
            return msg.get("content"), msg.get("tool_calls"), r.get("provider"), None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            return None, None, None, err
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2)
    return None, None, None, err


def run(job):
    model, level, source, case, setting, attacker_out = job
    if source == "fixed_injection":
        inj = case["Attacker Instruction"] if setting == "base" else ENHANCED_PREFIX + case["Attacker Instruction"]
    else:
        inj = extract_injection(attacker_out)
    msgs, schemas = build_toolcall_messages(case, inj, tools, level)
    text, calls, prov, err = call(model, msgs, schemas)
    g = grade_toolcall(case, text, calls, attacker_out) if err is None else dict(
        reward=0.0, reason="target_error", eval=None, invalid_reason=err, attacker_tool_args=None)
    return {
        "run_id": "precheck", "format": "toolcall", "source": source, "setting": setting, "target_level": level,
        "task_id": case["task_id"], "goal_id": case["goal_id"], "set": case["set"],
        "attack_type": case["Attack Type"], "attacker_tool": case["Attacker Tools"][0],
        "goal": case["Attacker Instruction"], "expected_achievement": case["Expected Achievements"],
        "user_tool": case["User Tool"], "user_instruction": case["User Instruction"],
        "attacker_model": "Qwen/Qwen2.5-3B-Instruct" if attacker_out else None,
        "attacker_messages": attacker_messages(case) if attacker_out else None,
        "attacker_output": attacker_out, "format_ok": format_ok(attacker_out) if attacker_out else None,
        "injection": inj, "tool_response_with_injection": tool_response_with_injection(case, inj),
        "target_model": model, "target_source": "openrouter", "target_provider": prov, "target_temperature": 0.0,
        "target_messages": msgs, "target_tools": schemas,
        "target_output": text or "", "target_tool_calls": calls,
        "eval": g["eval"], "invalid_reason": g["invalid_reason"], "parsed": None,
        "attacker_tool_args": g["attacker_tool_args"], "reward": g["reward"], "reward_reason": g["reason"],
        "completion_sha1": hashlib.sha1((attacker_out or "").encode()).hexdigest()[:12],
    }


with ThreadPoolExecutor(4) as ex:
    recs = list(ex.map(run, jobs))
with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
import collections  # noqa: E402
for m in MODELS:
    for lv in LEVELS_RUN:
        for src in ("fixed_injection", "full_episode"):
            rs = [r for r in recs if r["target_model"] == m and r["source"] == src and r["target_level"] == lv]
            print(f"{m:30s} {lv:11s} {src:16s} n={len(rs)} {dict(collections.Counter(r['reward_reason'] for r in rs))}")
