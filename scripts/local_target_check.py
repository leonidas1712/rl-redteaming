"""Run extra target models locally (Mac MPS, HF transformers, greedy) on the same check episodes as OpenRouter:
  - 16 fixed-injection cases (InjecAgent base + enhanced), same sample as or_check.py (random.seed(0))
  - the 30 Qwen2.5-3B full-episode injections already in runs/precheck/rollouts.jsonl
Appends records to runs/precheck/local_targets.jsonl (same schema). Then merge with merge_precheck.py.

Usage: python -I scripts/local_target_check.py google/gemma-2-2b-it,google/gemma-3-1b-it
"""
import hashlib
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from redteam.episode import (ENHANCED_PREFIX, attacker_messages, build_target_messages, format_ok, grade,
                             load_cases, load_tools, tool_response_with_injection)

MODELS = sys.argv[1].split(",")
OUT = os.path.join(ROOT, "runs", "precheck", "local_targets.jsonl")
tools = load_tools()
cases = load_cases(("dh",))
by_id = {c["task_id"]: c for c in cases}
random.seed(0)
fixed = random.sample(cases, 16)  # same draw as or_check.py
full = {}
for l in open(os.path.join(ROOT, "runs", "precheck", "rollouts.jsonl")):
    r = json.loads(l)
    if r["source"] == "full_episode":
        full[(r["task_id"], r["attacker_output"])] = r

jobs = [("fixed_injection", c, s, None) for c in fixed for s in ("base", "enhanced")]
jobs += [("full_episode", by_id[tid], None, out) for (tid, out) in full]


def chat_input(tok, messages):
    """Gemma-2's template rejects a system role: fold it into the first user message (what Gemma-3's template does)."""
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True), False
    except Exception:  # noqa: BLE001
        merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
        return tok.apply_chat_template(merged, add_generation_prompt=True, return_tensors="pt", return_dict=True), True


with open(OUT, "a") as f:
    for name in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()
        except Exception as e:  # noqa: BLE001
            print(f"{name}: could not load ({type(e).__name__}: {str(e)[:200]})", flush=True)
            continue
        t0 = time.time()
        for source, case, setting, attacker_out in jobs:
            if source == "fixed_injection":
                inj = case["Attacker Instruction"] if setting == "base" else ENHANCED_PREFIX + case["Attacker Instruction"]
            else:
                inj = attacker_out.split("<prompt>")[1].split("</prompt>")[0]
            msgs = build_target_messages(case, inj, tools)
            ids, merged = chat_input(tok, msgs)
            ids = ids.to("mps")
            with torch.no_grad():
                o = model.generate(**ids, do_sample=False, max_new_tokens=512, pad_token_id=tok.eos_token_id)
            text = tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            g = grade(case, inj, text, attacker_out)
            f.write(json.dumps({
                "run_id": "precheck", "source": source, "setting": setting,
                "task_id": case["task_id"], "goal_id": case["goal_id"], "set": case["set"],
                "attack_type": case["Attack Type"], "attacker_tool": case["Attacker Tools"][0],
                "goal": case["Attacker Instruction"], "expected_achievement": case["Expected Achievements"],
                "user_tool": case["User Tool"], "user_instruction": case["User Instruction"],
                "attacker_model": "Qwen/Qwen2.5-3B-Instruct" if attacker_out else None,
                "attacker_messages": attacker_messages(case) if attacker_out else None,
                "attacker_output": attacker_out, "format_ok": format_ok(attacker_out) if attacker_out else None,
                "injection": inj, "tool_response_with_injection": tool_response_with_injection(case, inj),
                "target_model": name, "target_source": "local-mps" + (" (system folded into user)" if merged else ""),
                "target_provider": None, "target_temperature": 0.0, "target_messages": msgs,
                "target_output": text, "eval": g["eval"], "invalid_reason": g["invalid_reason"],
                "parsed": g["info_list"], "attacker_tool_args": g["attacker_tool_args"],
                "reward": g["reward"], "reward_reason": g["reason"],
                "completion_sha1": hashlib.sha1((attacker_out or "").encode()).hexdigest()[:12],
            }) + "\n")
            f.flush()
        print(f"{name}: {len(jobs)} episodes in {time.time()-t0:.0f}s", flush=True)
        del model
        torch.mps.empty_cache()
