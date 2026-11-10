"""Phase 2: base-rate sweep. Untrained attacker (served by vLLM: fine, nothing is trained) writes n attacks per dev
task; every attack is sent to the target at each difficulty level; graded; per-level statistics.

Runs inside the Modal container with two vLLM servers up: attacker (ATTACKER_URL) and target (TARGET_URL).
"""
import collections
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from redteam.attacker_prompts import attacker_messages
from redteam.episode import (build_toolcall_messages, extract_injection, format_ok, grade_toolcall, load_cases,
                             load_tools, tool_response_with_injection)
from redteam.splits import load_split
from redteam.stats import summarize
from redteam.target import VLLMTarget


def run(out_dir, run_id, attacker_model, attacker_url, target_model, target_url, split="dev",
        variant="v2_context_hints", levels=("L0_obey", "L1_lenient", "L2_default", "L3_guarded"),
        n=8, max_tokens=256, temperature=1.0, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    tools = load_tools()
    tasks = load_split(split, load_cases(("dh", "ds")))
    att = OpenAI(base_url=attacker_url, api_key="EMPTY", timeout=600)
    summary = {"run_id": run_id, "split": split, "variant": variant, "n": n, "levels": list(levels),
               "attacker_model": attacker_model, "target_model": target_model, "tasks": len(tasks)}

    # 1. attacks: one request per task, n samples each, per-task seed for reproducibility
    def gen(i_task):
        i, c = i_task
        r = att.chat.completions.create(model=attacker_model, messages=attacker_messages(c, variant), n=n,
                                        temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                                        # vLLM seeds the j-th of n choices with seed+j, and all 17 tasks of a goal share
                                        # one attacker prompt: seeds spaced by 1 made task i sample k == task i+1 sample
                                        # k-1 (sweep 20261110-1619: 1018/1632 distinct). Space them by n.
                                        seed=seed * 1_000_000 + i * n)
        return [(ch.message.content or "", ch.finish_reason) for ch in r.choices]
    t0 = time.time()
    with ThreadPoolExecutor(64) as ex:
        gens = list(ex.map(gen, enumerate(tasks)))
    summary["t_attacks_s"] = round(time.time() - t0, 1)

    # 2. every attack x every level -> target
    target = VLLMTarget(target_model, base_url=target_url, concurrency=128)
    jobs = []
    for c, samples in zip(tasks, gens):
        for k, (txt, fin) in enumerate(samples):
            inj = extract_injection(txt)
            for lv in levels:
                msgs, schemas = build_toolcall_messages(c, inj, tools, lv)
                jobs.append((c, k, txt, fin, inj, lv, msgs, schemas))
    t0 = time.time()
    results = target.batch([(j[6], j[7]) for j in jobs])
    summary["t_target_s"] = round(time.time() - t0, 1)
    summary["target_calls"] = len(jobs)

    rows = []
    for (c, k, txt, fin, inj, lv, msgs, schemas), res in zip(jobs, results):
        g = (dict(reward=0.0, reason="target_error", eval=None, invalid_reason=res["error"], attacker_tool_args=None)
             if res["error"] else grade_toolcall(c, res["text"], res["tool_calls"], txt))
        rows.append({
            "run_id": run_id, "phase": "phase2_sweep", "split": split, "format": "toolcall", "source": f"sweep {variant}",
            "attacker_prompt_variant": variant, "target_level": lv, "sample_idx": k, "seed": seed,
            "task_id": c["task_id"], "goal_id": c["goal_id"], "set": c["set"], "attack_type": c["Attack Type"],
            "attacker_tool": c["Attacker Tools"][0], "goal": c["Attacker Instruction"],
            "expected_achievement": c["Expected Achievements"], "user_tool": c["User Tool"],
            "user_instruction": c["User Instruction"], "attacker_model": attacker_model,
            "attacker_messages": attacker_messages(c, variant), "attacker_output": txt,
            "attacker_finish_reason": fin, "format_ok": format_ok(txt), "injection": inj,
            "tool_response_with_injection": tool_response_with_injection(c, inj),
            "target_model": target_model, "target_source": "vllm-modal", "target_temperature": 0.0,
            "target_messages": msgs, "target_tools": schemas, "target_output": res["text"] or "",
            "target_tool_calls": res["tool_calls"], "target_error": res["error"],
            "eval": g["eval"], "invalid_reason": g["invalid_reason"], "attacker_tool_args": g["attacker_tool_args"],
            "reward": g["reward"], "reward_reason": g["reason"]})
    with open(os.path.join(out_dir, "sweep_rollouts.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # 3. statistics per level, overall and per set (dh / ds)
    summary["format_ok"] = f"{sum(r['format_ok'] for r in rows if r['target_level'] == levels[0])}/{len(tasks) * n}"
    summary["target_errors"] = sum(bool(r["target_error"]) for r in rows)
    summary["by_level"] = {}
    for lv in levels:
        entry = {}
        for name, keep in (("all", lambda r: True), ("dh", lambda r: r["set"] == "dh"), ("ds", lambda r: r["set"] == "ds")):
            per_task = collections.defaultdict(list)
            for r in rows:
                if r["target_level"] == lv and keep(r):
                    per_task[r["task_id"]].append(r["reward"])
            entry[name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summarize(per_task).items()}
        entry["reasons"] = dict(collections.Counter(r["reward_reason"] for r in rows if r["target_level"] == lv))
        summary["by_level"][lv] = entry
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
