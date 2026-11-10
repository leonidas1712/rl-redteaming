"""Evaluate attacker checkpoints (LoRA adapters) on a split: n attacks per task, each sent to the target at the given
ladder levels; per-adapter, per-level statistics plus per-task successes for the paired bootstrap.

The attacker is served by vLLM with LoRA support (fine for evaluation: nothing is trained). Adapter "base" = no LoRA,
i.e. the untrained attacker. Runs inside a Modal container with two vLLM servers up (attacker, target).
"""
import collections
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from redteam.attacker_prompts import attacker_messages
from redteam.episode import (build_toolcall_messages, extract_injection, format_ok, grade_toolcall, load_cases,
                             load_tools)
from redteam.splits import load_split
from redteam.stats import summarize
from redteam.target import VLLMTarget


def run(out_dir, run_id, attacker_url, base_model, adapters, target_model, target_url, split="dev",
        variant="v2_context_hints", levels=("L2_default", "L0_obey"), n=8, max_tokens=256, seed=0):
    """adapters: list of served model names ("base" means base_model itself; others are vLLM LoRA module names)."""
    os.makedirs(out_dir, exist_ok=True)
    tools = load_tools()
    tasks = load_split(split, load_cases(("dh", "ds")))
    att = OpenAI(base_url=attacker_url, api_key="EMPTY", timeout=900)
    target = VLLMTarget(target_model, base_url=target_url, concurrency=128)
    summary = {"run_id": run_id, "split": split, "n": n, "levels": list(levels), "adapters": adapters, "by": {}}
    per_task_out = {}
    f = open(os.path.join(out_dir, "eval_rollouts.jsonl"), "w")
    for ad in adapters:
        served = base_model if ad == "base" else ad

        def gen(i_task):
            i, c = i_task
            r = att.chat.completions.create(model=served, messages=attacker_messages(c, variant), n=n, temperature=1.0,
                                            top_p=1.0, max_tokens=max_tokens, seed=seed * 1_000_000 + i * n)
            return [ch.message.content or "" for ch in r.choices]
        t0 = time.time()
        with ThreadPoolExecutor(64) as ex:
            gens = list(ex.map(gen, enumerate(tasks)))
        t_gen = time.time() - t0
        for lv in levels:
            jobs = [(c, k, txt, extract_injection(txt)) for c, samples in zip(tasks, gens) for k, txt in enumerate(samples)]
            res = target.batch([build_toolcall_messages(c, inj, tools, lv) for c, _, _, inj in jobs])
            per_task = collections.defaultdict(list)
            for (c, k, txt, inj), r in zip(jobs, res):
                g = (dict(reward=0.0, reason="target_error") if r["error"]
                     else grade_toolcall(c, r["text"], r["tool_calls"], txt))
                per_task[c["task_id"]].append(g["reward"])
                f.write(json.dumps({"run_id": run_id, "adapter": ad, "split": split, "target_level": lv,
                                    "task_id": c["task_id"], "goal_id": c["goal_id"], "set": c["set"], "sample_idx": k,
                                    "attacker_output": txt, "format_ok": format_ok(txt), "injection": inj,
                                    "target_output": r["text"] or "", "target_tool_calls": r["tool_calls"],
                                    "target_error": r["error"], "reward": g["reward"], "reward_reason": g["reason"]}) + "\n")
            f.flush()
            key = f"{ad}@{lv}"
            summary["by"][key] = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in summarize(per_task).items()},
                                  "format_ok": round(sum(format_ok(t) for g in gens for t in g) / (len(tasks) * n), 4),
                                  "distinct_attacks": len({t for g in gens for t in g}), "t_gen_s": round(t_gen, 1)}
            per_task_out[key] = {t: sum(v) / len(v) for t, v in per_task.items()}
            print("EVAL", key, summary["by"][key], flush=True)
    json.dump(per_task_out, open(os.path.join(out_dir, "per_task_pass1.json"), "w"))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summary
