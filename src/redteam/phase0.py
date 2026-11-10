"""Phase 0: the pipeline end to end on one GPU, plus three checks. Runs inside the Modal container.

  A. End-to-end episodes: untrained attacker (HF + LoRA) -> injection slot -> vLLM target -> grader -> JSONL.
  B. Replay the 248 OpenRouter Ministral check episodes on our vLLM: same conversations, compare verdicts.
  C. Repeatability: send B's 248 conversations again, shuffled, at a different concurrency; count text/reward changes.
  D. Timing of each part.

Everything goes to out_dir; the Modal entrypoint copies it back to runs/<run_id>/ locally.
"""
import json
import os
import random
import time

from redteam.episode import (CALL_ID, attacker_messages, build_toolcall_messages, extract_injection, format_ok,
                             grade_toolcall, load_cases, load_tools, tool_response_with_injection)
from redteam.target import VLLMTarget


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def toolcall_record(run_id, phase, case, injection, target_model, level, msgs, tools, res, g, extra):
    rec = {
        "run_id": run_id, "phase": phase, "format": "toolcall", "target_level": level,
        "task_id": case["task_id"], "goal_id": case["goal_id"], "set": case["set"],
        "attack_type": case["Attack Type"], "attacker_tool": case["Attacker Tools"][0],
        "goal": case["Attacker Instruction"], "expected_achievement": case["Expected Achievements"],
        "user_tool": case["User Tool"], "user_instruction": case["User Instruction"],
        "injection": injection, "tool_response_with_injection": tool_response_with_injection(case, injection),
        "target_model": target_model, "target_source": "vllm-modal", "target_temperature": 0.0,
        "target_messages": msgs, "target_tools": tools,
        "target_output": res["text"] or "", "target_tool_calls": res["tool_calls"],
        "target_error": res["error"], "target_finish_reason": res["finish_reason"], "t_target_s": round(res["seconds"], 3),
        "eval": g["eval"], "invalid_reason": g["invalid_reason"], "attacker_tool_args": g["attacker_tool_args"],
        "reward": g["reward"], "reward_reason": g["reason"],
    }
    rec.update(extra)
    return rec


def grade(case, res, attacker_output=None):
    if res["error"]:
        return dict(reward=0.0, reason="target_error", eval=None, invalid_reason=res["error"], attacker_tool_args=None)
    return grade_toolcall(case, res["text"], res["tool_calls"], attacker_output)


def run(out_dir, run_id, target_model, attacker_model, replay_path, attacker_revision=None, n_tasks=8, n_samples=2, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    tools = load_tools()
    cases = load_cases(("dh",))
    by_id = {c["task_id"]: c for c in cases}
    target = VLLMTarget(target_model)
    summary = {"run_id": run_id, "target_model": target_model, "attacker_model": attacker_model}

    # ---- A. end-to-end episodes ----
    from redteam.attacker import HFAttacker
    t0 = time.time()
    attacker = HFAttacker(attacker_model, revision=attacker_revision)
    summary["t_attacker_load_s"] = round(time.time() - t0, 1)
    tasks = random.Random(seed).sample(cases, n_tasks)
    gens = attacker.generate([attacker_messages(c) for c in tasks], n=n_samples, seed=seed)
    summary["t_attacker_generate_s"] = round(attacker.last_seconds, 1)
    items, meta = [], []
    for c, samples in zip(tasks, gens):
        for k, s in enumerate(samples):
            inj = extract_injection(s["text"])
            msgs, schemas = build_toolcall_messages(c, inj, tools, "L2_default")
            items.append((msgs, schemas))
            meta.append((c, k, s, inj, msgs, schemas))
    t0 = time.time()
    results = target.batch(items)
    summary["t_target_batch_s"] = round(time.time() - t0, 2)
    rows = []
    for (c, k, s, inj, msgs, schemas), res in zip(meta, results):
        g = grade(c, res, s["text"])
        rows.append(toolcall_record(run_id, "phase0_e2e", c, inj, target_model, "L2_default", msgs, schemas, res, g, {
            "source": "full_episode", "sample_idx": k, "seed": seed, "attacker_model": attacker_model,
            "attacker_messages": attacker_messages(c), "attacker_output": s["text"], "format_ok": format_ok(s["text"]),
            "attacker_tokens": s["tokens"], "truncated": s["truncated"]}))
    write_jsonl(os.path.join(out_dir, "e2e_rollouts.jsonl"), rows)
    summary["e2e"] = {"n": len(rows), "format_ok": sum(r["format_ok"] for r in rows),
                      "reward": sum(r["reward"] for r in rows),
                      "reasons": _count(r["reward_reason"] for r in rows),
                      "target_errors": sum(bool(r["target_error"]) for r in rows)}
    del attacker
    import torch
    torch.cuda.empty_cache()

    # ---- B. replay the OpenRouter Ministral checks on our vLLM ----
    old = [json.loads(l) for l in open(replay_path)]
    items = []
    for r in old:
        msgs = json.loads(json.dumps(r["target_messages"]))
        for m in msgs:  # Mistral needs 9-char tool-call ids; the OpenRouter run used the old long id
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tc["id"] = CALL_ID
            if m.get("tool_call_id"):
                m["tool_call_id"] = CALL_ID
        items.append((msgs, r["target_tools"]))
    t0 = time.time()
    res_a = target.batch(items, concurrency=64)
    summary["t_replay_A_s"] = round(time.time() - t0, 2)

    # ---- C. repeatability: same conversations, shuffled order, different concurrency ----
    order = list(range(len(items)))
    random.Random(seed + 1).shuffle(order)
    t0 = time.time()
    res_b_shuffled = target.batch([items[i] for i in order], concurrency=16)
    summary["t_replay_B_s"] = round(time.time() - t0, 2)
    res_b = [None] * len(items)
    for pos, i in enumerate(order):
        res_b[i] = res_b_shuffled[pos]

    replay_rows, same_text, flips, agree_or = [], 0, 0, 0
    for r, (msgs, schemas), a, b in zip(old, items, res_a, res_b):
        c = by_id[r["task_id"]]
        att = r.get("attacker_output")
        ga, gb = grade(c, a, att), grade(c, b, att)
        same_text += (a["text"] == b["text"] and json.dumps(a["tool_calls"]) == json.dumps(b["tool_calls"]))
        flips += (ga["reward"] != gb["reward"])
        agree_or += (ga["reward"] == r["reward"])
        replay_rows.append(toolcall_record(run_id, "phase0_replay", c, r["injection"], target_model, r["target_level"],
                                           msgs, schemas, a, ga, {
            "source": r["source"], "setting": r.get("setting"), "attacker_model": r.get("attacker_model"),
            "attacker_output": att, "format_ok": r.get("format_ok"),
            "openrouter_reward": r["reward"], "openrouter_reason": r["reward_reason"],
            "openrouter_output": r.get("target_output"), "openrouter_tool_calls": r.get("target_tool_calls"),
            "repeat_output": b["text"], "repeat_tool_calls": b["tool_calls"], "repeat_reward": gb["reward"]}))
    write_jsonl(os.path.join(out_dir, "replay_rollouts.jsonl"), replay_rows)
    by_level = {}
    for row in replay_rows:
        d = by_level.setdefault(row["target_level"], {"n": 0, "vllm_fooled": 0, "openrouter_fooled": 0})
        d["n"] += 1
        d["vllm_fooled"] += row["reward"]
        d["openrouter_fooled"] += row["openrouter_reward"]
    n_err = sum(bool(a["error"]) for a in res_a) + sum(bool(b["error"]) for b in res_b)
    if n_err:  # repeatability/agreement numbers are meaningless if requests failed
        summary["replay_INVALID"] = f"{n_err} target errors in replay; agreement and repeatability not computed"
    summary["replay"] = {"n": len(replay_rows), "by_level": by_level, "target_errors_total": n_err,
                         "agree_with_openrouter_reward": agree_or,
                         "repeat_identical_text_and_calls": same_text, "repeat_reward_flips": flips,
                         "target_errors": sum(bool(a["error"]) for a in res_a)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out
