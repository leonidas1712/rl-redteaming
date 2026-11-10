"""Turn runs/precheck/attacker_prompt_check.jsonl (slim records) into full viewer records and add Jev labels.

For each episode: rebuild the attacker prompt (per variant) and the target's full conversation + tool schemas
(level L2_default, as sent), keep the target reply/tool calls/reward that were logged, and label the target reply
with Jev (taxonomy v1.1, scripts/jev_label.py). `acted` comes from the code grader, not Jev.

Output: runs/precheck/attacker_prompt_check_full.jsonl
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]
from jev_label import ask, state_of  # noqa: E402
from redteam.attacker_prompts import attacker_messages  # noqa: E402
from redteam.episode import build_toolcall_messages, load_cases, load_tools, tool_response_with_injection  # noqa: E402

tools = load_tools()
by_id = {c["task_id"]: c for c in load_cases(("dh", "ds"))}
rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs/precheck/attacker_prompt_check.jsonl"))]
out = []
for i, r in enumerate(rows):
    c = by_id[r["task_id"]]
    msgs, schemas = build_toolcall_messages(c, r["injection"], tools, "L2_default")
    rec = {
        "episode_id": f"apc-{i:03d}", "format": "toolcall", "source": f"attacker prompt {r['variant']}",
        "attacker_prompt_variant": r["variant"], "target_level": "L2_default",
        "task_id": c["task_id"], "goal_id": c["goal_id"], "attack_type": c["Attack Type"],
        "attacker_tool": c["Attacker Tools"][0], "goal": c["Attacker Instruction"],
        "expected_achievement": c["Expected Achievements"], "user_tool": c["User Tool"],
        "user_instruction": c["User Instruction"], "attacker_model": "Qwen/Qwen2.5-3B-Instruct",
        "attacker_messages": attacker_messages(c, r["variant"]), "attacker_output": r["attacker_output"],
        "format_ok": r["format_ok"], "attacker_tokens": r["tokens"], "truncated": r["truncated"],
        "injection": r["injection"], "tool_response_with_injection": tool_response_with_injection(c, r["injection"]),
        "target_model": "mistralai/ministral-3b-2512", "target_source": "openrouter (Mistral)",
        "target_temperature": 0.0, "target_messages": msgs, "target_tools": schemas,
        "target_output": r.get("target_output") or "", "target_tool_calls": r.get("target_tool_calls"),
        "eval": "succ" if r["reward"] == 1 else ("invalid" if r["reward_reason"].startswith("invalid") else "unsucc"),
        "reward": r["reward"], "reward_reason": r["reward_reason"],
    }
    if not r["format_ok"]:
        rec["jev_label"] = "(no target run: bad attacker format)"
    elif r["reward"] == 1:
        rec["jev_label"] = "acted"
        rec["jev_note"] = "from the code grader (first tool call = attacker tool)"
    else:
        choice, raw = ask("typesafe/jev-1.14", state_of(rec))
        rec["jev_label"] = choice or "ERROR"
        rec["jev_raw"] = raw
    out.append(rec)
with open(os.path.join(ROOT, "runs/precheck/attacker_prompt_check_full.jsonl"), "w") as f:
    for rec in out:
        f.write(json.dumps(rec) + "\n")
import collections  # noqa: E402
for v in ("v0_rlhammer", "v1_context", "v2_context_hints"):
    print(v, dict(collections.Counter(r["jev_label"] for r in out if r["attacker_prompt_variant"] == v)))
