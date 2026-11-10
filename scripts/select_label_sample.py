"""Pick a fixed, stratified sample of tool-call episodes for hand labelling (seed 0) and write it compactly.

Output: runs/labels/sample_v1.jsonl, one line per episode with only what a labeller needs:
episode_id, log_file, target_model, level, source, user_request, attacker_tool, injection, tool_result,
target_text, target_tool_calls, code_reward. Viewer ids match viewer/build.py ('tc_<stem>-NNNN').
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {"runs/precheck/toolcall_openrouter_ministral_levels.jsonl": 34,
         "runs/precheck/toolcall_qwen_local.jsonl": 20,
         "runs/precheck/toolcall_openrouter_lfm.jsonl": 6}

rng = random.Random(0)
out = []
for rel, k in FILES.items():
    stem = os.path.splitext(os.path.basename(rel))[0].replace("toolcall_", "tc_")
    rows = [json.loads(l) for l in open(os.path.join(ROOT, rel))]
    for i, r in enumerate(rows):
        r["_id"] = f"{stem}-{i:04d}"
    if "qwen" in rel:  # mostly the 3B (the only Qwen that ever acts), a few smaller ones
        rows = [r for r in rows if "3B" in r["target_model"]] * 3 + [r for r in rows if "3B" not in r["target_model"]]
    fooled = [r for r in rows if r["reward"] == 1]
    rest = [r for r in rows if r["reward"] != 1]
    pick = rng.sample(fooled, min(len(fooled), max(1, k // 6))) if fooled else []
    seen = {r["_id"] for r in pick}
    for r in rng.sample(rest, len(rest)):
        if len(pick) >= k:
            break
        if r["_id"] not in seen:
            pick.append(r)
            seen.add(r["_id"])
    for r in pick:
        out.append({"episode_id": r["_id"], "log_file": rel, "target_model": r["target_model"],
                    "level": r.get("target_level", "L2_default"), "source": r["source"],
                    "user_request": r["user_instruction"], "attacker_tool": r["attacker_tool"],
                    "injection": r["injection"], "tool_result": r["tool_response_with_injection"],
                    "target_text": r.get("target_output") or "", "target_tool_calls": r.get("target_tool_calls"),
                    "code_reward": r["reward"]})
rng.shuffle(out)
os.makedirs(os.path.join(ROOT, "runs/labels"), exist_ok=True)
with open(os.path.join(ROOT, "runs/labels/sample_v1.jsonl"), "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(len(out), "episodes ->", "runs/labels/sample_v1.jsonl")
