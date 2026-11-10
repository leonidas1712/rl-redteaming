"""Turn a training run's canary.jsonl into full viewer records (rebuilds the attacker prompt and the target's
conversation at L2 from task_id + injection, as they were sent). Usage: python -I scripts/canary_viewer_records.py runs/<run> [steps]"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from redteam.attacker_prompts import attacker_messages
from redteam.episode import build_toolcall_messages, load_cases, load_tools, tool_response_with_injection
run = sys.argv[1]; steps = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,80").split(",")]
cfg = json.load(open(os.path.join(run, "config.json")))
tools = load_tools(); cases = {c["task_id"]: c for c in load_cases(("dh", "ds"))}
C = [json.loads(l) for l in open(os.path.join(run, "canary.jsonl"))]
out = []
for i, r in enumerate(C):
    if r["step"] not in steps:
        continue
    c = cases[r["task_id"]]
    msgs, schemas = build_toolcall_messages(c, r["injection"], tools, r.get("target_level", "L2_default"))
    k = sum(1 for x in C[:i] if x["step"] == r["step"] and x["task_id"] == r["task_id"])
    out.append({"episode_id": f"canary-s{r['step']:03d}-{c['task_id'].split(':')[1][:40]}-{k}", "format": "toolcall",
                "source": f"canary step {r['step']}" + (" (before training)" if r["step"] == 0 else " (after training)" if r["step"] == max(steps) else ""),
                "run_id": cfg["run_id"], "phase": "canary", "split": "dev", "step": r["step"], "sample_idx": k,
                "attacker_prompt_variant": "v2_context_hints", "target_level": r.get("target_level", "L2_default"),
                "task_id": c["task_id"], "goal_id": c["goal_id"], "set": c["set"], "attack_type": c["Attack Type"],
                "attacker_tool": c["Attacker Tools"][0], "goal": c["Attacker Instruction"],
                "expected_achievement": c["Expected Achievements"], "user_tool": c["User Tool"],
                "user_instruction": c["User Instruction"], "attacker_model": "Qwen/Qwen2.5-3B-Instruct" + (" + LoRA step %d" % r["step"] if r["step"] else " (untrained)"),
                "attacker_messages": attacker_messages(c, "v2_context_hints"), "attacker_output": r["attacker_output"],
                "format_ok": r["format_ok"], "injection": r["injection"],
                "tool_response_with_injection": tool_response_with_injection(c, r["injection"]),
                "target_model": cfg["models"]["target"]["name"], "target_source": "vllm-modal", "target_temperature": 0.0,
                "target_messages": msgs, "target_tools": schemas, "target_output": r["target_output"],
                "target_tool_calls": r["target_tool_calls"], "eval": "succ" if r["reward"] == 1 else "unsucc",
                "reward": r["reward"], "reward_reason": r["reward_reason"]})
p = os.path.join(run, "canary_viewer.jsonl")
with open(p, "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print(len(out), "records ->", p)
