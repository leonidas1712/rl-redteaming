"""Build a static, self-contained episode viewer from one or more rollout JSONL files. Read-only.

Usage: python -I viewer/build.py OUT.html IN1.jsonl [IN2.jsonl ...]
Links: index.html#ep=ep0383   index.html#format=toolcall&target=Qwen/Qwen2.5-3B-Instruct&level=L0_obey
Episode ids: records that already have one keep it; others get '<file stem>-NNNN'.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from redteam.episode import TARGET_LEVELS  # noqa: E402

args = sys.argv[1:]
title, desc = "pre-GPU checks", None
if "--title" in args:
    i = args.index("--title"); title = args[i + 1]; del args[i:i + 2]
if "--desc" in args:
    i = args.index("--desc"); desc = args[i + 1]; del args[i:i + 2]
dst, srcs = args[0], args[1:]
records = []
for src in srcs:
    stem = os.path.splitext(os.path.basename(src))[0].replace("toolcall_", "tc_")
    for i, l in enumerate(open(src)):
        r = json.loads(l)
        r.setdefault("format", "text")
        if r["format"] == "toolcall":
            r.setdefault("target_level", "L2_default")
            r.setdefault("target_level_text", TARGET_LEVELS.get(r["target_level"], ""))
        r.setdefault("episode_id", f"{stem}-{i:04d}")
        r["log_file"] = os.path.relpath(src, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        records.append(r)

# Text-format episodes share one long system prompt: store it once.
system_prompts = sorted({r["target_messages"][0]["content"] for r in records
                         if r["format"] == "text" and r.get("target_messages")})
for r in records:
    if r["format"] == "text" and r.get("target_messages"):
        r["target_system_idx"] = system_prompts.index(r["target_messages"][0]["content"])
        r["target_user"] = r["target_messages"][1]["content"]
        del r["target_messages"]
payload = json.dumps({"records": records, "system_prompts": system_prompts}).replace("</", "<\\/")
html = open(os.path.join(os.path.dirname(__file__), "template.html")).read()
html = html.replace("__RUN__", title).replace("__DATA__", payload)
if desc:
    import re
    html = re.sub(r"(<header>.*?<p>).*?(</p>)", lambda m: m.group(1) + desc + m.group(2), html, count=1, flags=re.S)
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w").write(html)
print(f"wrote {dst} ({len(records)} episodes, {len(html) / 1e6:.1f} MB)")
