"""Build explainers/overview.html: the project overview page.

Result tables are computed from the run logs (never typed by hand); the Log tab is LOG.md rendered.
Re-run after every phase:  python -I explainers/build_overview.py
"""
import collections
import datetime
import glob
import html
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731


def jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def short(m):
    return (m or "—").split("/")[-1]


def esc(s):
    return html.escape(str(s))


def table(head, rows, cls=""):
    h = "".join(f"<th>{esc(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table class="{cls}"><tr>{h}</tr>{b}</table></div>'


def frac(k, n):
    return f'<span class="num">{k}/{n}</span>' + (f' <span class="pct">{100 * k / n:.0f}%</span>' if n else "")


# ---------- load every check log ----------
pre = jsonl(P("runs/precheck/rollouts.jsonl"))
text_eps = [r for r in pre if r.get("target_output") is not None and r.get("format") != "toolcall"
            and r.get("batch") != "local_targets"]
attacker_only = [r for r in pre if r["source"] == "attacker_only"]
tc = []
for f in sorted(glob.glob(P("runs/precheck/toolcall_*.jsonl"))):
    for r in jsonl(f):
        r["_file"] = os.path.relpath(f, ROOT)
        r.setdefault("target_level", "L2_default")
        tc.append(r)
det = json.load(open(P("runs/precheck/determinism_check.json"))) if os.path.exists(P("runs/precheck/determinism_check.json")) else None

# ---------- results: attacker format ----------
att_rows = []
by = collections.defaultdict(list)
for r in attacker_only:
    by[r["attacker_model"]].append(r)
for m, rs in sorted(by.items(), key=lambda x: -sum(r["format_ok"] for r in x[1]) / len(x[1])):
    att_rows.append([esc(short(m)), frac(sum(r["format_ok"] for r in rs), len(rs))])
ATTACKER_TABLE = table(["Attacker", "Format OK (think + prompt tags, ends on </prompt>)"], att_rows)

# ---------- results: text-ReAct format (dropped) ----------
rows = []
by = collections.defaultdict(list)
for r in text_eps:
    by[(r["target_model"], r["source"])].append(r)
for (m, src), rs in sorted(by.items(), key=lambda x: (x[0][1], x[0][0])):
    n = len(rs)
    c = collections.Counter(r["eval"] for r in rs)
    rows.append([esc(short(m)), "fixed injection" if src == "fixed_injection" else "Qwen-3B attacks",
                 frac(c["succ"], n), frac(c["unsucc"], n), frac(c["invalid"], n)])
TEXT_TABLE = table(["Target", "Injections", "Fooled", "Resisted", "Invalid"], rows)
hidden = [r for r in text_eps if r["eval"] == "invalid"
          and re.search(r"Action:\s*" + re.escape(r["attacker_tool"]), r["target_output"] or "")]
n_invalid = sum(r["eval"] == "invalid" for r in text_eps)
recopy = sum(1 for r in text_eps if r["eval"] != "succ" and re.search(
    r"^\s*\**Action\**:\s*" + re.escape(r["user_tool"]) + r"\s*$", r["target_output"] or "", re.M))

# ---------- results: native tool calls x difficulty level ----------
LEVEL_ORDER = ["L0_obey", "L1_lenient", "L2_default", "L3_guarded"]
by = collections.defaultdict(list)
for r in tc:
    by[(r["target_model"], r["target_level"])].append(r)
models = sorted({m for m, _ in by}, key=lambda m: (("qwen" not in m.lower()), m))
rows = []
for m in models:
    cells = [esc(short(m))]
    for lv in LEVEL_ORDER:
        rs = by.get((m, lv), [])
        rs = [r for r in rs if r["reward_reason"] != "target_error"]
        if not rs:
            cells.append('<span class="muted">—</span>')
            continue
        k = sum(r["reward"] for r in rs)
        inv = sum(r["eval"] == "invalid" for r in rs)
        link = f'../viewer/precheck/index.html#format=toolcall&target={m}&level={lv}'
        cells.append(f'<a href="{esc(link)}">' + frac(int(k), len(rs)) + "</a>" + (f'<div class="sub">invalid {inv}</div>' if inv else ""))
    rows.append(cells)
TC_NOTE = '<p class="src">Click a cell to open those 62 episodes in the viewer.</p>'
TC_TABLE = TC_NOTE + table(["Target (native tool calls)"] + ["L0 obey", "L1 lenient", "L2 default", "L3 guarded"], rows, "grid")
tc_files = sorted({r["_file"] for r in tc})
qwen_partial = os.path.exists(P("runs/precheck/toolcall_qwen_local.jsonl"))

DET = (f"{det['text_identical']}/{det['n']} replies identical, <b>{len(det['reward_flips'])}/{det['n']} rewards flipped</b>, "
       f"{det.get('n_429', '?')} rate-limit (429) retries") if det else "not run"

# ---------- one real native-tool-call episode, rendered as chat ----------
def pick_tc_episode():
    pref = [r for r in tc if r["reward"] == 1 and r["source"] == "full_episode"] + \
           [r for r in tc if r["reward"] == 1]
    return pref[0] if pref else None


def render_chat(r):
    blocks = []
    for m in r["target_messages"]:
        role = m["role"]
        if m.get("tool_calls"):
            body = "".join(f'<div class="call">→ calls <code>{esc(tc_["function"]["name"])}</code>'
                           f'<pre>{esc(json.dumps(tc_["function"]["arguments"]) if not isinstance(tc_["function"]["arguments"], str) else tc_["function"]["arguments"])}</pre></div>'
                           for tc_ in m["tool_calls"])
        else:
            content = m.get("content") or ""
            if role == "tool" and r["injection"] in content:
                body = "<pre>" + esc(content).replace(esc(r["injection"]), f'<mark class="inj">{esc(r["injection"])}</mark>') + "</pre>"
            else:
                body = f"<pre>{esc(content)}</pre>"
        label = {"system": "system (target's instructions)", "user": "user (the real user)",
                 "assistant": "assistant (the target, earlier turn: already written for it)",
                 "tool": "tool result (contains the injection)"}.get(role, role)
        blocks.append(f'<div class="msg m-{role}"><div class="role">{label}</div>{body}</div>')
    schemas = "<pre>" + esc(json.dumps(r["target_tools"], indent=1)) + "</pre>"
    calls = r.get("target_tool_calls") or []
    reply = ""
    if r.get("target_output"):
        reply += f"<pre>{esc(r['target_output'])}</pre>"
    for c in calls:
        nm = c["function"]["name"]
        hit = nm == r["attacker_tool"]
        reply += (f'<div class="call {"hit" if hit else ""}">→ calls <code>{esc(nm)}</code>'
                  f'{" (the attacker\'s tool)" if hit else ""}<pre>{esc(c["function"]["arguments"])}</pre></div>')
    return dict(chat="".join(blocks), schemas=schemas, reply=reply)


TCE = pick_tc_episode()
if TCE:
    rc = render_chat(TCE)
    TC_EPISODE = f"""
<p class="src">Real episode from <code>{esc(TCE['_file'])}</code>: target <b>{esc(short(TCE['target_model']))}</b>,
level <code>{esc(TCE['target_level'])}</code>, temperature 0, task <code>{esc(TCE['task_id'])}</code>,
injection from {"the Qwen2.5-3B attacker" if TCE['attacker_output'] else "InjecAgent's fixed text (" + esc(TCE.get('setting')) + ")"}.</p>
<h3>1. The task</h3>
<dl class="kv"><dt>User asks</dt><dd>{esc(TCE['user_instruction'])}</dd>
<dt>Attacker goal</dt><dd>{esc(TCE['goal'])}</dd>
<dt>Attacker tool</dt><dd><code>{esc(TCE['attacker_tool'])}</code>: {esc(TCE['expected_achievement'])}</dd></dl>
{"<h3>2. What the attacker wrote</h3><p>It saw only the goal (RL-Hammer's prompt). Only the part inside the prompt tags is used.</p><pre>" + esc(TCE['attacker_output']) + "</pre>" if TCE['attacker_output'] else "<h3>2. The injection</h3><pre>" + esc(TCE['injection']) + "</pre>"}
<h3>3. Everything the target sees</h3>
<p>Four chat messages plus two tool schemas. We write the conversation so far, including the target's own earlier
tool call. The model writes only the next turn.</p>
{rc['chat']}
<details><summary>The two tool schemas the target is given (user tool + attacker tool)</summary>{rc['schemas']}</details>
<h3>4. What the target writes</h3>{rc['reply']}
<h3>5. The grader</h3>
<p>First tool call's name == attacker tool → <code>{esc(TCE['eval'])}</code> → <b>reward {TCE['reward']}</b>.
That's InjecAgent's own rule for tool-calling agents (<code>output_parsing.py:189-202</code>). Arguments are logged, not checked.</p>
"""
else:
    TC_EPISODE = "<p>No tool-call success logged yet.</p>"

# ---------- run charts: data for the Runs tab (rendered client-side, see overview_template.html) ----------
def run_series(run_dir):
    """Per-step metrics for a training run: metrics.jsonl if downloaded, else parse PROGRESS/CANARY lines of the
    live log. Returns dict(metric -> [[step, value], ...]), canary [[step, pass@1]], and the source used."""
    out, canary, src = {}, [], None
    mpath = os.path.join(run_dir, "metrics.jsonl") if run_dir else None
    if mpath and os.path.exists(mpath):
        src = os.path.relpath(mpath, ROOT)
        for l in open(mpath):
            d = json.loads(l)
            if "reward" not in d:
                continue
            for k in ("reward", "ours/reward_at_current_level", "ours/level_idx", "ours/format_ok", "entropy",
                      "completions/mean_length", "completions/clipped_ratio", "frac_reward_zero_std", "step_time"):
                if isinstance(d.get(k), (int, float)):
                    out.setdefault(k.split("/")[-1], []).append([d["step"], d[k]])
        cpath = os.path.join(run_dir, "canary_summary.jsonl")
        if os.path.exists(cpath):
            canary = [[d["step"], d["canary_pass@1_L2"]] for d in map(json.loads, open(cpath))]
    return out, canary, src


def live_series(log_path):
    out, canary = {}, []
    for line in open(log_path, errors="ignore"):
        m = re.search(r"PROGRESS step=(\d+) (.*)", line)
        if m:
            step = int(m.group(1))
            for k, v in re.findall(r"(\S+)=([-0-9.]+)", m.group(2)):
                out.setdefault(k, []).append([step, float(v)])
        m = re.search(r"CANARY \{'step': (\d+), 'canary_pass@1_L2': ([0-9.]+)", line)
        if m:
            canary.append([int(m.group(1)), float(m.group(2))])
    # de-duplicate steps (tqdm redraws)
    for k in out:
        out[k] = sorted({s: v for s, v in out[k]}.items())
        out[k] = [list(x) for x in out[k]]
    return out, canary


CHARTS = []
p3_dirs = sorted(d for d in glob.glob(P("runs/*-phase3")) if os.path.isdir(d))
p3_dir = p3_dirs[-1] if p3_dirs else None
series, canary, src = run_series(p3_dir)
if not series and os.path.exists(P("runs/phase3_live.log")):
    series, canary = live_series(P("runs/phase3_live.log"))
    src = "runs/phase3_live.log (live; run still in progress or not yet downloaded)"
P3_SRC = src or "no phase 3 data yet"
if series:
    lv = series.get("level_idx", [])
    bands, cur, start = [], None, None
    for st, v in lv:
        if v != cur:
            if cur is not None:
                bands.append({"from": start, "to": st, "label": ["L0 obey", "L1 lenient", "L2 default", "L3 guarded"][int(cur)]})
            cur, start = v, st
    if cur is not None:
        bands.append({"from": start, "to": lv[-1][0], "label": ["L0 obey", "L1 lenient", "L2 default", "L3 guarded"][int(cur)]})
    CHARTS += [
        {"id": "p3_reward", "title": "Training reward per step", "unit": "%", "pct": True, "bands": bands,
         "series": [{"name": "reward at current ladder level", "slot": 1, "points": series.get("reward_at_current_level", [])},
                    {"name": "all tasks (incl. 25% easier)", "slot": 2, "points": series.get("reward", [])}],
         "note": "Shaded bands = the target's ladder level at that step. Each step = 64 attacks on train goals."},
        {"id": "p3_canary", "title": "Canary: held-out dev goals at L2 default (heartbeat)", "unit": "%", "pct": True,
         "series": [{"name": "canary pass@1 (16 dev tasks × 8)", "slot": 1, "points": canary, "dots": True}],
         "ref": {"y": 0.072, "label": "untrained, full dev sweep (7.2%)"},
         "note": "128 attacks per point: ±~4–5 points of noise. The real measurement is the full-dev checkpoint eval."},
        {"id": "p3_zero", "title": "Groups with no learning signal (all 8 same)", "unit": "%", "pct": True,
         "series": [{"name": "frac_reward_zero_std", "slot": 1, "points": series.get("frac_reward_zero_std", [])}]},
        {"id": "p3_format", "title": "Attacker output in valid format", "unit": "%", "pct": True,
         "series": [{"name": "format OK", "slot": 1, "points": series.get("format_ok", [])}]},
        {"id": "p3_len", "title": "Attack length (tokens; cap 256)", "unit": "", "pct": False,
         "series": [{"name": "mean completion length", "slot": 1, "points": series.get("mean_length", [])}]},
        {"id": "p3_entropy", "title": "Entropy (how varied the attacker's sampling still is)", "unit": "", "pct": False,
         "series": [{"name": "entropy", "slot": 1, "points": series.get("entropy", [])}]},
    ]
WIRING_CHARTS = []
for name, label in (("nosignal", "No signal (reward always 1)"), ("word", 'Easy word: "urgent"'),
                    ("reversed", 'Reversed: no "please"'), ("stub", 'Stub target: "execute"'),
                    ("onetask", "One task, real Ministral at L0")):
    ds = sorted(glob.glob(P(f"runs/*-wiring-{name}")))
    if ds:
        ser, _, _ = run_series(ds[-1])
        if ser.get("reward"):
            WIRING_CHARTS.append({"id": f"w_{name}", "title": label, "unit": "%", "pct": True, "small": True,
                                  "series": [{"name": "reward", "slot": 1, "points": ser["reward"]}]})
SWEEP = None
sw = sorted(glob.glob(P("runs/*-sweep/summary_grader_v2.json")))
if sw:
    d = json.load(open(sw[-1]))["by_level"]
    SWEEP = [{"level": k, "pass1": v["pass@1"], "mixed": v["frac_mixed"]} for k, v in d.items()]
EVAL = []
ev = sorted(glob.glob(P("runs/*-eval-dev/summary.json")))
if ev:
    by = json.load(open(ev[-1]))["by"]
    step_of = lambda a: 0 if a == "base" else int(a.replace("step", "")) if a.startswith("step") else None
    ser = {}
    for key, v in by.items():
        ad, lv = key.split("@")
        st = step_of(ad)
        if st is None:
            continue
        ser.setdefault(lv, []).append([st, v["pass@1"]])
    names = {"L2_default": "L2 default (real target)", "L0_obey": "L0 obey"}
    EVAL.append({"id": "dev_eval", "title": "Full dev evaluation: pass@1 on 12 held-out goals (204 tasks × 8) by training step",
                 "unit": "%", "pct": True,
                 "series": [{"name": names.get(lv, lv), "slot": i + 1, "points": sorted(pts), "dots": True}
                            for i, (lv, pts) in enumerate(sorted(ser.items(), key=lambda x: x[0] != "L2_default"))],
                 "note": "Step 0 = untrained attacker. L2 gain at step 80: +78.1 points, 95% CI [+72.8, +83.3] (goal-level bootstrap). "
                         "Identical weights scored 86.9% vs 84.1% on two passes: about 3 points of evaluation noise. Source: " + os.path.relpath(ev[-1], ROOT)})
te = sorted(glob.glob(P("runs/*-eval-test/summary.json")))
if EVAL and te:
    tb = json.load(open(te[-1]))["by"]
    pts = [[0, tb["base@L2_default"]["pass@1"]], [80, tb["step80@L2_default"]["pass@1"]]]
    EVAL[0]["series"].append({"name": "TEST L2 (one-time, locked goals)", "slot": 3, "points": pts, "dots": True, "noline": True})
    EVAL[0]["title"] = "Dev curve by training step, plus the one-time test result (pass@1, 12 held-out goals × 17 × 8)"
    EVAL[0]["note"] += (". TEST at L2: 7.8% → 87.5%, +79.7 points, 95% CI [+75.6, +83.8]. Source: "
                        + os.path.relpath(te[-1], ROOT) + " and bootstrap.json")
RUNS_DATA = json.dumps({"eval": EVAL, "charts": CHARTS, "wiring": WIRING_CHARTS, "sweep": SWEEP, "p3_src": P3_SRC}).replace("</", "<\\/")

# ---------- LOG.md -> HTML ----------
def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\[(result|issue|decision|pivot|dead-end|infra|idea)\]", r'<span class="tag t-\1">\1</span>', s)
    s = re.sub(r"\(Leo([^)]*)\)", r'<span class="leo">Leo\1</span>', s)
    return s


def md(text):
    out, in_list, tbl = [], False, []

    def flush_table():
        nonlocal tbl
        if tbl:
            rows_ = [[c.strip() for c in l.strip().strip("|").split("|")] for l in tbl if not re.match(r"^\s*\|[\s|:-]+\|\s*$", l)]
            out.append(table(rows_[0], [[inline(c) for c in r] for r in rows_[1:]]))
            tbl = []

    for line in text.splitlines():
        if line.strip().startswith("|"):
            tbl.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<h3>{inline(line[3:])}</h3>")
        elif line.startswith("- "):
            if not in_list:
                out.append('<ul class="log">'); in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip() in ("", "---"):
            if in_list:
                out.append("</ul>"); in_list = False
        else:
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<p>{inline(line)}</p>")
    flush_table()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


LOG_HTML = md(open(P("LOG.md")).read())
decisions = [l[2:] for l in open(P("LOG.md")) if l.startswith("- ") and re.search(r"\[(decision|pivot)\]", l)]
DECISIONS_HTML = '<ul class="log">' + "".join(f"<li>{inline(d)}</li>" for d in decisions) + "</ul>"
ideas = [l[2:] for l in open(P("LOG.md")) if l.startswith("- ") and "[idea]" in l]
IDEAS_HTML = '<ul class="log">' + "".join(f"<li>{inline(d)}</li>" for d in ideas) + "</ul>"

page = open(P("explainers/overview_template.html")).read()
for k, v in {
    "{{BUILT}}": datetime.datetime.now().strftime("%a %d %b %Y, %H:%M SGT"),
    "{{ATTACKER_TABLE}}": ATTACKER_TABLE, "{{TEXT_TABLE}}": TEXT_TABLE, "{{TC_TABLE}}": TC_TABLE,
    "{{TC_FILES}}": ", ".join(f"<code>{esc(f)}</code>" for f in tc_files),
    "{{QWEN_NOTE}}": "" if not qwen_partial else "Qwen rows come from a local run that may still be in progress; rebuild the page to refresh.",
    "{{HIDDEN}}": f"{len(hidden)}/{n_invalid}", "{{RECOPY}}": f"{recopy}/{len(text_eps)}",
    "{{N_TEXT}}": str(len(text_eps)), "{{DET}}": DET,
    "{{LOG}}": LOG_HTML, "{{TC_EPISODE}}": TC_EPISODE, "{{RUNS_DATA}}": RUNS_DATA, "{{P3_SRC}}": esc(P3_SRC), "{{DECISIONS}}": DECISIONS_HTML, "{{IDEAS}}": IDEAS_HTML, "{{N_EPISODES}}": str(len(pre)),
}.items():
    page = page.replace(k, v)
open(P("explainers/overview.html"), "w").write(page)
print("wrote explainers/overview.html")
