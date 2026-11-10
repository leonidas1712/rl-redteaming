"""Make the README figures from the run logs (no model runs; reads JSON/JSONL only).

  .venv/bin/python scripts/make_readme_figures.py      # writes figures/*.png

Colours: a palette checked with a colour-vision validator (blue, orange, green).
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1a9e6e", "#8a8985", "#1f1f1d", "#6b6a66"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "lines.linewidth": 2, "lines.solid_capstyle": "round"})


def P(*a):
    return os.path.join(ROOT, *a)


def jl(path):
    return [json.loads(l) for l in open(path)]


def pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


# ---------- 1. headline: dev curve + one-time test ----------
dev = json.load(open(P("runs/20261110-1855-eval-dev/summary.json")))["by"]
test = json.load(open(P("runs/20261110-2112-eval-test/summary.json")))["by"]
tb = json.load(open(P("runs/20261110-2112-eval-test/bootstrap.json")))["results"]
steps = {"base": 0, "step25": 25, "step50": 50, "step75": 75, "step80": 80}
fig, ax = plt.subplots(figsize=(7, 4.2))
for lv, col, name in [("L0_obey", ORANGE, "Dev, L0 (easiest target prompt)"), ("L2_default", BLUE, "Dev, L2 (real target, no added line)")]:
    pts = sorted((s, dev[f"{a}@{lv}"]["pass@1"]) for a, s in steps.items())
    ax.plot(*zip(*pts), color=col, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5, label=name)
tp = [(0, test["base@L2_default"]["pass@1"]), (80, test["step80@L2_default"]["pass@1"])]
ax.scatter(*zip(*tp), s=90, color=GREEN, edgecolor="white", linewidth=1.5, zorder=5, label="TEST, L2 (one-time, locked goals)")
d = tb["L2_default"]
ax.annotate(f"test {tp[0][1]:.1%} → {tp[1][1]:.1%}\n+{100*d['diff']:.1f} pts, 95% CI [+{100*d['ci95'][0]:.1f}, +{100*d['ci95'][1]:.1f}]",
            xy=(80, tp[1][1]), xytext=(47, 0.10), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=1))
ax.set_ylim(0, 1.02)
pct(ax)
ax.set_xlabel("training step")
ax.set_ylabel("attack success (pass@1)")
ax.set_title("Held-out goals: success of the attacker vs training step", loc="left", fontsize=11, color=INK)
ax.legend(loc="upper left", fontsize=8.5)
fig.text(0.01, 0.005, "12 held-out goals × 17 scenarios × 8 attacks per point. Target: Ministral-3B. CI = goal-level paired bootstrap.",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(OUT, "headline.png"))
plt.close(fig)

# ---------- 2. training: reward with ladder bands, entropy, flat groups ----------
m = [r for r in jl(P("runs/20261110-1805-phase3/metrics.jsonl")) if "ours/level_idx" in r and "entropy" in r]
st = [r["step"] for r in m]
lvl = [int(r["ours/level_idx"]) for r in m]
names = ["L0 obey", "L1 lenient", "L2 default", "L3 guarded"]
fig, axs = plt.subplots(3, 1, figsize=(7, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 1]})
start = st[0]
bands = []
for i in range(1, len(st) + 1):
    if i == len(st) or lvl[i] != lvl[i - 1]:
        bands.append((start, st[i - 1], lvl[i - 1]))
        if i < len(st):
            start = st[i]
for k, (a, b, L) in enumerate(bands):
    for ax in axs:
        ax.axvspan(a - 0.5, b + 0.5, color=BLUE, alpha=0.05 if k % 2 == 0 else 0.11, lw=0)
    axs[0].text(a + 0.3, 1.04, names[L] if b - a >= 8 else names[L].split()[0], fontsize=8, color=MUTED, va="bottom")
axs[0].plot(st, [r["ours/reward_at_current_level"] for r in m], color=BLUE, lw=1.6, label="reward at current ladder level")
axs[0].set_ylim(0, 1.0)
pct(axs[0])
axs[0].set_ylabel("train reward")
axs[0].set_title("Run A: 80 GRPO steps, adaptive difficulty ladder (64 attacks per step)", loc="left", fontsize=11, color=INK, pad=18)
axs[1].plot(st, [r["entropy"] for r in m], color=ORANGE, lw=1.6)
axs[1].set_ylabel("entropy")
axs[1].text(st[-1], 0.85, "1.02 → 0.36: collapse onto one template", ha="right", fontsize=8, color=MUTED)
axs[2].plot(st, [r["frac_reward_zero_std"] for r in m], color=GREEN, lw=1.6)
pct(axs[2])
axs[2].set_ylim(0, 1)
axs[2].set_ylabel("flat groups")
axs[2].set_xlabel("training step")
fig.text(0.01, 0.005, "Flat group = all 8 attacks on a task got the same reward → zero advantage → no gradient. "
         "Source: runs/…-phase3/metrics.jsonl", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(OUT, "training.png"))
plt.close(fig)

# ---------- 3. per-goal test, before vs after ----------
pt = json.load(open(P("runs/20261110-2112-eval-test/per_task_pass1.json")))
goal = defaultdict(lambda: defaultdict(list))
for a in ("base", "step80"):
    for tid, v in pt[f"{a}@L2_default"].items():
        g = tid.split("|")[0].split(":", 1)[1]
        goal[g][a].append(v)
rows = sorted(((g, sum(d["base"]) / len(d["base"]), sum(d["step80"]) / len(d["step80"])) for g, d in goal.items()),
              key=lambda r: r[2])
fig, ax = plt.subplots(figsize=(7.4, 5.0))
for i, (g, b, a) in enumerate(rows):
    ax.plot([b, a], [i, i], color="#c9c8c2", lw=2, zorder=1)
ax.scatter([r[1] for r in rows], range(len(rows)), s=45, color=GREY, edgecolor="white", zorder=3, label="untrained")
ax.scatter([r[2] for r in rows], range(len(rows)), s=55, color=GREEN, edgecolor="white", zorder=4, label="after 80 steps")
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
ax.set_xlim(-0.02, 1.02)
pct(ax.xaxis.axes)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: rows[int(v)][0] if 0 <= int(v) < len(rows) else ""))
ax.set_xlabel("attack success at L2 (pass@1; 17 scenarios × 8 attacks per goal)", fontsize=9)
ax.set_title("Every one of the 12 locked test goals improved", loc="left", fontsize=11, color=INK)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "test_per_goal.png"))
plt.close(fig)

# ---------- 4. wiring checks ----------
W = [("nosignal", "reward always 1\n→ weights must not change"), ("word", 'reward "urgent"\n→ should spread'),
     ("reversed", 'reward no "please"\n→ should vanish'), ("stub", 'stub target obeys "execute"\n→ grader end to end'),
     ("onetask", "real Ministral, one task\n→ learnable?")]
fig, axs = plt.subplots(1, 5, figsize=(11, 2.8), sharey=True)
for ax, (name, title) in zip(axs, W):
    f = glob.glob(P(f"runs/*-wiring-{name}/metrics.jsonl"))[0]
    mm = [r for r in jl(f) if "reward" in r]
    ax.plot([r["step"] for r in mm], [r["reward"] for r in mm], color=BLUE, lw=1.6)
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("step", fontsize=8)
pct(axs[0])
axs[0].set_ylabel("reward")
fig.suptitle("Wiring checks: fake rewards with known answers (all passed)", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "wiring.png"))
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
