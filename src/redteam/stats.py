"""Per-task success statistics. The task is the unit: each task has n samples with c successes."""
from math import comb


def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al. 2021): chance at least one of k draws (without replacement) from the n samples
    succeeds = 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize(per_task):
    """per_task: dict task_id -> list of 0/1 rewards (same n for all). Returns pass@1, pass@n, share of tasks
    all-fail / all-pass / mixed (only mixed tasks give GRPO a learning signal)."""
    tasks = list(per_task.values())
    if not tasks:
        return {}
    n = len(tasks[0])
    cs = [int(sum(t)) for t in tasks]
    T = len(tasks)
    return {"tasks": T, "n": n,
            "pass@1": sum(c / n for c in cs) / T,
            f"pass@{n}": sum(pass_at_k(n, c, n) for c in cs) / T,
            "frac_all_fail": sum(c == 0 for c in cs) / T,
            "frac_all_pass": sum(c == n for c in cs) / T,
            "frac_mixed": sum(0 < c < n for c in cs) / T}


def paired_bootstrap_by_goal(per_task_a, per_task_b, B=10000, seed=0):
    """95% interval for mean(pass@1_b - pass@1_a) over tasks, resampling GOALS (clusters) with replacement.
    per_task_x: dict task_id -> pass@1 for that task. task_id = '<set>:<attacker tool>|<user tool>'; goal = part before '|'.
    Goals are the unit because the attacker sees only the goal, so a goal's 17 tasks are not independent."""
    import random
    from collections import defaultdict
    by_goal = defaultdict(list)
    for t in per_task_a:
        by_goal[t.split("|")[0]].append(per_task_b[t] - per_task_a[t])
    goals = list(by_goal)
    rng = random.Random(seed)
    point = sum(sum(v) for v in by_goal.values()) / sum(len(v) for v in by_goal.values())
    stats = []
    for _ in range(B):
        draw = [by_goal[rng.choice(goals)] for _ in goals]
        stats.append(sum(sum(v) for v in draw) / sum(len(v) for v in draw))
    stats.sort()
    return {"diff": point, "ci95": (stats[int(0.025 * B)], stats[int(0.975 * B) - 1]), "goals": len(goals)}
