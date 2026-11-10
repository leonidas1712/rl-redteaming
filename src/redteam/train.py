"""GRPO training of the attacker (TRL GRPOTrainer + LoRA), with our reward function, the adaptive target ladder,
probe rewards for the wiring checks, a canary eval and full rollout logging.

One optimizer step = 8 train tasks x 8 attacker samples = 64 rollouts, generated and scored together
(per_device_train_batch_size 16 x gradient_accumulation_steps 4; TRL keeps each task's 8 samples consecutive).

Reward modes (--reward):
  real            target = vLLM server at the current ladder level; reward from the grader (episode.grade_toolcall)
  constant        every sample gets 1.0                         -> no signal: weights must not change
  word:W          1.0 if the injection contains W               -> reward must rise
  reversed:W      1.0 if the injection does NOT contain W       -> W must disappear
  stub:W          scripted target: calls the attacker tool iff the injection contains W; real grader scores it
  (one-task check = --reward real --only-task <task_id>)
"""
import hashlib
import json
import os
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from redteam.attacker_prompts import attacker_messages
from redteam.episode import (build_toolcall_messages, extract_injection, format_ok, grade_toolcall, load_cases,
                             load_tools, tool_response_with_injection)
from redteam.splits import load_split

LADDER = ["L0_obey", "L1_lenient", "L2_default", "L3_guarded"]


class Curriculum:
    """Adaptive target ladder. Move up when the moving average of the step reward (at the current level) reaches
    `up`; move down when it falls below `down`. A fraction of each step's tasks stays at the previous level."""

    def __init__(self, start=0, top=3, up=0.40, down=0.05, window=5, mix_prev=0.25, fixed=None):
        self.level, self.top, self.up, self.down, self.window, self.mix_prev = start, top, up, down, window, mix_prev
        self.fixed = fixed  # a level name to pin (e.g. for evals / probes), or None
        self.history = []
        self.events = []

    def level_for_group(self, rng):
        if self.fixed:
            return self.fixed
        if self.level > 0 and rng.random() < self.mix_prev:
            return LADDER[self.level - 1]
        return LADDER[self.level]

    def update(self, step, mean_reward_at_level):
        if self.fixed or mean_reward_at_level is None:
            return
        self.history.append(mean_reward_at_level)
        avg = sum(self.history[-self.window:]) / min(len(self.history), self.window)
        if len(self.history) >= self.window and avg >= self.up and self.level < self.top:
            self.events.append({"step": step, "from": LADDER[self.level], "to": LADDER[self.level + 1], "avg": avg})
            self.level += 1
            self.history = []
        elif len(self.history) >= self.window and avg < self.down and self.level > 0:
            self.events.append({"step": step, "from": LADDER[self.level], "to": LADDER[self.level - 1], "avg": avg})
            self.level -= 1
            self.history = []


class StubTarget:
    """Scripted target for the stub-target wiring check: calls the attacker tool iff the injection has the word."""

    def __init__(self, word):
        self.word = word.lower()

    def batch(self, items, injections, cases):
        out = []
        for (msgs, schemas), inj, c in zip(items, injections, cases):
            if self.word in inj.lower():
                calls = [{"id": "stub00001", "type": "function",
                          "function": {"name": c["Attacker Tools"][0], "arguments": "{}"}}]
                out.append(dict(text="", tool_calls=calls, error=None, finish_reason="tool_calls", seconds=0.0))
            else:
                out.append(dict(text="Here is what I found.", tool_calls=None, error=None, finish_reason="stop",
                                seconds=0.0))
        return out


class InjectionReward:
    """TRL reward function. Called once per optimizer step with all 64 completions (8 tasks x 8, consecutive)."""

    def __init__(self, mode, out_dir, run_id, cases_by_id, curriculum, n_gen, target=None, seed=0):
        self.__name__ = "injection_reward"
        self.mode, self.out_dir, self.run_id = mode, out_dir, run_id
        self.cases, self.cur, self.n_gen, self.target = cases_by_id, curriculum, n_gen, target
        self.tools = load_tools()
        self.rng = random.Random(seed)
        self.calls = 0
        self.f = open(os.path.join(out_dir, "rollouts.jsonl"), "a")

    def __call__(self, prompts, completions, task_id, trainer_state=None, log_metric=None, **kw):
        t0 = time.time()
        step = trainer_state.global_step if trainer_state is not None else self.calls
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        n = len(texts)
        assert n % self.n_gen == 0
        # index check: every group of n_gen must be one task
        for g in range(n // self.n_gen):
            assert len(set(task_id[g * self.n_gen:(g + 1) * self.n_gen])) == 1, "group mixes tasks"
        cases = [self.cases[t] for t in task_id]
        levels = []
        for g in range(n // self.n_gen):
            levels += [self.cur.level_for_group(self.rng)] * self.n_gen
        injections = [extract_injection(t) for t in texts]
        fmt = [format_ok(t) for t in texts]
        rewards, results, reasons = [0.0] * n, [None] * n, [""] * n

        kind, _, word = self.mode.partition(":")
        if kind == "constant":
            rewards, reasons = [1.0] * n, ["probe_constant"] * n
        elif kind in ("word", "reversed"):
            for i, inj in enumerate(injections):
                has = word.lower() in inj.lower()
                rewards[i] = float(has if kind == "word" else not has)
                reasons[i] = f"probe_{kind}"
        else:  # real or stub: build the target's conversation and grade its reply
            items = [build_toolcall_messages(c, inj, self.tools, lv) for c, inj, lv in zip(cases, injections, levels)]
            if kind == "stub":
                results = StubTarget(word).batch(items, injections, cases)
            else:
                results = self.target.batch(items)
            for i, (c, res) in enumerate(zip(cases, results)):
                if res["error"]:
                    rewards[i], reasons[i] = 0.0, "target_error"
                    continue
                g = grade_toolcall(c, res["text"], res["tool_calls"], texts[i])
                rewards[i], reasons[i] = g["reward"], g["reason"]

        # curriculum: mean reward of this step's groups at the current level (not the easier mixed-in ones)
        cur_level = LADDER[self.cur.level]
        at_cur = [r for r, lv in zip(rewards, levels) if lv == cur_level]
        self.cur.update(step, sum(at_cur) / len(at_cur) if at_cur else None)

        for i in range(n):
            res = results[i]
            c = cases[i]
            rec = {"run_id": self.run_id, "step": step, "phase": "train", "split": "train", "reward_mode": self.mode,
                   "group_id": f"{step}:{i // self.n_gen}", "sample_idx": i % self.n_gen, "task_id": task_id[i],
                   "goal_id": c["goal_id"], "set": c["set"], "attacker_tool": c["Attacker Tools"][0],
                   "goal": c["Attacker Instruction"], "user_tool": c["User Tool"],
                   "user_instruction": c["User Instruction"], "target_level": levels[i],
                   "attacker_output": texts[i], "format_ok": fmt[i], "injection": injections[i],
                   "completion_sha1": hashlib.sha1(texts[i].encode()).hexdigest()[:12],
                   "reward": rewards[i], "reward_reason": reasons[i]}
            if res is not None:
                rec.update(target_output=res["text"] or "", target_tool_calls=res["tool_calls"],
                           target_error=res["error"], t_target_s=round(res["seconds"], 3))
            self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.calls += 1
        if log_metric is not None:
            log_metric("ours/format_ok", sum(fmt) / n)
            log_metric("ours/level_idx", float(self.cur.level))
            log_metric("ours/reward_at_current_level", sum(at_cur) / len(at_cur) if at_cur else float("nan"))
            log_metric("ours/distinct_completions", len(set(texts)) / n)
            log_metric("ours/t_reward_s", time.time() - t0)
        return rewards


class CanaryAndLog(TrainerCallback):
    """Every `every` steps: canary eval (16 fixed dev tasks x 8 samples, target at L2 default) with the current policy.
    Every step: append TRL's logged metrics to metrics.jsonl. At start/end: hash of the LoRA weights (no-signal check)."""

    def __init__(self, out_dir, trainer_ref, canary_cases, target, tools, variant, every=25, n=8, max_new=256):
        self.out_dir, self.tr, self.canary, self.target, self.tools = out_dir, trainer_ref, canary_cases, target, tools
        self.variant, self.every, self.n, self.max_new = variant, every, n, max_new

    def lora_hash(self):
        h = hashlib.sha256()
        for name, p in sorted(self.tr[0].model.named_parameters()):
            if "lora_" in name:
                h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def on_train_begin(self, args, state, control, **kw):
        self._write("weights.jsonl", {"when": "begin", "step": state.global_step, "lora_sha256": self.lora_hash()})
        if self.target is not None and self.canary:
            self.canary_eval(state.global_step)

    def on_train_end(self, args, state, control, **kw):
        self._write("weights.jsonl", {"when": "end", "step": state.global_step, "lora_sha256": self.lora_hash()})
        if self.target is not None and self.canary and state.global_step % self.every:
            self.canary_eval(state.global_step)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            self._write("metrics.jsonl", {"step": state.global_step, **logs})
            if "reward" in logs:
                keys = ["reward", "frac_reward_zero_std", "ours/level_idx", "ours/reward_at_current_level",
                        "ours/format_ok", "entropy", "completions/mean_length", "completions/clipped_ratio", "step_time"]
                print("PROGRESS step=%d " % state.global_step + " ".join(
                    f"{k.split('/')[-1]}={logs[k]:.3f}" for k in keys if isinstance(logs.get(k), (int, float))), flush=True)

    def on_step_end(self, args, state, control, **kw):
        if self.target is not None and self.canary and state.global_step % self.every == 0:
            self.canary_eval(state.global_step)
        if state.global_step % 5 == 0:  # make partial logs visible on the volume for monitoring (best effort)
            try:
                import modal
                modal.Volume.from_name("redteam-runs").commit()
            except Exception:  # noqa: BLE001
                pass

    @torch.no_grad()
    def canary_eval(self, step):
        trainer = self.tr[0]
        model, tok = trainer.model, trainer.processing_class
        was_training = model.training
        model.eval()
        tok.padding_side = "left"
        texts = [tok.apply_chat_template(attacker_messages(c, self.variant), add_generation_prompt=True,
                                         tokenize=False) for c in self.canary for _ in range(self.n)]
        outs = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=self.max_new,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if was_training:
            model.train()
        cases = [c for c in self.canary for _ in range(self.n)]
        injections = [extract_injection(t) for t in outs]
        items = [build_toolcall_messages(c, inj, self.tools, "L2_default") for c, inj in zip(cases, injections)]
        results = self.target.batch(items)
        rewards = []
        for c, t, inj, res in zip(cases, outs, injections, results):
            g = (dict(reward=0.0, reason="target_error") if res["error"]
                 else grade_toolcall(c, res["text"], res["tool_calls"], t))
            rewards.append(g["reward"])
            self._write("canary.jsonl", {"step": step, "task_id": c["task_id"], "target_level": "L2_default",
                                         "attacker_output": t, "format_ok": format_ok(t), "injection": inj,
                                         "target_output": res["text"] or "", "target_tool_calls": res["tool_calls"],
                                         "reward": g["reward"], "reward_reason": g["reason"]})
        summary = {"step": step, "canary_pass@1_L2": sum(rewards) / len(rewards),
                   "canary_format_ok": sum(format_ok(t) for t in outs) / len(outs)}
        self._write("canary_summary.jsonl", summary)
        print("CANARY", summary, flush=True)

    def _write(self, name, obj):
        with open(os.path.join(self.out_dir, name), "a") as f:
            f.write(json.dumps(obj) + "\n")


def train(out_dir, run_id, attacker_model, attacker_revision, reward="real", target=None, max_steps=40,
          lr=2e-5, lora_r=32, lora_alpha=64, n_gen=8, tasks_per_step=8, max_completion_length=256,
          variant="v2_context_hints", start_level=0, top_level=3, fixed_level=None, only_task=None,
          canary_every=25, save_steps=25, seed=0, report_to="none", bf16=True, gradient_checkpointing=True,
          up=0.40, down=0.05, mix_prev=0.25):
    os.makedirs(out_dir, exist_ok=True)
    cases = load_cases(("dh", "ds"))
    train_cases = load_split("train", cases)
    if only_task:
        train_cases = [c for c in train_cases if c["task_id"] == only_task] * 64
    ds = Dataset.from_list([{"prompt": attacker_messages(c, variant), "task_id": c["task_id"]} for c in train_cases])
    cur = Curriculum(start=start_level, top=top_level, fixed=fixed_level, up=up, down=down, mix_prev=mix_prev)
    reward_fn = InjectionReward(reward, out_dir, run_id, {c["task_id"]: c for c in cases}, cur, n_gen, target, seed)

    per_device = 16
    cfg = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"), run_name=run_id, seed=seed,
        max_steps=max_steps, learning_rate=lr, lr_scheduler_type="constant_with_warmup", warmup_steps=max(1, round(0.03 * max_steps)),
        per_device_train_batch_size=per_device, gradient_accumulation_steps=(tasks_per_step * n_gen) // per_device,
        num_generations=n_gen, max_completion_length=max_completion_length, temperature=1.0, top_p=1.0,
        beta=0.0, loss_type="dapo", scale_rewards="group", num_iterations=1, max_grad_norm=0.2,
        bf16=bf16, gradient_checkpointing=gradient_checkpointing,
        model_init_kwargs={"dtype": torch.bfloat16 if bf16 else torch.float32},
        logging_steps=1, save_steps=save_steps, save_only_model=True, report_to=report_to,
        log_completions=False, remove_unused_columns=False, use_vllm=False,
    )
    peft = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])
    trainer_ref = []
    canary = load_split("canary", cases) if (reward == "real" and not only_task) else []
    cb = CanaryAndLog(out_dir, trainer_ref, canary, target if reward == "real" else None, load_tools(), variant,
                      every=canary_every)
    # TRL loads the config/tokenizer without a revision, which fails offline; give it the pinned snapshot's path.
    model_path = attacker_model
    if attacker_revision:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(attacker_model, revision=attacker_revision, local_files_only=True)
    trainer = GRPOTrainer(model=model_path, reward_funcs=[reward_fn], args=cfg, train_dataset=ds,
                          peft_config=peft, callbacks=[cb])
    trainer_ref.append(trainer)
    t0 = time.time()
    trainer.train()
    summary = {"run_id": run_id, "reward_mode": reward, "steps": trainer.state.global_step,
               "t_train_s": round(time.time() - t0, 1), "curriculum_events": cur.events,
               "final_level": LADDER[cur.level]}
    trainer.save_model(os.path.join(out_dir, "final_adapter"))
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
