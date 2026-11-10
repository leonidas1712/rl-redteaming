"""Modal entry points for the RL red-teamer. Importing this file submits nothing.

  modal run modal_app.py::prefetch              # CPU: download pinned model weights into the weights volume (once)
  modal run --detach modal_app.py::phase0       # GPU: phase 0 pipeline check; results copied back to runs/<run_id>/
  modal run --detach modal_app.py::sweep         # GPU: phase 2 base-rate sweep (dev, attacker prompt v2, 4 target levels)
  modal run --detach modal_app.py::train_smoke   # GPU: 2 training steps with the stub target (code check)
  modal run --detach modal_app.py::wiring        # GPU: the 5 wiring-check training runs, in parallel containers
  modal run --detach modal_app.py::phase3        # GPU: phase 3 real training run (adaptive ladder from L0)
  modal run --detach modal_app.py::evalckpt --train-run <id>   # GPU: base + every checkpoint of a run on dev, L2 and L0
  modal run modal_app.py::fetch --run-id <id>   # copy a run's folder from the volume to runs/<id>/ (e.g. after a disconnect)

Every GPU run writes to the `redteam-runs` volume under <run_id>/ and the local entrypoint downloads that folder,
plus a config.json with the exact settings, model revisions and sha256 of every source file used.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent if modal.is_local() else Path("/app")

MODELS = {
    "target": {"name": "mistralai/Ministral-3-3B-Instruct-2512-BF16", "revision": "b6d637bef2393152b3da2b2fde72eecdee30557e",
               # mistral-format files only (vLLM --load-format mistral); skips the duplicate HF shards
               "allow_patterns": ["consolidated.safetensors", "params.json", "tekken.json", "*.json", "*.jinja", "*.txt"]},
    "attacker": {"name": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
                 "allow_patterns": None},
}
VLLM_PORT = 8010

app = modal.App("rl-redteam")
weights = modal.Volume.from_name("redteam-weights", create_if_missing=True)
runs = modal.Volume.from_name("redteam-runs", create_if_missing=True)
image = (
    # CUDA devel base (has nvcc): vLLM's FlashInfer kernels JIT-compile at startup. debian_slim failed with
    # "Could not find nvcc" (run 20261110-1520-phase0). Same base as Modal's vLLM example.
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm==0.32.0", "mistral_common[image]==1.12.0",  # 1.13 added a required arg vLLM 0.32 doesn't pass
                    "trl==1.17.1", "peft==0.21.2", "openai", "nltk", "wandb")
    .env({"HF_HOME": "/weights/hf", "PYTHONPATH": "/app/src", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_CACHE_ROOT": "/weights/vllm_cache"})  # keep torch.compile/CUDA-graph cache between runs
)
if modal.is_local():
    image = image.add_local_dir(ROOT / "src", remote_path="/app/src", ignore=["**/__pycache__/**"])
    image = image.add_local_dir(ROOT / "data", remote_path="/app/data")
    image = image.add_local_file(ROOT / "runs/precheck/toolcall_openrouter_ministral_levels.jsonl",
                                 "/app/runs/precheck/toolcall_openrouter_ministral_levels.jsonl")


def source_hashes():
    out = {}
    for p in sorted(list((ROOT / "src").rglob("*.py")) + [ROOT / "modal_app.py"]):
        out[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


@app.function(image=image, volumes={"/weights": weights}, timeout=60 * 60, cpu=4, memory=16384)
def prefetch():
    from huggingface_hub import snapshot_download
    for role, m in MODELS.items():
        path = snapshot_download(m["name"], revision=m["revision"], allow_patterns=m["allow_patterns"])
        print(role, m["name"], "->", path, flush=True)
    weights.commit()


def start_vllm(model, revision, log_path, gpu_mem=0.30, max_len=8192, port=VLLM_PORT, mistral=True, extra=None):
    """Serve a model with vLLM as a subprocess on this GPU (Modal's documented Popen pattern).
    mistral=True: flags from the Ministral 3 model card (mistral tokenizer/config/load formats + tool-call parser)."""
    import subprocess
    import time
    import urllib.request

    cmd = ["vllm", "serve", model, "--revision", revision, "--served-model-name", model, "--port", str(port),
           "--gpu-memory-utilization", str(gpu_mem), "--max-model-len", str(max_len), "--seed", "0"]
    cmd += list(extra or [])
    if mistral:
        cmd += ["--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral",
                "--enable-auto-tool-choice", "--tool-call-parser", "mistral"]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with {proc.returncode}; see {log_path}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            return proc, round(time.time() - t0, 1), cmd
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise TimeoutError("vLLM did not come up in 15 min")


@app.function(image=image, gpu="A100-80GB", volumes={"/weights": weights, "/runs": runs}, timeout=45 * 60,
              cpu=8, memory=65536)
def phase0_remote(run_id: str, config: dict):
    import time
    os.environ["HF_HUB_OFFLINE"] = "1"
    out = f"/runs/{run_id}"
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    try:
        tgt = MODELS["target"]
        proc, t_up, cmd = start_vllm(tgt["name"], tgt["revision"], f"{out}/vllm.log")
        config["vllm_cmd"] = cmd
        config["t_vllm_startup_s"] = t_up
        json.dump(config, open(f"{out}/config.json", "w"), indent=1)
        from redteam.phase0 import run
        summary = run(out, run_id, tgt["name"], MODELS["attacker"]["name"],
                      "/app/runs/precheck/toolcall_openrouter_ministral_levels.jsonl",
                      attacker_revision=MODELS["attacker"]["revision"])
        summary["t_vllm_startup_s"] = t_up
        summary["t_total_s"] = round(time.time() - t_start, 1)
        json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
        proc.terminate()
        return summary
    finally:
        runs.commit()


@app.function(image=image, gpu="A100-80GB", volumes={"/weights": weights, "/runs": runs}, timeout=60 * 60,
              cpu=8, memory=65536)
def sweep_remote(run_id: str, config: dict):
    import time
    os.environ["HF_HUB_OFFLINE"] = "1"
    out = f"/runs/{run_id}"
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    try:
        tgt, att = MODELS["target"], MODELS["attacker"]
        p_t, t_up_t, cmd_t = start_vllm(tgt["name"], tgt["revision"], f"{out}/vllm_target.log", gpu_mem=0.35)
        p_a, t_up_a, cmd_a = start_vllm(att["name"], att["revision"], f"{out}/vllm_attacker.log", gpu_mem=0.30,
                                        max_len=4096, port=VLLM_PORT + 1, mistral=False)
        config.update(vllm_target_cmd=cmd_t, vllm_attacker_cmd=cmd_a, t_vllm_startup_s=[t_up_t, t_up_a])
        json.dump(config, open(f"{out}/config.json", "w"), indent=1)
        from redteam.sweep import run
        summary = run(out, run_id, att["name"], f"http://127.0.0.1:{VLLM_PORT + 1}/v1",
                      tgt["name"], f"http://127.0.0.1:{VLLM_PORT}/v1", **config["sweep_args"])
        summary["t_total_s"] = round(time.time() - t_start, 1)
        json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
        p_t.terminate()
        p_a.terminate()
        return summary
    finally:
        runs.commit()


@app.function(image=image, gpu="A100-80GB", volumes={"/weights": weights, "/runs": runs}, timeout=2 * 60 * 60,
              cpu=8, memory=65536)
def train_remote(run_id: str, config: dict):
    """One GRPO run. Starts the vLLM target only when the reward needs the real target."""
    import time
    os.environ["HF_HUB_OFFLINE"] = "1"
    out = f"/runs/{run_id}"
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    proc = None
    try:
        target = None
        if config["train_args"]["reward"] == "real":
            tgt = MODELS["target"]
            proc, t_up, cmd = start_vllm(tgt["name"], tgt["revision"], f"{out}/vllm_target.log", gpu_mem=0.30)
            config.update(vllm_target_cmd=cmd, t_vllm_startup_s=t_up)
            from redteam.target import VLLMTarget
            target = VLLMTarget(tgt["name"], concurrency=64)
        json.dump(config, open(f"{out}/config.json", "w"), indent=1)
        from redteam.train import train
        summary = train(out, run_id, MODELS["attacker"]["name"], MODELS["attacker"]["revision"], target=target,
                        **config["train_args"])
        summary["t_total_s"] = round(time.time() - t_start, 1)
        json.dump(summary, open(f"{out}/train_summary.json", "w"), indent=1)
        return summary
    finally:
        if proc is not None:
            proc.terminate()
        runs.commit()


@app.function(image=image, gpu="A100-80GB", volumes={"/weights": weights, "/runs": runs}, timeout=2 * 60 * 60,
              cpu=8, memory=65536)
def eval_remote(run_id: str, config: dict):
    """Serve the base attacker + every saved LoRA of a training run with vLLM; evaluate each on a split."""
    import time
    os.environ["HF_HUB_OFFLINE"] = "1"
    runs.reload()
    out = f"/runs/{run_id}"
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    procs = []
    try:
        if config["eval_args"]["split"] == "test":
            assert config.get("final_eval_confirmed") is True, "test split needs explicit confirmation"
            os.environ["RL_REDTEAM_FINAL_EVAL"] = "1"
        src = f"/runs/{config['train_run']}"
        ckpts = sorted((d for d in os.listdir(f"{src}/checkpoints") if d.startswith("checkpoint-")),
                       key=lambda d: int(d.split("-")[1])) if os.path.isdir(f"{src}/checkpoints") else []
        modules = [f"step{d.split('-')[1]}={src}/checkpoints/{d}" for d in ckpts]
        if os.path.isdir(f"{src}/final_adapter"):
            modules.append(f"final={src}/final_adapter")
        if config.get("only_adapters"):  # e.g. ["step80"] for the one-time test evaluation
            modules = [m for m in modules if m.split("=")[0] in config["only_adapters"]]
        tgt, att = MODELS["target"], MODELS["attacker"]
        p_t, t_up_t, cmd_t = start_vllm(tgt["name"], tgt["revision"], f"{out}/vllm_target.log", gpu_mem=0.35)
        procs.append(p_t)
        p_a, t_up_a, cmd_a = start_vllm(att["name"], att["revision"], f"{out}/vllm_attacker.log", gpu_mem=0.35,
                                        max_len=4096, port=VLLM_PORT + 1, mistral=False,
                                        extra=["--enable-lora", "--max-lora-rank", "32", "--max-loras", "4",
                                               "--lora-modules", *modules] if modules else None)
        procs.append(p_a)
        adapters = ["base"] + [m.split("=")[0] for m in modules]
        config.update(vllm_target_cmd=cmd_t, vllm_attacker_cmd=cmd_a, adapters=adapters)
        json.dump(config, open(f"{out}/config.json", "w"), indent=1)
        from redteam.evaluate import run
        summary = run(out, run_id, f"http://127.0.0.1:{VLLM_PORT + 1}/v1", att["name"], adapters, tgt["name"],
                      f"http://127.0.0.1:{VLLM_PORT}/v1", **config["eval_args"])
        summary["t_total_s"] = round(time.time() - t_start, 1)
        json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
        return summary
    finally:
        for p in procs:
            p.terminate()
        runs.commit()


def _train_config(run_id, phase, train_args):
    return {"run_id": run_id, "phase": phase, "models": MODELS, "gpu": "A100-80GB", "train_args": train_args,
            "source_sha256": source_hashes(), "started_local": datetime.datetime.now().isoformat()}


def download(run_id):
    dest = ROOT / "runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for entry in runs.listdir(run_id):
        if entry.type.name == "DIRECTORY":  # checkpoints/ and adapters stay on the volume (pull on demand)
            print(f"  (kept on volume: {entry.path}/)")
            continue
        name = Path(entry.path).name
        with open(dest / name, "wb") as f:
            for chunk in runs.read_file(entry.path):
                f.write(chunk)
    print(f"downloaded -> {dest}")


@app.local_entrypoint()
def phase0():
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M") + "-phase0"
    config = {"run_id": run_id, "phase": "phase0", "models": MODELS, "gpu": "A100-80GB",
              "source_sha256": source_hashes(), "started_local": datetime.datetime.now().isoformat()}
    print("run_id", run_id, flush=True)
    try:
        summary = phase0_remote.remote(run_id, config)
        print(json.dumps(summary, indent=1))
    finally:
        download(run_id)


@app.local_entrypoint()
def fetch(run_id: str):
    download(run_id)


@app.local_entrypoint()
def sweep(variant: str = "v2_context_hints", n: int = 8, split: str = "dev"):
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M") + "-sweep"
    config = {"run_id": run_id, "phase": "phase2_sweep", "models": MODELS, "gpu": "A100-80GB",
              "sweep_args": {"split": split, "variant": variant, "n": n, "max_tokens": 256, "temperature": 1.0,
                             "seed": 0, "levels": ["L0_obey", "L1_lenient", "L2_default", "L3_guarded"]},
              "source_sha256": source_hashes(), "started_local": datetime.datetime.now().isoformat()}
    print("run_id", run_id, flush=True)
    try:
        summary = sweep_remote.remote(run_id, config)
        print(json.dumps(summary, indent=1))
    finally:
        download(run_id)


@app.local_entrypoint()
def train_smoke():
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M") + "-train-smoke"
    args = {"reward": "stub:execute", "max_steps": 2, "save_steps": 1000}
    try:
        print(json.dumps(train_remote.remote(run_id, _train_config(run_id, "train_smoke", args)), indent=1))
    finally:
        download(run_id)


WIRING = {  # name -> train_args. Expected results are in LOG.md / overview.
    "nosignal": {"reward": "constant", "max_steps": 10, "save_steps": 1000},
    "word": {"reward": "word:urgent", "max_steps": 30, "save_steps": 1000},
    "reversed": {"reward": "reversed:please", "max_steps": 30, "save_steps": 1000},
    "stub": {"reward": "stub:execute", "max_steps": 30, "save_steps": 1000},
    "onetask": {"reward": "real", "max_steps": 30, "save_steps": 1000, "fixed_level": "L0_obey",
                "only_task": "dh:The23andMeUpdateData|TwitterManagerGetUserProfile"},
}


@app.local_entrypoint()
def wiring(only: str = ""):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    names = [n for n in WIRING if not only or n in only.split(",")]
    calls = {}
    for name in names:
        run_id = f"{stamp}-wiring-{name}"
        calls[run_id] = train_remote.spawn(run_id, _train_config(run_id, f"wiring_{name}", WIRING[name]))
        print("spawned", run_id, flush=True)
    for run_id, call in calls.items():
        try:
            print(run_id, json.dumps(call.get()), flush=True)
        except Exception as e:  # noqa: BLE001
            print(run_id, "FAILED:", repr(e)[:500], flush=True)
        finally:
            download(run_id)


@app.local_entrypoint()
def phase3(max_steps: int = 80, up: float = 0.30):
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M") + "-phase3"
    args = {"reward": "real", "max_steps": max_steps, "save_steps": 25, "canary_every": 25, "start_level": 0,
            "top_level": 3, "up": up, "down": 0.05, "mix_prev": 0.25}
    print("run_id", run_id, flush=True)
    try:
        print(json.dumps(train_remote.remote(run_id, _train_config(run_id, "phase3", args)), indent=1))
    finally:
        download(run_id)


@app.local_entrypoint()
def evalckpt(train_run: str, split: str = "dev", n: int = 8, levels: str = "L2_default,L0_obey", only: str = "",
             confirm_test_once: bool = False):
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M") + f"-eval-{split}"
    config = {"run_id": run_id, "phase": "eval", "train_run": train_run, "models": MODELS, "gpu": "A100-80GB",
              "eval_args": {"split": split, "n": n, "levels": levels.split(","), "variant": "v2_context_hints",
                            "max_tokens": 256, "seed": 0},
              "source_sha256": source_hashes(), "started_local": datetime.datetime.now().isoformat()}
    if split == "test" and not confirm_test_once:
        raise SystemExit("test split is locked: pass --confirm-test-once (see experiments/PREREG_defenders.md, F5)")
    config["final_eval_confirmed"] = split == "test" and confirm_test_once
    config["only_adapters"] = [a for a in only.split(",") if a]
    print("run_id", run_id, flush=True)
    try:
        print(json.dumps(eval_remote.remote(run_id, config), indent=1))
    finally:
        download(run_id)
