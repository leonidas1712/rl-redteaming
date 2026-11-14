"""Two OpenAI-compatible vLLM endpoints on Modal, used by hud_env/ and anything else that wants our models over HTTPS.

  modal serve modal_serve.py      # dev mode: endpoints live only while this command runs (Ctrl-C stops billing)

  target   = Ministral-3-3B-Instruct-2512-BF16 (same weights/flags as training and evals), served name = repo id
  attacker = Qwen2.5-3B-Instruct, served name "base", plus LoRA adapter "step80" (Run A's last checkpoint)

Both require the API key VLLM_API_KEY, read locally from .env (gitignored) and passed to the containers.
Importing this file submits nothing.
"""
import os
import subprocess
from pathlib import Path

import modal

from modal_app import MODELS, image, runs, weights

ROOT = Path(__file__).resolve().parent
if modal.is_local():
    for line in open(ROOT / ".env"):
        if line.startswith("VLLM_API_KEY="):
            os.environ["VLLM_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
key_secret = modal.Secret.from_dict({"VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "")})

serve_image = image.add_local_python_source("modal_app")  # the container imports modal_app too (crash-looped without it)
app = modal.App("rl-redteam-serve")
ATTACKER_ADAPTER = "/runs/20261110-1805-phase3/checkpoints/checkpoint-80"
PORT = 8000


def _vllm(args):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    subprocess.Popen(["vllm", "serve", *args, "--port", str(PORT), "--host", "0.0.0.0",
                      "--api-key", os.environ["VLLM_API_KEY"], "--seed", "0"], env=env)


@app.function(image=serve_image, gpu="L40S", volumes={"/weights": weights}, secrets=[key_secret], timeout=2 * 60 * 60,
              scaledown_window=300)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def target():
    t = MODELS["target"]
    _vllm([t["name"], "--revision", t["revision"], "--served-model-name", t["name"],
           "--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral",
           "--enable-auto-tool-choice", "--tool-call-parser", "mistral",
           "--gpu-memory-utilization", "0.85", "--max-model-len", "8192"])


@app.function(image=serve_image, gpu="L40S", volumes={"/weights": weights, "/runs": runs}, secrets=[key_secret],
              timeout=2 * 60 * 60, scaledown_window=300)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def attacker():
    a = MODELS["attacker"]
    _vllm([a["name"], "--revision", a["revision"], "--served-model-name", "base",
           "--enable-lora", "--max-lora-rank", "32", "--lora-modules", f"step80={ATTACKER_ADAPTER}",
           "--gpu-memory-utilization", "0.85", "--max-model-len", "4096"])
