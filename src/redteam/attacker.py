"""The attacker: Qwen2.5-3B-Instruct (+ LoRA) generating injections with plain HF generate, batched.

Same loading path as training (bf16 + PEFT LoRA), so phase 0 times the real thing. An untrained LoRA has B = 0,
so its outputs equal the base model's.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFAttacker:
    def __init__(self, model_name, revision=None, lora_rank=32, lora_alpha=64, device="cuda"):
        from peft import LoraConfig, get_peft_model

        self.tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.bfloat16).to(device)
        cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])
        self.model = get_peft_model(base, cfg).eval()
        self.device = device

    @torch.no_grad()
    def generate(self, messages_list, n=1, temperature=1.0, top_p=1.0, max_new_tokens=256, seed=0, batch_size=64):
        """messages_list: one chat (list of dicts) per prompt. Returns, per prompt, a list of n dicts
        (text, tokens, truncated). Prompts are repeated n times and generated in batches."""
        torch.manual_seed(seed)
        texts = [self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages_list]
        flat = [t for t in texts for _ in range(n)]
        outs, t0 = [], time.time()
        for i in range(0, len(flat), batch_size):
            enc = self.tok(flat[i:i + batch_size], return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                                      max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                ids = [t for t in row.tolist() if t != self.tok.pad_token_id and t != self.tok.eos_token_id]
                outs.append(dict(text=self.tok.decode(ids, skip_special_tokens=True), tokens=len(ids),
                                 truncated=len(ids) >= max_new_tokens))
        self.last_seconds = time.time() - t0
        return [outs[k * n:(k + 1) * n] for k in range(len(messages_list))]
