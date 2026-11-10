"""The target: a fixed model behind an OpenAI-compatible server (our own vLLM on the training GPU).

One call per episode: the pre-written conversation (system, user, assistant tool call, tool result) plus two tool
schemas go in; the target's next turn (text and/or tool calls) comes out. Requests are sent concurrently; vLLM batches
them on the GPU.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


class VLLMTarget:
    def __init__(self, model, base_url="http://127.0.0.1:8010/v1", max_tokens=512, temperature=0.0,
                 concurrency=64, timeout=300, api_key="EMPTY"):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    def one(self, messages, tools):
        """Returns dict(text, tool_calls, error, seconds). tool_calls: list of {"function": {"name", "arguments"}}."""
        t0 = time.time()
        try:
            r = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools,
                                                    temperature=self.temperature, max_tokens=self.max_tokens)
            msg = r.choices[0].message
            calls = [{"id": c.id, "type": "function",
                      "function": {"name": c.function.name, "arguments": c.function.arguments}}
                     for c in (msg.tool_calls or [])]
            return dict(text=msg.content or "", tool_calls=calls or None, error=None,
                        finish_reason=r.choices[0].finish_reason, seconds=time.time() - t0)
        except Exception as e:  # noqa: BLE001 — logged per episode, scored as target_error (reward 0)
            return dict(text=None, tool_calls=None, error=f"{type(e).__name__}: {e}"[:500], finish_reason=None,
                        seconds=time.time() - t0)

    def batch(self, items, concurrency=None):
        """items: list of (messages, tools). Results come back in the same order as items."""
        with ThreadPoolExecutor(concurrency or self.concurrency) as ex:
            return list(ex.map(lambda it: self.one(*it), items))
