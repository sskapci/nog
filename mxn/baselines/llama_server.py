"""The LLM on llama.cpp's server (GGUF, CPU), with the same interface as LocalLLM.

The server is a child process bound to 127.0.0.1; prompts go through the model's own chat
template (/v1/chat/completions), greedy, with the repetition penalty LocalLLM gets from the model's
generation_config (1.1 over the whole context, prompt included: HF's RepetitionPenaltyLogitsProcessor;
found in the Q8_0 debug, decision 2 of 2026-10-24). P(yes) is read from the first answer token's top
log-probabilities (Yes vs No, renormalized), as LocalLLM.yes_prob_batch reads its logits.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path


class ServerLLM:
    def __init__(self, server: str, model: str, port: int = 7410, threads: int = 8, ctx: int = 4096,
                 log: str | None = None, repeat_penalty: float = 1.1):
        self.url = f"http://127.0.0.1:{port}"
        out = open(log, "a") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen([server, "-m", model, "--host", "127.0.0.1", "--port", str(port),
                                      "-t", str(threads), "-c", str(ctx), "--temp", "0"],
                                     stdout=out, stderr=subprocess.STDOUT)
        self.model = Path(model).name
        self.penalty = {"repeat_penalty": repeat_penalty, "repeat_last_n": ctx}
        self.last: dict = {}
        for _ in range(600):
            try:
                if json.load(urllib.request.urlopen(self.url + "/health", timeout=2)).get("status") == "ok":
                    return
            except Exception:
                time.sleep(1)
        self.close()
        raise RuntimeError("llama-server did not start")

    def _chat(self, prompt: str, **kw) -> dict:
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 0,
                           **kw}).encode()
        req = urllib.request.Request(self.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        u = r.get("usage", {})
        self.last = {"prompt_tokens": u.get("prompt_tokens"), "gen_tokens": u.get("completion_tokens")}
        return r

    def generate(self, prompt: str, max_new_tokens: int = 120) -> str:
        return self._chat(prompt, max_tokens=max_new_tokens, **self.penalty)["choices"][0]["message"]["content"]

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 120) -> list[str]:
        return [self.generate(p, max_new_tokens) for p in prompts]

    def yes_prob_batch(self, prompts: list[str]) -> list[float]:
        out = []
        for p in prompts:
            r = self._chat(p, max_tokens=1, logprobs=True, top_logprobs=20, repeat_penalty=1.0)   # HF: one forward pass, no penalty
            self.last["gen_tokens"] = 0
            top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            lp = {t["token"].strip(): t["logprob"] for t in top}
            y, n = math.exp(lp.get("Yes", -1e9)), math.exp(lp.get("No", -1e9))
            out.append(y / (y + n) if y + n > 0 else 0.5)
        return out

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate(); self.proc.wait()
