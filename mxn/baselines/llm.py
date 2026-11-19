"""Optional small local LLM (CPU). Used only by the optional rephraser and question
generator; both are off by default."""

from __future__ import annotations


class LocalLLM:
    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        # float32: bfloat16 (the checkpoint default) is several times slower on this CPU
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float32").to(device)
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 120) -> str:
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 120) -> list[str]:
        import torch

        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        ids = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        n = ids["input_ids"].shape[1]
        gen = [o[n:] for o in out]
        eos = {self.tok.eos_token_id, self.tok.pad_token_id}
        self.last = {"prompt_tokens": int(ids["attention_mask"][0].sum()),
                     "gen_tokens": int(sum(1 for t in gen[0].tolist() if t not in eos))}
        return [self.tok.decode(o, skip_special_tokens=True) for o in gen]

    def yes_prob_batch(self, prompts: list[str]) -> list[float]:
        """P("Yes") vs P("No") as the first answer token: one forward pass, no generation."""
        import torch

        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        ids = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        self.last = {"prompt_tokens": int(ids["attention_mask"][0].sum()), "gen_tokens": 0}
        yes = self.tok.encode("Yes", add_special_tokens=False)[0]
        no = self.tok.encode("No", add_special_tokens=False)[0]
        with torch.inference_mode():
            logits = self.model(**ids, logits_to_keep=1).logits[:, -1, [yes, no]]   # last position only
        return torch.softmax(logits.float(), -1)[:, 0].tolist()
