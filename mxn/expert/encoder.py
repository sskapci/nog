"""Our bi-encoder, trained from scratch (step 2 of 2026-10-24b; no pretrained weights).

Transformer encoder over our 16k SentencePiece pieces: 4 layers, d 256, 4 heads, FF 1024, max 128
pieces, learned positions, a query/passage type embedding, mean pooling over real tokens, L2 norm.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiEncoder(nn.Module):
    def __init__(self, vocab: int = 16000, d: int = 256, layers: int = 4, heads: int = 4, ff: int = 1024,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.typ = nn.Embedding(2, d)                     # 0 query, 1 passage
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor, kind: int) -> torch.Tensor:
        mask = ids != 0
        pos = torch.arange(ids.shape[1], device=ids.device)[None, :]
        x = self.tok(ids) + self.pos(pos) + self.typ.weight[kind]
        x = self.norm(self.enc(x, src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).float()
        return F.normalize((x * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


class Encoder:
    """Text → vectors with our tokenizer and a trained BiEncoder (inference, CPU)."""

    def __init__(self, model_dir: str, tokenizer: str, threads: int = 8):
        import sentencepiece as spm

        torch.set_num_threads(threads)
        ck = torch.load(f"{model_dir}/model.pt", map_location="cpu")
        self.net = BiEncoder(**ck["config"]); self.net.load_state_dict(ck["state"]); self.net.eval()
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer)
        self.passage_title = ck.get("passage_title", False)

    def ids(self, texts: list[str]) -> torch.Tensor:
        return pad([self.sp.encode(t)[: self.net.max_len] for t in texts])

    def encode(self, texts: list[str], kind: str = "query", batch: int = 256) -> np.ndarray:
        out = []
        with torch.inference_mode():
            order = np.argsort([len(t) for t in texts])
            for i in range(0, len(texts), batch):
                idx = order[i: i + batch]
                out.append((idx, self.net(self.ids([texts[j] for j in idx]), 0 if kind == "query" else 1).numpy()))
        V = np.zeros((len(texts), self.net.norm.normalized_shape[0]), np.float32)
        for idx, v in out:
            V[idx] = v
        return V


def pad(seqs: list[list[int]]) -> torch.Tensor:
    n = max(1, max(len(s) for s in seqs))
    return torch.tensor([s + [0] * (n - len(s)) for s in seqs], dtype=torch.long)


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
