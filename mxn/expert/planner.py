"""Our planner (step 3 of 2026-10-24b): question → plan in the fixed plan language.

The simplest design of the step 3 rules: the step 2 encoder's transformer trunk (our weights) with
heads on top: template (3), r1 and r2 (relations + none), op (6 + none) from the mean-pooled
question, and a per-piece BIO tagger for the entity spans E1/E2 (O, B-E1, I-E1, B-E2, I-E2).
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from mxn.expert.encoder import BiEncoder, pad

TEMPLATES = ["chain", "compare", "bridge"]
OPS = ["none", "earlier", "later", "larger", "smaller", "same", "different"]
TAGS = ["O", "B-E1", "I-E1", "B-E2", "I-E2"]


class Planner(nn.Module):
    """templates / r3 (MetaQA reverse lookup, reports/design_reverse_lookup.md): a planner trained with more templates
    (e.g. "rev": hop 1 forward r1, hop 2 reverse r2, optional hop 3 forward r3; "prev", person heads,
    reports/design_metaqa_sets_persons.md: hop 1 reverse r1 for the head words, optional hop 2 forward r2) and an r3
    head; both are stored in its checkpoint (load_planner). Defaults = the 2Wiki planners, unchanged."""

    def __init__(self, relations: list[str], trunk: dict | None = None, templates: list | None = None, r3: bool = False):
        super().__init__()
        self.relations = ["none"] + list(relations)
        self.templates = list(templates or TEMPLATES)
        self.trunk = BiEncoder(**(trunk or {}))
        d = self.trunk.norm.normalized_shape[0]
        self.template = nn.Linear(d, len(self.templates))
        self.r1, self.r2 = nn.Linear(d, len(self.relations)), nn.Linear(d, len(self.relations))
        self.has_r3 = r3
        if r3:
            self.r3 = nn.Linear(d, len(self.relations))
        self.op = nn.Linear(d, len(OPS))
        self.tag = nn.Linear(d, len(TAGS))

    def forward(self, ids: torch.Tensor) -> dict:
        t = self.trunk
        mask = ids != 0
        pos = torch.arange(ids.shape[1], device=ids.device)[None, :]
        x = t.tok(ids) + t.pos(pos) + t.typ.weight[0]
        x = t.norm(t.enc(x, src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).float()
        h = (x * m).sum(1) / m.sum(1).clamp(min=1)
        o = {"template": self.template(h), "r1": self.r1(h), "r2": self.r2(h), "op": self.op(h), "tag": self.tag(x)}
        if self.has_r3:
            o["r3"] = self.r3(h)
        return o


def load_planner(ck: dict) -> "Planner":
    """A planner from its checkpoint dict (templates and r3 head when it was trained with them)."""
    net = Planner(ck["relations"], ck["trunk"], ck.get("templates"), ck.get("r3", False))
    net.load_state_dict(ck["state"]); net.eval()
    return net


def pieces(sp, text: str, max_len: int):
    """Piece ids and their (begin, end) character offsets in text. Pieces never cross whitespace (the
    tokenizer splits on it), so each word is encoded alone; inside a word, offsets follow the pieces'
    surface lengths (byte-fallback pieces count 1/n of a character each: a run of them covers one)."""
    ids, offs = [], []
    for m in re.finditer(r"\S+", text):
        w, a = m.group(), m.start()
        pid = sp.encode(w)
        surf = [sp.id_to_piece(i).replace("▁", "") for i in pid]
        size = [0 if sp.is_byte(i) else len(s) for i, s in zip(pid, surf)]
        nbyte = sum(sp.is_byte(i) for i in pid)
        if sum(size) + (nbyte > 0) * (len(w) - sum(size)) != len(w) or (nbyte and sum(size) >= len(w)):
            size = None                                    # normalization changed the length: spread evenly
        cur = a
        for k, i in enumerate(pid):
            if size is None:
                b, e = a + len(w) * k // len(pid), a + len(w) * (k + 1) // len(pid)
                e = max(e, b + 1)
            elif sp.is_byte(i):
                b, e = cur, min(cur + 1, a + len(w))
                if k + 1 == len(pid) or not sp.is_byte(pid[k + 1]):
                    cur = e
            else:
                b, e = cur, cur + size[k]; cur = e
            ids.append(i); offs.append((b, e))
    return ids[:max_len], offs[:max_len]


def tag_targets(offsets, spans) -> list[int]:
    """BIO label per piece from character spans [E1, (E2)]."""
    out = [0] * len(offsets)
    for k, (a, b) in enumerate(spans):
        first = True
        for i, (s, e) in enumerate(offsets):
            if (e > a and s < b and e > s) or (e == s and a <= s < b):   # zero-width "▁" pieces inside count
                out[i] = TAGS.index(("B-" if first else "I-") + f"E{k + 1}")
                first = False
    return out


def decode_spans(tags, offsets, text) -> list:
    """Character spans for E1 and E2 (None if absent): the longest run per entity."""
    spans = []
    for k in (1, 2):
        runs, cur = [], None
        for i, t in enumerate(tags):
            name = TAGS[t]
            if name == f"B-E{k}" or (name == f"I-E{k}" and cur is None):
                if cur:
                    runs.append(cur)
                cur = [i, i]
            elif name == f"I-E{k}" and cur is not None:
                cur[1] = i
            elif cur is not None:
                runs.append(cur); cur = None
        if cur:
            runs.append(cur)
        runs = [(a, b) for a, b in runs if offsets[b][1] > offsets[a][0]]
        if not runs:
            spans.append(None); continue
        a, b = max(runs, key=lambda r: offsets[r[1]][1] - offsets[r[0]][0])
        s, e = offsets[a][0], offsets[b][1]
        while s < e and text[s].isspace():
            s += 1
        spans.append([s, e])
    return spans


def predict(net: Planner, sp, questions: list[str], batch: int = 256) -> list[dict]:
    """Plans for questions (same dict form as the targets)."""
    out = []
    net.eval()
    with torch.inference_mode():
        for i in range(0, len(questions), batch):
            qs = questions[i: i + batch]
            enc = [pieces(sp, q, net.trunk.max_len) for q in qs]
            o = net(pad([e[0] for e in enc]))
            for j, q in enumerate(qs):
                tpl = net.templates[int(o["template"][j].argmax())]
                rel = lambda head: net.relations[1 + int(o[head][j][1:].argmax())]
                op = OPS[1 + int(o["op"][j][1:].argmax())]
                n = len(enc[j][0])
                spans = decode_spans(o["tag"][j][:n].argmax(-1).tolist(), enc[j][1], q)
                spans = spans[:1] if tpl in ("chain", "rev", "prev") else spans
                # plan confidence (step 7): chosen-class probabilities of the used heads x per span the mean
                # max tag probability over its pieces (an absent span → 0)
                pr = {h: o[h][j].softmax(-1) for h in ("template", "r1", "r2", "op")}
                # one-hop chain (MetaQA, 2026-11-04): r2 is none only when "none" is the r2 head's argmax
                one_hop = tpl in ("chain", "prev") and int(o["r2"][j].argmax()) == 0
                conf = float(pr["template"].max()) * float(pr["r1"][1:].max())
                conf *= (float(pr["r2"][0]) if one_hop else float(pr["r2"][1:].max())) if tpl != "compare" else 1.0
                conf *= float(pr["op"][1:].max()) if tpl not in ("chain", "rev", "prev") else 1.0
                r3 = None
                if net.has_r3:                    # rev: hop 3 is none when "none" is the r3 head's argmax
                    pr["r3"] = o["r3"][j].softmax(-1)
                    if tpl == "rev" and int(o["r3"][j].argmax()) != 0:
                        r3 = rel("r3")
                    if tpl == "rev":
                        conf *= float(pr["r3"].max())
                tp = o["tag"][j][:n].softmax(-1).max(-1).values
                for s_ in spans:
                    if s_ is None:
                        conf = 0.0; continue
                    k = [i for i, (a, b) in enumerate(enc[j][1]) if b > s_[0] and a < s_[1]]
                    conf *= float(tp[k].mean()) if k else 0.0
                out.append({"template": tpl, "r1": rel("r1"), "r2": None if tpl == "compare" or one_hop else rel("r2"),
                            "op": None if tpl in ("chain", "rev", "prev") else op, "spans": spans, "conf": conf,
                            **({"r3": r3} if net.has_r3 else {})})
    return out
