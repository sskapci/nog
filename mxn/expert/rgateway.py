"""The expert's gateway with KU-owned resolution (decisions 2026-10-28, central resolver fix).

Architecture check (roadmap "share-nothing knowledge units"; decision 2026-10-24b §6): the gateway holds the planner,
tokenizer, encoder (language expert: it encodes the span and the question) and the operations' country table, plus
the list of host addresses. It holds no titles, title vectors, key tables or placement map: an entity is resolved by
broadcasting RESOLVE to every host (each host's own title index proposes, each KU scores itself) and merging the
returned scores (mxn/expert/hostindex.merge); a KU's title and host are known only for the request that resolved it.
A hop is first asked of the KU that produced the value (VIA, its own outgoing connections), then broadcast.

Degraded mode (decision 2026-10-28 Part A2): if any host does not answer a RESOLVE broadcast (refused or timeout), the
gateway accepts only a unique exact-title match among the replies it got and abstains on anything resolved by a weaker
rule (exact ties included: the missing host may hold another exact-title KU that would have won). Architecture check:
the gateway still holds no KU state; it reacts only to a reply missing from that one request.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from mxn.expert.fhost import DROP, FIND, RESOLVE, VIA, WHO
from mxn.expert.hostindex import merge
from mxn.expert.ops import answer as template
from mxn.expert.pipeline import Expert
from mxn.common.textkeys import _DISAMBIG
from mxn.common.frames import frame, open_conn, read_frame


class HostsClient:
    """One connection per host; requests to one host answered in order; RESOLVE goes to all hosts at once.
    No placement: every message names its host."""

    def __init__(self, hosts: dict[str, str], timeout: float = 10.0):
        self.hosts, self.timeout = dict(hosts), timeout
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.conns, self.locks, self.rid, self.down = {}, {}, 0, set()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    async def _request(self, host: str, kind: int, msg: dict) -> dict | None:
        lock = self.locks.setdefault(host, asyncio.Lock())
        async with lock:
            try:
                if host not in self.conns:
                    self.conns[host] = await asyncio.wait_for(open_conn(self.hosts[host]), self.timeout)
                reader, writer = self.conns[host]
                writer.write(frame(kind, json.dumps(msg).encode())); await writer.drain()
                _, payload = await asyncio.wait_for(read_frame(reader), self.timeout)
                return json.loads(payload)
            except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
                self.conns.pop(host, None); self.down.add(host)
                return None

    def request(self, host: str, kind: int, **msg) -> dict | None:
        self.rid += 1
        return self._run(self._request(host, kind, {"rid": self.rid, **msg}))

    def broadcast(self, kind: int, **msg) -> list[dict]:
        return self.broadcast_all(kind, **msg)[0]

    def broadcast_all(self, kind: int, **msg) -> tuple[list[dict], list[str]]:
        """→ (replies, hosts that did not answer this request)."""
        self.rid += 1
        m = {"rid": self.rid, **msg}
        hs = sorted(self.hosts)

        async def all_hosts():
            return await asyncio.gather(*(self._request(h, kind, m) for h in hs))
        rs = self._run(all_hosts())
        return [r for r in rs if r is not None], [h for h, r in zip(hs, rs) if r is None]

    def manage(self, kind: int, **msg) -> dict | None:
        """EDIT / EDITV / DROP / DETACH / ATTACH / MOVE without a placement map: a read-only WHO broadcast finds the
        holders first; with exactly one, the operation goes to that host only. → the owner's reply, or None if no host
        holds the KU. Several holders: nothing is applied, RuntimeError (see repair_duplicate)."""
        holders = [r["host"] for r in self.broadcast(WHO, k=msg["k"]) if r.get("applied")]
        if len(holders) > 1:
            raise RuntimeError(f"KU {msg['k']} held by several hosts: {sorted(holders)}; nothing applied")
        if not holders:
            return None
        r = self.request(holders[0], kind, **msg)
        return r if r and r.get("applied") else None

    def repair_duplicate(self, k) -> dict:
        """After manage() reported several holders of KU k: if every copy is identical and exactly one holder imported
        it from another holder (an interrupted MOVE: the destination confirmed, the source died before deleting), the
        move is completed by dropping the source's copy. Anything else is left alone and reported (no automatic
        choice between differing copies)."""
        holders = [r for r in self.broadcast(WHO, k=k) if r.get("applied")]
        rep = {"k": k, "holders": [{x: r.get(x) for x in ("host", "sha1", "detached", "imported_from")} for r in holders]}
        names = {r["host"] for r in holders}
        dest = [r for r in holders if r.get("imported_from") in names]
        if len(holders) < 2:
            return {**rep, "resolved": True, "action": "none (one holder)"}
        if len({r["sha1"] for r in holders}) == 1 and len(holders) == 2 and len(dest) == 1:
            src = dest[0]["imported_from"]
            r = self.request(src, DROP, k=k)
            return {**rep, "resolved": bool(r and r.get("applied")), "action": f"dropped the source copy on {src}",
                    "kept": dest[0]["host"]}
        return {**rep, "resolved": False, "action": "none: copies differ or the move origin is unclear"}

    def close(self) -> None:
        async def shut():
            for _, w in self.conns.values():
                w.close()
        self._run(shut())
        self.loop.call_soon_threadsafe(self.loop.stop)


class ResolveGateway(Expert):
    def __init__(self, planner, sp, enc, equiv: dict | None, entity_form: str = "title", client: HostsClient = None,
                 degraded: bool = True):
        self.planner, self.sp, self.enc, self.equiv, self.entity_form, self.client = planner, sp, enc, equiv, entity_form, client
        self.degraded = degraded
        self.stats = {"resolve_broadcasts": 0, "via_asked": 0, "via_answered": 0, "resolve_missing_reply": 0,
                      "degraded_abstained": {}}

    def resolve(self, span: str, question: str) -> dict | None:
        qs = self.enc.encode([span], "query")[0]
        qq = self.enc.encode([question], "query")[0]
        self.stats["resolve_broadcasts"] += 1
        replies, missing = self.client.broadcast_all(RESOLVE, span=span, qv_span=qs.tolist(), qv_question=qq.tolist())
        e = merge(replies)
        if missing:
            self.stats["resolve_missing_reply"] += 1
            if self.degraded and e is not None and e["rule"] != "exact":     # degraded mode: unique exact match only
                d = self.stats["degraded_abstained"]; d[e["rule"]] = d.get(e["rule"], 0) + 1
                return None
        return e

    def find(self, ent: dict, relation: str) -> dict:
        r = self.client.request(ent["host"], FIND, k=ent["k"], relation=relation)
        v, p = ("", 0.0) if r is None else (r["value"], r["p_ext"])
        return {"ku": ent["k"], "title": ent["title"], "relation": relation, "value": v, "p_ext": p, "p_ku": 1.0,
                "host": ent["host"]}

    def hop_find(self, src: dict, r1: str, r2: str, question: str) -> dict | None:
        """Hop from the source KU's r1 value to the next KU and read r2 there. → the FIND record with resolved_by,
        or None if the hop finds no KU."""
        e2 = self.hop(src, r1, src["value"], question)
        if e2 is None:
            return None
        f2 = self.find(e2, r2); f2["resolved_by"] = e2["rule"]
        return f2

    def hop(self, src: dict, relation: str, value: str, question: str) -> dict | None:
        """Ask the source KU first (its own outgoing connections), then broadcast."""
        if not value:
            return None
        self.stats["via_asked"] += 1
        r = self.client.request(src["host"], VIA, k=src["ku"], relation=relation, value=value)
        if r is not None and r.get("target"):
            self.stats["via_answered"] += 1
            k, title, host = r["target"]
            return {"k": k, "title": title, "host": host, "rule": "via_source"}
        return self.resolve(value, question)

    def answer(self, q: str, plan: dict) -> dict:
        rec = {"plan": plan, "finds": [], "entities": [], "answer": None, "conf": 0.0}
        spans = plan["spans"]
        need = 1 if plan["template"] == "chain" else 2
        if len(spans) < need or any(s is None for s in spans[:need]):
            return rec
        ents = [self.resolve(q[a:b], q) for a, b in spans[:need]]
        if any(e is None for e in ents):
            return rec
        rec["entities"] = [[e["title"], e["rule"]] for e in ents]       # as the pipeline: c2 looks rules up by title
        f1 = [self.find(e, plan["r1"]) for e in ents]
        rec["finds"] += f1
        if plan["template"] == "chain":
            f2 = self.hop_find(f1[0], plan["r1"], plan["r2"], q)
            if f2 is None:
                return rec
            rec["finds"].append(f2)
            values = {"v1": [f1[0]["value"]], "v2": [f2["value"]] if f2["value"] else []}
        else:
            values = {"v1": [f1[0]["value"]] if f1[0]["value"] else [], "v2": [f1[1]["value"]] if f1[1]["value"] else []}
            if plan["template"] == "bridge":
                for side, var in ((0, "v3"), (1, "v4")):
                    f2 = self.hop_find(f1[side], plan["r1"], plan["r2"], q)
                    if f2 is None:
                        values[var] = []; continue
                    rec["finds"].append(f2)
                    values[var] = [f2["value"]] if f2["value"] else []
        forms = {"title": [_DISAMBIG.sub("", e["title"]) for e in ents], "span": [q[a:b] for a, b in spans[:need]]}
        rec["answer_forms"] = {f: template(plan, values, names, self.equiv) for f, names in forms.items()}
        rec["answer"] = rec["answer_forms"][self.entity_form]
        c = plan["conf"]
        for f in rec["finds"]:
            c *= f["p_ext"] * f["p_ku"]
        rec["conf"] = c if rec["answer"] is not None else 0.0
        return rec


def load_resolve_gateway(ex, entity_form: str = "title", planner_dir: str = "planner2", client=None,
                         degraded: bool = True) -> ResolveGateway:
    """Planner, tokenizer, encoder and the country table; nothing from the KUs and no titles."""
    import sentencepiece as spm
    import torch

    from mxn.expert.encoder import Encoder
    from mxn.expert.planner import Planner

    ex = Path(ex)
    sp = spm.SentencePieceProcessor(model_file=str(ex / "tok16k.model"))
    enc = Encoder(str(ex / "enc"), str(ex / "tok16k.model"))
    ck = torch.load(ex / planner_dir / "planner.pt", map_location="cpu", weights_only=False)
    planner = Planner(ck["relations"], ck["trunk"]); planner.load_state_dict(ck["state"]); planner.eval()
    equiv = json.load(open(ex / "country_table.json"))["table"]
    return ResolveGateway(planner, sp, enc, equiv, entity_form, client, degraded)
