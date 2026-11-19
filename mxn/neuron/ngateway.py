"""Gateway and resultant of the neuron system (reports/design_neuron_e2e.md §3; supplements only).

Gateway (entry point / scheduler): the planner breaks each question into signals (template, entity spans, relation
per hop, operation, plan confidence); the language expert encodes the spans and the question; the first hop of each
chain is found by one batched RESOLVE broadcast (each host's KUs score themselves; merge as everywhere); the first
tasks go to the winners' hosts in one TASKS batch per host per wave (Spark-style stage). From there the KUs answer and
forward by themselves; the gateway does not orchestrate hops.

Resultant: collects findings per question; a chain is complete when a final finding arrives; when all chains of a
question are complete it applies the exact operations (mxn/expert/ops.py) or abstains. Confidence = plan confidence
× Π f over the question's findings; confident = confidence ≥ τ, τ from the frozen table (calibrated once on dev,
ruling 2026-11-01). It reads fire strengths and the KUs' answer words, never KU text.

Retry (cold copies, reports/design_neuron_cold.md): when a wave goes idle with questions incomplete and the driver
gave an on_incomplete hook (it waits until a confirmed dead host's KUs are restored and returns the new host list),
those questions are asked again from the start (attempt number in their tasks; findings of an older attempt are
ignored), at most MAX_RETRY times. Without the hook nothing changes.

Template → chains: chain: one chain, relations [r1, r2]; compare: two chains, [r1]; bridge: two chains, [r1, r2];
rev (MetaQA, reports/design_reverse_lookup.md): one chain, hop 0 r1, then the KU's host broadcasts REVERSE r2 and
(with r3) sends hop 2 to each KU in the set; the answer is the set (2-hop: titles; 3-hop: values), one or wrong.
prev (MetaQA person heads, reports/design_metaqa_sets_persons.md §3): no RESOLVE; one start task (k None, value =
the head words) goes to one host, which broadcasts REVERSE r1 for the head; with r2 every yes-KU answers r2 forward.
Set mode (sets = {theta_f}, §2): every task carries it; the answer is the set: chain = the last-hop KU's listed values
(hop 0; with r2, hop 1: one pointer at hop 0, demo corpus A parent -> X, reports/design_demo_rules.md §2);
rev 2-hop / prev 1-hop = the reverse finding's titles; rev 3-hop / prev 2-hop = the union of the forward findings'
listed values; the head is excluded (normalized); exact operations only (union, difference, deduplication).
Without sets nothing changes.
Variables: chain v1 = hop 0, v2 = hop 1; compare v1 / v2 = chain 0 / 1 hop 0; bridge v1 / v2 hop 0, v3 / v4 hop 1.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import time
import zlib

import numpy as np

from mxn.common.frames import frame, read_frame, serve
from mxn.common.textkeys import _DISAMBIG
from mxn.expert.ops import answer as template
from mxn.neuron.nhost import FINDINGS, REPLY, RESOLVEB, TASKS, ask, merge_once

MAX_RETRY = 2


class Resultant:
    def __init__(self, equiv: dict | None, form: str, tau: float | None, sets: dict | None = None):
        self.equiv, self.form, self.tau, self.sets = equiv, form, tau, sets
        self.q: dict = {}
        self.done: dict = {}
        self.last = time.time()
        self.on_done = None                             # early sink (E2E_SINK_EARLY): complete rows leave at close

    def register(self, qid: str, plan: dict, chains: list, names: dict) -> None:
        att = self.q[qid]["att"] + 1 if qid in self.q else 0
        self.done.pop(qid, None)
        self.q[qid] = {"plan": plan, "chains": chains, "names": names, "findings": [], "closed": set(), "att": att}
        if not chains:
            self.close(qid)

    def add(self, fs: list) -> None:
        self.last = time.time()
        for f in fs:
            r = self.q.get(f["qid"])
            if r is None or f["qid"] in self.done or f.get("att", 0) != r["att"]:
                continue
            r["findings"].append(f)
            if f.get("lost"):                         # a RESOLVE broadcast missed a host: the chain cannot be trusted
                self.close(f["qid"], incomplete=True); continue
            tpl = r["plan"]["template"]
            if tpl in ("rev", "prev"):                # reverse lookup: the reverse finding names how many forward
                fh = 2 if tpl == "rev" else 1         # findings (hop fh) follow (expect); closed when all have arrived
                if f.get("kind") == "reverse":
                    r["expect"] = f["expect"]
                elif f["hop"] == fh:
                    r["got"] = r.get("got", 0) + 1
                if (tpl == "rev" and f["final"] and f["hop"] == 0) or (f.get("kind") == "reverse" and f["final"]) or \
                        ("expect" in r and r.get("got", 0) >= r["expect"] > 0):
                    r["closed"].add(f["chain"])
            elif f["final"]:
                r["closed"].add(f["chain"])
            if len(r["closed"]) == len(r["chains"]):
                self.close(f["qid"])

    def close(self, qid: str, incomplete: bool = False) -> None:
        self._close(qid, incomplete)
        if self.on_done is not None and not incomplete:  # memory: stream the row now, not at wave end (answers same)
            self.done[qid] = self.on_done(self.done[qid])
            self.q[qid] = {"att": self.q[qid]["att"]}   # late findings: dropped by add()

    def _close(self, qid: str, incomplete: bool = False) -> None:
        r = self.q[qid]; plan = r["plan"]
        fs = sorted(r["findings"], key=lambda f: (f["chain"], f["hop"], -1 if f["k"] is None else f["k"]))
        val = {(f["chain"], f["hop"]): f["value"] for f in fs}
        L = lambda c, h: [val[(c, h)]] if val.get((c, h)) else []
        if self.sets is not None:                     # set mode (design_metaqa_sets_persons.md §2)
            from mxn.expert.ops import normalize
            tpl, rv = plan["template"], [f for f in fs if f.get("kind") == "reverse"]
            if tpl == "chain":                         # the last hop's values (hop 0; hop 1 with r2: demo corpus A)
                lh = 1 if plan.get("r2") else 0
                vals = [v for f in fs if f["hop"] == lh for v in f.get("values", [f["value"]] if f["value"] else [])]
                head = None
            elif (tpl == "rev" and not plan.get("r3")) or (tpl == "prev" and not plan.get("r2")):
                vals, head = [t for f in rv for t in f["titles"]], None
            else:
                fh = 2 if tpl == "rev" else 1
                vals = [v for f in fs if f["hop"] == fh and f.get("kind") != "reverse"
                        for v in f.get("values", [f["value"]] if f["value"] else [])]
                head = r["names"]["span"][0] if tpl == "prev" else r["names"]["title"][0] if r["names"]["title"] else None
            uniq = sorted({normalize(v): v for v in vals if v and (head is None or normalize(v) != normalize(head))}
                          .values())
            ans = None if incomplete or not uniq else " | ".join(uniq)
            self.done[qid] = {"id": qid, "plan": plan, "findings": fs, "answer": ans,
                              "conf": plan["conf"] if ans else 0.0, "confident": False, "complete": not incomplete,
                              "answer_set": [] if incomplete else uniq}
            return
        if plan["template"] == "rev":                 # set answers (design_reverse_lookup.md): exactly one or wrong
            from mxn.expert.ops import normalize
            rv = [f for f in fs if f.get("kind") == "reverse"]
            vals = [t for f in rv for t in f["titles"]] if not plan.get("r3") else \
                [f["value"] for f in fs if f["hop"] == 2 and f["value"]]
            uniq = sorted({normalize(v): v for v in vals}.values())
            ans = None if incomplete or not uniq else uniq[0] if len(uniq) == 1 else " | ".join(uniq)
            c = plan["conf"]
            self.done[qid] = {"id": qid, "plan": plan, "findings": fs, "answer": ans, "conf": c if ans else 0.0,
                              "confident": False, "complete": not incomplete, "answer_set": uniq}
            return
        if plan["template"] == "chain":
            values = {"v1": L(0, 0), "v2": L(0, 1)} if plan["r2"] else {"v1": [], "v2": L(0, 0)}   # one-hop chain
        elif plan["template"] == "compare":
            values = {"v1": L(0, 0), "v2": L(1, 0)}
        else:
            values = {"v1": L(0, 0), "v2": L(1, 0), "v3": L(0, 1), "v4": L(1, 1)}
        ans = template(plan, values, r["names"][self.form], self.equiv) if r["chains"] and not incomplete else None
        c = plan["conf"]
        for f in fs:
            c *= f["f"]
        c = c if ans is not None else 0.0
        self.done[qid] = {"id": qid, "plan": plan, "findings": fs, "answer": ans, "conf": c,
                          "confident": ans is not None and self.tau is not None and c >= self.tau,
                          "complete": not incomplete}


class Gateway:
    def __init__(self, planner, sp, enc, hosts: dict, res_addr: str, resultant: Resultant, wave: int = 250,
                 fmin: float = 0.5, rev: tuple | None = None, plans: dict | None = None, sets: dict | None = None):
        self.planner, self.sp, self.enc, self.hosts, self.fmin = planner, sp, enc, hosts, fmin
        self.sets = sets                    # set mode {theta_f} (design_metaqa_sets_persons.md), None = off
        self.rev = rev                      # (rule, theta) of the KUs' reverse-lookup yes rule (frozen setting)
        self.gold_plans = plans             # dry runs only: qid → plan instead of the planner (mechanics check)
        self.res_addr, self.R, self.wave = res_addr, resultant, wave

    async def on_findings(self, reader, writer):
        try:
            kind, payload = await read_frame(reader)
            if kind == FINDINGS:
                self.R.add(json.loads(payload)["findings"])
            writer.write(frame(REPLY, b'{"ok": true}')); await writer.drain()
        finally:
            writer.close()

    async def run(self, questions: list, say=print, idle_timeout: float = 120.0, on_incomplete=None,
                  sink=None) -> list:
        from mxn.expert.planner import predict
        srv = await serve(self.res_addr, self.on_findings)
        self.retried = []
        if sink is not None and os.environ.get("E2E_SINK_EARLY", "1") != "0":   # default (owner, 2026-11-11, 9991c)
            self.R.on_done = sink
        self.trim = {"calls": 0, "freed_kb": 0, "cpu_us": 0, "wall_us": 0}
        try:
            for w in range(0, len(questions), self.wave):
                batch = questions[w: w + self.wave]
                plans = [self.gold_plans[i] for i, _ in batch] if self.gold_plans is not None else \
                    predict(self.planner, self.sp, [q for _, q in batch])
                await self.dispatch(batch, plans)
                for attempt in range(MAX_RETRY + 1):
                    self.R.last = time.time()
                    while any(i not in self.R.done for i, _ in batch):
                        await asyncio.sleep(0.05)
                        if time.time() - self.R.last > idle_timeout:
                            for i, _ in batch:
                                if i not in self.R.done:
                                    self.R.close(i, incomplete=True)
                    again = [j for j, (i, _) in enumerate(batch) if not self.R.done[i]["complete"]]
                    if not again or on_incomplete is None or attempt == MAX_RETRY:
                        break
                    self.hosts = await on_incomplete()          # a dead host's KUs are back on the survivors
                    self.retried += [batch[j][0] for j in again]
                    say(f"re-asking {len(again)} incomplete questions (attempt {attempt + 1})")
                    await self.dispatch([batch[j] for j in again], [plans[j] for j in again])
                if os.environ.get("E2E_MEMPROBE"):    # diagnostics only: driver memory at wave end, before the sink
                    rss = next((int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmRSS")), 0)
                    held = sum(len(q.get("findings", [])) for q in self.R.q.values())
                    say(f"memprobe driver: rss {rss >> 10} MB findings held {held} rows done {len(self.R.done)}")
                if sink is not None:                 # stream finished rows out; keep what sink returns (memory)
                    for i, _ in batch:
                        if "key" in self.R.done[i]:    # already streamed at close (E2E_SINK_EARLY)
                            continue
                        self.R.done[i] = sink(self.R.done[i])
                        self.R.q[i] = {"att": self.R.q[i]["att"]}   # late findings: dropped by add()
                if os.environ.get("E2E_TRIM", "1") != "0":   # driver: hand freed heap back at wave end (default since
                                                             # 9991d PASS, owner rule 2026-11-11; E2E_TRIM=0 off)
                    rss = lambda: next((int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmRSS")), 0)
                    c0, w0, b = time.thread_time_ns(), time.perf_counter_ns(), rss()
                    ctypes.CDLL("libc.so.6").malloc_trim(0); a = rss()
                    self.trim["calls"] += 1; self.trim["freed_kb"] += max(b - a, 0)
                    self.trim["cpu_us"] += (time.thread_time_ns() - c0) // 1000
                    self.trim["wall_us"] += (time.perf_counter_ns() - w0) // 1000
                    say(f"trim driver: rss {b >> 10} -> {a >> 10} MB")
                say(f"questions {min(w + self.wave, len(questions))}/{len(questions)}")
        finally:
            srv.close()
        return [self.R.done[i] for i, _ in questions]

    async def dispatch(self, batch: list, plans: list) -> None:
        items, owners, tasks = [], [], {}
        st = {"set": self.sets} if self.sets is not None else {}
        hn = sorted(self.hosts)
        for (qid, q), p in zip(batch, plans):
            if p["template"] == "prev":               # person head: no RESOLVE; one host starts with REVERSE
                sp_ = p["spans"][0] if p["spans"] else None
                if sp_ is None:
                    self.R.register(qid, p, [], {"title": [], "span": []}); continue
                self.R.register(qid, {**p, "entities": []}, [0], {"title": [], "span": [q[sp_[0]:sp_[1]]]})
                att = self.R.q[qid]["att"]
                h = hn[zlib.crc32(qid.encode()) % len(hn)]
                tasks.setdefault(h, []).append({"qid": qid, "chain": 0, "hop": 0, "k": None, "value": q[sp_[0]:sp_[1]],
                                                "rels": [p["r1"]] + ([p["r2"]] if p.get("r2") else []),
                                                "prev": {"r2": p.get("r2"), "rule": self.rev[0], "theta": self.rev[1]},
                                                "res": self.res_addr, "how": "start", "fmin": self.fmin,
                                                **({"att": att} if att else {}), **st})
                continue
            need = 1 if p["template"] in ("chain", "rev") else 2
            spans = p["spans"][:need]
            if len(spans) < need or any(s is None for s in spans):
                continue
            qq = self.enc.encode([q], "query")[0].tolist()
            for e, (a, b) in enumerate(spans):
                items.append({"span": q[a:b], "qv_span": self.enc.encode([q[a:b]], "query")[0].tolist(),
                              "qv_question": qq})
                owners.append((qid, e))
        replies = await asyncio.gather(*[ask(a, RESOLVEB, {"items": items}) for a in self.hosts.values()]) \
            if items else []
        ok = [r for r in replies if r is not None]
        partial = len(ok) < len(self.hosts)                 # a host did not answer: winners may be wrong → re-ask
        ents = {}
        for i, (qid, e) in enumerate(owners):
            ents[(qid, e)] = merge_once([{"host": r["host"], **r["props"][i]} for r in ok]) if ok else None
        for (qid, q), p in zip(batch, plans):
            if p["template"] == "prev":
                continue
            need = 1 if p["template"] in ("chain", "rev") else 2
            es = [ents.get((qid, e)) for e in range(need)]
            rels = [p["r1"], p["r2"]] if p["template"] not in ("compare", "rev") else [p["r1"]]
            rels = [r for r in rels if r]           # one-hop chain (r2 none; MetaQA): one relation
            rv = {"rev": {"r2": p["r2"], "r3": p.get("r3"), "rule": self.rev[0], "theta": self.rev[1]}} \
                if p["template"] == "rev" else {}
            if partial and any(o[0] == qid for o in owners):
                self.R.register(qid, p, [None], {"title": [], "span": []}); self.R.close(qid, incomplete=True)
                continue
            if any(x is None for x in es):
                self.R.register(qid, p, [], {"title": [], "span": []}); continue
            names = {"title": [_DISAMBIG.sub("", x["title"]) for x in es],
                     "span": [q[a:b] for a, b in p["spans"][:need]]}
            self.R.register(qid, {**p, "entities": [[x["title"], x["rule"]] for x in es]}, list(range(need)), names)
            att = self.R.q[qid]["att"]
            qq = items[owners.index((qid, 0))]["qv_question"]
            for e, x in enumerate(es):
                tasks.setdefault(x["host"], []).append({"qid": qid, "chain": e, "hop": 0, "rels": rels, "k": x["k"],
                                                        "qq": qq, "res": self.res_addr, "how": "first",
                                                        "fmin": self.fmin, **({"att": att} if att else {}), **rv,
                                                        **st})
        await asyncio.gather(*[ask(self.hosts[h], TASKS, {"tasks": ts}) for h, ts in tasks.items()])
