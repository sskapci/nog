"""A host of KU neurons (reports/design_neuron_e2e.md; first end-to-end run, user 2026-11-01).

The host holds only its own KUs (rendezvous placement over the static host list, R = 1), trains them there, and
runs them. KUs answer, point and forward by themselves; the host is their runtime: it batches their messages and
runs them on its worker threads (Spark-style: tasks arrive in batches per host, independent tasks run in parallel).
The host's language expert (the shared frozen encoder, a supplement: words → signals) encodes each KU's training
inputs and the sub-question signal a KU receives; it reads no KU knowledge.

Frames (mxn/common/frames.py), JSON payloads; every request gets one reply frame (OK unless stated):
  TRAIN    {settings}            → {n, stats}: every own KU trains alone on its own article
  WIRE     {}                    → {n_slots, pointers}: each own KU broadcasts RESOLVE for its name slots (exact /
                                   no-disambiguator title match, unique over all hosts) and keeps the target as its
                                   pointer
  RESOLVEB {items: [{span, qv_span?, qv_question?}]} → {host, props: [...]}: this host's title index proposes, each
                                   KU scores itself (mxn/expert/hostindex.py)
  DROPTEXT {}                    → {dropped}: every own KU's raw text deleted (and the host's copy of the shard)
  AUDIT    {}                    → {host, sentences_in_memory, name_sentences_in_memory}: live strings equal to a KU
                                   sentence (sha1); a sentence that is only a name held as KU state counted apart
  TASKS    {tasks: [...]}        → {accepted}: queued; KUs answer, report findings to the resultant and forward the
                                   next hop themselves (pointer → the target's host; no pointer → RESOLVE broadcast)
  STATS    {}                    → counters
  STOP     {}
  MEMBERS  {phase, hosts}        → rebalancing (reports/design_neuron_rebalance.md): the new host list in four
                                   phases, each sent to every host and finished on all before the next:
                                   copy: every own KU whose rendezvous owner under the new list is another host is
                                   sent there whole (MOVE, its package: trained parameters, slots, pointers; no text);
                                   add: KUs received join this host (tasks, RESOLVE); broadcasts reach old and new hosts;
                                   drop: routing switches to the new list; KUs that left are forgotten here;
                                   done: broadcasts reach the new hosts only.
                                   A task for a KU this host does not hold goes on to its owner under this host's list.
  MOVE     {from, blob}          → {stored}: KU packages (torch.save, base64) kept until the add phase
  COLD     {phase, gen, hosts?}  → cold copies (reports/design_neuron_cold.md), R = 1:
                                   write: every own KU's package (no text) is sent to its successor, the second host
                                   in its rendezvous order over the current list (COLDPUT), which stores it on its own
                                   disk, not loaded; commit: copies of older generations deleted; reply: copies held
                                   and a hash of their KU ids; restore {hosts: survivors}: after a confirmed death,
                                   each survivor loads the copies whose rendezvous owner under the survivors' list is
                                   itself (then MEMBERS add / drop / done, as rebalancing); audit: KU sentences in
                                   this host's cold files (must be 0)
  COLDPUT  {from, gen, i, blob}  → {stored}
  FORGET   {ids}                 → {deleted, cold_deleted, pointers_cleared} (reports/design_forget_edit.md §1; broadcast):
                                   this host deletes the KUs it holds among ids (parameters, slots, package, caches),
                                   removes them from its own title index, slot-value cache and cold store, and clears
                                   every own KU's pointer to them. Known-bad (dry runs): NHOST_FORGET_KEEP_COLD=1,
                                   NHOST_FORGET_NO_SCRUB=1.
  FAUDIT   {mode, ids, titles?, values?, sha?, phash?} → hash: {phash: {id: sha1 of parameters}} for own KUs among ids;
                                   audit: counts of what of the forgotten KUs is still on this host (held, index,
                                   cache, unique slot labels, pointers, sentences, parameter tensors, cold store).
  EDIT     {k, sentences}        → {held, retrained, seconds} (§2): if this host holds KU k, k retrains alone on its own
                                   edited article (TRAIN settings), rebuilds its slots, re-resolves its own name slots
                                   by its own RESOLVE broadcast, refreshes its caches and drops the text. Known-bad:
                                   NHOST_EDIT_NO_TRAIN=1.
  LOCAL    {phase, quota?}       → locality placement (reports/design_neuron_locality.md): plan: each own KU chooses
                                   its anchor (nku.anchor: its strongest pointer's target); reply: KUs held and how
                                   many would go to each host (rendezvous of the anchor); copy {quota: {host: n}}: up
                                   to n of them (by KU id) take the anchor as their key and are sent (MOVE); the rest
                                   keep their own id; then MEMBERS add / drop / done; keys: every own KU learns its
                                   pointer targets' keys (KEYS broadcast) and forwards by them
  KEYS     {ids}                 → {keys: {id: key}} for the KUs held here
  REVERSE  {items: [{rel, value, rule, theta, exclude}]} → {host, yes: [[[k, title, p], ...] per item], evaluated,
                                   forward, cpu_s}: reverse lookup (reports/design_reverse_lookup.md): every own KU
                                   checks its own slot table for value (rel's kind) and, if it holds it, decides with
                                   its own forward pass (nku.lists; its p per relation cached here); exclude = the head
                                   KU's id. No value index: each KU is asked in turn.
Arms: neuron (the system); first (control: first slot of the fitting kind); all (set control: every slot of the
fitting kind, reports/design_metaqa_sets_persons.md); oracle (KUs trained on their gold triples: the ceiling,
measurement only); leak (dry runs only, known-bad for K: reads raw text when present).
Set mode (task field set: {theta_f}; design_metaqa_sets_persons.md §2): a KU answering a last forward hop, or hop 0 of
a rev task, lists its values (nku.answer_set); the finding carries values and fs; REVERSE then carries all hop-0
values (a KU says yes if it lists any). Person heads (task field prev: {r2, rule, theta}, k None, value = the head
words; §3): the receiving host broadcasts REVERSE (rels[0], value) first; the reverse finding is hop 0 and, with r2,
every yes-KU answers r2 forward (hop 1).
Task: {qid, chain, hop, rels, k, qq, res, how, fmin[, rev: {r2, r3, rule, theta}]}. A rev task's hop-0 KU answers
r1; its host then broadcasts REVERSE (r2, the answer) and reports one finding {kind: "reverse", set, titles, fs,
expect} for hop 1; with r3 it sends a hop-2 task (rels [r1, r2, r3]) to every KU in the set. Finding (to the resultant, FINDINGS {findings: [...]}): {qid, chain,
hop, k, title, value, kind, target, f, how, final, host}.

Security: frames are not authenticated or encrypted. Use only on one machine or inside a trusted LAN, with the
hosts' ports (7400-7409, resultant 7499) open between the machines and closed to everything else.

    python -m mxn.neuron.nhost --name h0 --hostlist HOSTS.json --shard SHARD.pkl --expert artifacts/expert
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import ctypes
import gc
import hashlib
import io
import json
import os
import pickle
import threading
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from mxn.common.frames import frame, open_conn, read_frame, serve
from mxn.common.textkeys import _DISAMBIG
from mxn.expert.hostindex import HostTitleIndex, KUTitle, merge, title_vector
from mxn.expert.extractor import dates_rule
from mxn.neuron.nku import Settings, build, package, unpackage

TRAIN, WIRE, RESOLVEB, DROPTEXT, AUDIT, TASKS, STATS, STOP, FINDINGS, REPLY = range(40, 50)
MEMBERS, MOVE = 50, 51
COLD, COLDPUT = 52, 53
LOCAL, KEYS = 54, 55
REVERSE = 56
FORGET, FAUDIT, EDIT = 57, 58, 59
MOVE_CHUNK = 50                     # KU packages per MOVE frame
MAX_FORWARD = 3                     # a task for a KU not held here goes on at most this often
CHUNK = 250                         # KUs per checkpoint file
_LIBC = ctypes.CDLL("libc.so.6")


async def ask(addr: str, kind: int, msg: dict, timeout: float = 600.0) -> dict | None:
    try:
        reader, writer = await asyncio.wait_for(open_conn(addr), timeout)
        writer.write(frame(kind, json.dumps(msg).encode())); await writer.drain()
        _, payload = await asyncio.wait_for(read_frame(reader), timeout)
        writer.close()
        return json.loads(payload)
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError):
        return None


def merge_once(replies: list) -> dict | None:
    """merge (mxn/expert/hostindex.py) with each KU counted once: while a KU moves (rebalancing) two hosts may hold it
    for a moment; its proposals are kept at the first host in sorted order. With no KU on two hosts = merge."""
    first: dict = {}
    for r in sorted(replies, key=lambda r: r["host"]):
        for tier in ("exact", "nodis"):
            for x in r[tier]:
                first.setdefault(x[0], r["host"])
        if r.get("fallback"):
            first.setdefault(r["fallback"][0], r["host"])
    return merge([{**r, "exact": [x for x in r["exact"] if first[x[0]] == r["host"]],
                   "nodis": [x for x in r["nodis"] if first[x[0]] == r["host"]],
                   "fallback": r["fallback"] if r.get("fallback") and first[r["fallback"][0]] == r["host"] else None}
                  for r in replies])


def _strings(o, seen: set):
    """Strings held directly by o; descends into containers the garbage collector does not track (dicts, tuples
    holding only atomic values, e.g. a small object's __dict__), which gc.get_objects() never returns."""
    for r in gc.get_referents(o):
        if isinstance(r, str):
            yield r
        elif isinstance(r, (dict, list, tuple, set, frozenset)) and not gc.is_tracked(r) and id(r) not in seen:
            seen.add(id(r)); yield from _strings(r, seen)


def _all_strings(o):
    """Every string inside a loaded package list (cold-copy audit)."""
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from _all_strings(k); yield from _all_strings(v)
    elif isinstance(o, (list, tuple, set)):
        for v in o:
            yield from _all_strings(v)


def name_kinds(titles: list, labels: list, sha: set) -> dict:
    """sha1 → kind for KU sentences whose whole text is a name: a KU title, a title with its disambiguator stripped
    (the RESOLVE key form), or a slot label (first kind wins in that order)."""
    out: dict = {}
    for kind, xs in (("title", titles), ("title_stripped", [_DISAMBIG.sub("", t) for t in titles]), ("slot_label", labels)):
        for x in xs:
            h = hashlib.sha1(x.encode()).hexdigest()
            if h in sha:
                out.setdefault(h, kind)
    return out


def text_in_memory(sha: set, names: dict | None = None) -> dict:
    """Live strings equal to a KU sentence (sha1). A sentence whose whole text is a name held as KU state (names:
    sha1 → kind: "title", "title_stripped" (disambiguator removed, the RESOLVE key form) or "slot_label") is counted
    apart; text = every other hit (must be 0 after the drop). The sha1 of every hit is recorded with its kind."""
    names = names or {}
    gc.collect(); hits: dict = {}; seen: set = set()
    for o in gc.get_objects():
        for x in _strings(o, seen):
            if len(x) > 20:
                h = hashlib.sha1(x.encode()).hexdigest()
                if h in sha:
                    hits[h] = names.get(h, "text")
    kinds = Counter(hits.values())
    return {"text": kinds.get("text", 0), "names": sum(v for k, v in kinds.items() if k != "text"),
            "by_kind": dict(kinds), "hits": hits}


class NeuronHost:
    def __init__(self, name: str, hosts: dict, shard: dict, expert_dir: str, workers: int, arm: str,
                 ckpt: str | None = None, sha: str | None = None, cold: str | None = None):
        from mxn.expert.encoder import Encoder
        torch.set_num_threads(1)
        self.name, self.hosts, self.arm = name, hosts, arm
        self.R = 1                      # replicas per KU (MEMBERS r); the first live one in rendezvous order routes
        self.members = sorted(hosts)    # routing list (rendezvous); self.hosts = broadcast reach (both change only
        self.incoming: dict = {}        # by MEMBERS); KUs received by MOVE, not yet active
        self.leaving: set = set()       # own KUs sent to their new owner, forgotten at the drop phase
        self.ckpt = Path(ckpt) if ckpt else None      # trained KU packages (no raw text), for resume
        self.cold = Path(cold) if cold else None      # cold copies of other hosts' KUs (this host's own disk)
        self.cold_ids: dict = defaultdict(set)        # generation → KU ids held as cold copies
        self.moved: dict = {}           # KU id → host it was sent to by a locality move (this host's own forwarding)
        self.plan: dict = {}            # locality: own KU id → (anchor, destination host)
        self.shard = shard          # {k: (title, sentences[, gold])}: training input only (gold: oracle arm only)
        self.sha = {hashlib.sha1(s.encode()).hexdigest() for v in shard.values() for s in v[1]}
        if sha:                         # rebalancing: KUs may arrive from other hosts; audit against every sentence
            self.sha |= set(json.load(open(sha)))
        self.names: dict = {}           # sha1 → kind for sentences that are only a name (see name_kinds), set by TRAIN
        self.equiv = json.load(open(f"{expert_dir}/country_table.json"))["table"] if arm == "oracle" else None
        self.enc = Encoder(f"{expert_dir}/enc", f"{expert_dir}/tok16k.model", threads=1)
        self.pool = ThreadPoolExecutor(workers)
        self.kus: dict = {}
        self.index = None
        self.qcache: dict = {}
        self.pcache: dict = {}          # (KU id, relation) → that KU's own probs (reverse lookup; parameters fixed)
        self._cache: dict = {}          # slot-value cache: normalized slot label → own KU ids (design_slot_cache.md)
        self._cache_lab: dict = {}      # own KU id → its labels in the cache (to remove it when it leaves or changes)
        self._cache_ids: set = set()
        self._cache_lock = threading.Lock()
        self._cache_bad = False         # known-bad dry run (NHOST_CACHE_DROP_ONE) applied
        self.c = Counter()

    # ---- building (each KU alone, on this host)
    def train(self, st: Settings) -> dict:
        self.st = st                                          # EDIT retrains with the same settings
        t0 = time.process_time()
        keys, tag, resumed = sorted(self.shard), {"settings": vars(st), "arm": self.arm}, 0
        for c in range(0, len(keys), CHUNK):
            f = self.ckpt / f"chunk_{c // CHUNK:04d}.pt" if self.ckpt else None
            if f is not None and f.exists():                 # resume: same settings, same KUs, text from the shard
                d = torch.load(f, weights_only=False)
                assert d["tag"] == tag and [u["k"] for u in d["kus"]] == keys[c: c + CHUNK], f"checkpoint mismatch {f}"
                for u in map(unpackage, d["kus"]):
                    u.text = list(self.shard[u.k][1]); self.kus[u.k] = u
                resumed += len(d["kus"])
            else:
                for k in keys[c: c + CHUNK]:
                    v = self.shard[k]
                    gold = (v[2] if len(v) > 2 else []) if self.arm == "oracle" else None
                    self.kus[k] = build(k, v[0], v[1], self.enc, st, gold, self.equiv)
                if f is not None:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"tag": tag, "kus": [package(self.kus[k]) for k in keys[c: c + CHUNK]]}, f.with_suffix(".tmp"))
                    f.with_suffix(".tmp").rename(f)
            for k in keys[c: c + CHUNK]:                     # memory only (answers unchanged): no grads kept,
                for q in self.kus[k].net.parameters():       # freed training buffers returned to the system
                    q.grad = None; q.requires_grad_(False)
            gc.collect(); _LIBC.malloc_trim(0)
            print(f"{time.strftime('%H:%M:%S')} host {self.name}: {len(self.kus)}/{len(keys)} KUs trained "
                  f"(resumed {resumed})", flush=True)
        self.index = HostTitleIndex([KUTitle(k, u.title, title_vector(self.enc, u.title)) for k, u in self.kus.items()])
        self.names = name_kinds([u.title for u in self.kus.values()], [sl.label for u in self.kus.values() for sl in u.slots],
                                self.sha)
        n = len(self.kus)
        return {"n": n, "resumed": resumed, "cpu_s_per_ku": round((time.process_time() - t0) / max(n - resumed, 1), 4),
                "params_mean": float(np.mean([u.stats["params"] for u in self.kus.values()])) if n else 0}

    async def wire(self) -> dict:
        items = [(k, j, u.slots[j].label) for k, u in sorted(self.kus.items()) for j in sorted(u.conns)]
        labels = sorted({lab for _, _, lab in items})
        replies = await asyncio.gather(*[ask(a, RESOLVEB, {"items": [{"span": s} for s in labels]})
                                         for a in self.hosts.values()])
        assert all(r is not None for r in replies), "a host did not answer the wiring RESOLVE"
        win = {}
        for i, lab in enumerate(labels):
            e = merge_once([{"host": r["host"], **r["props"][i]} for r in replies])
            if e is not None and e["rule"] in ("exact", "no_disambiguator"):      # unique title match only
                win[lab] = e["k"]
        n = 0
        for k, j, lab in items:
            t = win.get(lab)
            if t is not None and t != k:
                self.kus[k].conns[j]["target"] = t; n += 1
        return {"n_slots": len(items), "pointers": n}

    def resolve_props(self, items: list) -> list:
        out = []
        for it in items:
            if it.get("qv_span") is None:                     # wiring: title keys only
                p = self.index.propose(it["span"], np.zeros(self.index.TV.shape[1], np.float32),
                                       np.zeros(self.index.TV.shape[1], np.float32))
                p = {"exact": [[k, t, 0.0] for k, t, _ in p["exact"]],
                     "nodis": [[k, t, 0.0] for k, t, _ in p["nodis"]], "fallback": None}
            else:
                p = self.index.propose(it["span"], np.asarray(it["qv_span"], np.float32),
                                       np.asarray(it["qv_question"], np.float32))
            out.append(p)
        return out

    def drop_text(self) -> int:
        if os.environ.get("NHOST_KEEP_TEXT_COPY"):          # audit known-bad only: a hidden copy survives the drop
            self.hidden = types.SimpleNamespace(t=tuple(x for v in self.shard.values() for x in v[1]))
        n = 0
        for u in self.kus.values():
            n += u.text is not None; u.text = None
        self.pcache = {}                                    # reverse lookup: recomputed from parameters after the drop
        if os.environ.get("NHOST_COLD_TEXT"):               # cold-copy known-bad only: text goes into the copies
            self.cold_text = {k: list(v[1]) for k, v in self.shard.items()}
        self.shard = None
        gc.collect()
        return n

    # ---- answering (KUs forward by themselves)
    def qvec(self, u, rel: str) -> np.ndarray:
        t = u.query_text(rel)
        if t not in self.qcache:
            self.qcache[t] = self.enc.encode([t])[0]
        return self.qcache[t]

    def step(self, t: dict) -> tuple[dict, dict | None, str | None]:
        """One KU answers one task. → (finding, next task or None, label to resolve by broadcast or None)."""
        u = self.kus.get(t["k"])
        base = {"qid": t["qid"], "chain": t["chain"], "hop": t["hop"], "k": t["k"], "how": t["how"], "host": self.name}
        if t.get("att"):                                    # re-asked question (gateway retry after a host death)
            base["att"] = t["att"]
        if u is None:                                       # not held here, and its owner under this host's list
            self.c["not_here"] += 1
            return {**base, "title": None, "value": None, "kind": None, "target": None, "f": 0.0, "final": True}, None, None
        rel = t["rels"][t["hop"]]
        last = t["hop"] + 1 >= len(t["rels"]) or (t.get("rev") is not None and t["hop"] == 0)
        if t.get("set") is not None and last:               # set mode (design_metaqa_sets_persons.md §2)
            arm = self.arm if self.arm in ("first", "all") else "neuron"
            f, js = u.answer_set(self.qvec(u, rel) if arm == "neuron" else None, rel, arm, t.get("fmin", 0.5),
                                 t["set"]["theta_f"])
            fin = {**base, "title": u.title, "value": u.slots[js[0][0]].label if js else None, "kind": None,
                   "target": None, "f": f, "final": True, "values": [u.slots[j].label for j, _ in js],
                   "fs": [p for _, p in js]}
            if not js:
                self.c["dont_know"] += 1
            else:
                fin["kind"] = "literal"
            return fin, None, None
        f, j = u.answer(self.qvec(u, rel) if self.arm not in ("first", "all") else None, rel,
                        "first" if self.arm in ("first", "all") else "neuron", t.get("fmin", 0.5))
        if self.arm == "leak" and u.text:                     # known-bad (dry runs only): reads raw text if present
            v = dates_rule(rel, u.text)
            if v is not None:
                return {**base, "title": u.title, "value": v, "kind": "literal", "target": None, "f": 1.0,
                        "final": True}, None, None
        fin = {**base, "title": u.title, "value": None, "kind": None, "target": None, "f": f, "final": True}
        if j is None:
            self.c["dont_know"] += 1
            return fin, None, None
        s = u.slots[j]; c = u.conns.get(j)
        fin.update(value=s.label, kind="pointer" if c and c["target"] is not None else "literal",
                   target=c["target"] if c else None)
        if last:                                            # rev hop 0: the host goes on by REVERSE (run_tasks)
            return fin, None, None
        nxt = {**t, "hop": t["hop"] + 1}
        if c and c["target"] is not None:
            fin["final"] = False; self.c["hop_pointer"] += 1
            return fin, {**nxt, "k": c["target"], "how": "pointer", "key": c.get("tkey", c["target"])}, None
        if s.kind == "name":
            fin["final"] = False; self.c["hop_broadcast"] += 1
            return fin, nxt, s.label                          # target found by RESOLVE broadcast
        self.c["hop_literal_end"] += 1
        return fin, None, None

    async def run_tasks(self, tasks: list) -> None:
        loop = asyncio.get_running_loop()
        start = [t for t in tasks if t.get("prev") and t["k"] is None]
        if start:                                             # person heads: this host starts with REVERSE
            ids = {id(a) for a in start}; tasks = [t for t in tasks if id(t) not in ids]
            findings, out = defaultdict(list), defaultdict(list)
            await self.reverse([(t, None) for t in start], findings, out)
            await asyncio.gather(*[ask(r, FINDINGS, {"findings": fs}) for r, fs in findings.items()],
                                 *[ask(self.hosts[h], TASKS, {"tasks": ts}) for h, ts in out.items()])
            if not tasks:
                return
        away = [t for t in tasks if t["k"] not in self.kus and self.route(t) != self.name
                and t.get("fwd", 0) < MAX_FORWARD and not os.environ.get("NHOST_NO_FORWARD")]
        if away:                                              # the KU moved: the task goes on to its owner
            ids = {id(a) for a in away}; tasks = [t for t in tasks if id(t) not in ids]
            fw = defaultdict(list)
            for t in away:
                fw[self.route(t)].append({**t, "fwd": t.get("fwd", 0) + 1})
            self.c["forward_moved"] += len(away)
            await asyncio.gather(*[ask(self.hosts[h], TASKS, {"tasks": ts}) for h, ts in fw.items()])
            if not tasks:
                return
        res = await asyncio.gather(*[loop.run_in_executor(self.pool, self.step, t) for t in tasks])
        findings = defaultdict(list); out = defaultdict(list); need = []
        for t, (fin, nxt, label) in zip(tasks, res):
            findings[t["res"]].append(fin)
            if nxt is not None and label is None:
                out[self.route(nxt)].append(nxt)
            elif nxt is not None:
                need.append((nxt, label, fin))
        rev = [(t, fin) for t, (fin, _, _) in zip(tasks, res) if t.get("rev") and t["hop"] == 0 and fin["value"]
               and t.get("rev", {}).get("r2")]
        if rev:                                               # after the hop-0 findings (the resultant sees them first)
            await self.reverse(rev, findings, out)
        if need:                                              # KUs without a pointer: RESOLVE broadcast
            items = [{"span": lab, "qv_span": self.enc.encode([lab], "query")[0].tolist(), "qv_question": n["qq"]}
                     for n, lab, _ in need]
            replies = await asyncio.gather(*[ask(a, RESOLVEB, {"items": items}) for a in self.hosts.values()])
            ok = [r for r in replies if r is not None]
            for i, (n, lab, fin) in enumerate(need):
                e = merge_once([{"host": r["host"], **r["props"][i]} for r in ok]) if ok else None
                if len(ok) < len(replies):                    # a host did not answer: the winner may be wrong
                    fin["final"] = True; fin["lost"] = True; self.c["resolve_partial"] += 1
                    continue
                if e is None:
                    fin["final"] = True                       # nowhere to forward
                    continue
                fin["target"] = e["k"]
                n.pop("key", None)                            # the winner's host holds it; key unknown here
                out[e["host"]].append({**n, "k": e["k"], "how": "broadcast"})
        for host, ts in out.items():
            self.c["forward_same_host" if host == self.name else "forward_other_host"] += len(ts)
        await asyncio.gather(*[ask(r, FINDINGS, {"findings": fs}) for r, fs in findings.items()],
                             *[ask(self.hosts[h], TASKS, {"tasks": ts}) for h, ts in out.items()])
        self.trim()

    def trim(self) -> None:
        """Hand freed heap back to the OS (malloc_trim) at most once per NHOST_TRIM seconds, after a batch; default 5
        (owner decision 2026-11-11, after 9991c: answers identical, host mean RSS -35%); NHOST_TRIM=0 turns it off.
        Counts the RSS released and the time spent (STATS trim_*). Answers and messages unchanged."""
        every = float(os.environ.get("NHOST_TRIM", "5"))
        only = os.environ.get("NHOST_TRIM_HOSTS")              # comparison runs: trim on these hosts only
        if every <= 0 or (only and self.name not in only.split(",")) or time.time() - self.c.get("_trim_t", 0) < every:
            return
        rss = lambda: next((int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmRSS")), 0)
        c0, w0 = time.thread_time_ns(), time.perf_counter_ns()
        before = rss(); _LIBC.malloc_trim(0); after = rss()
        self.c["trim_cpu_us"] += (time.thread_time_ns() - c0) // 1000
        self.c["trim_wall_us"] += (time.perf_counter_ns() - w0) // 1000
        self.c["_trim_t"] = int(time.time()); self.c["trim_calls"] += 1
        self.c["trim_freed_kb"] += max(before - after, 0)
        self.c["trim_max_freed_kb"] = max(self.c["trim_max_freed_kb"], before - after)
        print(f"{time.strftime('%H:%M:%S')} trim {self.name}: rss {before >> 10} -> {after >> 10} MB", flush=True)

    async def reverse(self, rev: list, findings: dict, out: dict) -> None:
        """REVERSE broadcast for rev tasks whose hop-0 KU answered (fin = its finding) and for person-head start tasks
        (fin None): one batch to every host; per task one reverse finding (the set of KUs that say yes, head excluded)
        and, with a next relation (rev: r3; prev: r2), a forward task to each of them."""
        def spec(t, fin):                                   # (reverse relation, value(s), excluded KU, rule, theta,
            if fin is None:                                 #  reverse hop, next relation, rels of the next task)
                pv = t["prev"]
                return (t["rels"][0], t["value"], None, pv["rule"], pv["theta"], 0, pv.get("r2"),
                        [t["rels"][0], pv.get("r2")])
            rv = t["rev"]
            vals = fin["values"] if t.get("set") is not None else fin["value"]
            return (rv["r2"], vals, t["k"], rv["rule"], rv["theta"], 1, rv.get("r3"),
                    [t["rels"][0], rv["r2"], rv.get("r3")])
        specs = [spec(t, fin) for t, fin in rev]
        items = [{"rel": a, "value": v, "rule": ru, "theta": th, "exclude": x} for a, v, x, ru, th, _, _, _ in specs]
        t0 = time.time()
        replies = await asyncio.gather(*[ask(a, REVERSE, {"items": items}, 3600) for a in self.hosts.values()])
        ok = [r for r in replies if r is not None]
        self.c["reverse_batches"] += 1; self.c["reverse_items"] += len(items)
        self.c["reverse_messages"] += 2 * len(replies)
        self.c["reverse_wall_ms"] += int(1000 * (time.time() - t0))
        self.c["reverse_kus_evaluated"] += sum(r["evaluated"] for r in ok)
        self.c["reverse_forward_passes"] += sum(r["forward"] for r in ok)
        self.c["reverse_cpu_ms"] += int(1000 * sum(r["cpu_s"] for r in ok))
        for part in ("total", "forward", "lists", "scan"):    # per-thread CPU of the REVERSE work itself
            self.c[f"reverse_thread_{part}_ms"] += sum(r.get("thread_ms", {}).get(part, 0) for r in ok)
        for i, (t, fin) in enumerate(rev):
            if fin is not None:
                fin["final"] = False
            _, _, _, _, _, rhop, r3, nrels = specs[i]
            base = {"qid": t["qid"], "chain": t["chain"], "hop": rhop, "k": None, "how": "reverse", "host": self.name,
                    "value": None, "kind": "reverse", "target": None, **({"att": t["att"]} if t.get("att") else {})}
            if len(ok) < len(replies):                       # a host did not answer: the set may be incomplete
                findings[t["res"]].append({**base, "title": None, "f": 0.0, "final": True, "lost": True,
                                           "set": [], "titles": [], "fs": [], "expect": 0})
                self.c["reverse_partial"] += 1
                continue
            yes = sorted({tuple(x) for r in ok for x in r["yes"][i]})
            findings[t["res"]].append({**base, "title": None, "f": max([x[2] for x in yes], default=0.0),
                                       "final": not (r3 and yes), "set": [x[0] for x in yes],
                                       "titles": [x[1] for x in yes], "fs": [x[2] for x in yes],
                                       "expect": len(yes) if r3 else 0})
            if r3:
                for k, _, _ in yes:
                    nxt = {kk: v for kk, v in t.items() if kk not in ("key", "value")}
                    nxt.update(hop=rhop + 1, rels=nrels, k=k, how="reverse")
                    out[self.route(nxt)].append(nxt)

    def _cache_add(self, k) -> None:
        labs = [x for x in self.kus[k].labels() if x]
        self._cache_lab[k] = labs
        for x in labs:
            self._cache.setdefault(x, set()).add(k)

    def _cache_drop(self, k) -> None:
        for x in self._cache_lab.pop(k, ()):
            ids = self._cache.get(x)
            if ids is not None:
                ids.discard(k)
                if not ids:
                    del self._cache[x]

    def cache_rebuild(self) -> dict:
        """The cache recomputed from the KUs held (check and rebuild)."""
        out: dict = {}
        for k, u in self.kus.items():
            for x in u.labels():
                if x:
                    out.setdefault(x, set()).add(k)
        return out

    def cache_invalidate(self, k) -> None:
        """A KU's slots changed (EDIT): its labels are taken out and read again."""
        with self._cache_lock:
            self._cache_drop(k)
            if k in self.kus:
                self._cache_add(k)

    def slot_cache(self) -> dict:
        """Per-host slot-value cache (reports/design_slot_cache.md; host-cache rule 2026-11-09): this host's own KUs
        only, brought up to date with the KUs held (arrived: labels added; left or forgotten: removed). It proposes
        candidates; every candidate KU still decides with its own slot check and forward pass."""
        with self._cache_lock:
            ids = set(self.kus)
            if ids != self._cache_ids:
                for k in self._cache_ids - ids:
                    self._cache_drop(k)
                for k in ids - self._cache_ids:
                    self._cache_add(k)
                self._cache_ids = ids
                if os.environ.get("NHOST_CACHE_CHECK") and not self._cache_bad:     # dry runs: exact rebuild check
                    assert self._cache == self.cache_rebuild(), f"{self.name}: slot cache differs from its KUs"
            return self._cache

    def reverse_answer(self, items: list) -> dict:
        """REVERSE on this host: every own KU decides for itself (nku.lists) whether it lists each item's value."""
        t0 = time.process_time(); yes = []; ev = fw = 0
        tt0 = time.thread_time_ns(); tf = tl = 0              # this thread's own CPU: total, forward passes, lists
        use = os.environ.get("NHOST_SLOT_CACHE", "1") != "0"
        own = None if use else sorted(self.kus.items())
        cache = self.slot_cache() if use else None
        from mxn.expert.ops import normalize
        for it in items:
            got = []
            vals = it["value"] if isinstance(it["value"], list) else [it["value"]]     # set mode: several values
            if use:                                           # candidates: own KUs holding a slot labelled v
                if os.environ.get("NHOST_CACHE_DROP_ONE") and not self._cache_bad:  # known-bad dry run only
                    with self._cache_lock:
                        cache.pop(normalize(vals[0]), None); self._cache_bad = True
                ids = set()
                for v in vals:
                    ids |= cache.get(normalize(v), set())
                cand = [(k, self.kus[k]) for k in sorted(ids) if k in self.kus]
            else:
                cand = own
            self.c["reverse_kus_owned"] += len(self.kus)
            for k, u in cand:
                ev += 1
                if (k == it["exclude"] and not os.environ.get("NHOST_REV_KEEP_HEAD")) or \
                        not any(u.holds(it["rel"], v) for v in vals):  # NHOST_REV_KEEP_HEAD: dry-run known-bad only
                    continue
                p = None
                if self.arm not in ("first", "all"):
                    if (k, it["rel"]) not in self.pcache:
                        a = time.thread_time_ns()
                        self.pcache[(k, it["rel"])] = u.probs(self.qvec(u, it["rel"]), it["rel"]); fw += 1
                        tf += time.thread_time_ns() - a
                    p = self.pcache[(k, it["rel"])]
                a = time.thread_time_ns()
                f = u.lists(p, it["rel"], it["value"], it["rule"], it["theta"],
                            self.arm if self.arm in ("first", "all") else "neuron")
                tl += time.thread_time_ns() - a
                if f is not None:
                    got.append([k, _DISAMBIG.sub("", u.title), f])
            yes.append(got)
        tt = time.thread_time_ns() - tt0
        return {"host": self.name, "yes": yes, "evaluated": ev, "forward": fw,
                "cpu_s": round(time.process_time() - t0, 4),          # process CPU (all threads) during the call
                "thread_ms": {"total": tt // 10**6, "forward": tf // 10**6, "lists": tl // 10**6,
                              "scan": (tt - tf - tl) // 10**6}}

    def owner(self, k) -> str:
        from mxn.expert.replica import order
        return order(k, self.members)[0]                      # rendezvous of a placement key (default: the KU id)

    def pkey(self, k) -> int:
        u = self.kus.get(k) or self.incoming.get(k)
        return u.key if u is not None and u.key is not None else k

    def route(self, t: dict) -> str:
        """Host for a task: this host's own forwarding note for a KU it moved away, else rendezvous of the key the
        task carries (the pointing KU's copy of the target's key), else of the KU id."""
        if t["k"] in self.moved:
            return self.moved[t["k"]]
        return self.owner(t.get("key", t["k"]))

    # ---- rebalancing (MEMBERS / MOVE)
    async def members_phase(self, phase: str, hosts: dict, r: int | None = None, dead: list | None = None) -> dict:
        """r: replicas per KU under the new list (default: unchanged); dead: hosts gone without a goodbye (their
        copies cannot be sent; another holder sends). Each KU lives on the first r hosts of its rendezvous order
        (reports/design_neuron_replication.md); with r = 1 this is rebalancing (design_neuron_rebalance.md)."""
        from mxn.expert.replica import order
        new, r, dead = sorted(hosts), r or self.R, set(dead or ())
        if phase == "copy":
            out = defaultdict(list)
            for k in sorted(self.kus):
                holders = [h for h in order(self.pkey(k), self.members)[:self.R] if h not in dead]
                if not holders or holders[0] != self.name:          # another live holder sends it
                    continue
                for o in order(self.pkey(k), new)[:r]:
                    if o not in holders:
                        out[o].append(k)
            loop = asyncio.get_running_loop()
            for h, ks in out.items():
                for c in range(0, len(ks), MOVE_CHUNK):
                    blob = await loop.run_in_executor(None, self.pack, ks[c: c + MOVE_CHUNK])
                    a = await ask(hosts[h], MOVE, {"from": self.name, "blob": blob}, 3600)
                    assert a is not None and a["stored"] == len(ks[c: c + MOVE_CHUNK]), f"MOVE to {h} failed"
            self.leaving = {k for k in self.kus if self.name not in order(self.pkey(k), new)[:r]}
            return {"host": self.name, "sent": {h: len(ks) for h, ks in out.items()}}
        if phase == "add":
            n = len(self.incoming)
            if os.environ.get("NHOST_SKIP_ADD"):                 # known-bad (dry runs only): copies never activated
                self.incoming = {}
            loop = asyncio.get_running_loop()
            vec = await loop.run_in_executor(None, lambda: {k: title_vector(self.enc, u.title)
                                                            for k, u in self.incoming.items()})
            idx = HostTitleIndex(list(self.index.kus.values()) + [KUTitle(k, u.title, vec[k])
                                                                   for k, u in sorted(self.incoming.items())])
            self.kus.update(self.incoming); self.index = idx            # swapped whole: RESOLVE never sees a half index
            self.names = name_kinds([u.title for u in self.kus.values()],
                                    [sl.label for u in self.kus.values() for sl in u.slots], self.sha)
            self.incoming = {}
            self.hosts = {**self.hosts, **hosts}
            if os.environ.get("NHOST_CACHE_CHECK"):
                self.slot_cache()
            return {"host": self.name, "added": n}
        if phase == "drop":
            self.members, self.R = new, r
            self.index = HostTitleIndex([v for k, v in self.index.kus.items() if k not in self.leaving])
            for k in self.leaving:
                self.kus.pop(k, None)
            n = len(self.leaving); self.leaving = set()
            gc.collect(); _LIBC.malloc_trim(0)
            if os.environ.get("NHOST_CACHE_CHECK"):           # dry runs: slot cache follows the move exactly
                self.slot_cache()
            return {"host": self.name, "dropped": n, "kus": len(self.kus)}
        if phase == "done":
            self.hosts = dict(hosts)
            return {"host": self.name, "kus": len(self.kus), "r": self.R, "ids_sha1": hashlib.sha1(
                json.dumps(sorted(self.kus)).encode()).hexdigest()}
        raise ValueError(phase)

    # ---- locality placement (LOCAL / KEYS)
    async def local_phase(self, phase: str, quota: dict | None) -> dict:
        from mxn.expert.replica import order
        from mxn.neuron.nku import anchor
        if phase == "plan":
            self.plan = {}
            for k, u in sorted(self.kus.items()):
                a = anchor(u)
                self.plan[k] = (a, order(a, self.members)[0])
            want = Counter(d for a, d in self.plan.values() if d != self.name)
            return {"host": self.name, "resident": len(self.kus), "want": dict(want)}
        if phase == "copy":
            out = defaultdict(list)
            for k, (a, d) in sorted(self.plan.items()):
                if d == self.name:
                    self.kus[k].key = a                          # its anchor's host is this one: stays
                elif len(out[d]) < quota.get(d, 0) and not os.environ.get("NHOST_LOCAL_NOKEY"):
                    self.kus[k].key = a; out[d].append(k)
            if os.environ.get("NHOST_LOCAL_NOKEY"):              # known-bad (dry runs only): KUs sent, keys unchanged
                for k, (a, d) in sorted(self.plan.items()):
                    if d != self.name and len(out[d]) < quota.get(d, 0):
                        out[d].append(k)
            loop = asyncio.get_running_loop()
            for h, ks in out.items():
                for c in range(0, len(ks), MOVE_CHUNK):
                    blob = await loop.run_in_executor(None, self.pack, ks[c: c + MOVE_CHUNK])
                    a = await ask(self.hosts[h], MOVE, {"from": self.name, "blob": blob}, 3600)
                    assert a is not None and a["stored"] == len(ks[c: c + MOVE_CHUNK]), f"MOVE to {h} failed"
                for k in ks:
                    self.moved[k] = h
            self.leaving = {k for ks in out.values() for k in ks}
            return {"host": self.name, "sent": {h: len(ks) for h, ks in out.items()}}
        if phase == "keys":
            ids = sorted({c["target"] for u in self.kus.values() for c in u.conns.values() if c["target"] is not None})
            rs = await asyncio.gather(*[ask(a, KEYS, {"ids": ids}, 600) for a in self.hosts.values()])
            assert all(r is not None for r in rs), "a host did not answer KEYS"
            keys = {}
            for r in rs:
                keys.update({int(i): v for i, v in r["keys"].items()})
            n = missing = 0
            for u in self.kus.values():
                for c in u.conns.values():
                    if c["target"] is None or os.environ.get("NHOST_LOCAL_NOTKEY"):           # known-bad switch
                        continue
                    if c["target"] not in keys:                   # no host holds the target: report, keep the id
                        missing += 1; continue
                    c["tkey"] = keys[c["target"]]; n += c["tkey"] != c["target"]
            return {"host": self.name, "pointers_with_other_key": n, "targets_missing": missing}
        raise ValueError(phase)

    def pack(self, ks: list) -> str:
        b = io.BytesIO(); torch.save([package(self.kus[k]) for k in ks], b)
        return base64.b64encode(b.getvalue()).decode()

    # ---- cold copies (COLD / COLDPUT), R = 1
    async def cold_phase(self, phase: str, gen: int, hosts: dict | None) -> dict:
        from mxn.expert.replica import order
        if phase == "write":
            out = defaultdict(list)
            for k in sorted(self.kus):
                o = order(self.pkey(k), self.members)
                if len(o) > 1:
                    out[o[1]].append(k)                           # successor: the owner if this host is gone
            loop = asyncio.get_running_loop()
            for h, ks in out.items():
                for c in range(0, len(ks), MOVE_CHUNK):
                    blob = await loop.run_in_executor(None, self.pack_cold, ks[c: c + MOVE_CHUNK])
                    r = await ask(self.hosts[h], COLDPUT, {"from": self.name, "gen": gen, "i": c // MOVE_CHUNK,
                                                           "blob": blob}, 3600)
                    assert r is not None and r["stored"] == len(ks[c: c + MOVE_CHUNK]), f"COLDPUT to {h} failed"
            return {"host": self.name, "sent": {h: len(ks) for h, ks in out.items()}}
        if phase == "commit":
            for f in self.cold.glob("g*_*.pt"):
                if int(f.name[1:].split("_")[0]) < gen:
                    f.unlink()
            for g in [g for g in self.cold_ids if g < gen]:
                del self.cold_ids[g]
            ids = sorted(self.cold_ids[gen])
            return {"host": self.name, "copies": len(ids), "ids_sha1": hashlib.sha1(json.dumps(ids).encode()).hexdigest()}
        if phase == "restore":
            new = sorted(hosts); n = 0
            loop = asyncio.get_running_loop()
            for f in sorted(self.cold.glob("g*_*.pt")):          # every copy on this disk (older ones: deleted at commit)
                ps = await loop.run_in_executor(None, lambda f=f: torch.load(f, weights_only=False))
                for d in ps:
                    if order(d.get("key", d["k"]), new)[0] == self.name and d["k"] not in self.kus \
                            and d["k"] not in self.incoming:
                        self.incoming[d["k"]] = unpackage(d); n += 1
            return {"host": self.name, "restored": n}
        if phase == "audit":
            hits, names = set(), set()
            for f in sorted(self.cold.glob("g*_*.pt")):
                ps = torch.load(f, weights_only=False)
                own = name_kinds([d["title"] for d in ps], [sl["label"] for d in ps for sl in d["slots"]], self.sha)
                for x in _all_strings(ps):
                    h = hashlib.sha1(x.encode()).hexdigest()
                    if len(x) > 20 and h in self.sha:                 # as text_in_memory; names held as KU state apart
                        (names if h in own else hits).add(h)
            return {"host": self.name, "cold_text": len(hits), "cold_names": len(names),
                    "files": len(list(self.cold.glob("g*_*.pt")))}
        raise ValueError(phase)

    # ---- forgetting and editing (reports/design_forget_edit.md)
    def forget(self, ids: list) -> dict:
        ids = set(ids); deleted = 0
        for k in [k for k in self.kus if k in ids]:
            del self.kus[k]; deleted += 1
            if self.index is not None and k in self.index.kus:
                self.index.remove(k)
        for k in [k for k in self.incoming if k in ids]:
            del self.incoming[k]
        for d in (self.moved, self.plan):
            for k in [k for k in d if k in ids]:
                del d[k]
        self.pcache = {key: v for key, v in self.pcache.items() if key[0] not in ids}
        with self._cache_lock:
            for k in ids:
                self._cache_drop(k)
            self._cache_ids -= ids
        cleared = 0
        if not os.environ.get("NHOST_FORGET_NO_SCRUB"):          # known-bad only: other KUs keep their pointers
            for u in self.kus.values():
                for c in u.conns.values():
                    if c.get("target") in ids:
                        c["target"] = None; c.pop("tkey", None); cleared += 1
        cold_deleted = 0
        if self.cold is not None and self.cold.exists() and not os.environ.get("NHOST_FORGET_KEEP_COLD"):
            for f in sorted(self.cold.glob("g*_*.pt")):
                ps = torch.load(f, weights_only=False)
                keep = [d for d in ps if d["k"] not in ids]
                if len(keep) != len(ps):
                    cold_deleted += len(ps) - len(keep)
                    if keep:
                        torch.save(keep, f.with_suffix(".tmp")); f.with_suffix(".tmp").rename(f)
                    else:
                        f.unlink()
            for g in self.cold_ids:
                self.cold_ids[g] -= ids
        gc.collect(); _LIBC.malloc_trim(0)
        return {"host": self.name, "deleted": deleted, "cold_deleted": cold_deleted, "pointers_cleared": cleared}

    @staticmethod
    def phash(u) -> str:
        h = hashlib.sha1()
        for q in u.net.parameters():
            h.update(q.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def faudit(self, m: dict) -> dict:
        ids = set(m["ids"])
        if m["mode"] == "hash":
            return {"host": self.name, "phash": {str(k): self.phash(u) for k, u in self.kus.items() if k in ids}}
        titles, values = set(m.get("titles", [])), set(m.get("values", []))
        sha, ph = set(m.get("sha", [])), set(m.get("phash", []))
        r = {"host": self.name, "held": sum(k in ids for k in list(self.kus) + list(self.incoming)),
             "index": sum(k in ids for k in (self.index.kus if self.index is not None else ())),
             "index_titles": sum(u.title in titles for u in self.index.kus.values()) if self.index is not None else 0}
        with self._cache_lock:
            r["cache_listing"] = sum(bool(v & ids) for v in self._cache.values())
        r["unique_labels"] = sum(sl.label in values for u in self.kus.values() for sl in u.slots)
        r["pointers"] = sum(c.get("target") in ids for u in self.kus.values() for c in u.conns.values())
        r["sentences"] = text_in_memory(sha)["text"] if sha else 0
        gc.collect(); tens = 0
        for o in gc.get_objects():                               # parameter tensors of the forgotten KUs, anywhere
            if isinstance(o, torch.nn.Module) and any(True for _ in o.parameters()):   # same hash as phash(u.net)
                h = hashlib.sha1()
                for q in o.parameters():
                    h.update(q.detach().cpu().numpy().tobytes())
                tens += h.hexdigest() in ph
        r["parameter_modules"] = tens
        cold = 0
        if self.cold is not None and self.cold.exists():
            for f in sorted(self.cold.glob("g*_*.pt")):
                cold += sum(d["k"] in ids or d.get("title") in titles for d in torch.load(f, weights_only=False))
        r["cold"] = cold
        r["total"] = sum(v for k, v in r.items() if k != "host")
        return r

    async def edit(self, k, sentences: list) -> dict:
        if k not in self.kus:
            return {"host": self.name, "held": False}
        t0 = time.time()
        if os.environ.get("NHOST_EDIT_NO_TRAIN"):                  # known-bad only: nothing changes
            return {"host": self.name, "held": True, "retrained": False, "seconds": round(time.time() - t0, 3)}
        old = self.kus[k]
        loop = asyncio.get_running_loop()
        u = await loop.run_in_executor(None, build, k, old.title, list(sentences), self.enc, self.st, None, self.equiv)
        for q in u.net.parameters():
            q.grad = None; q.requires_grad_(False)
        self.sha |= {hashlib.sha1(x.encode()).hexdigest() for x in sentences}      # audit covers the edited text too
        self.kus[k] = u
        self.pcache = {key: v for key, v in self.pcache.items() if key[0] != k}
        self.cache_invalidate(k)
        await self.wire_one(k)                                      # its own pointers, by its own RESOLVE broadcast
        u.text = None
        gc.collect()
        return {"host": self.name, "held": True, "retrained": True, "seconds": round(time.time() - t0, 3)}

    async def wire_one(self, k) -> int:
        u = self.kus[k]
        labels = sorted({u.slots[j].label for j in u.conns})
        if not labels:
            return 0
        replies = await asyncio.gather(*[ask(a, RESOLVEB, {"items": [{"span": s} for s in labels]})
                                         for a in self.hosts.values()])
        win = {}
        for i, lab in enumerate(labels):
            e = merge_once([{"host": r["host"], **r["props"][i]} for r in replies if r is not None])
            if e is not None and e["rule"] in ("exact", "no_disambiguator"):
                win[lab] = e["k"]
        n = 0
        for j in u.conns:
            t = win.get(u.slots[j].label)
            if t is not None and t != k:
                u.conns[j]["target"] = t; n += 1
        return n

    def pack_cold(self, ks: list) -> str:
        ps = [package(self.kus[k]) for k in ks]
        if getattr(self, "cold_text", None):                     # known-bad only (NHOST_COLD_TEXT)
            for d in ps:
                d["stats"] = {**d["stats"], "text": self.cold_text.get(d["k"], [])}
        b = io.BytesIO(); torch.save(ps, b)
        return base64.b64encode(b.getvalue()).decode()

    def cold_put(self, m: dict) -> int:
        raw = base64.b64decode(m["blob"])
        ks = [d["k"] for d in torch.load(io.BytesIO(raw), weights_only=False)]
        if os.environ.get("NHOST_COLD_SKIP"):                     # known-bad only: the copy is never stored
            return len(ks)
        self.cold.mkdir(parents=True, exist_ok=True)
        f = self.cold / f"g{m['gen']}_{m['from']}_{m['i']:05d}.pt"
        f.with_suffix(".tmp").write_bytes(raw); f.with_suffix(".tmp").rename(f)
        self.cold_ids[m["gen"]] |= set(ks)
        return len(ks)

    def move_in(self, blob: str) -> int:
        ps = torch.load(io.BytesIO(base64.b64decode(blob)), weights_only=False)
        for d in ps:
            u = unpackage(d)
            if os.environ.get("NHOST_PERTURB_MOVE"):              # known-bad (dry runs only): a moved KU changes
                with torch.no_grad():
                    for q in u.net.parameters():
                        q.add_(1e-2)
            self.incoming[d["k"]] = u
        return len(ps)

    # ---- frames
    async def on_conn(self, reader, writer):
        try:
            kind, payload = await read_frame(reader)
            m = json.loads(payload)
            loop = asyncio.get_running_loop()
            if kind == TRAIN:
                r = await loop.run_in_executor(None, self.train, Settings(**m["settings"]))
            elif kind == WIRE:
                r = await self.wire()
            elif kind == RESOLVEB:
                r = {"host": self.name, "props": await loop.run_in_executor(None, self.resolve_props, m["items"])}
            elif kind == DROPTEXT:
                r = {"dropped": self.drop_text()}
            elif kind == AUDIT:
                a = await loop.run_in_executor(None, text_in_memory, self.sha, self.names)
                r = {"host": self.name, "sentences_in_memory": a["text"], "name_sentences_in_memory": a["names"],
                     "by_kind": a["by_kind"], "hits": a["hits"]}
            elif kind == TASKS:
                self.c["tasks_in"] += len(m["tasks"]); self.c["batches_in"] += 1
                asyncio.ensure_future(self.run_tasks(m["tasks"]))
                r = {"accepted": len(m["tasks"])}
            elif kind == MEMBERS:
                r = await self.members_phase(m["phase"], m["hosts"], m.get("r"), m.get("dead"))
            elif kind == MOVE:
                r = {"stored": await loop.run_in_executor(None, self.move_in, m["blob"])}
            elif kind == LOCAL:
                r = await self.local_phase(m["phase"], m.get("quota"))
            elif kind == REVERSE:
                r = await loop.run_in_executor(None, self.reverse_answer, m["items"])
                self.trim()
            elif kind == KEYS:
                r = {"keys": {str(k): self.pkey(k) for k in m["ids"] if k in self.kus}}
            elif kind == COLD:
                r = await self.cold_phase(m["phase"], m["gen"], m.get("hosts"))
            elif kind == COLDPUT:
                r = {"stored": await loop.run_in_executor(None, self.cold_put, m)}
            elif kind == FORGET:
                r = await loop.run_in_executor(None, self.forget, m["ids"])
            elif kind == FAUDIT:
                r = await loop.run_in_executor(None, self.faudit, m)
            elif kind == EDIT:
                r = await self.edit(m["k"], m["sentences"])
            elif kind == STATS:
                r = {"host": self.name, "kus": len(self.kus), **self.c,
                     "proc_cpu_ms": int(1000 * time.process_time()), "threads": len(os.listdir("/proc/self/task")),
                     "cpus": sorted(os.sched_getaffinity(0))}
            elif kind == STOP:
                r = {"stopping": True}
                writer.write(frame(REPLY, json.dumps(r).encode())); await writer.drain(); writer.close()
                asyncio.get_running_loop().stop(); return
            else:
                r = {"error": f"unknown kind {kind}"}
            writer.write(frame(REPLY, json.dumps(r).encode())); await writer.drain()
        finally:
            writer.close()


def memprobe(name: str, every: float) -> None:
    """Diagnostics only (NHOST_MEMPROBE=<seconds>): log RSS, traced Python memory and its top allocation sites."""
    import threading, tracemalloc
    tracemalloc.start(1)

    def loop():
        while True:
            time.sleep(every)
            rss = next((int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmRSS")), 0)
            cur, peak = tracemalloc.get_traced_memory()
            top = tracemalloc.take_snapshot().statistics("lineno")[:8]
            sites = "; ".join(f"{s.traceback[0].filename.rsplit('/', 1)[-1]}:{s.traceback[0].lineno} {s.size >> 20}MB/{s.count}"
                              for s in top)
            print(f"{time.strftime('%H:%M:%S')} memprobe {name}: rss {rss >> 10} MB traced {cur >> 20} MB interval-peak {peak >> 20} MB"
                  f" | {sites}", flush=True)
            tracemalloc.reset_peak()                    # peak per interval
    threading.Thread(target=loop, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name"); ap.add_argument("--hostlist"); ap.add_argument("--shard")
    ap.add_argument("--expert", default="artifacts/expert"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arm", default="neuron", choices=("neuron", "first", "all", "oracle", "leak"))
    ap.add_argument("--ckpt", default=None); ap.add_argument("--sha", default=None); ap.add_argument("--cold", default=None)
    a = ap.parse_args()
    if os.environ.get("NHOST_MEMPROBE"):
        memprobe(a.name, float(os.environ["NHOST_MEMPROBE"]))
    hosts = json.load(open(a.hostlist))
    shard = pickle.load(open(a.shard, "rb"))
    workers = a.workers
    if os.environ.get("NHOST_PIN"):     # CPU investigation: host i of H gets its own share of the cores; threads <= share
        cpus = sorted(os.sched_getaffinity(0)); H = len(hosts); i = sorted(hosts).index(a.name)
        mine = [c for j, c in enumerate(cpus) if j * H // len(cpus) == i] or [cpus[i % len(cpus)]]
        os.sched_setaffinity(0, mine); workers = len(mine)
        print(f"{time.strftime('%H:%M:%S')} pin {a.name}: cpus {mine}, workers {workers}", flush=True)
    h = NeuronHost(a.name, hosts, shard, a.expert, workers, a.arm, a.ckpt, a.sha, a.cold)
    del shard
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    if os.environ.get("NHOST_PIN"):     # REVERSE runs in the default executor: same cap
        loop.set_default_executor(ThreadPoolExecutor(workers))
    loop.run_until_complete(serve(hosts[a.name], h.on_conn))
    print(f"host {a.name} ready on {hosts[a.name]} with {len(h.shard)} KUs", flush=True)
    loop.run_forever()


if __name__ == "__main__":
    main()
