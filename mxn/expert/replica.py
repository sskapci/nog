"""Primary-backup replication of knowledge units (chapter 3 step 1; design: reports/design_replication.md, 69a315d).

Architecture check (roadmap "share-nothing"; decision 2026-10-24b §6): every copy is a complete, independent KU held by
one host; hosts exchange copies by message only. Per KU copy the holding host keeps role (primary | backup), epoch
(takeovers), version (edits) and the list of hosts holding copies: KU state, owned by the copy. A host's title index
covers only the KUs it is primary for. The gateway holds no KU state: it keeps the read-only host list and, at the
transport level, which hosts failed to answer (to send FAILOVER once and REJOIN when a host is back). Placement is
rendezvous hashing over the static host list (names and machine labels); no placement map, coordinator or vote.

Messages (numbers continue mxn/expert/fhost.py):
  REPLICATE host → host: {"k", "epoch", "version", "role", "primary", "copies", "package" | "drop": true}
  FAILOVER  gateway → live hosts: carried as RESOLVE {"dead": [...]} — each backup whose primary is dead and which is
            the first live copy in the KU's order promotes itself (epoch + 1), re-replicates, then proposes
  SYNC      host → host: {"ks": [...]} → {"state": [[k, role, epoch, version, copies], ...]}
  PULL      host → host: {"k"} → the copy's package and state
  REJOIN    gateway → a host that failed to answer before: the host checks each of its copies against the other
            holders (SYNC) and demotes, catches up or drops stale copies before it answers anything else
  FEEDBACK  verifier → the source KU's primary: {"k", "relation", "value_sha1", "target", "outcome", "t", "source"} —
            a verified hop outcome (success | failure | correction: the verified target of a failed hop); the primary
            updates its own connection, versions and forwards it (chapter 3 step 2, reports/design_connections.md)

Connections (chapter 3 step 2): KU state of the source KU, a list per KU of {"relation", "value_sha1", "target",
"succ", "fail", "corr", "t", "version", "epoch"} (corr: the part of succ that came from corrections); score = (succ + 1) / (succ + fail + 2) after lazy decay (half-life in the
time unit the messages carry). Only the primary answers VIA from them, and only with tau_c set (None: never).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time

from mxn.expert.fhost import (ADD, DROP, EDIT, EDITV, FIND, FOUND, OK, RESOLVE, RESOLVED, VIA, VIAR, WHO, FindHost)
from mxn.common.frames import frame, open_conn, read_frame
from mxn.common.safeload import pickle_load  # no code execution from data files (2026-11-20)

REPLICATE, SYNC, PULL, REJOIN = 19, 20, 21, 22
FEEDBACK = 23


def value_sha1(value: str) -> str:
    return hashlib.sha1((value or "").encode()).hexdigest()


def conn_score(c: dict, t: float, half_life: float | None) -> float:
    """Score after lazy decay to time t (nothing is written)."""
    f = 0.5 ** (max(0.0, t - c["t"]) / half_life) if half_life else 1.0
    s, x = c["succ"] * f, c["fail"] * f
    return (s + 1) / (s + x + 2)
MIN_R = 2


def check_r(R: int) -> int:
    if R < MIN_R:
        raise ValueError(f"replication factor {R} < {MIN_R}: refusing to run with fewer than {MIN_R} copies per KU")
    return R


def rv_score(k, h: str) -> int:
    return int(hashlib.sha1(f"{k}|{h}".encode()).hexdigest()[:16], 16)


def order(k, names) -> list[str]:
    """The KU's rendezvous order over host names (ties by name): the same everywhere, from the id alone."""
    return sorted(names, key=lambda h: (-rv_score(k, h), h))


def replicas(k, machines: dict[str, str], R: int, exclude=()) -> list[str]:
    """The KU's R hosts: walk its order, skipping a host whose machine label is taken, as long as R distinct machines
    remain (machine awareness). machines: {host: machine label}. First = initial primary."""
    names = [h for h in machines if h not in set(exclude)]
    n_m = len({machines[h] for h in names})
    out, used = [], set()
    for h in order(k, names):
        if n_m >= R and machines[h] in used:
            continue
        out.append(h); used.add(machines[h])
        if len(out) == R:
            break
    if len(out) < R:                                          # fewer distinct machines than R: fill in order
        out += [h for h in order(k, names) if h not in out][: R - len(out)]
    return out


class RepHost(FindHost):
    """A FindHost whose KUs are copies with a role. Construct with FindHost's arguments, then call setup()."""

    def setup(self, hostlist: dict[str, dict], R: int, meta: dict, vecs: dict) -> None:
        """hostlist: {name: {"addr", "machine"}}; meta: {k: {"role", "epoch", "version", "primary", "copies"}};
        vecs: {k: title vector} for every copy held (backups' vectors are kept to be indexed on promotion)."""
        from mxn.expert.hostindex import HostTitleIndex, KUTitle

        self.hostlist, self.R, self.meta = hostlist, check_r(R), meta
        self.vecs = {k: [float(x) for x in v] for k, v in vecs.items()}
        self.index = HostTitleIndex([KUTitle(k, self.units[k][0], _arr(self.vecs[k])) for k in self.units
                                     if meta[k]["role"] == "primary"])
        self.alive, self.hold_forward, self.log = True, False, []
        self.rejoining = None
        if not isinstance(self.connections, dict) or self.connections is FindHost.connections:
            self.connections = {}
        self.tau_c = getattr(self, "tau_c", None)            # read-only parameters (chapter 3 step 2)
        self.half_life = getattr(self, "half_life", None)

    # ---- helpers
    def machines(self) -> dict[str, str]:
        return {h: v["machine"] for h, v in self.hostlist.items()}

    def is_primary(self, k) -> bool:
        return k in self.meta and self.meta[k]["role"] == "primary"

    def state(self, k) -> list:
        m = self.meta[k]
        return [k, m["role"], m["epoch"], m["version"], m["copies"]]

    def pkg(self, k) -> dict:
        t, ss = self.units[k]
        return {"title": t, "sentences": list(ss), "vec": self.vecs.get(k),
                "connections": [dict(c) for c in self.connections.get(k, [])]}

    def _promote(self, k) -> None:
        from mxn.expert.hostindex import KUTitle
        self.meta[k].update(role="primary", primary=self.name)
        if k not in self.index.kus:
            self.index.add(KUTitle(k, self.units[k][0], _arr(self.vecs[k])))

    def _demote(self, k, primary: str) -> None:
        self.meta[k].update(role="backup", primary=primary)
        if k in self.index.kus:
            self.index.remove(k)

    def _store(self, k, p: dict, meta: dict) -> None:
        from mxn.expert.hostindex import KUTitle
        self.units[k] = (p["title"], p["sentences"])
        if p.get("vec") is not None:
            self.vecs[k] = p["vec"]
        elif k not in self.vecs:
            self.vecs[k] = [float(x) for x in self._ku_title(k, p["title"]).vec]
        if not isinstance(self.connections, dict) or self.connections is FindHost.connections:
            self.connections = {}
        if p.get("connections"):
            self.connections[k] = [dict(c) for c in p["connections"]]
        else:                                                   # the package is the whole KU: no connections = none
            self.connections.pop(k, None)
        self.meta[k] = dict(meta)
        if meta["role"] == "primary":
            if k in self.index.kus:
                self.index.remove(k)
            self.index.add(KUTitle(k, p["title"], _arr(self.vecs[k])))
        elif k in self.index.kus:
            self.index.remove(k)

    def _forget(self, k) -> None:
        self.units.pop(k, None); self.vecs.pop(k, None); self.meta.pop(k, None); self.connections.pop(k, None)
        if k in self.index.kus:
            self.index.remove(k)

    async def ask(self, h: str, kind: int, msg: dict) -> dict | None:
        """One request to another host; None if it cannot be reached."""
        try:
            reader, writer = await asyncio.wait_for(open_conn(self.hostlist[h]["addr"]), 5)
            try:
                writer.write(frame(kind, json.dumps({"rid": 0, **msg}).encode())); await writer.drain()
                _, payload = await asyncio.wait_for(read_frame(reader), 10)
                return json.loads(payload)
            finally:
                writer.close()
        except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            return None

    async def forward(self, k) -> dict:
        """Send the primary's copy (or its deletion) to every other copy; → {host: accepted (None: unreachable)}."""
        m = self.meta.get(k)
        out = {}
        if m is None or self.hold_forward:
            return out
        msg = {"k": k, "epoch": m["epoch"], "version": m["version"], "role": "backup", "primary": self.name,
               "copies": m["copies"]}
        msg.update({"drop": True} if m.get("dropped") else {"package": self.pkg(k)})
        for h in m["copies"]:
            if h != self.name:
                r = await self.ask(h, REPLICATE, msg)
                out[h] = None if r is None else r.get("applied")
        if m.get("dropped"):
            self._forget(k)
        return out

    async def failover(self, dead: list[str]) -> list:
        """Promote the copies this host is first in line for; re-replicate every KU it is primary of that lost a copy.
        → [[k, epoch, version], ...] promoted."""
        dead = set(dead)
        promoted, todo = [], []
        for k in sorted(self.meta, key=str):
            m = self.meta[k]
            if m["role"] == "backup" and m["primary"] in dead:
                live = [h for h in order(k, m["copies"]) if h not in dead]
                if live and live[0] == self.name:
                    m["epoch"] += 1
                    self._promote(k)
                    promoted.append([k, m["epoch"], m["version"]])
            if self.is_primary(k) and dead & set(m["copies"]):
                keep = [h for h in m["copies"] if h not in dead]
                new = [h for h in replicas(k, self.machines(), self.R, exclude=dead) if h not in keep]
                m["copies"] = keep + new[: self.R - len(keep)]
                todo.append(k)
        # re-replication after the reply: new epoch / copies to the old live copies and the new ones
        self.rereplicating = asyncio.ensure_future(self._forward_all(todo))
        return promoted

    async def _forward_all(self, ks) -> None:
        for k in ks:
            await self.forward(k)

    async def rejoin(self) -> dict:
        """Back after failing to answer: check every copy against the other holders before answering anything."""
        rep = {"demoted": 0, "caught_up": 0, "dropped": 0}
        peers = {}
        for k, m in self.meta.items():
            for h in set(m["copies"]) | set(replicas(k, self.machines(), self.R)):
                if h != self.name:
                    peers.setdefault(h, []).append(k)
        best = {}
        for h, ks in peers.items():
            r = await self.ask(h, SYNC, {"ks": ks})
            for k, role, e, v, copies in (r or {}).get("state", []):
                if (e, v, role == "primary") > best.get(k, ((-1, -1, False), None, None))[0]:
                    best[k] = ((e, v, role == "primary"), h, copies)
        for k in sorted(list(self.meta), key=str):
            if k not in best:
                continue
            (e, v, is_p), h, copies = best[k]
            m = self.meta[k]
            if (e, v) > (m["epoch"], m["version"]):
                if self.name not in copies and len(copies) >= self.R:
                    self._forget(k); rep["dropped"] += 1; continue
                was_p = m["role"] == "primary"
                p = await self.ask(h, PULL, {"k": k})
                if p and p.get("package"):
                    self._store(k, p["package"], {"role": "backup", "epoch": e, "version": v, "primary": h if is_p else p["primary"],
                                                  "copies": copies})
                    rep["caught_up"] += 1
                rep["demoted"] += was_p
        return rep

    # ---- messages
    async def on_conn(self, reader, writer):
        try:
            while True:
                kind, payload = await read_frame(reader)
                if not self.alive:                              # killed / paused in a test: as if unreachable
                    writer.close(); return
                if self.rejoining is not None and kind not in (SYNC, PULL, REPLICATE):
                    await self.rejoining
                m = json.loads(payload)
                r = await self.handle(kind, m)
                writer.write(frame(r[0], json.dumps({"rid": m.get("rid"), "host": self.name, **r[1]}).encode()))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()

    async def handle(self, kind: int, m: dict):
        import numpy as np

        k = m.get("k")
        if kind == REJOIN:
            self.rejoining = asyncio.ensure_future(self.rejoin())
            rep = await self.rejoining
            self.rejoining = None
            return OK, {"applied": True, **rep}
        if kind == RESOLVE:
            t0 = time.perf_counter()
            promoted = await self.failover(m["dead"]) if m.get("dead") else []
            t1 = time.perf_counter()
            r = self.index.propose(m["span"], np.asarray(m["qv_span"], np.float32), np.asarray(m["qv_question"], np.float32))
            ks = {c[0] for t in ("exact", "nodis") for c in r[t]} | ({r["fallback"][0]} if r["fallback"] else set())
            timing = {"failover_s": t1 - t0, "propose_s": time.perf_counter() - t1} if m.get("dead") else {}
            return RESOLVED, {**r, "epochs": [[x, self.meta[x]["epoch"]] for x in sorted(ks, key=str)], "promoted": promoted,
                              **timing}
        if kind == FIND:
            if not self.is_primary(k):
                return FOUND, {"value": "", "p_ext": 0.0, "not_primary": True}
            v, p = self.find(k, m["relation"])
            return FOUND, {"value": v, "p_ext": p, "epoch": self.meta[k]["epoch"], "version": self.meta[k]["version"],
                           "title": self.units[k][0]}
        if kind == VIA:                                         # the source KU's best valid connection ≥ tau_c
            if not self.is_primary(k) or self.tau_c is None:
                return VIAR, {"target": None}
            h, t, best = value_sha1(m.get("value")), m.get("t", 0.0), None
            for c in self.connections.get(k, []):
                if c["relation"] == m["relation"] and c["value_sha1"] == h:
                    sc = conn_score(c, t, self.half_life)
                    if sc >= self.tau_c and (best is None or (sc, c["succ"], str(c["target"])) > best[:3]):
                        best = (sc, c["succ"], str(c["target"]), c["target"], c.get("corr", 0.0))
            return VIAR, {"target": None if best is None else [best[3], best[0], best[1], best[4]],
                          "epoch": self.meta[k]["epoch"], "version": self.meta[k]["version"]}
        if kind == FEEDBACK:                                    # verified outcome only (design §5)
            if not self.is_primary(k):
                return OK, {"applied": False, "not_primary": k in self.meta}
            cs = self.connections.setdefault(k, [])
            key = (m["relation"], m["value_sha1"], m["target"])
            c = next((c for c in cs if (c["relation"], c["value_sha1"], c["target"]) == key), None)
            if c is None:
                if m["outcome"] == "failure":                   # a failure never creates a connection
                    return OK, {"applied": False, "reason": "no connection"}
                c = {"relation": m["relation"], "value_sha1": m["value_sha1"], "target": m["target"],
                     "succ": 0.0, "fail": 0.0, "corr": 0.0, "t": m["t"]}
                cs.append(c)
            f = 0.5 ** (max(0.0, m["t"] - c["t"]) / self.half_life) if self.half_life else 1.0
            c["succ"] *= f; c["fail"] *= f; c["t"] = max(c["t"], m["t"])
            c["fail" if m["outcome"] == "failure" else "succ"] += 1
            if m["outcome"] == "correction":                    # the verified target of a failed hop (amendment)
                c["corr"] = c.get("corr", 0.0) * f + 1
            self.meta[k]["version"] += 1
            c["version"], c["epoch"] = self.meta[k]["version"], self.meta[k]["epoch"]
            asyncio.ensure_future(self.forward(k))
            return OK, {"applied": True, "version": c["version"], "epoch": c["epoch"]}
        if kind == WHO:
            if k not in self.meta:
                return OK, {"applied": False}
            t, ss = self.units[k]
            return OK, {"applied": True, "role": self.meta[k]["role"], "epoch": self.meta[k]["epoch"],
                        "version": self.meta[k]["version"], "copies": self.meta[k]["copies"],
                        "sha1": hashlib.sha1("\n".join([t, *ss]).encode()).hexdigest()}
        if kind in (EDIT, EDITV, DROP):
            if not self.is_primary(k):
                return OK, {"applied": False, "not_primary": k in self.meta}
            if kind == EDIT:
                self.units[k] = (self.units[k][0], m["sentences"]); r = {}
                self.connections.pop(k, None)                   # the KU's facts replaced: all connections reset
            elif kind == EDITV:
                r = self.edit_value(m)
                if "error" in r:
                    return OK, {"applied": False, **r}
                if k in self.connections:                       # its fact for this relation changed: reset them
                    self.connections[k] = [c for c in self.connections[k] if c["relation"] != m["relation"]]
            else:
                r = {}; self.meta[k]["dropped"] = True
                if k in self.index.kus:
                    self.index.remove(k)
            self.meta[k]["version"] += 1
            v, e = self.meta[k]["version"], self.meta[k]["epoch"]
            asyncio.ensure_future(self.forward(k))            # after the reply: the lost-edit window (design §4.5)
            return OK, {"applied": True, "version": v, "epoch": e, **r}
        if kind == ADD:
            k = m.get("k") or self.new_id()
            reps = replicas(k, self.machines(), self.R)
            vec = self.vecs.get(k) or [float(x) for x in self._ku_title(k, m["title"]).vec]
            p = {"title": m["title"], "sentences": m["sentences"], "vec": vec, "connections": []}
            meta = {"epoch": 0, "version": 0, "primary": reps[0], "copies": reps}
            acc = {}
            for i, h in enumerate(reps):
                role = "primary" if i == 0 else "backup"
                if h == self.name:
                    self._store(k, p, {**meta, "role": role}); acc[h] = True
                else:
                    r = await self.ask(h, REPLICATE, {"k": k, **meta, "role": role, "package": p})
                    acc[h] = None if r is None else r.get("applied")
            return OK, {"applied": acc.get(reps[0]) is True, "k": k, "copies": reps, "accepted": acc}
        if kind == REPLICATE:
            own = (self.meta[k]["epoch"], self.meta[k]["version"]) if k in self.meta else (-1, -1)
            new = (m["epoch"], m["version"])
            if m.get("drop"):
                ok = new >= own
                if ok:
                    self._forget(k)
                return OK, {"applied": ok}
            if new < own or (new == own and k in self.meta and self.meta[k]["role"] == m["role"]
                             and self.meta[k]["copies"] == m["copies"]):
                return OK, {"applied": new == own, "epoch": own[0], "version": own[1]}
            self._store(k, m["package"], {"role": m["role"], "epoch": m["epoch"], "version": m["version"],
                                          "primary": m["primary"], "copies": m["copies"]})
            return OK, {"applied": True}
        if kind == SYNC:
            return OK, {"state": [self.state(x) for x in m["ks"] if x in self.meta]}
        if kind == PULL:
            if k not in self.meta:
                return OK, {"package": None}
            return OK, {"package": self.pkg(k), "primary": self.meta[k]["primary"], **{x: self.meta[k][x] for x in ("epoch", "version")}}
        return OK, {"applied": False, "error": f"unknown kind {kind}"}


def _arr(v):
    import numpy as np
    return np.asarray(v, np.float32)


def placement(ks, machines: dict[str, str], R: int) -> dict[str, dict]:
    """Prep: per host, the copies it holds with their initial state → {host: {k: meta}}."""
    out = {h: {} for h in machines}
    for k in ks:
        reps = replicas(k, machines, R)
        for i, h in enumerate(reps):
            out[h][k] = {"role": "primary" if i == 0 else "backup", "epoch": 0, "version": 0, "primary": reps[0],
                         "copies": reps}
    return out


# ---- gateway side: no KU state; transport-level knowledge of which hosts failed to answer
from mxn.expert.hostindex import merge                              # noqa: E402
from mxn.expert.rgateway import HostsClient, ResolveGateway         # noqa: E402


class RepClient(HostsClient):
    """HostsClient that sends REJOIN to a host that failed to answer before, ahead of anything else, and remembers
    (transport level) the hosts it already sent a failover round for."""

    def __init__(self, hosts: dict[str, str], timeout: float = 10.0):
        super().__init__(hosts, timeout)
        self.failed_over: set[str] = set()
        self.trace: list | None = None                  # measurement only: [(host, kind, seconds, answered), ...]

    async def _request(self, host: str, kind: int, msg: dict) -> dict | None:
        t0 = time.perf_counter()
        r = await self._request_rejoin_first(host, kind, msg)
        if self.trace is not None:
            self.trace.append((host, kind, time.perf_counter() - t0, r is not None))
        return r

    async def _request_rejoin_first(self, host: str, kind: int, msg: dict) -> dict | None:
        if host in self.down and kind != REJOIN:
            r = await super()._request(host, REJOIN, {"rid": 0})
            if r is None:
                return None
            self.down.discard(host); self.failed_over.discard(host)
        return await super()._request(host, kind, msg)

    def manage(self, kind: int, **msg) -> dict | None:
        """Edits go to the KU's primary only: a read-only WHO finds the holders; the primary with the highest epoch."""
        hs = [r for r in self.broadcast(WHO, k=msg["k"]) if r.get("applied")]
        ps = sorted((r for r in hs if r.get("role") == "primary"), key=lambda r: -r["epoch"])
        if not ps:
            return None
        r = self.request(ps[0]["host"], kind, **msg)
        return r if r and r.get("applied") else None

    def edit_status(self, k, version: int) -> dict:
        """Lost-edit report (design §4.5): the version the primary confirmed vs what the current primary holds."""
        hs = [r for r in self.broadcast(WHO, k=k) if r.get("applied")]
        ps = sorted((r for r in hs if r.get("role") == "primary"), key=lambda r: -r["epoch"])
        now = ps[0]["version"] if ps else None
        return {"k": k, "confirmed_version": version, "current_version": now, "primary": ps[0]["host"] if ps else None,
                "lost": now is not None and now < version}


class RepGateway(ResolveGateway):
    """The compliant gateway with failover: a RESOLVE missing a host's reply is asked again of the live hosts with
    {"dead": [...]}, so the backups first in line promote themselves and answer. Degraded mode only when R or more
    hosts are missing (then some KU may have lost every copy). Two copies of one KU in one request: logged, higher
    epoch used (safety check; expected never)."""

    def __init__(self, *a, R: int = 2, **kw):
        super().__init__(*a, **kw)
        self.R = check_r(R)
        self.stats.update({"failover_rounds": 0, "promoted": 0, "dedupe_errors": 0})
        self.errors: list = []
        self.failover_timing: list = []                 # measurement only: one entry per failover round (no KU data)
        # connections (chapter 3 step 2): read-only band edges for c2; a clock for decay (experiments: traffic index)
        self.stats.update({"conn_proposed": 0, "conn_accepted": 0, "conn_conflicts": 0, "conn_check_failed": 0})
        self.conn_bands: list = [[0.0, "b0"]]
        self.exact_guard = True                         # a unique exact-title match overrides a connection (False: off)
        self.clock = time.time

    def band(self, score: float) -> str:
        return [name for lo, name in self.conn_bands if score >= lo][-1]

    def find_primary(self, k, relation: str) -> dict | None:
        """FIND at KU k's current primary (WHO first; no placement kept). None if no primary answers with a value."""
        hs = [r for r in self.client.broadcast(WHO, k=k) if r.get("applied") and r.get("role") == "primary"]
        for h in sorted(hs, key=lambda r: -r["epoch"])[:1]:
            r = self.client.request(h["host"], FIND, k=k, relation=relation)
            if r and r.get("value") and not r.get("not_primary"):
                return {"ku": k, "title": r["title"], "relation": relation, "value": r["value"], "p_ext": r["p_ext"],
                        "p_ku": 1.0, "host": h["host"]}
        return None

    def hop_find(self, src: dict, r1: str, r2: str, question: str) -> dict | None:
        """Hop (design_connections §6): VIA to the source KU; RESOLVE(value) as before. A proposed connection is used
        unless RESOLVE found a different unique exact-title match (logged conflict; only with exact_guard on) or its
        target does not answer r2 as primary (check failed: RESOLVE's result is used). Every hop record names its source and RESOLVE's
        choice (ids and rules only) for the verifier and the report."""
        value = src["value"]
        if not value:
            return None
        self.stats["via_asked"] += 1
        t0 = time.perf_counter()
        v = self.client.request(src["host"], VIA, k=src["ku"], relation=r1, value=value, t=self.clock())
        t1 = time.perf_counter()
        e = self.resolve(value, question)
        info = {"src": src["ku"], "src_relation": r1, "value_sha1": value_sha1(value),
                "resolve_k": None if e is None else e["k"], "resolve_rule": None if e is None else e["rule"],
                "conn": None, "conn_used": False, "via_s": t1 - t0, "resolve_s": time.perf_counter() - t1}
        tgt = v.get("target") if v else None
        if tgt:
            self.stats["via_answered"] += 1; self.stats["conn_proposed"] += 1
            info["conn"] = [tgt[0], tgt[1], tgt[3] if len(tgt) > 3 else 0.0]     # target, score, corrections
            differs = e is not None and e["rule"] == "exact" and e["k"] != tgt[0]
            if differs and self.exact_guard:
                self.stats["conn_conflicts"] += 1; info["conn_conflict"] = True
            else:
                t2 = time.perf_counter()
                f = self.find_primary(tgt[0], r2)
                info["check_s"] = time.perf_counter() - t2
                if differs:
                    info["conn_exact_differs"] = True               # guard off: recorded, not acted on
                if f is not None:
                    self.stats["conn_accepted"] += 1
                    f.update(info, resolved_by=f"connection {self.band(tgt[1])}", conn_used=True)
                    return f
                self.stats["conn_check_failed"] += 1; info["conn_check_failed"] = True
        if e is None:
            return None
        f = self.find(e, r2); f["resolved_by"] = e["rule"]; f.update(info)
        return f

    def resolve(self, span: str, question: str) -> dict | None:
        t0 = time.perf_counter()
        qs = self.enc.encode([span], "query")[0]; qq = self.enc.encode([question], "query")[0]
        msg = {"span": span, "qv_span": qs.tolist(), "qv_question": qq.tolist()}
        self.stats["resolve_broadcasts"] += 1
        t1 = time.perf_counter()
        replies, missing = self.client.broadcast_all(RESOLVE, **msg)
        t2 = time.perf_counter()
        new = [h for h in missing if h not in self.client.failed_over]
        if new:                                                 # second round: backups of the missing hosts take over
            self.stats["failover_rounds"] += 1
            self.client.failed_over |= set(missing)
            replies, missing2 = self.client.broadcast_all(RESOLVE, dead=sorted(missing), **msg)
            missing = sorted(set(missing) | set(missing2))
            self.stats["promoted"] += sum(len(r.get("promoted", [])) for r in replies)
            self.failover_timing.append({
                "encode_s": t1 - t0, "round1_s": t2 - t1, "round2_s": time.perf_counter() - t2, "missing": missing,
                "host_failover_s": {r["host"]: r.get("failover_s") for r in replies},
                "host_propose_s": {r["host"]: r.get("propose_s") for r in replies},
                "promoted": {r["host"]: len(r.get("promoted", [])) for r in replies}})
        replies = self.dedupe(replies)
        e = merge(replies)
        if missing:
            self.stats["resolve_missing_reply"] += 1
            if len(missing) >= self.R and self.degraded and e is not None and e["rule"] != "exact":
                d = self.stats["degraded_abstained"]; d[e["rule"]] = d.get(e["rule"], 0) + 1
                return None
        return e

    def dedupe(self, replies: list[dict]) -> list[dict]:
        seen = {}
        for r in replies:
            for k, ep in r.get("epochs", []):
                seen.setdefault(json.dumps(k), []).append((ep, r["host"]))
        bad = {k: v for k, v in seen.items() if len(v) > 1}
        if not bad:
            return replies
        self.stats["dedupe_errors"] += len(bad)
        self.errors.append({"duplicate_copies": bad})
        keep = {k: max(v)[1] for k, v in bad.items()}
        out = []
        for r in replies:
            r = dict(r)
            ok = lambda k: json.dumps(k) not in keep or keep[json.dumps(k)] == r["host"]
            r["exact"] = [c for c in r["exact"] if ok(c[0])]; r["nodis"] = [c for c in r["nodis"] if ok(c[0])]
            if r.get("fallback") and not ok(r["fallback"][0]):
                r["fallback"] = None
            out.append(r)
        return out


def main():
    """A replicated host process: python -m mxn.expert.replica --package P --name h0 --hostlist HL.json --R 2 ..."""
    import argparse
    import pickle

    from mxn.common.frames import serve

    ap = argparse.ArgumentParser()
    ap.add_argument("--package"); ap.add_argument("--name"); ap.add_argument("--hostlist"); ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--expert"); ap.add_argument("--extract", default="extract2"); ap.add_argument("--state", default=None)
    ap.add_argument("--tau-c", type=float, default=None); ap.add_argument("--half-life", type=float, default=None)
    a = ap.parse_args()
    check_r(a.R)
    p = pickle_load(a.package)                     # {"units", "title_vectors", "meta"}: this host's copies
    hostlist = json.load(open(a.hostlist))
    host = RepHost(p["units"], a.expert, a.name, a.extract, True, {k: p["title_vectors"][k] for k in p["units"]})
    host.state_dir = a.state
    host.tau_c, host.half_life = a.tau_c, a.half_life
    host.setup(hostlist, a.R, p["meta"], p["title_vectors"])

    async def run():
        srv = await serve(hostlist[a.name]["addr"], host.on_conn)
        n_p = sum(m["role"] == "primary" for m in host.meta.values())
        print(f"{a.name} ready on {hostlist[a.name]['addr']} with {len(host.units)} copies ({n_p} primary)", flush=True)
        async with srv:
            await srv.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
