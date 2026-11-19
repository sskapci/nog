"""Our expert's KUs as host processes (step 9: SCALABLE, UPDATABLE): each host holds some KUs (title, sentences) and
a read-only copy of the extractor; the gateway sends FIND(KU, relation) to the host holding that KU.

Frames (mxn/common/frames.py: u32 length, u8 kind, payload), JSON payloads:
  FIND   gateway → host: {"rid", "k", "relation"}       FOUND  host → gateway: {"rid", "value", "p_ext", "host"}
  EDIT   gateway → host: {"rid", "k", "sentences"}      (replace a KU's sentences)
  ADD    gateway → host: {"rid", "k", "title", "sentences"}   (a new KU)
  DROP   gateway → host: {"rid", "k"}                   (detach a KU)
  OK     host → gateway: {"rid"}
  EDITV  gateway → host: {"rid", "k", "relation", "old", "new", "mode", "forms"?, "sent"?}   the owning host applies
         the edit to its own copy with kuedit (mode "v2": edit_ku with the country table; "clean": edit_ku_clean with
         forms; "v1": first mention in sentence `sent`) → {"rid", "replaced", "sha1"} or {"rid", "error"}. The host
         keeps no vectors or index of its KUs (FIND reads the current sentences), so nothing else is rebuilt.
  DETACH / ATTACH gateway → host: {"rid", "k"}         hide / restore a KU (the host keeps its copy)
  Locating the owner (compliant gateway, no placement map): EDIT, EDITV, DROP, DETACH, ATTACH and MOVE are broadcast to
         every host; only the host that holds the KU acts, and every host replies {"rid", "host", "applied": bool}, so
         the sender learns the owner from the one "applied": true reply. ADD goes to a host the sender chooses; without
         "k" the host names the KU "<host name>:<n>" (unique as long as host names are unique; the id does not change
         when the KU moves, so it is a name, not an address).
  MOVE   gateway → every host: {"rid", "k", "to", "to_name"} (the destination host ignores it)   the owner sends the KU's package (title, sentences, title vector,
         attached or not, connections) straight to the destination host (IMPORT) and drops its copy only after the
         destination's OK; the package never passes through the gateway.
  IMPORT host → host: {"rid", "package"}
  WHO    gateway → every host: {"rid", "k"} → {"applied": held, "sha1", "detached", "imported_from"}: finds every holder
         of a KU, to detect and repair a duplicate left by a move whose source died after the destination's OK.
  ANNOUNCE gateway → host: {"rid"}                    TITLES host → gateway: {"rid", "host", "kus": [[k, title], ...]}
         (names only: KU id and title of every KU the host holds; no sentences, values or confidences)
  RESOLVE gateway → every host: {"rid", "span", "qv_span", "qv_question"}   RESOLVED host → gateway: {"rid", "host",
         "exact", "nodis", "fallback"} — the host's title index (a cache over its own KUs, mxn/expert/hostindex.py)
         proposes candidates and each KU scores itself; the gateway merges (hostindex.merge).
  VIA    gateway → the source KU's host: {"rid", "k", "relation", "value"}   VIAR host → gateway: {"rid", "host",
         "target": [k, title] | null} — the source KU answers a hop from its own outgoing connections (none are kept
         yet: always null; connection scores are chapter 3), else the gateway broadcasts RESOLVE.
A host that cannot be reached or does not answer within `timeout` seconds counts as down for that request: the FIND
returns an empty value with p_ext 0 (the question then abstains), and nothing hangs.
Security: frames are not authenticated or encrypted. Use only on one machine or inside a trusted LAN, with the
hosts' ports open between the machines and closed to everything else.

    python -m mxn.expert.fhost --shard SHARD.pkl --addr tcp:127.0.0.1:7100 --name h0 --expert artifacts/expert
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import threading

from mxn.common.frames import frame, open_conn, read_frame, serve

FIND, FOUND, EDIT, ADD, DROP, OK, ANNOUNCE, TITLES, EDITV, DETACH, ATTACH = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
RESOLVE, RESOLVED, VIA, VIAR, MOVE, IMPORT, WHO = 12, 13, 14, 15, 16, 17, 18


class FindHost:
    index = None                                                  # the host's title index (set with index=True)
    connections: dict = {}                                        # replaced per instance in __init__

    def __init__(self, units: dict, expert_dir: str, name: str, extract_dir: str = "extract2", index: bool = False,
                 title_vectors: dict | None = None):
        import sentencepiece as spm
        import torch

        from mxn.expert.extractor import Extractor

        torch.set_num_threads(1)
        self.units, self.name = units, name                      # {k: (title, sentences)}
        self.detached, self.expert_dir = {}, expert_dir
        self.sp = spm.SentencePieceProcessor(model_file=f"{expert_dir}/tok16k.model")
        ck = torch.load(f"{expert_dir}/{extract_dir}/extractor.pt", map_location="cpu", weights_only=False)
        self.net = Extractor(ck["trunk"]); self.net.load_state_dict(ck["state"]); self.net.eval()
        self.t = json.load(open(f"{expert_dir}/{extract_dir}/threshold.json"))["t"]
        self.connections: dict[int, list] = {}                    # per source KU: outgoing connections (chapter 3)
        self.index = None
        if index:                                                 # the host's title index over its own KUs
            from mxn.expert.encoder import Encoder
            from mxn.expert.hostindex import HostTitleIndex

            self.enc = Encoder(f"{expert_dir}/enc", f"{expert_dir}/tok16k.model")    # for KUs added later
            if title_vectors is not None:                         # each KU's own title vector, from its package
                import numpy as np

                from mxn.expert.hostindex import KUTitle
                self.index = HostTitleIndex([KUTitle(k, t, np.asarray(title_vectors[k], np.float32))
                                             for k, (t, _) in sorted(self.units.items())])
            else:
                self.index = HostTitleIndex.from_units(
                    self.units, self.enc, lambda d, n: print(f"{name}: title index {d}/{n}", flush=True))

    def _ku_title(self, k: int, title: str):
        from mxn.expert.hostindex import KUTitle, title_vector

        return KUTitle(k, title, title_vector(self.enc, title))

    def find(self, k: int, relation: str) -> tuple[str, float]:
        from mxn.expert.extractor import dates_rule, decode, piece_scores

        title, sents = self.units[k]
        v = dates_rule(relation, sents)
        if v is not None:
            return v, 1.0
        return decode(piece_scores(self.net, self.sp, relation, title, sents), sents, self.t)

    def edit_value(self, m: dict) -> dict:
        import hashlib

        from mxn.expert.kuedit import edit_ku, edit_ku_clean

        title, sents = self.units[m["k"]]
        if m["mode"] == "v1":
            new = list(sents); new[m["sent"]] = new[m["sent"]].replace(m["old"], m["new"], 1); n = 1
        elif m["mode"] == "v2":
            if not hasattr(self, "equiv"):
                self.equiv = json.load(open(f"{self.expert_dir}/country_table.json"))["table"]
            new, n = edit_ku(list(sents), m["old"], m["new"], m["relation"], self.equiv)
        elif m["mode"] == "clean":
            r = edit_ku_clean(list(sents), m["forms"], m["relation"])
            if r is None:
                return {"error": "not standalone"}
            new, n = r
        else:
            return {"error": f"unknown mode {m['mode']}"}
        self.units[m["k"]] = (title, new)
        return {"replaced": n, "sha1": hashlib.sha1("\n".join(new).encode()).hexdigest()}

    def new_id(self) -> str:
        """Next host-named KU id "<host>:<n>". The counter continues from the highest n this host holds (attached or
        detached) and from the persisted counter (<state_dir>/<host>.counter, written before the id is used), so ids
        are never reused, also after a restart or after the KU with the highest n was dropped or moved away."""
        import os
        import re

        if getattr(self, "next_local", None) is None:
            pat = re.compile(rf"^{re.escape(self.name)}:(\d+)$")
            held = [int(mm.group(1)) for k in list(self.units) + list(self.detached)
                    if isinstance(k, str) and (mm := pat.match(k))]
            n = max(held, default=0)
            sd = getattr(self, "state_dir", None)
            if sd and os.path.exists(f"{sd}/{self.name}.counter"):
                n = max(n, int(open(f"{sd}/{self.name}.counter").read().strip() or 0))
            self.next_local = n
        self.next_local += 1
        sd = getattr(self, "state_dir", None)
        if sd:
            os.makedirs(sd, exist_ok=True)
            tmp = f"{sd}/{self.name}.counter.tmp"
            with open(tmp, "w") as f:
                f.write(str(self.next_local)); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, f"{sd}/{self.name}.counter")
        return f"{self.name}:{self.next_local}"

    def package(self, k) -> dict:
        """Everything the KU owns, to move it: title, sentences, its title vector, attached or not, connections."""
        detached = k in self.detached
        title, sents = (self.detached if detached else self.units)[k]
        vec = None
        if self.index is not None:
            u = (self.index.off if detached else self.index.kus).get(k)
            vec = None if u is None else [float(x) for x in u.vec]
        return {"k": k, "title": title, "sentences": list(sents), "vec": vec, "detached": detached,
                "connections": list(self.connections.get(k, [])), "from": self.name}

    async def move_out(self, k, to: str) -> dict | None:
        """Host to host: send the KU's package to the destination (IMPORT), and only after its OK drop the local copy.
        The package never passes through the gateway."""
        reader, writer = await open_conn(to)
        try:
            writer.write(frame(IMPORT, json.dumps({"rid": 0, "package": self.package(k)}).encode())); await writer.drain()
            _, payload = await read_frame(reader)
            if not json.loads(payload).get("applied"):
                return None
        finally:
            writer.close()
        self.units.pop(k, None); self.detached.pop(k, None); self.connections.pop(k, None)
        if self.index is not None:
            self.index.remove(k)
        return {"moved_to": to}

    def move_in(self, p: dict) -> None:
        import numpy as np

        from mxn.expert.hostindex import KUTitle

        k = p["k"]
        (self.detached if p["detached"] else self.units)[k] = (p["title"], p["sentences"])
        if not hasattr(self, "imported"):
            self.imported = {}
        self.imported[k] = p.get("from")                          # where it came from (duplicate repair after a crash)
        if p["connections"]:
            if not isinstance(self.connections, dict) or self.connections is FindHost.connections:
                self.connections = {}
            self.connections[k] = p["connections"]
        if self.index is not None:
            u = KUTitle(k, p["title"], np.asarray(p["vec"], np.float32)) if p["vec"] is not None else self._ku_title(k, p["title"])
            self.index.add(u)
            if p["detached"]:
                self.index.detach(k)

    async def on_conn(self, reader, writer):
        try:
            while True:
                kind, payload = await read_frame(reader)
                m = json.loads(payload)
                if kind == FIND:
                    if m["k"] in self.units:
                        v, p = self.find(m["k"], m["relation"])
                    else:
                        v, p = "", 0.0
                    writer.write(frame(FOUND, json.dumps({"rid": m["rid"], "value": v, "p_ext": p, "host": self.name}).encode()))
                elif kind == EDIT:
                    ok = m["k"] in self.units
                    if ok:
                        self.units[m["k"]] = (self.units[m["k"]][0], m["sentences"])
                    writer.write(frame(OK, json.dumps({"rid": m["rid"], "host": self.name, "applied": ok}).encode()))
                elif kind == ADD:
                    k = m.get("k")
                    if k is None:                                 # the host names the new KU: "<host>:<n>", unique
                        k = self.new_id()
                    self.units[k] = (m["title"], m["sentences"])
                    if self.index is not None:
                        self.index.add(self._ku_title(k, m["title"]))
                    writer.write(frame(OK, json.dumps({"rid": m["rid"], "host": self.name, "applied": True, "k": k}).encode()))
                elif kind == DROP:
                    ok = m["k"] in self.units or m["k"] in self.detached
                    self.units.pop(m["k"], None); self.detached.pop(m["k"], None)
                    self.connections.pop(m["k"], None) if isinstance(self.connections, dict) else None
                    if ok and self.index is not None:
                        self.index.remove(m["k"])
                    writer.write(frame(OK, json.dumps({"rid": m["rid"], "host": self.name, "applied": ok}).encode()))
                elif kind == EDITV:
                    r = self.edit_value(m) if m["k"] in self.units else {"error": "unknown KU"}
                    writer.write(frame(OK, json.dumps({"rid": m["rid"], "host": self.name,
                                                       "applied": m["k"] in self.units and "error" not in r, **r}).encode()))
                elif kind == DETACH:
                    ok = m["k"] in self.units
                    if ok:
                        self.detached[m["k"]] = self.units.pop(m["k"])
                        if self.index is not None:
                            self.index.detach(m["k"])
                    writer.write(frame(OK, json.dumps({"rid": m["rid"], "host": self.name, "applied": ok}).encode()))
                elif kind == ATTACH:
                    ok = m["k"] in self.detached
                    if ok:
                        self.units[m["k"]] = self.detached.pop(m["k"])
                        if self.index is not None:
                            self.index.attach(m["k"])
                    writer.write(frame(OK, json.dumps({"rid": m["rid"], "host": self.name, "applied": ok}).encode()))
                elif kind == MOVE:
                    held = m["k"] in self.units or m["k"] in self.detached
                    # the destination may already hold the KU when its own copy of the broadcast MOVE arrives: never
                    # act on a MOVE that names this host as the destination
                    r = await self.move_out(m["k"], m["to"]) if held and m.get("to_name") != self.name else None
                    writer.write(frame(OK, json.dumps({"rid": m["rid"], "host": self.name, "applied": r is not None,
                                                       **(r or {})}).encode()))
                elif kind == WHO:                                 # does this host hold the KU? (duplicate check)
                    import hashlib

                    held = m["k"] in self.units or m["k"] in self.detached
                    r = {"rid": m["rid"], "host": self.name, "applied": held}
                    if held:
                        t, ss = (self.units if m["k"] in self.units else self.detached)[m["k"]]
                        r.update({"sha1": hashlib.sha1("\n".join([t, *ss]).encode()).hexdigest(),
                                  "detached": m["k"] in self.detached,
                                  "imported_from": getattr(self, "imported", {}).get(m["k"])})
                    writer.write(frame(OK, json.dumps(r).encode()))
                elif kind == IMPORT:
                    self.move_in(m["package"])
                    writer.write(frame(OK, json.dumps({"rid": m["rid"], "host": self.name, "applied": True}).encode()))
                elif kind == RESOLVE:
                    import numpy as np

                    r = self.index.propose(m["span"], np.asarray(m["qv_span"], np.float32),
                                           np.asarray(m["qv_question"], np.float32))
                    writer.write(frame(RESOLVED, json.dumps({"rid": m["rid"], "host": self.name, **r}).encode()))
                elif kind == VIA:
                    target = None                                 # no connection memory yet (chapter 3)
                    for c in self.connections.get(m["k"], []):
                        if c["relation"] == m["relation"]:
                            target = c["target"]; break
                    writer.write(frame(VIAR, json.dumps({"rid": m["rid"], "host": self.name, "target": target}).encode()))
                elif kind == ANNOUNCE:
                    from mxn.expert.hostindex import id_key
                    kus = [[k, self.units[k][0]] for k in sorted(self.units, key=id_key)]
                    writer.write(frame(TITLES, json.dumps({"rid": m["rid"], "host": self.name, "kus": kus}).encode()))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()


class FindClient:
    """Synchronous client for the pipeline: one connection per host, requests answered in order per host."""

    def __init__(self, hosts: dict[str, str], placement: dict[int, str], timeout: float = 10.0):
        self.hosts, self.placement, self.timeout = dict(hosts), dict(placement), timeout
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.conns, self.rid, self.down = {}, 0, set()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    async def _request(self, host: str, kind: int, msg: dict) -> dict | None:
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

    def send(self, k: int, kind: int, **msg) -> dict | None:
        self.rid += 1
        return self._run(self._request(self.placement[k], kind, {"rid": self.rid, "k": k, **msg}))

    def announce(self, host: str) -> dict | None:
        """The host's TITLES answer ({"host", "kus": [[k, title], ...]}), or None if it is down."""
        self.rid += 1
        return self._run(self._request(host, ANNOUNCE, {"rid": self.rid}))

    def find(self, k: int, relation: str) -> tuple[str, float]:
        r = self.send(k, FIND, relation=relation)
        return ("", 0.0) if r is None else (r["value"], r["p_ext"])

    def close(self) -> None:
        async def shut():
            for _, w in self.conns.values():
                w.close()
        self._run(shut())
        self.loop.call_soon_threadsafe(self.loop.stop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard"); ap.add_argument("--addr"); ap.add_argument("--name"); ap.add_argument("--expert")
    ap.add_argument("--extract", default="extract2"); ap.add_argument("--index", action="store_true")
    ap.add_argument("--state", default=None, help="directory for the host's persisted id counter")
    a = ap.parse_args()
    shard = pickle.load(open(a.shard, "rb"))
    if "units" in shard and "title_vectors" in shard:          # KU packages: sentences + each KU's own title vector
        host = FindHost(shard["units"], a.expert, a.name, a.extract, a.index, shard["title_vectors"])
    else:
        host = FindHost(shard, a.expert, a.name, a.extract, a.index)
    host.state_dir = a.state

    async def run():
        srv = await serve(a.addr, host.on_conn)
        print(f"{a.name} ready on {a.addr} with {len(host.units)} KUs", flush=True)
        async with srv:
            await srv.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
