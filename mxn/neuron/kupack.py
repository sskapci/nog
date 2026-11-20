"""KU package encoding without pickle (security fix 2026-11-20, owner order "Pickle fix NOW").

Every KU package that leaves a host process (MOVE / COLDPUT messages, cold copies and training checkpoints on disk)
used to be `torch.save` / `torch.load(weights_only=False)`, i.e. Python pickle: a crafted package executed code on
the receiving host. This module replaces it with a format that cannot execute anything:

    MAGIC (8 bytes "NOGKU1\\n\\0") | u64 header length | header (UTF-8 JSON) | safetensors bytes (all tensors)

The header holds every non-tensor field as plain JSON (validated against a fixed schema on load); tensors are
stored by safetensors (raw arrays + a JSON index; no code). Loading rejects: a wrong magic, unknown or missing
fields, wrong types, tensor names not declared in the header, non-float32 state tensors, and sizes beyond the caps.

API: dumps(packages, extra=None) -> bytes; loads(data) -> (packages, extra). `extra` is a small JSON dict (the
checkpoint tag). Packages are the dicts of mxn.neuron.nku.package(); conns keys (slot indices) and slot occurrence
tuples are restored to int / tuple so a round trip returns equal packages.
"""

from __future__ import annotations

import json
import struct

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

MAGIC = b"NOGKU1\n\0"
MAX_BYTES = 1 << 31                  # 2 GiB per blob
MAX_HEADER = 1 << 28
MAX_PACKAGES = 100_000
FIELDS = {"k": int, "title": str, "slots": list, "hidden": int, "d_query": int, "conns": dict, "rich": bool,
          "stats": dict, "kind": bool, "key": (int, type(None)), "vocab": (str, type(None))}
REQUIRED = {"k"}                      # minimal packages (k, title) occur in tests; unpackage() needs the rest
SLOT_FIELDS = {"label": str, "kind": str, "occ": list, "pointer": (int, type(None))}


class PackageError(ValueError):
    """A blob that is not a valid KU package: refused, nothing loaded."""


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x)
    if hasattr(x, "item") and getattr(x, "ndim", 1) == 0:            # numpy / torch scalars in stats
        return x.item()
    raise TypeError(f"not storable in a KU package header: {type(x).__name__}")


def dumps(packages: list[dict], extra: dict | None = None) -> bytes:
    heads, tensors = [], {}
    for i, p in enumerate(packages):
        unknown = set(p) - set(FIELDS) - {"state"}
        if unknown:
            raise TypeError(f"package fields not in the schema: {sorted(unknown)}")
        h = {f: _json_safe(v) for f, v in p.items() if f != "state"}
        if "conns" in p:
            h["conns"] = {str(int(k)): _json_safe(v) for k, v in p["conns"].items()}
        if "state" in p:
            names = []
            for n, t in p["state"].items():
                tensors[f"{i}/{n}"] = t.detach().contiguous().cpu()
                names.append(n)
            h["state"] = names
        heads.append(h)
    hb = json.dumps({"packages": heads, "extra": _json_safe(extra or {})}).encode()
    body = st_save(tensors) if tensors else b""
    return MAGIC + struct.pack("<Q", len(hb)) + hb + body


def _check(cond: bool, why: str) -> None:
    if not cond:
        raise PackageError(why)


def _typed(v, t, what: str) -> None:
    _check(isinstance(v, t) and not (t is int and isinstance(v, bool)), f"{what}: wrong type {type(v).__name__}")


def loads(data: bytes) -> tuple[list[dict], dict]:
    _check(isinstance(data, (bytes, bytearray)), "not bytes")
    _check(len(data) <= MAX_BYTES, "too large")
    _check(data[:8] == MAGIC, "not a KU package (bad magic; pickle and torch.save blobs are refused)")
    _check(len(data) >= 16, "truncated")
    (n,) = struct.unpack("<Q", data[8:16])
    _check(n <= MAX_HEADER and 16 + n <= len(data), "bad header length")
    try:
        head = json.loads(data[16:16 + n].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PackageError(f"header is not JSON: {e}") from None
    _check(isinstance(head, dict) and set(head) == {"packages", "extra"}, "bad header")
    _check(isinstance(head["packages"], list) and len(head["packages"]) <= MAX_PACKAGES, "bad package list")
    _check(isinstance(head["extra"], dict), "bad extra")
    body = bytes(data[16 + n:])
    try:
        tensors = st_load(body) if body else {}
    except Exception as e:                                             # safetensors parse error
        raise PackageError(f"tensor block invalid: {e}") from None
    used, out = set(), []
    for i, h in enumerate(head["packages"]):
        _check(isinstance(h, dict), "package is not an object")
        unknown = set(h) - set(FIELDS) - {"state"}
        _check(not unknown, f"unknown package fields {sorted(unknown)}")
        _check(REQUIRED <= set(h), "package without k")
        p = {}
        for f, v in h.items():
            if f == "state":
                continue
            _typed(v, FIELDS[f], f)
            p[f] = v
        if "slots" in p:
            slots = []
            for s in p["slots"]:
                _check(isinstance(s, dict) and set(s) == set(SLOT_FIELDS), "bad slot")
                for f, t in SLOT_FIELDS.items():
                    _typed(s[f], t, f"slot.{f}")
                _check(all(isinstance(o, list) and len(o) == 3 and all(isinstance(x, int) for x in o) for o in s["occ"]),
                       "bad slot occurrence")
                slots.append({**s, "occ": [tuple(o) for o in s["occ"]]})
            p["slots"] = slots
        if "conns" in p:
            conns = {}
            for k, c in p["conns"].items():
                _check(k.lstrip("-").isdigit(), "bad conns key")
                _check(isinstance(c, dict) and {"target", "rels", "weight"} <= set(c) <= {"target", "rels", "weight", "tkey"},
                       "bad conn")
                _typed(c["target"], (int, type(None)), "conn.target"); _typed(c["weight"], int, "conn.weight")
                if "tkey" in c:
                    _typed(c["tkey"], int, "conn.tkey")
                _check(isinstance(c["rels"], dict) and all(isinstance(v, int) for v in c["rels"].values()), "bad rels")
                conns[int(k)] = c
            p["conns"] = conns
        if "state" in h:
            _check(isinstance(h["state"], list) and all(isinstance(x, str) for x in h["state"]), "bad state names")
            st = {}
            for name in h["state"]:
                key = f"{i}/{name}"
                _check(key in tensors, f"missing tensor {key}")
                _check(tensors[key].dtype == torch.float32, f"tensor {key} is not float32")
                st[name] = tensors[key]; used.add(key)
            p["state"] = st
        out.append(p)
    _check(used == set(tensors), "tensors not declared in the header")
    return out, head["extra"]
