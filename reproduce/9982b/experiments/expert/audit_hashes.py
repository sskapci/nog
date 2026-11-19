"""Hash sets for the two audits (mxn/expert/audits.py), made outside the audited processes from the hosts' packages.

usage: audit_hashes.py EXPERT_DIR [POOL] → EXPERT_DIR/audit/{test,mdev,fresh,vktest}_{content,title}_sha1.json (fresh, vktest: when packages exist)
"""
import json, pickle, sys
from pathlib import Path

from mxn.expert.audits import content_hashes, title_hashes

ex = Path(sys.argv[1]); (ex / "audit").mkdir(exist_ok=True)
for pool, pkgs in (("test", sorted((ex / "replication").glob("package_h8_R2_h*.pkl"))),
                   ("mdev", sorted((ex / "connections" / "mdev").glob("package_h*.pkl"))),
                   ("fresh", sorted((ex / "connections" / "fresh").glob("package_h*.pkl"))),
                   ("vktest", sorted((ex / "connections" / "vktest").glob("package_h*.pkl")))):
    if not pkgs or (len(sys.argv) > 2 and pool != sys.argv[2]):     # optional POOL: only that pool's sets
        continue
    units = {}
    for p in pkgs:
        units.update(pickle.load(open(p, "rb"))["units"])
    kus = list(units.values())
    c, t = content_hashes(kus), title_hashes(kus)
    json.dump(sorted(c), open(ex / "audit" / f"{pool}_content_sha1.json", "w"))
    json.dump(sorted(t), open(ex / "audit" / f"{pool}_title_sha1.json", "w"))
    print(pool, {"packages": len(pkgs), "kus": len(kus), "content_lines": len(c), "title_names": len(t)})
