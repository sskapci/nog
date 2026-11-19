"""Process audit: what KU content a process (gateway or test driver) holds.

Moved from mxn/expert/kugateway.py (cleanup 2026-10-30). The banned classes are the KU-holding classes of the
current system: KU (mxn/ku; it replaces the walk's KnowledgeUnit / Network / UnitCalibration, which the current
system no longer imports; KU calibration is retired) and the extractors. Sentences are found by sha1 as before.
"""

from __future__ import annotations


def ku_content_hashes(kus) -> set[str]:
    """sha1 of every KU sentence, for audit_process. kus: [(title, sentences)]. Every sentence counts, a sentence
    equal to the KU's title included (the 2026-10-28 title amendment is revoked: titles belong in the host indexes,
    not in the gateway)."""
    import hashlib

    return {hashlib.sha1(s.encode()).hexdigest() for _, ss in kus for s in ss}


def audit_process(sentence_hashes: set[str]) -> dict:
    """What KU content this process holds: instances of KU or extractor classes, and KU sentences (by sha1, from a
    sample made outside this process) inside any tracked container.
    Empty lists mean clean."""
    import gc
    import os
    import sys
    import hashlib

    from mxn.expert.extractor import Extractor, ReadOnlyExtractor
    from mxn.ku.store import KU

    gc.collect()
    banned = (KU, Extractor, ReadOnlyExtractor)
    objs, sents = [], set()
    for o in gc.get_objects():
        if isinstance(o, banned):
            objs.append(type(o).__name__)
        elif isinstance(o, (list, tuple, set, frozenset, dict)):
            for x in (o.values() if isinstance(o, dict) else o):
                if isinstance(x, str) and len(x) > 20 and hashlib.sha1(x.encode()).hexdigest() in sentence_hashes:
                    sents.add(hashlib.sha1(x.encode()).hexdigest())
                    if os.environ.get("AUDIT_SHOW"):                # diagnosis only: where a found sentence sits
                        print("AUDIT_FOUND", type(o).__name__, len(o), repr(x[:90]), file=sys.stderr)
    return {"ku_objects": sorted(set(objs)), "ku_object_count": len(objs), "ku_sentences_found": len(sents)}
