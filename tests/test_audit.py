"""Process audit (mxn/common/audit.py) and the KU store (mxn/ku/store.py)."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mxn.common.audit import audit_process, ku_content_hashes
from mxn.ku.store import KU, load_store, write_store

EX = Path(__file__).resolve().parents[1] / "artifacts" / "expert"


def test_audit_clean_process_is_clean():                              # known-good
    a = audit_process({hashlib.sha1(b"a sentence nobody in this process holds, long enough").hexdigest()})
    assert a == {"ku_objects": [], "ku_object_count": 0, "ku_sentences_found": 0}


def test_audit_detects_ku_sentences():                                # known-bad: a KU sentence held
    s = "Robert Vernay was a French film director and screenwriter."
    hold = [s]
    assert audit_process({hashlib.sha1(s.encode()).hexdigest()})["ku_sentences_found"] == 1
    del hold


def test_audit_detects_ku_objects():                                  # known-bad: a KU held
    u = KU("w0", "Some Title", ["One sentence of this unit."], np.ones((1, 4), np.float32))
    a = audit_process(set())
    assert a["ku_objects"] == ["KU"] and a["ku_object_count"] == 1, a
    del u


def test_title_only_sentence_counts_as_ku_content():
    """The title amendment is revoked: a sentence equal to its KU's title is KU content for the audit."""
    kus = [("Princess Amelia of Great Britain", ["Princess Amelia of Great Britain",
                                                 "Princess Amelia was a daughter of King George II."])]
    hs = ku_content_hashes(kus)
    del kus
    assert len(hs) == 2
    held = ["Princess Amelia of Great Britain"]
    assert audit_process(hs)["ku_sentences_found"] == 1
    del held


def test_store_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    kus = [KU(f"w{i}", f"Title {i}", [f"Sentence {j} of unit {i}." for j in range(i + 1)],
              rng.normal(size=(i + 1, 8)).astype(np.float16).astype(np.float32)) for i in range(3)]
    write_store(tmp_path / "net", kus, "ours/x")
    back = load_store(tmp_path / "net", "ours/x")
    assert [(u.id, u.title, u.sentences) for u in back] == [(u.id, u.title, u.sentences) for u in kus]
    assert all(np.array_equal(a.chunk_vecs, b.chunk_vecs) for a, b in zip(back, kus))
    assert sorted(p.name for p in (tmp_path / "net").iterdir()) == ["chunks.npy", "meta.json", "units.jsonl"]
    with pytest.raises(ValueError):
        load_store(tmp_path / "net", "ours/other")


@pytest.mark.skipif(not (EX / "kus" / "net" / "questions.jsonl").exists(), reason="needs artifacts/expert")
def test_old_store_with_calibration_loads():
    """A store built before the retirement (with questions.jsonl etc.) loads; the calibration field is not read."""
    kus = load_store(EX / "kus" / "net", "ours/enc16k-s2030")
    assert len(kus) == 14128 and not hasattr(kus[0], "calibration")


@pytest.mark.skipif(not (EX / "tok16k.model").exists(), reason="needs artifacts/expert")
def test_gateway_process_holds_no_ku_content(tmp_path):
    """The compliant gateway (rgateway) loaded in a fresh process holds no KU objects or sentences."""
    P = pickle.load(open(EX / "pool_test.pkl", "rb"))
    json.dump(sorted(ku_content_hashes([(P["titles"][o], [s]) for s, o in zip(P["sentences"], P["owner"])])),
              open(tmp_path / "sha1.json", "w"))
    code = ("import json\n"
            "from mxn.common.audit import audit_process\n"
            "from mxn.expert.rgateway import load_resolve_gateway\n"
            f"G = load_resolve_gateway({str(EX)!r})\n"
            f"print(json.dumps(audit_process(set(json.load(open({str(tmp_path / 'sha1.json')!r}))))))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert r.returncode == 0, r.stderr[-2000:]
    a = json.loads(r.stdout.strip().splitlines()[-1])
    assert a == {"ku_objects": [], "ku_object_count": 0, "ku_sentences_found": 0}, a
