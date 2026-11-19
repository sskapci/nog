"""The two audits (mxn/expert/audits.py): content audit ignores name lines; index audit finds each index kind."""
import numpy as np

from mxn.expert.audits import audit_content, audit_index, content_hashes, sha1, title_hashes

KUS = [("Alpha Beta Gamma Delta (film)", ["Alpha Beta Gamma Delta", "Alpha Beta Gamma Delta is a 1999 film by someone."]),
       ("Some Person Name Here", ["Some Person Name Here was born in 1900 in a small town."])] + \
      [(f"Title number {i:05d} of the pool", [f"Sentence {i} of the pool, long enough to count."]) for i in range(1200)]


class Client:
    def __init__(self):
        self.hosts = {"h0": "tcp:127.0.0.1:1", "h1": "tcp:127.0.0.1:2"}


class Gw:
    def __init__(self):
        self.client, self.equiv, self.stats = Client(), {"Some Person Name Here": ["x"]}, {}


def test_content_audit_counts_sentences_not_names():
    name = lambda: " ".join(["Alpha", "Beta", "Gamma", "Delta"])          # built at run time: no constant holds them
    sent = lambda: " ".join(["Alpha", "Beta", "Gamma", "Delta", "is", "a", "1999", "film."])
    c = content_hashes([(name() + " (film)", [name(), sent()])])
    assert sha1(name()) not in c and sha1(sent()) in c
    keep = [name()]                                                       # an answer / edit value equal to a name
    assert audit_content(c)["clean"]
    keep.append(sent())                                                   # a real KU line
    a = audit_content(c)
    assert not a["clean"] and a["ku_lines_found"] == 1 and keep


def test_index_audit_finds_each_index_kind_and_skips_the_country_table():
    t = title_hashes(KUS)
    g = Gw()
    a = audit_index(g, t)
    assert a["clean"] and a["country_table_title_strings_skipped"] == 1
    for name, val, field in (("x1", ["Some Person Name Here"], "titles_found"),
                             ("x2", np.zeros((500, 256), np.float32), "title_vector_matrices"),
                             ("x3", {f"key {i}": [i] for i in range(1500)}, "key_tables"),
                             ("x5", {("key", str(i)): [i] for i in range(1500)}, "key_tables"),
                             ("x4", {i: ["h0", "h1"][i % 2] for i in range(1500)}, "placement_tables"),
                             ("titles", [], "banned_attributes")):
        g = Gw(); setattr(g, name, val)
        a = audit_index(g, t)
        assert not a["clean"] and a[field], (name, a)
