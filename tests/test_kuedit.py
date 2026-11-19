from mxn.expert.kuedit import edit_ku, leading_name_match, mentions

TABLE = {"france": 1, "french": 1, "australia": 2, "australian": 2, "sweden": 3, "swedish": 3}


def test_every_mention_in_every_sentence():
    s = ["Robert Vernay (May 30, 1907 in Paris – October 17, 1979 in Paris) was a director.", "He lived in Paris."]
    out, n = edit_ku(s, "Paris", "Caracas")
    assert n == 3 and all("Paris" not in x for x in out) and out[1] == "He lived in Caracas."


def test_whole_words_only():
    s = ["Rod Hardy (born in Melbourne, Australia) is an Australian director.", "A Parisian café in Paris."]
    out, n = edit_ku(s, "Australia", "Sweden")
    assert n == 1 and "Australian director" in out[0] and "Swedenn" not in out[0]
    out, n = edit_ku(s, "Paris", "Rome")
    assert n == 1 and out[1] == "A Parisian café in Rome."


def test_country_class_forms_replaced():
    s = ["Michel Auder (born 1945 in Soissons, France) is a French photographer.", "He moved back to france."]
    out, n = edit_ku(s, "France", "Chile", "country of citizenship", TABLE)
    assert n == 3 and not any(w in x.lower() for x in out for w in ("france", "french"))
    assert mentions(out, "France", "country of citizenship", TABLE) == 0


def test_non_country_relation_ignores_table():
    s = ["She was born in France and is French."]
    out, n = edit_ku(s, "France", "Chile", "place of birth", TABLE)
    assert n == 1 and "French" in out[0]


def test_multiword_and_punctuation_values():
    s = ["He died of Huntington's disease in 1967.", "Huntington's disease runs in families."]
    out, n = edit_ku(s, "Huntington's disease", "poison")
    assert n == 2 and out[0] == "He died of poison in 1967."


def test_leading_name_part_is_segment_before_first_comma():
    assert leading_name_match("Kirksville", "Kirksville, Missouri")
    assert leading_name_match("kirksville", "Kirksville, Missouri")
    assert leading_name_match("Ernest I", "Ernest I, Prince of Anhalt-Dessau")
    assert leading_name_match("New York", "New York, New York")
    assert not leading_name_match("New", "New York")              # no comma: not a first-word match
    assert not leading_name_match("New", "New York, New York")    # partial segment does not count
    assert not leading_name_match("Missouri", "Kirksville, Missouri")
    assert not leading_name_match(None, "Kirksville, Missouri")
    assert not leading_name_match("", "Kirksville, Missouri")


def test_clean_edit_standalone_mentions():
    from mxn.expert.kuedit import edit_ku_clean
    out, n = edit_ku_clean(["He died in Paris in 1979.", "(1907 in Paris – 1979)"], {"Paris": "Lyon"}, "place of death")
    assert n == 2 and out == ["He died in Lyon in 1979.", "(1907 in Lyon – 1979)"]
    assert edit_ku_clean(["In Paris he worked."], {"Paris": "Lyon"}, "place of death")[1] == 1   # sentence start


def test_clean_edit_rejects_mentions_inside_larger_names():
    from mxn.expert.kuedit import edit_ku_clean
    for s, old, rel in [("He founded London Films.", "London", "place of death"),
                        ("He studied at the Conservatoire de Paris.", "Paris", "place of death"),
                        ("Born in New York City.", "New York", "place of birth"),
                        ("He died in the Battle of Göllheim.", "Göllheim", "place of death"),
                        ("An Italian-American director.", "American", "country of citizenship"),
                        ("Born in Cedofeita, Porto.", "Porto", "place of birth"),
                        ("He was born in Soissons, France.", "France", "country of citizenship"),
                        ("She ran London's Southwark Playhouse.", "London", "place of death"),
                        ("Son of the Hongwu Emperor.", "Hongwu", "father"),
                        ("Daughter of John II, Count of Holland.", "John II", "father"),
                        ("Her husband was Giovanni Matteo de Candia.", "Giovanni Matteo", "spouse")]:
        assert edit_ku_clean([s], {old: "X"}, rel) is None, s


def test_clean_edit_replaces_region_with_place():
    from mxn.expert.kuedit import edit_ku_clean
    out, n = edit_ku_clean(["He was born in Shreveport, Louisiana.", "Shreveport is a city."],
                           {"Shreveport": "Hawaii"}, "place of birth")
    assert n == 2 and out == ["He was born in Hawaii.", "Hawaii is a city."]
    out, _ = edit_ku_clean(["(1889, Philadelphia – June 24, 1984, New York, New York)"], {"New York": "Italy"}, "place of death")
    assert out == ["(1889, Philadelphia – June 24, 1984, Italy)"]


def test_clean_edit_country_adjective_for_adjective():
    from mxn.expert.kuedit import edit_ku_clean
    forms = {"Austria": "Argentina", "Austrian": "Argentine"}
    out, n = edit_ku_clean(["Leo Mittler was an Austrian playwright.", "He left Austria in 1938."], forms,
                           "country of citizenship")
    assert n == 2 and out == ["Leo Mittler was an Argentine playwright.", "He left Argentina in 1938."]
