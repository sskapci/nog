from mxn.expert.ops import answer, compare, normalize, parse_date, parse_number


def test_parse_date_formats():
    assert parse_date("13 June 1946") == (1946, 6, 13)
    assert parse_date("June 13, 1946") == (1946, 6, 13)
    assert parse_date("April 19, 1850") == (1850, 4, 19)
    assert parse_date("June 1946") == (1946, 6, 0)
    assert parse_date("1946") == (1946, 0, 0)
    assert parse_date("sometime") is None


def test_parse_number():
    assert parse_number("1,250 km") == 1250.0
    assert parse_number("about 3.5") == 3.5
    assert parse_number("none") is None


def test_compare_dates():
    assert compare(["25 April 1948"], ["22 December 1993"], "earlier") == 0
    assert compare(["25 April 1948"], ["22 December 1993"], "later") == 1
    assert compare(["1951"], ["1915"], "earlier") == 1
    assert compare(["1951"], ["1951"], "earlier") is None          # tie → abstain
    assert compare(["?"], ["1951"], "earlier") is None             # unparsable → abstain
    assert compare([], ["1951"], "later") is None


def test_compare_numbers():
    assert compare(["10"], ["20"], "larger") == 1
    assert compare(["10"], ["20"], "smaller") == 0


def test_compare_sets():
    assert compare(["American"], ["american"], "same") is True
    assert compare(["Croatian", "Yugoslavia"], ["Czech"], "same") is False
    assert compare(["Croatian", "Yugoslavia"], ["Yugoslavia"], "different") is False
    assert compare(["French"], ["Czech"], "different") is True
    assert compare([], ["Czech"], "same") is None


def test_templates():
    chain = {"template": "chain", "op": None}
    assert answer(chain, {"v1": ["Rune Gerhardsen"], "v2": ["13 June 1946"]}, ["Mina Gerhardsen"]) == "13 June 1946"
    assert answer(chain, {"v1": ["x"], "v2": []}, ["e"]) is None
    cmp_ = {"template": "compare", "op": "earlier"}
    assert answer(cmp_, {"v1": ["1915"], "v2": ["2009"]}, ["The Love Route", "Engal Aasan"]) == "The Love Route"
    bridge = {"template": "bridge", "op": "same"}
    vals = {"v1": ["A"], "v2": ["B"], "v3": ["American"], "v4": ["American"]}
    assert answer(bridge, vals, ["F1", "F2"]) == "yes"
    bridge_later = {"template": "bridge", "op": "later"}
    vals = {"v1": ["A"], "v2": ["B"], "v3": ["December 6, 1905"], "v4": ["August 25, 1877"]}
    assert answer(bridge_later, vals, ["King of the Gamblers", "A Bowery Cinderella"]) == "King of the Gamblers"


def test_normalize_official():
    assert normalize("The  Love Route!") == "love route"


def test_compare_equivalence_table():
    eq = {"american": 0, "united states": 0, "usa": 0, "french": 1}
    assert compare(["American"], ["United States"], "same") is False
    assert compare(["American"], ["United States"], "same", eq) is True
    assert compare(["USA"], ["French"], "different", eq) is True
