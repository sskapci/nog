"""Stop words and word pattern for lexical features and answer checks.

Moved unchanged from mxn/units/training.py (cleanup 2026-10-30); used by the KU confidence (mxn/ku/store.py) and
the answer check (mxn/common/answer_check.py).
"""

from __future__ import annotations

import re

_STOP = set("""a an the of in on at to for by with from and or but is are was were be been being
it its this that these those as he she they his her their there which who whom what when where
how also not no than then into over after before about between during under such can could
may might will would has have had do does did""".split())
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
