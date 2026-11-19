"""Title keys: word tokens and the trailing "(...)" disambiguator of a title.

Moved unchanged from mxn/graph/network.py (cleanup 2026-10-30); used by the resolvers, the host indexes and the
operations' answer forms.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+")
_DISAMBIG = re.compile(r"\s*\([^)]*\)\s*$")
