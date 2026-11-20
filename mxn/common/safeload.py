"""Loading pickled data files and torch checkpoints without executing code (security fix 2026-11-20).

pickle.load / torch.load(weights_only=False) run whatever callable a file names. Every data file and checkpoint the
system reads that another process wrote goes through this module instead:

- pickle_load(path): a restricted unpickler; only the globals in ALLOWED can be named (plain containers, numpy
  array / scalar / dtype reconstruction). Any other global (os.system, builtins.eval, subprocess, ...) → UnpicklingError,
  nothing runs.
- torch_load(path): torch.load(weights_only=True) (PyTorch's own restricted unpickler) with the same numpy allowlist.

KU packages between hosts do not use pickle at all (mxn/neuron/kupack.py).
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np

ALLOWED = {
    ("collections", "OrderedDict"), ("collections", "defaultdict"), ("builtins", "set"), ("builtins", "frozenset"),
    ("builtins", "slice"), ("builtins", "complex"), ("builtins", "bytearray"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
}
_DTYPES_MODULE = "numpy.dtypes"            # numpy >= 1.25 dtype classes (Float64DType, ...): data types only


class _Restricted(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in ALLOWED or (module == _DTYPES_MODULE and name.endswith("DType")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a data file")


def pickle_loads(data: bytes):
    return _Restricted(io.BytesIO(data)).load()


def pickle_load(path):
    with open(path, "rb") as f:
        return _Restricted(f).load()


def _numpy_globals() -> list:
    out = [np.ndarray, np.dtype]
    for mod in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            m = __import__(mod, fromlist=["_reconstruct", "scalar"])
            out += [m._reconstruct, m.scalar]
            break
        except (ImportError, AttributeError):
            continue
    import numpy.dtypes as nd
    out += [getattr(nd, n) for n in dir(nd) if n.endswith("DType")]
    return out


def torch_load(path, map_location="cpu"):
    import torch
    with torch.serialization.safe_globals(_numpy_globals()):
        return torch.load(path if not isinstance(path, Path) else str(path), map_location=map_location, weights_only=True)
