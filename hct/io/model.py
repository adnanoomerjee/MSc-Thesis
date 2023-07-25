
"""Loading/saving of inference functions."""

import dill
from typing import Any
from etils import epath

def load(path: str) -> Any:
  with epath.Path(path).open('rb') as fin:
    buf = fin.read()
  return dill.loads(buf)


def save(path: str, obj: Any):
  """Saves parameters in flax format."""
  with epath.Path(path).open('wb') as fout:
    fout.write(dill.dumps(obj, protocol=dill.HIGHEST_PROTOCOL))
