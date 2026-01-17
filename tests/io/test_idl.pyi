from typing import Any, assert_type

from scipy.io import readsav
from scipy.io._idl import AttrDict

###

# readsav
assert_type(readsav("file.sav"), AttrDict[Any])
assert_type(readsav("file.sav", python_dict=True), dict[str, Any])
