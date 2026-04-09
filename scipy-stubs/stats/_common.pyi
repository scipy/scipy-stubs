from typing import Generic, NamedTuple
from typing_extensions import TypeVar

import numpy as np
import optype.numpy.compat as npc

_FltT = TypeVar("_FltT", bound=npc.floating, default=np.float64)

class ConfidenceInterval(NamedTuple, Generic[_FltT]):
    low: _FltT
    high: _FltT
