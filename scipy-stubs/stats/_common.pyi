from typing import Generic, NamedTuple
from typing_extensions import TypeVar

import numpy as np
import optype.numpy.compat as npc

_FltT_co = TypeVar("_FltT", bound=npc.floating, default=np.float64, covariant=True)

class ConfidenceInterval(NamedTuple, Generic[_FltT]):
    low: _FltT_co
    high: _FltT_co
