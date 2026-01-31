from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar
import numpy as np
import optype.numpy as onp

#  the TypeVar to handle both scalars and ND arrays
_T = TypeVar("_T", bound=onp.ArrayND[np.float64] | np.float64)

@dataclass(frozen=True)
class ConfidenceInterval(Generic[_T]):
    low: _T
    high: _T