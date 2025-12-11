from _typeshed import Incomplete
from types import ModuleType
from typing import Literal, TypeAlias, overload

import numpy as np
import optype as op
import optype.numpy as onp

_Mode: TypeAlias = Literal["mirror", "constant", "nearest", "wrap", "interp"]

@overload
def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: int | None = None,
    use: Literal["conv", "dot"] = "conv",
    xp: None = None,
    device: None = None,
) -> onp.Array1D[np.float64]: ...
@overload
def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: int | None = None,
    use: Literal["conv", "dot"] = "conv",
    *,
    xp: ModuleType,
    device: object | None = None,
) -> Incomplete: ...

#
def savgol_filter(
    x: onp.ToFloatND,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    axis: op.CanIndex = -1,
    mode: _Mode = "interp",
    cval: float = 0.0,
) -> onp.ArrayND[np.float32 | np.float64]: ...
