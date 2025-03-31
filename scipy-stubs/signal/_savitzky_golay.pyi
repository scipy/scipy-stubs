from typing import Any, Literal, TypeAlias

import numpy as np
import optype as op
import optype.numpy as onp

_Mode: TypeAlias = Literal["mirror", "constant", "nearest", "wrap", "interp"]

def savgol_coeffs(
    window_length: int | bool,
    polyorder: int | bool,
    deriv: int | bool = 0,
    delta: float | int | bool = 1.0,
    pos: int | bool | None = None,
    use: Literal["conv", "dot"] = "conv",
) -> onp.Array1D[np.floating[Any]]: ...

#
def savgol_filter(
    x: onp.ToFloatND,
    window_length: int | bool,
    polyorder: int | bool,
    deriv: int | bool = 0,
    delta: float | int | bool = 1.0,
    axis: op.CanIndex = -1,
    mode: _Mode = "interp",
    cval: float | int | bool = 0.0,
) -> onp.ArrayND[np.float32 | np.float64]: ...
