# defined in scipy/optimize/_directmodule.c

from collections.abc import Callable
from typing import Literal

import numpy as np
import optype.numpy as onp

from scipy._lib._ccallback import LowLevelCallable

def direct(
    f: Callable[..., float] | LowLevelCallable,
    lv: onp.Array1D[np.float64],
    ub: onp.Array1D[np.float64],
    f_args: tuple[object, ...],
    disp: bool,
    magic_eps: float,
    max_feval: int,
    max_iter: int,
    algorithm: bool | Literal[0, 1],
    fglobal: float,
    fglobal_reltol: float,
    volume_reltol: float,
    sigma_reltol: float,
    callback: Callable[..., object] | None,
    /,
) -> tuple[
    list[float],  # x_seq
    float,  # minf
    int,  # ret_code
    int,  # numfunc
    int,  # numiter
]: ...
