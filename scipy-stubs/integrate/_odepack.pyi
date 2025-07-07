from _typeshed import Incomplete
from collections.abc import Callable

import optype.numpy as onp

class error(Exception): ...

def odeint(
    fun: Callable[..., Incomplete],
    y0: onp.ToFloat | onp.ToFloatND,
    t: float,
    args: tuple[object, ...] = (),
    Dfun: Callable[..., Incomplete] | None = None,
    col_deriv: int = 0,
    ml: int = ...,
    mu: int = ...,
    full_output: onp.ToBool = 0,
    rtol: float = ...,
    atol: float = ...,
    tcrit: float = ...,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: float = 0.0,
    mxstep: float = 0.0,
    mxhnil: int = 0,
    mxordn: int = 0,
    mxords: int = 0,
) -> tuple[Incomplete, ...]: ...
