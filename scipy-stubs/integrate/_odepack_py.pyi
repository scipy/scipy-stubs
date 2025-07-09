from typing import Literal, Protocol, overload, type_check_only
from typing_extensions import TypeVar, TypeVarTuple, Unpack

import optype.numpy as onp
import optype.numpy.compat as npc

from ._typing import ODEInfoDict

__all__ = ["ODEintWarning", "odeint"]

_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[()]])
_YT = TypeVar("_YT", bound=onp.ToFloat1D | onp.ToFloat)

@type_check_only
class _ODEFunc(Protocol[_YT, *_Ts]):
    def __call__(self, y: _YT, t: float, /, *args: *_Ts) -> _YT: ...

@type_check_only
class _ODEFuncInv(Protocol[_YT, *_Ts]):
    def __call__(self, t: float, y: _YT, /, *args: *_Ts) -> _YT: ...

###

class ODEintWarning(Warning): ...

# unspecified args
@overload
def odeint(
    func: _ODEFunc[_YT],
    y0: _YT,
    t: onp.ToInt1D,
    args: tuple[()] = (),
    Dfun: _ODEFunc[_YT] | None = None,
    col_deriv: Literal[0, 1] | bool = 0,
    full_output: onp.ToFalse = 0,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: onp.ArrayND[npc.integer | npc.floating] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[0, 1] | bool = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[0, 1] | bool = 0,
    tfirst: onp.ToFalse = False,
) -> onp.Array2D[npc.floating]: ...
@overload
def odeint(
    func: _ODEFunc[_YT],
    y0: _YT,
    t: onp.ToInt1D,
    args: tuple[()] = (),
    Dfun: _ODEFunc[_YT] | None = None,
    col_deriv: Literal[0, 1] | bool = 0,
    full_output: onp.ToFalse = 0,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: onp.ArrayND[npc.integer | npc.floating] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[0, 1] | bool = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[0, 1] | bool = 0,
    *,
    tfirst: onp.ToTrue,
) -> onp.Array2D[npc.floating]: ...
@overload
def odeint(
    func: _ODEFunc[_YT],
    y0: _YT,
    t: onp.ToInt1D,
    args: tuple[()] = (),
    Dfun: _ODEFunc[_YT] | None = None,
    col_deriv: Literal[0, 1] | bool = 0,
    *,
    full_output: onp.ToTrue,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: onp.ArrayND[npc.integer | npc.floating] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[0, 1] | bool = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[0, 1] | bool = 0,
    tfirst: onp.ToFalse = False,
) -> tuple[onp.Array2D[npc.floating], ODEInfoDict]: ...
@overload
def odeint(
    func: _ODEFunc[_YT],
    y0: _YT,
    t: onp.ToInt1D,
    args: tuple[()] = (),
    Dfun: _ODEFunc[_YT] | None = None,
    col_deriv: Literal[0, 1] | bool = 0,
    *,
    full_output: onp.ToTrue,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: onp.ArrayND[npc.integer | npc.floating] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[0, 1] | bool = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[0, 1] | bool = 0,
    tfirst: onp.ToTrue,
) -> tuple[onp.Array2D[npc.floating], ODEInfoDict]: ...

# specified args
@overload
def odeint(
    func: _ODEFunc[_YT, *_Ts],
    y0: _YT,
    t: onp.ToInt1D,
    args: tuple[*_Ts] = ...,
    Dfun: _ODEFunc[_YT, *_Ts] | None = None,
    col_deriv: Literal[0, 1] | bool = 0,
    full_output: onp.ToFalse = 0,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: onp.ArrayND[npc.integer | npc.floating] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[0, 1] | bool = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[0, 1] | bool = 0,
    tfirst: onp.ToFalse = False,
) -> onp.Array2D[npc.floating]: ...
@overload
def odeint(
    func: _ODEFuncInv[_YT, *_Ts],
    y0: _YT,
    t: onp.ToInt1D,
    args: tuple[*_Ts] = ...,
    Dfun: _ODEFuncInv[_YT, *_Ts] | None = None,
    col_deriv: Literal[0, 1] | bool = 0,
    full_output: onp.ToFalse = 0,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: onp.ArrayND[npc.integer | npc.floating] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[0, 1] | bool = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[0, 1] | bool = 0,
    *,
    tfirst: onp.ToTrue,
) -> onp.Array2D[npc.floating]: ...
@overload
def odeint(
    func: _ODEFunc[_YT, *_Ts],
    y0: _YT,
    t: onp.ToInt1D,
    args: tuple[*_Ts] = ...,
    Dfun: _ODEFunc[_YT, *_Ts] | None = None,
    col_deriv: Literal[0, 1] | bool = 0,
    *,
    full_output: onp.ToTrue,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: onp.ArrayND[npc.integer | npc.floating] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[0, 1] | bool = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[0, 1] | bool = 0,
    tfirst: onp.ToFalse = False,
) -> tuple[onp.Array2D[npc.floating], ODEInfoDict]: ...
@overload
def odeint(
    func: _ODEFuncInv[_YT, *_Ts],
    y0: _YT,
    t: onp.ToInt1D,
    args: tuple[*_Ts] = ...,
    Dfun: _ODEFuncInv[_YT, *_Ts] | None = None,
    col_deriv: Literal[0, 1] | bool = 0,
    *,
    full_output: onp.ToTrue,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: onp.ArrayND[npc.integer | npc.floating] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[0, 1] | bool = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[0, 1] | bool = 0,
    tfirst: onp.ToTrue,
) -> tuple[onp.Array2D[npc.floating], ODEInfoDict]: ...
