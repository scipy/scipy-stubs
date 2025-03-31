from collections.abc import Callable
from typing import Final, TypeAlias

import optype.numpy as onp

_Fun: TypeAlias = Callable[[float | int | bool], onp.ToFloat]

###

class DCSRCH:
    phi: Final[_Fun]
    derphi: Final[_Fun]
    ftol: Final[float | int | bool]
    gtol: Final[float | int | bool]
    xtol: Final[float | int | bool]
    stpmin: Final[float | int | bool]
    stpmax: Final[float | int | bool]

    stage: int | bool | None
    ginit: float | int | bool | None
    gtest: float | int | bool | None
    gx: float | int | bool | None
    gy: float | int | bool | None
    finit: float | int | bool | None
    fx: float | int | bool | None
    fy: float | int | bool | None
    stx: float | int | bool | None
    sty: float | int | bool | None
    stmin: float | int | bool | None
    stmax: float | int | bool | None
    width: float | int | bool | None
    width1: float | int | bool | None

    def __init__(
        self,
        /,
        phi: _Fun,
        derphi: _Fun,
        ftol: float | int | bool,
        gtol: float | int | bool,
        xtol: float | int | bool,
        stpmin: float | int | bool,
        stpmax: float | int | bool,
    ) -> None: ...
    def __call__(
        self,
        /,
        alpha1: float | int | bool,
        phi0: float | int | bool | None = None,
        derphi0: float | int | bool | None = None,
        maxiter: int | bool = 100,
    ) -> tuple[float | int | bool | None, float | int | bool, float | int | bool, bytes]: ...  # alpha, phi(alpha), phi(0), task

def dcstep(
    stx: float | int | bool,
    fx: float | int | bool,
    dx: float | int | bool,
    sty: float | int | bool,
    fy: float | int | bool,
    dy: float | int | bool,
    stp: float | int | bool,
    fp: float | int | bool,
    dp: float | int | bool,
    brackt: bool,
    stpmin: float | int | bool,
    stpmax: float | int | bool,
) -> tuple[
    float | int | bool,
    float | int | bool,
    float | int | bool,
    float | int | bool,
    float | int | bool,
    float | int | bool,
    float | int | bool,
    bool,
]: ...  # stx, fx, dx, sty, fy, dy, stp, brackt
