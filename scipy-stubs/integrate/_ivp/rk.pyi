from collections.abc import Callable
from typing import ClassVar, Final, Generic, overload
from typing_extensions import Never, TypeVar

import numpy as np
import optype.numpy as onp
from .base import DenseOutput, OdeSolver

_SCT_fc = TypeVar("_SCT_fc", bound=np.float64 | np.complex128, default=np.float64 | np.complex128)

###

SAFETY: Final = 0.9
MIN_FACTOR: Final = 0.2
MAX_FACTOR: Final = 10

class RungeKutta(OdeSolver, Generic[_SCT_fc]):
    C: ClassVar[onp.ArrayND[np.float64]]
    A: ClassVar[onp.ArrayND[np.float64]]
    B: ClassVar[onp.ArrayND[np.float64]]
    E: ClassVar[onp.ArrayND[np.float64]]
    P: ClassVar[onp.ArrayND[np.float64]]
    order: ClassVar[int | bool]
    error_estimator_order: ClassVar[int | bool]
    n_stages: ClassVar[int | bool]

    y_old: onp.ArrayND[_SCT_fc] | None
    f: onp.ArrayND[_SCT_fc]
    K: onp.ArrayND[_SCT_fc]
    max_step: float | int | bool
    h_abs: float | int | bool
    error_exponent: float | int | bool
    h_previous: float | int | bool | None

    def __init__(
        self,
        /,
        fun: Callable[[float | int | bool, onp.ArrayND[_SCT_fc]], onp.ArrayND[_SCT_fc]],
        t0: float | int | bool,
        y0: onp.ArrayND[_SCT_fc],
        t_bound: float | int | bool,
        max_step: float | int | bool = ...,
        rtol: float | int | bool = 0.001,
        atol: float | int | bool = 1e-06,
        vectorized: bool = False,
        first_step: float | int | bool | None = None,
        **extraneous: Never,
    ) -> None: ...

class RK23(RungeKutta[_SCT_fc], Generic[_SCT_fc]): ...
class RK45(RungeKutta[_SCT_fc], Generic[_SCT_fc]): ...

class DOP853(RungeKutta[_SCT_fc], Generic[_SCT_fc]):
    E3: ClassVar[onp.ArrayND[np.float64]]
    E5: ClassVar[onp.ArrayND[np.float64]]
    D: ClassVar[onp.ArrayND[np.float64]]
    A_EXTRA: ClassVar[onp.ArrayND[np.float64]]
    C_EXTRA: ClassVar[onp.ArrayND[np.float64]]

    K_extended: onp.ArrayND[_SCT_fc]

class RkDenseOutput(DenseOutput, Generic[_SCT_fc]):
    h: float | int | bool
    order: int | bool
    Q: onp.ArrayND[_SCT_fc]
    y_old: onp.ArrayND[_SCT_fc]

    def __init__(
        self, /, t_old: float | int | bool, t: float | int | bool, y_old: onp.ArrayND[_SCT_fc], Q: onp.ArrayND[_SCT_fc]
    ) -> None: ...

class Dop853DenseOutput(DenseOutput, Generic[_SCT_fc]):
    h: float | int | bool
    F: onp.ArrayND[_SCT_fc]
    y_old: onp.ArrayND[_SCT_fc]

    def __init__(
        self, /, t_old: float | int | bool, t: float | int | bool, y_old: onp.ArrayND[_SCT_fc], F: onp.ArrayND[_SCT_fc]
    ) -> None: ...

@overload
def rk_step(
    fun: Callable[[float | int | bool, onp.ArrayND[np.float64]], onp.ArrayND[np.float64]],
    t: float | int | bool,
    y: onp.ArrayND[np.float64],
    f: onp.ArrayND[np.float64],
    h: float | int | bool,
    A: onp.ArrayND[np.float64],
    B: onp.ArrayND[np.float64],
    C: onp.ArrayND[np.float64],
    K: onp.ArrayND[np.float64],
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
@overload
def rk_step(
    fun: Callable[[float | int | bool, onp.ArrayND[np.complex128]], onp.ArrayND[np.complex128]],
    t: float | int | bool,
    y: onp.ArrayND[np.complex128],
    f: onp.ArrayND[np.complex128],
    h: float | int | bool,
    A: onp.ArrayND[np.float64],
    B: onp.ArrayND[np.float64],
    C: onp.ArrayND[np.float64],
    K: onp.ArrayND[np.float64],
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]]: ...
