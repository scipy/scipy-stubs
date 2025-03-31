from collections.abc import Callable
from typing import Any, ClassVar, Final, Generic, Literal, TypeVar, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Truthy

_VT = TypeVar("_VT", bound=onp.ArrayND[np.inexact[Any]], default=onp.ArrayND[np.inexact[Any]])

class OdeSolver:
    TOO_SMALL_STEP: ClassVar[str] = ...

    t: float | int | bool
    t_old: float | int | bool
    t_bound: float | int | bool
    vectorized: bool
    fun: Callable[[float | int | bool, onp.ArrayND[np.float64]], onp.ArrayND[np.float64]]
    fun_single: Callable[[float | int | bool, onp.ArrayND[np.float64]], onp.ArrayND[np.float64]]
    fun_vectorized: Callable[[float | int | bool, onp.ArrayND[np.float64]], onp.ArrayND[np.float64]]
    direction: float | int | bool
    n: int | bool
    status: Literal["running", "finished", "failed"]
    nfev: int | bool
    njev: int | bool
    nlu: int | bool

    @overload
    def __init__(
        self,
        /,
        fun: Callable[[float | int | bool, onp.ArrayND[np.float64]], onp.ToFloatND],
        t0: onp.ToFloatND,
        y0: onp.ToFloatND,
        t_bound: onp.ToFloat,
        vectorized: bool,
        support_complex: onp.ToBool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        fun: Callable[[float | int | bool, onp.ArrayND[np.float64 | np.complex128]], onp.ToComplexND],
        t0: onp.ToFloat,
        y0: onp.ToComplexND,
        t_bound: onp.ToFloat,
        vectorized: bool,
        support_complex: Truthy,
    ) -> None: ...
    @property
    def step_size(self, /) -> float | int | bool | None: ...
    def step(self, /) -> str | None: ...
    def dense_output(self, /) -> ConstantDenseOutput: ...

class DenseOutput:
    t_old: Final[float | int | bool]
    t: Final[float | int | bool]
    t_min: Final[float | int | bool]
    t_max: Final[float | int | bool]

    def __init__(self, /, t_old: onp.ToFloat, t: onp.ToFloat) -> None: ...
    @overload
    def __call__(self, /, t: onp.ToFloat) -> onp.Array1D[np.inexact[Any]]: ...
    @overload
    def __call__(self, /, t: onp.ToFloatND) -> onp.ArrayND[np.inexact[Any]]: ...

class ConstantDenseOutput(DenseOutput, Generic[_VT]):
    value: _VT
    def __init__(self, /, t_old: onp.ToFloat, t: onp.ToFloat, value: _VT) -> None: ...

def check_arguments(
    fun: Callable[[float | int | bool, onp.ArrayND[np.float64]], onp.ToComplexND],
    y0: onp.ToComplexND,
    support_complex: bool,
) -> (
    Callable[[float | int | bool, onp.ArrayND[np.float64]], onp.ArrayND[np.float64]]
    | Callable[[float | int | bool, onp.ArrayND[np.float64]], onp.ArrayND[np.complex128]]
): ...
