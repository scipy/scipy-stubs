import collections
from collections.abc import Callable
from typing import Any, Concatenate, Final, Generic, Literal, NoReturn, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import Never, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Falsy, Truthy

_S = TypeVar("_S")
_T = TypeVar("_T")
_VT = TypeVar("_VT", default=object)
_NDT_co = TypeVar("_NDT_co", bound=_FloatingND, default=_FloatingND, covariant=True)
_SCT_co = TypeVar("_SCT_co", bound=np.floating[Any], default=np.float64, covariant=True)

_Floating: TypeAlias = float | int | bool | np.floating[Any]
_FloatingND: TypeAlias = onp.ArrayND[np.floating[Any]] | _Floating

_Fun: TypeAlias = Callable[Concatenate[float | int | bool, ...], _T] | Callable[Concatenate[np.float64, ...], _T]

_Norm: TypeAlias = Literal["max", "2"]
_Quadrature: TypeAlias = Literal["gk21", "gk15", "trapezoid"]

@type_check_only
class _DoesMap(Protocol):
    def __call__(
        self,
        func: Callable[[_S], _T],
        iterable: op.CanIter[op.CanNext[_S]],
        /,
    ) -> op.CanIter[op.CanIterSelf[_T]]: ...

@type_check_only
class _InfiniteFunc(Protocol[_NDT_co]):
    def get_t(self, /, x: float | int | bool) -> float | int | bool: ...
    def __call__(self, /, t: float | int | bool) -> _NDT_co: ...

###

class LRUDict(collections.OrderedDict[tuple[float | int | bool, float | int | bool], _VT], Generic[_VT]):
    def __init__(self, /, max_size: int | bool) -> None: ...
    @override
    def update(self, other: Never, /) -> NoReturn: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

class SemiInfiniteFunc(_InfiniteFunc[_NDT_co], Generic[_NDT_co]):
    def __init__(self, /, func: Callable[[float | int | bool], _NDT_co], start: float | int | bool, infty: bool) -> None: ...

class DoubleInfiniteFunc(_InfiniteFunc, Generic[_NDT_co]):
    def __init__(self, /, func: Callable[[float | int | bool], _NDT_co]) -> None: ...

# NOTE: This is only used as "info dict" for `quad_vec(..., full_output=True)`,
# even though, confusingly, it is not even even a mapping.
# NOTE: Because this "bunch" is only used as "info dict" (and nowhere else),
# its the ~keys~ attributes have been annotated right here.
class _Bunch(Generic[_SCT_co]):
    def __init__(
        self,
        /,
        *,
        success: bool,
        status: Literal[0, 1, 2],
        neval: int | bool,
        message: str,
        intervals: onp.Array2D[np.float64],
        errors: onp.Array1D[np.float64],
        integrals: onp.Array2D[_SCT_co],
    ) -> None: ...
    success: Final[bool]
    status: Final[Literal[0, 1, 2]]
    neval: Final[int | bool]
    message: Final[str]
    intervals: Final[onp.Array2D[np.float64]]
    errors: Final[onp.Array1D[np.float64]]
    integrals: onp.Array2D[_SCT_co]

@overload
def quad_vec(  # scalar function, full_output=False (default)
    f: _Fun[onp.ToFloat],
    a: onp.ToFloat,
    b: onp.ToFloat,
    epsabs: _Floating = 1e-200,
    epsrel: _Floating = 1e-08,
    norm: _Norm = "2",
    cache_size: onp.ToJustInt | float | int | bool = 100_000_000,
    limit: onp.ToFloat = 10_000,
    workers: onp.ToJustInt | _DoesMap = 1,
    points: onp.ToFloat1D | None = None,
    quadrature: _Quadrature | None = None,
    full_output: Falsy = False,
    *,
    args: tuple[object, ...] = (),
) -> tuple[_Floating, float | int | bool]: ...
@overload  # scalar function, full_output=True
def quad_vec(
    f: _Fun[onp.ToFloat],
    a: onp.ToFloat,
    b: onp.ToFloat,
    epsabs: _Floating = 1e-200,
    epsrel: _Floating = 1e-08,
    norm: _Norm = "2",
    cache_size: onp.ToJustInt | float | int | bool = 100_000_000,
    limit: onp.ToFloat = 10_000,
    workers: onp.ToJustInt | _DoesMap = 1,
    points: onp.ToFloat1D | None = None,
    quadrature: _Quadrature | None = None,
    *,
    full_output: Truthy,
    args: tuple[object, ...] = (),
) -> tuple[np.floating[Any], float | int | bool, _Bunch[np.floating[Any]]]: ...
@overload  # vector function, full_output=False (default)
def quad_vec(
    f: _Fun[onp.ToFloat1D],
    a: onp.ToFloat,
    b: onp.ToFloat,
    epsabs: _Floating = 1e-200,
    epsrel: _Floating = 1e-08,
    norm: _Norm = "2",
    cache_size: onp.ToJustInt | float | int | bool = 100_000_000,
    limit: onp.ToFloat = 10_000,
    workers: onp.ToJustInt | _DoesMap = 1,
    points: onp.ToFloat1D | None = None,
    quadrature: _Quadrature | None = None,
    *,
    full_output: Falsy,
    args: tuple[object, ...] = (),
) -> tuple[onp.Array1D[np.floating[Any]], float | int | bool]: ...
@overload  # vector function, full_output=True
def quad_vec(
    f: _Fun[onp.ToFloat1D],
    a: onp.ToFloat,
    b: onp.ToFloat,
    epsabs: _Floating = 1e-200,
    epsrel: _Floating = 1e-08,
    norm: _Norm = "2",
    cache_size: onp.ToJustInt | float | int | bool = 100_000_000,
    limit: onp.ToFloat = 10_000,
    workers: onp.ToJustInt | _DoesMap = 1,
    points: onp.ToFloat1D | None = None,
    quadrature: _Quadrature | None = None,
    *,
    full_output: Truthy,
    args: tuple[object, ...] = (),
) -> tuple[onp.Array1D[np.floating[Any]], float | int | bool, _Bunch[np.floating[Any]]]: ...
