from collections.abc import Callable, Iterable, Iterator
from typing import Any, Concatenate, Final, Generic, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._lib._ccallback import LowLevelCallable
from scipy._typing import Falsy, Truthy
from ._typing import QuadInfoDict, QuadOpts, QuadWeights

__all__ = ["IntegrationWarning", "dblquad", "nquad", "quad", "tplquad"]

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_f_contra = TypeVar("_T_f_contra", contravariant=True, default=float | int | bool)
_BT_co = TypeVar("_BT_co", bound=bool, covariant=True, default=bool)

# NOTE: Technically `integer[Any]` and `bool_` are also allowed, but there's no valid usecase for that.
_IntLike: TypeAlias = int | bool | np.integer[Any]
_FloatLike: TypeAlias = float | int | bool | np.floating[Any]
_ComplexLike: TypeAlias = complex | float | int | bool | np.inexact[Any]

# NOTE: Technically allowing `x: float64` here is type-unsafe. But in practice that isn't likely to be a problem at all.
_QuadFunc10: TypeAlias = Callable[[float | int | bool], _T] | Callable[[np.float64], _T] | LowLevelCallable
_QuadFunc1N: TypeAlias = (
    Callable[Concatenate[float | int | bool, ...], _T] | Callable[Concatenate[np.float64, ...], _T] | LowLevelCallable
)

_QuadFunc20: TypeAlias = (
    Callable[[float | int | bool, float | int | bool], _FloatLike]
    | Callable[[np.float64, np.float64], _FloatLike]
    | LowLevelCallable
)
_QuadFunc2N: TypeAlias = (
    Callable[Concatenate[float | int | bool, float | int | bool, ...], _FloatLike]
    | Callable[Concatenate[np.float64, np.float64, ...], _FloatLike]
    | LowLevelCallable
)  # fmt: skip

_QuadFunc30: TypeAlias = (
    Callable[[float | int | bool, float | int | bool, float | int | bool], _FloatLike]
    | Callable[[np.float64, np.float64, np.float64], _FloatLike]
    | LowLevelCallable
)  # fmt: skip
_QuadFunc3N: TypeAlias = (
    Callable[Concatenate[float | int | bool, float | int | bool, float | int | bool, ...], _FloatLike]
    | Callable[Concatenate[np.float64, np.float64, np.float64, ...], _FloatLike]
    | LowLevelCallable
)  # fmt: skip

_QuadFuncN: TypeAlias = (
    Callable[Concatenate[float | int | bool, ...], _FloatLike]
    | Callable[Concatenate[np.float64, ...], _FloatLike]
    | LowLevelCallable
)  # fmt: skip

_GHFunc: TypeAlias = _FloatLike | Callable[[float | int | bool], _FloatLike] | Callable[[np.float64], _FloatLike]
_QRFunc: TypeAlias = (
    _FloatLike | Callable[[float | int | bool, float | int | bool], _FloatLike] | Callable[[np.float64, np.float64], _FloatLike]
)

@type_check_only
class _QuadOutput1C_1(TypedDict):
    real: tuple[QuadInfoDict]
    imag: tuple[QuadInfoDict]

@type_check_only
class _QuadOutput1C_2(TypedDict):
    real: tuple[QuadInfoDict, str]
    imag: tuple[QuadInfoDict, str]

@type_check_only
class _QuadOutput1C_3(TypedDict):
    real: tuple[QuadInfoDict, str, _QuadExplain]
    imag: tuple[QuadInfoDict, str, _QuadExplain]

@type_check_only
class _QuadOutputNC(TypedDict):
    neval: int | bool

_QuadComplexFullOutput: TypeAlias = _QuadOutput1C_1 | _QuadOutput1C_2 | _QuadOutput1C_3
_QuadExplain = TypedDict("_QuadExplain", {0: str, 1: str, 2: str, 3: str, 4: str, 5: str})  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]

@type_check_only
class _CanLenAndIter(Protocol[_T_co]):
    def __len__(self, /) -> int | bool: ...
    def __iter__(self, /) -> Iterator[_T_co]: ...

_SizedIterable: TypeAlias = _CanLenAndIter[_T] | op.CanSequence[int | bool, _T]
_QuadRange: TypeAlias = _SizedIterable[float | int | bool]
_RangeT = TypeVar("_RangeT", bound=_QuadRange, default=_QuadRange)
_RangeT_co = TypeVar("_RangeT_co", bound=_QuadRange, covariant=True, default=_QuadRange)

@type_check_only
class _RangeCallable(Protocol[_T_f_contra, _RangeT_co]):
    def __call__(self, /, *args: _T_f_contra) -> _RangeT_co: ...

_OptT = TypeVar("_OptT", bound=QuadOpts, default=QuadOpts)
_OptT_co = TypeVar("_OptT_co", bound=QuadOpts, covariant=True, default=QuadOpts)

@type_check_only
class _OptCallable(Protocol[_T_f_contra, _OptT_co]):
    def __call__(self, /, *args: _T_f_contra) -> _OptT_co: ...

###

class _RangeFunc(_RangeCallable[_T_f_contra, _RangeT], Generic[_T_f_contra, _RangeT]):
    range_: _RangeT
    def __init__(self, /, range_: _RangeT) -> None: ...

class _OptFunc(_OptCallable[_T_f_contra, _OptT], Generic[_T_f_contra, _OptT]):
    opt: _OptT
    def __init__(self, /, opt: _OptT) -> None: ...

class _NQuad(Generic[_BT_co]):
    abserr: Final[float | int | bool]
    maxdepth: Final[int | bool]
    out_dict: Final[_QuadOutputNC]

    func: _QuadFuncN
    ranges: list[_RangeFunc]
    opts: list[_OptFunc]
    full_output: _BT_co

    def __init__(self, /, func: _QuadFuncN, ranges: list[_RangeFunc], opts: list[_OptFunc], full_output: _BT_co) -> None: ...
    @overload
    def integrate(self: _NQuad[Literal[False]], /, *args: object) -> tuple[float | int | bool, float | int | bool]: ...
    @overload
    def integrate(
        self: _NQuad[Literal[True]], /, *args: object
    ) -> tuple[float | int | bool, float | int | bool, _QuadOutputNC]: ...

class IntegrationWarning(UserWarning): ...

# 1-dimensional quadrature

@overload
def quad(
    func: _QuadFunc10[_FloatLike],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[()] = (),
    full_output: Falsy = 0,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToFloat1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    complex_func: Falsy = False,
) -> tuple[float | int | bool, float | int | bool]: ...
@overload
def quad(
    func: _QuadFunc1N[_FloatLike],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...],
    full_output: Falsy = 0,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToFloat1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    complex_func: Falsy = False,
) -> tuple[float | int | bool, float | int | bool]: ...
@overload
def quad(
    func: _QuadFunc10[_FloatLike],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[()],
    full_output: Truthy,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToFloat1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    complex_func: Falsy = False,
) -> (
    tuple[float | int | bool, float | int | bool, QuadInfoDict]
    | tuple[float | int | bool, float | int | bool, QuadInfoDict, str]
    | tuple[float | int | bool, float | int | bool, QuadInfoDict, str, _QuadExplain]
): ...
@overload
def quad(
    func: _QuadFunc10[_FloatLike],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[()] = (),
    *,
    full_output: Truthy,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToFloat1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    complex_func: Falsy = False,
) -> (
    tuple[float | int | bool, float | int | bool, QuadInfoDict]
    | tuple[float | int | bool, float | int | bool, QuadInfoDict, str]
    | tuple[float | int | bool, float | int | bool, QuadInfoDict, str, _QuadExplain]
): ...
@overload
def quad(
    func: _QuadFunc1N[_FloatLike],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...],
    full_output: Truthy,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToFloat1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    complex_func: Falsy = False,
) -> (
    tuple[float | int | bool, float | int | bool, QuadInfoDict]
    | tuple[float | int | bool, float | int | bool, QuadInfoDict, str]
    | tuple[float | int | bool, float | int | bool, QuadInfoDict, str, _QuadExplain]
): ...
@overload
def quad(
    func: _QuadFunc10[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[()],
    full_output: Falsy,
    epsabs: _FloatLike,
    epsrel: _FloatLike,
    limit: _IntLike,
    points: onp.ToComplex1D | None,
    weight: QuadWeights | None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None,
    maxp1: _IntLike,
    limlst: _IntLike,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool]: ...
@overload
def quad(
    func: _QuadFunc10[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[()] = (),
    full_output: Falsy = 0,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToComplex1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    *,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool]: ...
@overload
def quad(
    func: _QuadFunc10[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[()],
    full_output: Truthy,
    epsabs: _FloatLike,
    epsrel: _FloatLike,
    limit: _IntLike,
    points: onp.ToComplex1D | None,
    weight: QuadWeights | None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None,
    maxp1: _IntLike,
    limlst: _IntLike,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool, _QuadComplexFullOutput]: ...
@overload
def quad(
    func: _QuadFunc10[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[()],
    full_output: Truthy,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToComplex1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    *,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool, _QuadComplexFullOutput]: ...
@overload
def quad(
    func: _QuadFunc10[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[()] = (),
    *,
    full_output: Truthy,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToComplex1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool, _QuadComplexFullOutput]: ...
@overload
def quad(
    func: _QuadFunc1N[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[object, ...],
    full_output: Falsy,
    epsabs: _FloatLike,
    epsrel: _FloatLike,
    limit: _IntLike,
    points: onp.ToComplex1D | None,
    weight: QuadWeights | None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None,
    maxp1: _IntLike,
    limlst: _IntLike,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool]: ...
@overload
def quad(
    func: _QuadFunc1N[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[object, ...],
    full_output: Falsy = 0,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToComplex1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    *,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool]: ...
@overload
def quad(
    func: _QuadFunc1N[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[object, ...],
    full_output: Truthy,
    epsabs: _FloatLike,
    epsrel: _FloatLike,
    limit: _IntLike,
    points: onp.ToComplex1D | None,
    weight: QuadWeights | None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None,
    maxp1: _IntLike,
    limlst: _IntLike,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool, _QuadComplexFullOutput]: ...
@overload
def quad(
    func: _QuadFunc1N[_ComplexLike],
    a: onp.ToComplex,
    b: onp.ToComplex,
    args: tuple[object, ...],
    full_output: Truthy,
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
    limit: _IntLike = 50,
    points: onp.ToComplex1D | None = None,
    weight: QuadWeights | None = None,
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike] | None = None,
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]] | None = None,
    maxp1: _IntLike = 50,
    limlst: _IntLike = 50,
    *,
    complex_func: Truthy,
) -> tuple[complex | float | int | bool, complex | float | int | bool, _QuadComplexFullOutput]: ...

# 2-dimensional quadrature

@overload
def dblquad(
    func: _QuadFunc20,
    a: onp.ToFloat,
    b: onp.ToFloat,
    gfun: _GHFunc,
    hfun: _GHFunc,
    args: tuple[()] = (),
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
) -> tuple[float | int | bool, float | int | bool]: ...
@overload
def dblquad(
    func: _QuadFunc2N,
    a: onp.ToFloat,
    b: onp.ToFloat,
    gfun: _GHFunc,
    hfun: _GHFunc,
    args: tuple[object, ...],
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
) -> tuple[float | int | bool, float | int | bool]: ...

# 3-dimensional quadrature

@overload
def tplquad(
    func: _QuadFunc30,
    a: onp.ToFloat,
    b: onp.ToFloat,
    gfun: _GHFunc,
    hfun: _GHFunc,
    qfun: _QRFunc,
    rfun: _QRFunc,
    args: tuple[()] = (),
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
) -> tuple[float | int | bool, float | int | bool]: ...
@overload
def tplquad(
    func: _QuadFunc3N,
    a: onp.ToFloat,
    b: onp.ToFloat,
    gfun: _GHFunc,
    hfun: _GHFunc,
    qfun: _QRFunc,
    rfun: _QRFunc,
    args: tuple[object, ...],
    epsabs: _FloatLike = 1.49e-08,
    epsrel: _FloatLike = 1.49e-08,
) -> tuple[float | int | bool, float | int | bool]: ...

# N-dimensional quadrature

@overload
def nquad(
    func: _QuadFuncN,
    ranges: _SizedIterable[_QuadRange | _RangeCallable[float | int | bool]],
    args: Iterable[object] | None = None,
    opts: QuadOpts | Callable[..., QuadOpts] | Iterable[QuadOpts | Callable[..., QuadOpts]] | None = None,
    full_output: Falsy = False,
) -> tuple[float | int | bool, float | int | bool]: ...
@overload
def nquad(
    func: _QuadFuncN,
    ranges: _SizedIterable[_QuadRange | _RangeCallable[float | int | bool]],
    args: Iterable[object] | None,
    opts: QuadOpts | _OptCallable | Iterable[QuadOpts | _OptCallable] | None,
    full_output: Truthy,
) -> tuple[float | int | bool, float | int | bool, _QuadOutputNC]: ...
@overload
def nquad(
    func: _QuadFuncN,
    ranges: _SizedIterable[_QuadRange | _RangeCallable[float | int | bool]],
    args: Iterable[object] | None = None,
    opts: QuadOpts | _OptCallable | Iterable[QuadOpts | _OptCallable] | None = None,
    *,
    full_output: Truthy,
) -> tuple[float | int | bool, float | int | bool, _QuadOutputNC]: ...
