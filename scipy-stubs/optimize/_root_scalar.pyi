from collections.abc import Callable, Mapping
from typing import Concatenate, Final, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from ._typing import MethodRootScalar
from ._zeros_py import RootResults

__all__ = ["root_scalar"]

_ToFloat2: TypeAlias = tuple[onp.ToFloat, onp.ToFloat]
_ToFloat3: TypeAlias = tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat]

_RT = TypeVar("_RT", bound=onp.ToFloat | _ToFloat2 | _ToFloat3)
_RT2_co = TypeVar("_RT2_co", bound=_ToFloat2 | _ToFloat3, default=_ToFloat2 | _ToFloat3, covariant=True)

_Fun: TypeAlias = Callable[Concatenate[float | int | bool, ...], _RT] | Callable[Concatenate[np.float64, ...], _RT]
_Fun1: TypeAlias = _Fun[onp.ToFloat]
_Fun2: TypeAlias = _Fun[_ToFloat2]
_Fun3: TypeAlias = _Fun[_ToFloat3]

###

ROOT_SCALAR_METHODS: Final[list[MethodRootScalar]] = ...

class MemoizeDer(Generic[_RT2_co]):  # undocumented
    fun: _Fun[_RT2_co]  # readonly
    vals: _RT2_co | None
    x: onp.ToFloat
    n_calls: int | bool

    def __init__(self, /, fun: _Fun[_RT2_co]) -> None: ...
    def __call__(self, /, x: onp.ToFloat, *args: object) -> onp.ToFloat: ...
    def fprime(self, /, x: onp.ToFloat, *args: object) -> onp.ToFloat: ...
    def fprime2(self: MemoizeDer[_ToFloat3], /, x: onp.ToFloat, *args: object) -> onp.ToFloat: ...
    def ncalls(self, /) -> int | bool: ...

@overload  # bisect | brentq | brenth | ridder | toms748  (positional)
def root_scalar(
    f: _Fun1,
    args: tuple[object, ...],
    method: Literal["bisect", "brentq", "brenth", "ridder", "toms748"] | None,
    bracket: tuple[onp.ToFloat, onp.ToFloat],
    fprime: Falsy | None = None,
    fprime2: Falsy | None = None,
    x0: None = None,
    x1: None = None,
    xtol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    maxiter: onp.ToJustInt | None = None,
    options: Mapping[str, object] | None = None,
) -> RootResults[float | int | bool]: ...
@overload  # bisect | brentq | brenth | ridder | toms748  (keyword)
def root_scalar(
    f: _Fun1,
    args: tuple[object, ...] = (),
    method: Literal["bisect", "brentq", "brenth", "ridder", "toms748"] | None = None,
    *,
    bracket: tuple[onp.ToFloat, onp.ToFloat],
    fprime: Falsy | None = None,
    fprime2: Falsy | None = None,
    x0: None = None,
    x1: None = None,
    xtol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    maxiter: onp.ToJustInt | None = None,
    options: Mapping[str, object] | None = None,
) -> RootResults[float | int | bool]: ...
@overload  # secant
def root_scalar(
    f: _Fun1,
    args: tuple[object, ...] = (),
    method: Literal["secant"] | None = None,
    bracket: None = None,
    fprime: Falsy | None = None,
    fprime2: Falsy | None = None,
    *,
    x0: onp.ToFloat,
    x1: onp.ToFloat | None = None,
    xtol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    maxiter: onp.ToJustInt | None = None,
    options: Mapping[str, object] | None = None,
) -> RootResults[float | int | bool]: ...
@overload  # newton
def root_scalar(
    f: _Fun1,
    args: tuple[object, ...] = (),
    method: Literal["newton"] | None = None,
    bracket: None = None,
    fprime: _Fun1 | Falsy | None = None,
    fprime2: Falsy | None = None,
    *,
    x0: onp.ToFloat,
    x1: None = None,
    xtol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    maxiter: onp.ToJustInt | None = None,
    options: Mapping[str, object] | None = None,
) -> RootResults[float | int | bool]: ...
@overload  # newton  (fprime=True)
def root_scalar(
    f: _Fun2,
    args: tuple[object, ...] = (),
    method: Literal["newton"] | None = None,
    bracket: None = None,
    *,
    fprime: Truthy,
    fprime2: Falsy | None = None,
    x0: onp.ToFloat,
    x1: None = None,
    xtol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    maxiter: onp.ToJustInt | None = None,
    options: Mapping[str, object] | None = None,
) -> RootResults[float | int | bool]: ...
@overload  # halley
def root_scalar(
    f: _Fun1,
    args: tuple[object, ...] = (),
    method: Literal["halley"] | None = None,
    bracket: None = None,
    *,
    fprime: _Fun1,
    fprime2: _Fun1,
    x0: onp.ToFloat,
    x1: None = None,
    xtol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    maxiter: onp.ToJustInt | None = None,
    options: Mapping[str, object] | None = None,
) -> RootResults[float | int | bool]: ...
@overload  # halley (fprime=True)
def root_scalar(
    f: _Fun2,
    args: tuple[object, ...] = (),
    method: Literal["halley"] | None = None,
    bracket: None = None,
    *,
    fprime: Truthy,
    fprime2: _Fun1,
    x0: onp.ToFloat,
    x1: None = None,
    xtol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    maxiter: onp.ToJustInt | None = None,
    options: Mapping[str, object] | None = None,
) -> RootResults[float | int | bool]: ...
@overload  # halley (fprime=True, fprime2=True)
def root_scalar(
    f: _Fun3,
    args: tuple[object, ...] = (),
    method: Literal["halley"] | None = None,
    bracket: None = None,
    *,
    fprime: Truthy,
    fprime2: Truthy,
    x0: onp.ToFloat,
    x1: None = None,
    xtol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    maxiter: onp.ToJustInt | None = None,
    options: Mapping[str, object] | None = None,
) -> RootResults[float | int | bool]: ...

#
def _root_scalar_brentq_doc() -> None: ...  # undocumented
def _root_scalar_brenth_doc() -> None: ...  # undocumented
def _root_scalar_toms748_doc() -> None: ...  # undocumented
def _root_scalar_secant_doc() -> None: ...  # undocumented
def _root_scalar_newton_doc() -> None: ...  # undocumented
def _root_scalar_halley_doc() -> None: ...  # undocumented
def _root_scalar_ridder_doc() -> None: ...  # undocumented
def _root_scalar_bisect_doc() -> None: ...  # undocumented
