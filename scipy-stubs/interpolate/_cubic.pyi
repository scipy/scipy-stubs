from _typeshed import Incomplete
from types import ModuleType
from typing import Any, ClassVar, Generic, Literal, Never, overload, override
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._interpolate import PPoly

__all__ = ["Akima1DInterpolator", "CubicHermiteSpline", "CubicSpline", "PchipInterpolator", "pchip_interpolate"]

_CT_co = TypeVar("_CT_co", bound=np.float64 | np.complex128, default=np.float64, covariant=True)
_AxisT = TypeVar("_AxisT", bound=_ToAxis)

type _Tuple2[T] = tuple[T, T]
type _ToAxis = int | npc.integer

type _Akima1DMethod = Literal["akima", "makima"]
type _Extrapolate = Literal["periodic"] | bool
type _CubicBCName = Literal["not-a-knot", "clamped", "natural"]
type _CubicBCOrder = Literal[1, 2]
type _CubicBCType = Literal[_CubicBCName, "periodic"] | _Tuple2[_CubicBCName | tuple[_CubicBCOrder, onp.ToComplexND]]

type _PreparedInput[CT: np.float64 | np.complex128, AxisT: _ToAxis] = tuple[
    onp.Array1D[np.float64],  # x
    onp.Array1D[np.float64],  # dx
    onp.ArrayND[CT],  # y
    AxisT,  # axis
    onp.ArrayND[CT],  # dydx
]

###

class CubicHermiteSpline(PPoly[_CT_co]):
    @overload
    def __init__(
        self: CubicHermiteSpline[np.float64],
        /,
        x: onp.ToFloat1D,
        y: onp.ToFloatND,
        dydx: onp.ToFloatND,
        axis: _ToAxis = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: CubicHermiteSpline[np.complex128],
        /,
        x: onp.ToFloat1D,
        y: onp.ToJustComplexND,
        dydx: onp.ToComplexND,
        axis: _ToAxis = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: CubicHermiteSpline[Any],
        /,
        x: onp.ToFloat1D,
        y: onp.ToComplexND,
        dydx: onp.ToComplexND,
        axis: _ToAxis = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...

class PchipInterpolator(CubicHermiteSpline[np.float64]):
    # pyrefly: ignore [bad-override]
    __class_getitem__: ClassVar[None] = None  # type:ignore[assignment]  # pyright:ignore[reportIncompatibleMethodOverride]

    def __init__(self, /, x: onp.ToFloat1D, y: onp.ToFloatND, axis: _ToAxis = 0, extrapolate: bool | None = None) -> None: ...

class Akima1DInterpolator(CubicHermiteSpline[np.float64]):
    # pyrefly: ignore [bad-override]
    __class_getitem__: ClassVar[None] = None  # type:ignore[assignment]  # pyright:ignore[reportIncompatibleMethodOverride]

    def __init__(
        self,
        /,
        x: onp.ToFloat1D,
        y: onp.ToFloatND,
        axis: _ToAxis = 0,
        *,
        method: _Akima1DMethod = "akima",
        extrapolate: bool | None = None,
    ) -> None: ...

    # the following (class)methods will raise `NotImplementedError` when called
    @override
    # pyrefly: ignore [bad-override]
    def extend(self, /, c: Never, x: Never, right: bool = True) -> Never: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @classmethod
    @override
    # pyrefly: ignore [bad-override]
    def from_spline(cls, tck: Never, extrapolate: None = None) -> Never: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
    @classmethod
    @override
    # pyrefly: ignore [bad-override]
    def from_bernstein_basis(cls, bp: Never, extrapolate: None = None) -> Never: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]

class CubicSpline(CubicHermiteSpline[_CT_co], Generic[_CT_co]):
    @overload
    def __init__(
        self: CubicSpline[np.float64],
        /,
        x: onp.ToFloat1D,
        y: onp.ToFloatND,
        axis: _ToAxis = 0,
        bc_type: _CubicBCType = "not-a-knot",
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: CubicSpline[np.complex128],
        /,
        x: onp.ToFloat1D,
        y: onp.ToJustComplexND,
        axis: _ToAxis = 0,
        bc_type: _CubicBCType = "not-a-knot",
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: CubicSpline[Any],
        /,
        x: onp.ToFloat1D,
        y: onp.ToComplexND,
        axis: _ToAxis = 0,
        bc_type: _CubicBCType = "not-a-knot",
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...

@overload
def pchip_interpolate(
    xi: onp.ToFloat1D, yi: onp.ToFloat1D, x: onp.ToFloat, der: onp.ToInt = 0, axis: _ToAxis = 0
) -> np.float64: ...
@overload
def pchip_interpolate(
    xi: onp.ToFloat1D, yi: onp.ToFloat1D, x: onp.ToFloat1D, der: onp.ToInt | onp.ToInt1D = 0, axis: _ToAxis = 0
) -> onp.ArrayND[np.float64]: ...

# undocumented
@overload
def prepare_input(
    x: onp.ToFloat1D, y: onp.ToFloatND, axis: _AxisT, dydx: onp.ToFloatND | None = None, xp: None = None
) -> _PreparedInput[np.float64, _AxisT]: ...
@overload
def prepare_input(
    x: onp.ToFloat1D, y: onp.ToJustComplexND, axis: _AxisT, dydx: onp.ToComplexND | None = None, xp: None = None
) -> _PreparedInput[np.complex128, _AxisT]: ...
@overload
def prepare_input(
    x: onp.ToFloat1D, y: onp.ToComplexND, axis: _AxisT, dydx: onp.ToComplexND | None = None, xp: None = None
) -> _PreparedInput[Any, _AxisT]: ...
@overload
def prepare_input(
    x: onp.ToFloat1D, y: onp.ToComplexND, axis: _AxisT, dydx: onp.ToComplexND | None = None, *, xp: ModuleType
) -> tuple[Incomplete, Incomplete, Incomplete, _AxisT, Incomplete]: ...
