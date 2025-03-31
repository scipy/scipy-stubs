from collections.abc import Sequence
from typing import Any, Generic, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp

__all__ = ["AAA", "FloaterHormannInterpolator"]

_SCT = TypeVar("_SCT", bound=np.inexact[Any])
_SCT_co = TypeVar("_SCT_co", bound=np.inexact[Any], default=np.inexact[Any], covariant=True)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int | bool, ...])
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int | bool, ...], default=tuple[int | bool, ...], covariant=True)

_ToFloat16: TypeAlias = np.bool_ | np.integer[Any] | np.float16
_ToFloat64: TypeAlias = _ToFloat16 | np.float32 | np.float64
_ToComplex128: TypeAlias = _ToFloat64 | np.complex64 | np.complex128

###

class _BarycentricRational(Generic[_SCT_co, _ShapeT_co]):
    def __init__(self, /, x: onp.ToComplex1D, y: onp.ToComplexND) -> None: ...

    #
    @overload
    def __call__(
        self: _BarycentricRational[_SCT, tuple[int | bool]],
        /,
        z: onp.ArrayND[_SCT, _ShapeT],
    ) -> onp.ArrayND[_SCT, _ShapeT]: ...
    @overload
    def __call__(self, /, z: onp.ToFloat) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload
    def __call__(self, /, z: onp.ToComplex) -> onp.ArrayND[np.inexact[Any], _ShapeT_co]: ...
    @overload
    def __call__(self, /, z: onp.ToComplexND) -> onp.ArrayND[np.inexact[Any]]: ...

    #
    def poles(self, /) -> onp.Array1D[np.complexfloating[Any, Any]]: ...
    def roots(self, /) -> onp.Array1D[np.complexfloating[Any, Any]]: ...
    def residues(self, /) -> onp.ArrayND[np.inexact[Any]]: ...

class AAA(_BarycentricRational[_SCT_co, tuple[int | bool]], Generic[_SCT_co]):
    weights: onp.Array1D[_SCT_co]

    @property
    def support_points(self, /) -> onp.Array1D[_SCT_co]: ...
    @property
    def support_values(self, /) -> onp.Array1D[_SCT_co]: ...

    #
    @overload
    def __init__(
        self,
        /,
        x: onp.CanArrayND[_SCT_co] | Sequence[_SCT_co],
        y: onp.CanArrayND[_SCT_co | _ToFloat16] | Sequence[_SCT_co | _ToFloat16],
        *,
        rtol: float | int | bool | None = None,
        max_terms: int | bool = 100,
        clean_up: bool = True,
        clean_up_tol: float | int | bool = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self: AAA[np.float64],
        /,
        x: Sequence[float | int | bool],
        y: onp.CanArrayND[_ToFloat64] | Sequence[float | int | bool | _ToFloat64],
        *,
        rtol: float | int | bool | None = None,
        max_terms: int | bool = 100,
        clean_up: bool = True,
        clean_up_tol: float | int | bool = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self: AAA[np.float64],
        /,
        x: onp.CanArrayND[_ToFloat64] | Sequence[float | int | bool | _ToFloat64],
        y: Sequence[float | int | bool],
        *,
        rtol: float | int | bool | None = None,
        max_terms: int | bool = 100,
        clean_up: bool = True,
        clean_up_tol: float | int | bool = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self: AAA[np.complex128],
        /,
        x: Sequence[complex | float | int | bool],
        y: onp.CanArrayND[_ToFloat64] | Sequence[complex | float | int | bool | _ToComplex128],
        *,
        rtol: float | int | bool | None = None,
        max_terms: int | bool = 100,
        clean_up: bool = True,
        clean_up_tol: float | int | bool = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self: AAA[np.complex128],
        /,
        x: onp.CanArrayND[_ToComplex128] | Sequence[complex | float | int | bool | _ToComplex128],
        y: Sequence[complex | float | int | bool],
        *,
        rtol: float | int | bool | None = None,
        max_terms: int | bool = 100,
        clean_up: bool = True,
        clean_up_tol: float | int | bool = 1e-13,
    ) -> None: ...

    #
    def clean_up(self, /, cleanup_tol: float | int | bool = 1e-13) -> int | bool: ...

class FloaterHormannInterpolator(_BarycentricRational[_SCT_co, _ShapeT_co], Generic[_SCT_co, _ShapeT_co]):
    @overload
    def __init__(
        self,
        /,
        points: onp.CanArrayND[_SCT_co | _ToFloat16] | Sequence[_SCT_co | _ToFloat16 | int | bool],
        values: onp.CanArrayND[_SCT_co, _ShapeT_co],
        *,
        d: int | bool = 3,
    ) -> None: ...
    @overload
    def __init__(
        self: FloaterHormannInterpolator[np.floating[Any], tuple[int | bool]],
        /,
        points: onp.ToFloat1D,
        values: onp.ToFloatStrict1D,
        *,
        d: int | bool = 3,
    ) -> None: ...
    @overload
    def __init__(
        self: FloaterHormannInterpolator[np.floating[Any], tuple[int | bool, int | bool]],
        /,
        points: onp.ToFloat1D,
        values: onp.ToFloatStrict2D,
        *,
        d: int | bool = 3,
    ) -> None: ...
    @overload
    def __init__(
        self: FloaterHormannInterpolator[np.inexact[Any], tuple[int | bool]],
        /,
        points: onp.ToComplex1D,
        values: onp.ToComplexStrict1D,
        *,
        d: int | bool = 3,
    ) -> None: ...
    @overload
    def __init__(
        self: FloaterHormannInterpolator[np.inexact[Any], tuple[int | bool, int | bool]],
        /,
        points: onp.ToComplex1D,
        values: onp.ToComplexStrict2D,
        *,
        d: int | bool = 3,
    ) -> None: ...
    @overload
    def __init__(self, /, points: onp.ToComplex1D, values: onp.ToComplexND, *, d: int | bool = 3) -> None: ...
