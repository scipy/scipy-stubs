import types
from typing import Any, Generic, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["AAA", "FloaterHormannInterpolator"]

_ScalarT_co = TypeVar("_ScalarT_co", bound=npc.inexact, default=Any, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int, ...], default=tuple[Any, ...], covariant=True)

###

# NOTE: We disable `overload-overlap` here because these false positives only occur with numpy>=2.2
# mypy: disable-error-code="overload-overlap"

class _BarycentricRational(Generic[_ScalarT_co, _ShapeT_co]):
    @classmethod
    def __class_getitem__(cls, arg: object, /) -> types.GenericAlias: ...

    #
    def __init__(self, /, x: onp.ToComplex1D, y: onp.ToComplexND, axis: int = 0) -> None: ...

    #
    @overload
    def __call__[ST: npc.inexact, ShapeT: tuple[int, ...]](
        self: _BarycentricRational[ST, tuple[int]], /, z: onp.ArrayND[ST, ShapeT]
    ) -> onp.ArrayND[ST, ShapeT]: ...
    @overload
    def __call__(self, /, z: onp.ToInt) -> onp.ArrayND[_ScalarT_co, _ShapeT_co]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[np.float64, ShapeT], /, z: onp.ToFloat64
    ) -> onp.ArrayND[np.float64, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[np.float32, ShapeT], /, z: onp.ToFloat32
    ) -> onp.ArrayND[np.float32, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[np.float32, ShapeT], /, z: onp.ToJustFloat64
    ) -> onp.ArrayND[np.float64, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[np.float32, ShapeT], /, z: onp.ToJustComplex128
    ) -> onp.ArrayND[np.complex128, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.floating80, ShapeT], /, z: onp.ToFloat
    ) -> onp.ArrayND[np.longdouble, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[np.complex128, ShapeT], /, z: onp.ToComplex128
    ) -> onp.ArrayND[np.complex128, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[np.complex64, ShapeT], /, z: onp.ToComplex64
    ) -> onp.ArrayND[np.complex64, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[np.complex64, ShapeT], /, z: onp.ToJustComplex128
    ) -> onp.ArrayND[np.complex128, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.complexfloating160, ShapeT], /, z: onp.ToComplex
    ) -> onp.ArrayND[np.clongdouble, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.floating, ShapeT], /, z: onp.ToFloat
    ) -> onp.ArrayND[npc.floating, ShapeT]: ...
    @overload
    def __call__[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.complexfloating, ShapeT], /, z: onp.ToComplex
    ) -> onp.ArrayND[npc.complexfloating, ShapeT]: ...
    @overload
    def __call__(self, /, z: onp.ToJustComplex) -> onp.ArrayND[npc.complexfloating, _ShapeT_co]: ...
    @overload
    def __call__(self, /, z: onp.ToIntND) -> onp.ArrayND[_ScalarT_co]: ...
    @overload
    def __call__(self: _BarycentricRational[np.float64], /, z: onp.ToFloat64_ND) -> onp.ArrayND[np.float64]: ...
    @overload
    def __call__(self: _BarycentricRational[np.float32], /, z: onp.ToFloat32_ND) -> onp.ArrayND[np.float32]: ...
    @overload
    def __call__(self: _BarycentricRational[np.float32], /, z: onp.ToJustFloat64_ND) -> onp.ArrayND[np.float64]: ...
    @overload
    def __call__(self: _BarycentricRational[np.float32], /, z: onp.ToJustComplex128_ND) -> onp.ArrayND[np.complex128]: ...
    @overload
    def __call__(self: _BarycentricRational[npc.floating80], /, z: onp.ToFloatND) -> onp.ArrayND[np.longdouble]: ...
    @overload
    def __call__(self: _BarycentricRational[np.complex128], /, z: onp.ToComplex128_ND) -> onp.ArrayND[np.complex128]: ...
    @overload
    def __call__(self: _BarycentricRational[np.complex64], /, z: onp.ToComplex64_ND) -> onp.ArrayND[np.complex64]: ...
    @overload
    def __call__(self: _BarycentricRational[np.complex64], /, z: onp.ToJustComplex128_ND) -> onp.ArrayND[np.complex128]: ...
    @overload
    def __call__(self: _BarycentricRational[npc.complexfloating160], /, z: onp.ToComplexND) -> onp.ArrayND[np.clongdouble]: ...
    @overload
    def __call__(self: _BarycentricRational[npc.floating], /, z: onp.ToFloatND) -> onp.ArrayND[npc.floating]: ...
    @overload
    def __call__(self: _BarycentricRational[npc.complexfloating], /, z: onp.ToComplexND) -> onp.ArrayND[npc.complexfloating]: ...
    @overload
    def __call__(self, /, z: onp.ToJustComplexND) -> onp.ArrayND[npc.complexfloating]: ...

    #
    def residues(self, /) -> onp.ArrayND[_ScalarT_co, _ShapeT_co]: ...

    #
    @overload
    def poles[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.inexact64 | np.float16, ShapeT], /
    ) -> onp.ArrayND[np.complex128, ShapeT]: ...
    @overload
    def poles[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.inexact32, ShapeT], /
    ) -> onp.ArrayND[np.complex64, ShapeT]: ...
    @overload
    def poles[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.inexact80, ShapeT], /
    ) -> onp.ArrayND[np.clongdouble, ShapeT]: ...
    @overload
    def poles[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.inexact, ShapeT], /
    ) -> onp.ArrayND[npc.complexfloating, ShapeT]: ...

    #
    @overload
    def roots[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.inexact64 | np.float16, ShapeT], /
    ) -> onp.ArrayND[np.complex128, ShapeT]: ...
    @overload
    def roots[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.inexact32, ShapeT], /
    ) -> onp.ArrayND[np.complex64, ShapeT]: ...
    @overload
    def roots[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.inexact80, ShapeT], /
    ) -> onp.ArrayND[np.clongdouble, ShapeT]: ...
    @overload
    def roots[ShapeT: tuple[int, ...]](
        self: _BarycentricRational[npc.inexact, ShapeT], /
    ) -> onp.ArrayND[npc.complexfloating, ShapeT]: ...

class AAA(_BarycentricRational[_ScalarT_co, tuple[int]], Generic[_ScalarT_co]):
    weights: onp.Array1D[_ScalarT_co]

    @property
    def support_points(self, /) -> onp.Array1D[_ScalarT_co]: ...
    @property
    def support_values(self, /) -> onp.Array1D[_ScalarT_co]: ...

    #
    @overload
    def __init__(
        self: AAA[np.float64],
        /,
        x: onp.ToArray1D[float, npc.floating64 | npc.floating16 | npc.integer | np.bool],
        y: onp.ToArray1D[float, npc.floating64 | npc.floating16 | npc.integer | np.bool],
        *,
        rtol: float | None = None,
        max_terms: int = 100,
        clean_up: bool = True,
        clean_up_tol: float = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self: AAA[np.float64],
        /,
        x: onp.ToJustFloat64_1D | onp.ToJustFloat16_1D,
        y: onp.ToFloat64_1D,
        *,
        rtol: float | None = None,
        max_terms: int = 100,
        clean_up: bool = True,
        clean_up_tol: float = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self: AAA[np.float64],
        /,
        x: onp.ToFloat64_1D,
        y: onp.ToJustFloat64_1D | onp.ToJustFloat16_1D,
        *,
        rtol: float | None = None,
        max_terms: int = 100,
        clean_up: bool = True,
        clean_up_tol: float = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self: AAA[np.complex128],
        /,
        x: onp.ToJustComplex128_1D,
        y: onp.ToComplex128_1D,
        *,
        rtol: float | None = None,
        max_terms: int = 100,
        clean_up: bool = True,
        clean_up_tol: float = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self: AAA[np.complex128],
        /,
        x: onp.ToComplex128_1D,
        y: onp.ToJustComplex128_1D,
        *,
        rtol: float | None = None,
        max_terms: int = 100,
        clean_up: bool = True,
        clean_up_tol: float = 1e-13,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        x: onp.ToArray1D[_ScalarT_co, _ScalarT_co],
        y: onp.ToArray1D[_ScalarT_co, _ScalarT_co],
        *,
        rtol: float | None = None,
        max_terms: int = 100,
        clean_up: bool = True,
        clean_up_tol: float = 1e-13,
    ) -> None: ...

    #
    def clean_up(self, /, cleanup_tol: float = 1e-13) -> int: ...

class FloaterHormannInterpolator(_BarycentricRational[_ScalarT_co, _ShapeT_co], Generic[_ScalarT_co, _ShapeT_co]):
    @overload
    def __init__(
        self: FloaterHormannInterpolator[np.float64, tuple[int]],
        /,
        points: onp.ToArray1D[float, npc.floating64 | npc.floating16 | npc.integer | np.bool],
        values: onp.ToArrayStrict1D[float, npc.floating64 | npc.floating16 | npc.integer | np.bool],
        *,
        d: int = 3,
        axis: int = 0,
    ) -> None: ...
    @overload
    def __init__[ST: npc.inexact, ShapeT: tuple[int, ...]](
        self: FloaterHormannInterpolator[ST, ShapeT],
        /,
        points: onp.ToArray1D[int, ST],
        values: onp.ArrayND[ST, ShapeT],
        *,
        d: int = 3,
        axis: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: FloaterHormannInterpolator[npc.floating, tuple[int]],
        /,
        points: onp.ToFloat1D,
        values: onp.ToFloatStrict1D,
        *,
        d: int = 3,
        axis: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: FloaterHormannInterpolator[npc.floating, tuple[int, int]],
        /,
        points: onp.ToFloat1D,
        values: onp.ToFloatStrict2D,
        *,
        d: int = 3,
        axis: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: FloaterHormannInterpolator[npc.inexact, tuple[int]],
        /,
        points: onp.ToComplex1D,
        values: onp.ToComplexStrict1D,
        *,
        d: int = 3,
        axis: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: FloaterHormannInterpolator[npc.inexact, tuple[int, int]],
        /,
        points: onp.ToComplex1D,
        values: onp.ToComplexStrict2D,
        *,
        d: int = 3,
        axis: int = 0,
    ) -> None: ...
    @overload
    def __init__(self, /, points: onp.ToComplex1D, values: onp.ToComplexND, *, d: int = 3, axis: int = 0) -> None: ...
