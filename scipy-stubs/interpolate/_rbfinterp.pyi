from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp

__all__ = ["RBFInterpolator"]

_Kernel: TypeAlias = Literal[
    "thin_plate_spline",
    "linear",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
]

_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int | bool, ...], default=onp.AtLeast1D, covariant=True)
_SCT_co = TypeVar("_SCT_co", bound=np.float64 | np.complex128, default=np.float64, covariant=True)

###

class RBFInterpolator(Generic[_ShapeT_co, _SCT_co]):
    y: onp.Array2D[np.float64]
    d: onp.Array[_ShapeT_co, np.float64]
    d_shape: _ShapeT_co
    d_dtype: type[float | int | bool | complex | float | int | bool]
    neighbors: int | bool
    smoothing: onp.Array1D[np.float64]
    kernel: _Kernel
    epsilon: float | int | bool
    powers: int | bool

    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool], np.float64],
        /,
        y: onp.ToFloat2D,
        d: onp.ToFloatStrict1D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool], np.complex128],
        /,
        y: onp.ToFloat2D,
        d: onp.ToJustComplexStrict1D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool], Any],
        /,
        y: onp.ToFloat2D,
        d: onp.ToComplexStrict1D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool, int | bool], np.float64],
        /,
        y: onp.ToFloat2D,
        d: onp.ToFloatStrict2D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool, int | bool], np.complex128],
        /,
        y: onp.ToFloat2D,
        d: onp.ToJustComplexStrict2D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool, int | bool], Any],
        /,
        y: onp.ToFloat2D,
        d: onp.ToComplexStrict2D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool, int | bool, int | bool], np.float64],
        /,
        y: onp.ToFloat2D,
        d: onp.ToFloatStrict3D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool, int | bool, int | bool], np.complex128],
        /,
        y: onp.ToFloat2D,
        d: onp.ToJustComplexStrict3D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int | bool, int | bool, int | bool], Any],
        /,
        y: onp.ToFloat2D,
        d: onp.ToComplexStrict3D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[onp.AtLeast1D, np.float64],
        /,
        y: onp.ToFloat2D,
        d: onp.ToFloatND,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[onp.AtLeast1D, np.complex128],
        /,
        y: onp.ToFloat2D,
        d: onp.ToJustComplexND,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[onp.AtLeast1D, Any],
        /,
        y: onp.ToFloat2D,
        d: onp.ToComplexND,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...

    # TODO(jorenham): Return `onp.Array[tuple[int | bool, Unpack[_ShapeT_co]], _SCT_co]` once mypy supports it (if ever)
    def __call__(self, /, x: onp.ToFloat2D) -> onp.ArrayND[_SCT_co]: ...
