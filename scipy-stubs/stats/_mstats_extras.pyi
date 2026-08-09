from typing import Any, Never, SupportsIndex, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = [
    "compare_medians_ms",
    "hdmedian",
    "hdquantiles",
    "hdquantiles_sd",
    "idealfourths",
    "median_cihs",
    "mjci",
    "mquantiles_cimj",
    "rsh",
    "trimmed_mean_ci",
]

###

type _Tuple2[T] = tuple[T, T]
type _FloatND = onp.ArrayND[np.float64]

type _JustAnyShape = tuple[Never, Never, Never, Never]  # workaround for https://github.com/microsoft/pyright/issues/10232
type _ToFloatStrictND = onp.ArrayND[npc.floating | npc.integer | np.bool, _JustAnyShape]

type _ToProb = onp.ToFloat | onp.ToFloatND
type _ToPoints = onp.ToFloat | onp.ToFloat1D | None
type _ToAxis = SupportsIndex | None

###

@overload  # ?d, axis=None (default), var=False (default)
def hdquantiles(
    data: onp.ToFloatND, prob: _ToProb = (0.25, 0.5, 0.75), axis: None = None, var: onp.ToFalse = False
) -> onp.MArray1D[np.float64]: ...
@overload  # ?d, axis=None (default), var=True
def hdquantiles(data: onp.ToFloatND, prob: _ToProb, axis: None, var: onp.ToTrue) -> onp.MArray2D[np.float64]: ...
@overload  # ?d, axis=None (default), var=True
def hdquantiles(
    data: onp.ToFloatND, prob: _ToProb = (0.25, 0.5, 0.75), axis: None = None, *, var: onp.ToTrue
) -> onp.MArray2D[np.float64]: ...
@overload  # ?d, axis=<given>, var=False (default)
def hdquantiles(
    data: _ToFloatStrictND, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex, var: onp.ToFalse = False
) -> onp.MArray[np.float64]: ...
@overload  # ?d, axis=<given>, var=True
def hdquantiles(
    data: _ToFloatStrictND, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex, var: onp.ToTrue
) -> onp.MArray[np.float64]: ...
@overload  # 1d, axis=<given>, var=False (default)
def hdquantiles(
    data: onp.ToFloatStrict1D, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex, var: onp.ToFalse = False
) -> onp.MArray1D[np.float64]: ...
@overload  # 1d, axis=<given>, var=True
def hdquantiles(
    data: onp.ToFloatStrict1D, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex, var: onp.ToTrue
) -> onp.MArray2D[np.float64]: ...
@overload  # 2d, axis=<given>, var=False (default)
def hdquantiles(
    data: onp.ToFloatStrict2D, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex, var: onp.ToFalse = False
) -> onp.MArray2D[np.float64]: ...
@overload  # 2d, axis=<given>, var=True
def hdquantiles(
    data: onp.ToFloatStrict2D, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex, var: onp.ToTrue
) -> onp.MArray3D[np.float64]: ...
@overload  # fallback
def hdquantiles(
    data: onp.ToFloatND, prob: _ToProb = (0.25, 0.5, 0.75), axis: _ToAxis = None, var: bool = False
) -> onp.MArray[np.float64]: ...

#
@overload  # ?d, axis=None, var=False (default)
def hdmedian(data: onp.ToFloatND, axis: None, var: onp.ToFalse = False) -> onp.MArray0D[np.float64]: ...
@overload  # ?d, axis=None, var=True
def hdmedian(data: onp.ToFloatND, axis: None, var: onp.ToTrue) -> onp.MArray1D[np.float64]: ...
@overload  # ?d, axis=<given> (default), var=False (default)
def hdmedian(data: _ToFloatStrictND, axis: SupportsIndex = -1, var: onp.ToFalse = False) -> onp.MArray[np.float64]: ...
@overload  # ?d, axis=<given> (default), var=True
def hdmedian(data: _ToFloatStrictND, axis: SupportsIndex, var: onp.ToTrue) -> onp.MArray[np.float64]: ...
@overload  # ?d, axis=<given> (default), var=True
def hdmedian(data: _ToFloatStrictND, axis: SupportsIndex = -1, *, var: onp.ToTrue) -> onp.MArray[np.float64]: ...
@overload  # 1d, axis=<given> (default), var=False (default)
def hdmedian(data: onp.ToFloatStrict1D, axis: SupportsIndex = -1, var: onp.ToFalse = False) -> onp.MArray0D[np.float64]: ...
@overload  # 1d, axis=<given> (default), var=True
def hdmedian(data: onp.ToFloatStrict1D, axis: SupportsIndex, var: onp.ToTrue) -> onp.MArray1D[np.float64]: ...
@overload  # 1d, axis=<given> (default), var=True
def hdmedian(data: onp.ToFloatStrict1D, axis: SupportsIndex = -1, *, var: onp.ToTrue) -> onp.MArray1D[np.float64]: ...
@overload  # 2d, axis=<given> (default), var=False (default)
def hdmedian(data: onp.ToFloatStrict2D, axis: SupportsIndex = -1, var: onp.ToFalse = False) -> onp.MArray1D[np.float64]: ...
@overload  # 2d, axis=<given> (default), var=True
def hdmedian(data: onp.ToFloatStrict2D, axis: SupportsIndex, var: onp.ToTrue) -> onp.MArray2D[np.float64]: ...
@overload  # 2d, axis=<given> (default), var=True
def hdmedian(data: onp.ToFloatStrict2D, axis: SupportsIndex = -1, *, var: onp.ToTrue) -> onp.MArray2D[np.float64]: ...
@overload  # fallback
def hdmedian(data: onp.ToFloatND, axis: _ToAxis = -1, var: bool = False) -> onp.MArray[np.float64]: ...

#
def hdquantiles_sd(data: onp.ToFloatND, prob: _ToProb = (0.25, 0.5, 0.75), axis: _ToAxis = None) -> onp.MArray[np.float64]: ...

#
@overload  # ?d +f64, axis=None (default)
def trimmed_mean_ci(
    data: onp.ToFloat64_ND,
    limits: _Tuple2[onp.ToFloat] | None = (0.2, 0.2),
    inclusive: _Tuple2[bool] = (True, True),
    alpha: float | npc.floating = 0.05,
    axis: None = None,
) -> onp.Array1D[np.float64]: ...
@overload  # ?d ~f80, axis=None (default)
def trimmed_mean_ci(
    data: onp.ToJustLongDoubleND,
    limits: _Tuple2[onp.ToFloat] | None = (0.2, 0.2),
    inclusive: _Tuple2[bool] = (True, True),
    alpha: float | npc.floating = 0.05,
    axis: None = None,
) -> onp.Array1D[np.longdouble]: ...
@overload  # ?d, axis=<given>
def trimmed_mean_ci(
    data: _ToFloatStrictND,
    limits: _Tuple2[onp.ToFloat] | None = (0.2, 0.2),
    inclusive: _Tuple2[bool] = (True, True),
    alpha: float | npc.floating = 0.05,
    *,
    axis: SupportsIndex,
) -> onp.ArrayND[np.float64] | Any: ...
@overload  # 1d +f64, axis=<given>
def trimmed_mean_ci(
    data: onp.ToFloat64Strict1D,
    limits: _Tuple2[onp.ToFloat] | None = (0.2, 0.2),
    inclusive: _Tuple2[bool] = (True, True),
    alpha: float | npc.floating = 0.05,
    *,
    axis: SupportsIndex,
) -> onp.Array1D[np.float64]: ...
@overload  # 1d ~f80, axis=<given>
def trimmed_mean_ci(
    data: onp.ToJustLongDoubleStrict1D,
    limits: _Tuple2[onp.ToFloat] | None = (0.2, 0.2),
    inclusive: _Tuple2[bool] = (True, True),
    alpha: float | npc.floating = 0.05,
    *,
    axis: SupportsIndex,
) -> onp.Array1D[np.longdouble]: ...
@overload  # 2d +f64, axis=<given>
def trimmed_mean_ci(
    data: onp.ToFloat64Strict2D,
    limits: _Tuple2[onp.ToFloat] | None = (0.2, 0.2),
    inclusive: _Tuple2[bool] = (True, True),
    alpha: float | npc.floating = 0.05,
    *,
    axis: SupportsIndex,
) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~f80, axis=<given>
def trimmed_mean_ci(
    data: onp.ToJustLongDoubleStrict2D,
    limits: _Tuple2[onp.ToFloat] | None = (0.2, 0.2),
    inclusive: _Tuple2[bool] = (True, True),
    alpha: float | npc.floating = 0.05,
    *,
    axis: SupportsIndex,
) -> onp.Array2D[np.longdouble]: ...
@overload  # fallback
def trimmed_mean_ci(
    data: onp.ToFloatND,
    limits: _Tuple2[onp.ToFloat] | None = (0.2, 0.2),
    inclusive: _Tuple2[bool] = (True, True),
    alpha: float | npc.floating = 0.05,
    axis: SupportsIndex | None = None,
) -> onp.ArrayND[np.float64] | Any: ...

#
@overload  # ?d, axis=None (default)
def mjci(data: onp.ToFloatND, prob: _ToProb = (0.25, 0.5, 0.75), axis: None = None) -> onp.Array1D[np.float64]: ...
@overload  # ?d, axis=<given>
def mjci(data: _ToFloatStrictND, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex) -> onp.ArrayND[np.float64] | Any: ...
@overload  # 1d, axis=<given>
def mjci(data: onp.ToFloatStrict1D, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex) -> onp.MArray1D[np.float64]: ...
@overload  # 2d, axis=<given>
def mjci(data: onp.ToFloatStrict2D, prob: _ToProb = (0.25, 0.5, 0.75), *, axis: SupportsIndex) -> onp.MArray2D[np.float64]: ...
@overload  # fallback
def mjci(data: onp.ToFloatND, prob: _ToProb = (0.25, 0.5, 0.75), axis: _ToAxis = None) -> onp.ArrayND[np.float64] | Any: ...

#
@overload  # ?d +f64, axis=None (default)
def mquantiles_cimj(
    data: onp.ToFloat64_ND, prob: _ToProb = (0.25, 0.5, 0.75), alpha: float | npc.floating = 0.05, axis: None = None
) -> _Tuple2[onp.Array1D[np.float64]]: ...
@overload  # ?d ~f80, axis=None (default)
def mquantiles_cimj(
    data: onp.ToJustLongDoubleND, prob: _ToProb = (0.25, 0.5, 0.75), alpha: float | npc.floating = 0.05, axis: None = None
) -> _Tuple2[onp.Array1D[np.longdouble]]: ...
@overload  # ?d, axis=<given>
def mquantiles_cimj(
    data: _ToFloatStrictND, prob: _ToProb = (0.25, 0.5, 0.75), alpha: float | npc.floating = 0.05, *, axis: SupportsIndex
) -> _Tuple2[onp.ArrayND[np.float64] | Any]: ...
@overload  # 1d +f64, axis=<given>
def mquantiles_cimj(
    data: onp.ToFloat64Strict1D, prob: _ToProb = (0.25, 0.5, 0.75), alpha: float | npc.floating = 0.05, *, axis: SupportsIndex
) -> _Tuple2[onp.MArray1D[np.float64]]: ...
@overload  # 1d ~f80, axis=<given>
def mquantiles_cimj(
    data: onp.ToJustLongDoubleStrict1D,
    prob: _ToProb = (0.25, 0.5, 0.75),
    alpha: float | npc.floating = 0.05,
    *,
    axis: SupportsIndex,
) -> _Tuple2[onp.MArray1D[np.longdouble]]: ...
@overload  # 2d +f64, axis=<given>
def mquantiles_cimj(
    data: onp.ToFloat64Strict2D, prob: _ToProb = (0.25, 0.5, 0.75), alpha: float | npc.floating = 0.05, *, axis: SupportsIndex
) -> _Tuple2[onp.MArray2D[np.float64]]: ...
@overload  # 2d ~f80, axis=<given>
def mquantiles_cimj(
    data: onp.ToJustLongDoubleStrict2D,
    prob: _ToProb = (0.25, 0.5, 0.75),
    alpha: float | npc.floating = 0.05,
    *,
    axis: SupportsIndex,
) -> _Tuple2[onp.MArray2D[np.longdouble]]: ...
@overload  # fallback
def mquantiles_cimj(
    data: onp.ToFloatND, prob: _ToProb = (0.25, 0.5, 0.75), alpha: float | npc.floating = 0.05, axis: _ToAxis = None
) -> _Tuple2[onp.ArrayND[np.float64] | Any]: ...

#
@overload
def median_cihs(data: onp.ToFloatND, alpha: float | npc.floating = 0.05, axis: None = None) -> _Tuple2[np.float64]: ...
@overload
def median_cihs(data: onp.ToFloatND, alpha: float | npc.floating, axis: SupportsIndex) -> _Tuple2[np.float64 | _FloatND]: ...
@overload
def median_cihs(
    data: onp.ToFloatND, alpha: float | npc.floating = 0.05, *, axis: SupportsIndex
) -> _Tuple2[np.float64 | _FloatND]: ...

#
@overload
def compare_medians_ms(group_1: onp.ToFloatND, group_2: onp.ToFloatND, axis: None = None) -> np.float64: ...
@overload
def compare_medians_ms(group_1: onp.ToFloatND, group_2: onp.ToFloatND, axis: SupportsIndex) -> _FloatND: ...

#
@overload
def idealfourths(data: onp.ToFloatND, axis: None = None) -> list[np.float64]: ...
@overload
def idealfourths(data: onp.ToFloatND, axis: SupportsIndex) -> onp.MArray[np.float64]: ...

#
@overload  # ~f64
def rsh(data: onp.ToFloat64_1D, points: _ToPoints = None) -> onp.MArray1D[np.float64]: ...
@overload  # T@floating80
def rsh(data: onp.ToArray1D[npc.floating80, npc.floating80], points: _ToPoints = None) -> onp.MArray1D[np.longdouble]: ...
