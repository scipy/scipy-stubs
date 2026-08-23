from typing import Any, Literal as L, Never, SupportsIndex, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from . import _resampling
from ._stats_mstats_common import SiegelslopesResult, TheilslopesResult
from ._stats_py import SignificanceResult
from ._typing import NanPolicy

__all__ = ["chatterjeexi", "siegelslopes", "spearmanrho", "theilslopes"]

###

type _PermutationMethod = L["asymptotic"] | _resampling.PermutationMethod
type _Alternative = L["two-sided", "less", "greater"]

type _SlopesMethod = L["hierarchical", "separate"]

# workaround for https://github.com/microsoft/pyright/issues/10232
type _JustAnyShape = tuple[Never, Never, Never, Never]

type _AsF64_1D = onp.ToArrayStrict1D[float, npc.floating64 | npc.integer]
type _AsF64_2D = onp.ToArrayStrict2D[float, npc.floating64 | npc.integer]
type _AsF64_3D = onp.ToArrayStrict3D[float, npc.floating64 | npc.integer]
type _AsF64_ND = onp.ToArrayND[float, npc.floating64 | npc.integer]
type _AsF64StrictND = onp.ArrayND[npc.floating64 | npc.integer, _JustAnyShape]
type _AsF32StrictND = onp.ArrayND[np.float32, _JustAnyShape]

###

@overload  # 1d +f64, 1d +f64
def chatterjeexi(
    x: _AsF64_1D,
    y: _AsF64_1D,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float64]: ...
@overload  # 1d ~f32, 1d ~f32
def chatterjeexi(
    x: onp.ToJustFloat32Strict1D,
    y: onp.ToJustFloat32Strict1D,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float32]: ...
@overload  # <=2d +f64, 2d +f64
def chatterjeexi(
    x: _AsF64_1D | _AsF64_2D,
    y: _AsF64_2D,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[onp.Array1D[np.float64]]: ...
@overload  # 2d +f64, <=2d +f64
def chatterjeexi(
    x: _AsF64_2D,
    y: _AsF64_1D | _AsF64_2D,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[onp.Array1D[np.float64]]: ...
@overload  # 2d ~f32, 2d ~f32
def chatterjeexi(
    x: onp.ToJustFloat32Strict2D,
    y: onp.ToJustFloat32Strict2D,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[onp.Array1D[np.float32]]: ...
@overload  # ?d +f64, ?d|1d +f64
def chatterjeexi(
    x: _AsF64StrictND,
    y: _AsF64StrictND | _AsF64_1D,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float64 | onp.ArrayND[np.float64]]: ...
@overload  # ?d|1d +f64, ?d +f64
def chatterjeexi(
    x: _AsF64StrictND | _AsF64_1D,
    y: _AsF64StrictND,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float64 | onp.ArrayND[np.float64]]: ...
@overload  # ?d ~f32, ?d ~f32
def chatterjeexi(
    x: _AsF32StrictND,
    y: _AsF32StrictND,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float32 | onp.ArrayND[np.float32]]: ...
@overload  # ?d +f64, keepdims
def chatterjeexi(
    x: _AsF64_ND,
    y: _AsF64_ND,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[True],
) -> SignificanceResult[onp.ArrayND[np.float64]]: ...
@overload  # ?d ~f32, keepdims
def chatterjeexi(
    x: onp.ToJustFloat32_ND,
    y: onp.ToJustFloat32_ND,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[True],
) -> SignificanceResult[onp.ArrayND[np.float32]]: ...
@overload  # fallback
def chatterjeexi(
    x: onp.ToComplexND,
    y: onp.ToComplexND,
    *,
    axis: SupportsIndex = 0,
    y_continuous: bool = False,
    method: _PermutationMethod = "asymptotic",
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> SignificanceResult[np.float64 | Any]: ...

# keep in sync with `chatterjeexi` above
@overload  # 1d +f64, 1d +f64
def spearmanrho(
    x: _AsF64_1D,
    y: _AsF64_1D,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float64]: ...
@overload  # 1d ~f32, 1d ~f32
def spearmanrho(
    x: onp.ToJustFloat32Strict1D,
    y: onp.ToJustFloat32Strict1D,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float32]: ...
@overload  # <=2d +f64, 2d +f64
def spearmanrho(
    x: _AsF64_1D | _AsF64_2D,
    y: _AsF64_2D,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[onp.Array1D[np.float64]]: ...
@overload  # 2d +f64, <=2d +f64
def spearmanrho(
    x: _AsF64_2D,
    y: _AsF64_1D | _AsF64_2D,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[onp.Array1D[np.float64]]: ...
@overload  # 2d ~f32, 2d ~f32
def spearmanrho(
    x: onp.ToJustFloat32Strict2D,
    y: onp.ToJustFloat32Strict2D,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[onp.Array1D[np.float32]]: ...
@overload  # ?d +f64, ?d|1d +f64
def spearmanrho(
    x: _AsF64StrictND,
    y: _AsF64StrictND | _AsF64_1D,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float64 | onp.ArrayND[np.float64]]: ...
@overload  # ?d|1d +f64, ?d +f64
def spearmanrho(
    x: _AsF64StrictND | _AsF64_1D,
    y: _AsF64StrictND,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float64 | onp.ArrayND[np.float64]]: ...
@overload  # ?d ~f32, ?d ~f32
def spearmanrho(
    x: _AsF32StrictND,
    y: _AsF32StrictND,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> SignificanceResult[np.float32 | onp.ArrayND[np.float32]]: ...
@overload  # ?d +f64, keepdims
def spearmanrho(
    x: _AsF64_ND,
    y: _AsF64_ND,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[True],
) -> SignificanceResult[onp.ArrayND[np.float64]]: ...
@overload  # ?d ~f32, keepdims
def spearmanrho(
    x: onp.ToJustFloat32_ND,
    y: onp.ToJustFloat32_ND,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[True],
) -> SignificanceResult[onp.ArrayND[np.float32]]: ...
@overload  # fallback
def spearmanrho(
    x: onp.ToComplexND,
    y: onp.ToComplexND,
    /,
    *,
    alternative: _Alternative = "two-sided",
    method: _resampling.ResamplingMethod | None = None,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> SignificanceResult[np.float64 | Any]: ...

#
@overload  # ?d +f64, axis=None
def siegelslopes(
    y: _AsF64_ND,
    x: _AsF64_ND | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: None = None,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[np.float64]: ...
@overload  # ?d ~f32, axis=None
def siegelslopes(
    y: onp.ToJustFloat32_ND,
    x: onp.ToJustFloat32_ND | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: None = None,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[np.float32]: ...
@overload  # 1d +f64
def siegelslopes(
    y: _AsF64_1D,
    x: _AsF64_1D | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int | None = None,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[np.float64]: ...
@overload  # 1d ~f32
def siegelslopes(
    y: onp.ToJustFloat32Strict1D,
    x: onp.ToJustFloat32Strict1D | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int | None = None,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[np.float32]: ...
@overload  # 2d +f64
def siegelslopes(
    y: _AsF64_2D,
    x: _AsF64_2D | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[onp.Array1D[np.float64]]: ...
@overload  # 2d ~f32
def siegelslopes(
    y: onp.ToJustFloat32Strict2D,
    x: onp.ToJustFloat32Strict2D | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[onp.Array1D[np.float32]]: ...
@overload  # 3d +f64
def siegelslopes(
    y: _AsF64_3D,
    x: _AsF64_3D | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[onp.Array2D[np.float64]]: ...
@overload  # 3d ~f32
def siegelslopes(
    y: onp.ToJustFloat32Strict3D,
    x: onp.ToJustFloat32Strict3D | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[onp.Array2D[np.float32]]: ...
@overload  # ?d +f64, keepdims
def siegelslopes(
    y: _AsF64_ND,
    x: _AsF64_ND | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int | None = None,
    keepdims: L[True],
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[onp.ArrayND[np.float64]]: ...
@overload  # ?d ~f32, keepdims
def siegelslopes(
    y: onp.ToJustFloat32_ND,
    x: onp.ToJustFloat32_ND | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int | None = None,
    keepdims: L[True],
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[onp.ArrayND[np.float32]]: ...
@overload  # fallback
def siegelslopes(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    method: _SlopesMethod = "hierarchical",
    *,
    axis: int | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> SiegelslopesResult[np.float64 | Any]: ...

#
@overload  # ?d +f64, axis=None
def theilslopes(
    y: _AsF64_ND,
    x: _AsF64_ND | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: None = None,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[np.float64]: ...
@overload  # ?d ~f32, axis=None
def theilslopes(
    y: onp.ToJustFloat32_ND,
    x: onp.ToJustFloat32_ND | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: None = None,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[np.float32]: ...
@overload  # 1d +f64
def theilslopes(
    y: _AsF64_1D,
    x: _AsF64_1D | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int | None = None,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[np.float64]: ...
@overload  # 1d ~f32
def theilslopes(
    y: onp.ToJustFloat32Strict1D,
    x: onp.ToJustFloat32Strict1D | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int | None = None,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[np.float32]: ...
@overload  # 2d +f64
def theilslopes(
    y: _AsF64_2D,
    x: _AsF64_2D | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[onp.Array1D[np.float64]]: ...
@overload  # 2d ~f32
def theilslopes(
    y: onp.ToJustFloat32Strict2D,
    x: onp.ToJustFloat32Strict2D | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[onp.Array1D[np.float32]]: ...
@overload  # 3d +f64
def theilslopes(
    y: _AsF64_3D,
    x: _AsF64_3D | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[onp.Array2D[np.float64]]: ...
@overload  # 3d ~f32
def theilslopes(
    y: onp.ToJustFloat32Strict3D,
    x: onp.ToJustFloat32Strict3D | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int,
    keepdims: L[False] = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[onp.Array2D[np.float32]]: ...
@overload  # ?d +f64, keepdims
def theilslopes(
    y: _AsF64_ND,
    x: _AsF64_ND | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int | None = None,
    keepdims: L[True],
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[onp.ArrayND[np.float64]]: ...
@overload  # ?d ~f32, keepdims
def theilslopes(
    y: onp.ToJustFloat32_ND,
    x: onp.ToJustFloat32_ND | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int | None = None,
    keepdims: L[True],
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[onp.ArrayND[np.float32]]: ...
@overload  # fallback
def theilslopes(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    alpha: float | npc.floating = 0.95,
    method: _SlopesMethod = "separate",
    *,
    axis: int | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> TheilslopesResult[np.float64 | Any]: ...
