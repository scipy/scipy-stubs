from typing import Any, Never, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._resampling import PermutationMethod, PermutationTestResult
from ._typing import Alternative

###

type _JustAnyShape = tuple[Never, Never, Never, Never]

###

@overload  # Nd +f64, axis: None
def bws_test(
    x: onp.ToArrayND[float, npc.integer],
    y: onp.ToArrayND[float, npc.integer],
    *,
    alternative: Alternative = "two-sided",
    axis: None,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[np.float64, onp.Array1D[np.float64]]: ...
@overload  # Nd floating, axis: None
def bws_test[FloatT: (np.float64, np.float32, np.float16)](
    x: onp.ToArrayND[FloatT, FloatT],
    y: onp.ToArrayND[FloatT, FloatT],
    *,
    alternative: Alternative = "two-sided",
    axis: None,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[FloatT, onp.Array1D[FloatT]]: ...
@overload  # ?d +f64
def bws_test(
    x: onp.ArrayND[npc.integer, _JustAnyShape],
    y: onp.ArrayND[npc.integer, _JustAnyShape],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.ArrayND[np.float64] | np.float64, onp.ArrayND[np.float64]]: ...
@overload  # ?d floating
def bws_test[FloatT: (np.float64, np.float32, np.float16)](
    x: onp.ArrayND[FloatT, _JustAnyShape],
    y: onp.ArrayND[FloatT, _JustAnyShape],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.ArrayND[FloatT] | FloatT, onp.ArrayND[FloatT]]: ...
@overload  # 1d +f64
def bws_test(
    x: onp.ToArrayStrict1D[float, npc.integer],
    y: onp.ToArrayStrict1D[float, npc.integer],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[np.float64, onp.Array1D[np.float64]]: ...
@overload  # 1d floating
def bws_test[FloatT: (np.float64, np.float32, np.float16)](
    x: onp.ToArrayStrict1D[FloatT, FloatT],
    y: onp.ToArrayStrict1D[FloatT, FloatT],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[FloatT, onp.Array1D[FloatT]]: ...
@overload  # 2d +f64
def bws_test(
    x: onp.ToArrayStrict2D[float, npc.integer],
    y: onp.ToArrayStrict2D[float, npc.integer],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]]: ...
@overload  # 2d floating
def bws_test[FloatT: (np.float64, np.float32, np.float16)](
    x: onp.ToArrayStrict2D[FloatT, FloatT],
    y: onp.ToArrayStrict2D[FloatT, FloatT],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.Array1D[FloatT], onp.Array2D[FloatT]]: ...
@overload  # 3d +f64
def bws_test(
    x: onp.ToArrayStrict3D[float, npc.integer],
    y: onp.ToArrayStrict3D[float, npc.integer],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.Array2D[np.float64], onp.Array3D[np.float64]]: ...
@overload  # 3d floating
def bws_test[FloatT: (np.float64, np.float32, np.float16)](
    x: onp.ToArrayStrict3D[FloatT, FloatT],
    y: onp.ToArrayStrict3D[FloatT, FloatT],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.Array2D[FloatT], onp.Array3D[FloatT]]: ...
@overload  # Nd +f64
def bws_test(
    x: onp.ToArrayND[float, npc.integer],
    y: onp.ToArrayND[float, npc.integer],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.ArrayND[np.float64] | Any, onp.ArrayND[np.float64]]: ...
@overload  # Nd floating
def bws_test[FloatT: (np.float64, np.float32, np.float16)](
    x: onp.ToArrayND[FloatT, FloatT],
    y: onp.ToArrayND[FloatT, FloatT],
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.ArrayND[FloatT] | Any, onp.ArrayND[FloatT]]: ...
@overload  # fallback, axis=None
def bws_test(
    x: onp.ToComplexND,
    y: onp.ToComplexND,
    *,
    alternative: Alternative = "two-sided",
    axis: None,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[np.float64 | Any, onp.Array1D[np.float64 | Any]]: ...
@overload  # fallback
def bws_test(
    x: onp.ToComplexND,
    y: onp.ToComplexND,
    *,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: PermutationMethod | None = None,
) -> PermutationTestResult[onp.ArrayND[np.float64 | Any] | Any, onp.ArrayND[np.float64 | Any]]: ...
