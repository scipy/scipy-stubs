# type-tests for `binned_statistic`, `binned_statistic_2d`, and `binned_statistic_dd` from `stats/_binned_statistic.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_i64_1d: onp.Array1D[np.int64]
_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]
_g_1d: onp.Array1D[np.longdouble]

def _to_f64(a: onp.Array1D[np.float64], /) -> np.float64: ...
def _to_c128(a: onp.Array1D[np.float64], /) -> np.complex128: ...
def _to_c128_c(a: onp.Array1D[np.complex128], /) -> np.complex128: ...

###

# binned_statistic

assert_type(binned_statistic(_f64_1d, _f64_1d).bin_edges, onp.Array1D[np.float64 | Any])
assert_type(binned_statistic(_f64_1d, _f64_1d).binnumber, onp.Array1D[np.intp])

assert_type(binned_statistic(_f64_1d, _f64_1d).statistic, onp.Array1D[np.float64])
assert_type(binned_statistic(_f64_1d, _i64_1d, "sum").statistic, onp.Array1D[np.float64])
assert_type(binned_statistic(_f64_1d, [1, 2, 3], "median").statistic, onp.Array1D[np.float64])
assert_type(binned_statistic(_f64_1d, _c128_1d).statistic, onp.Array1D[np.complex128])
assert_type(binned_statistic(_f64_1d, [1j, 2j], "max").statistic, onp.Array1D[np.complex128])

# `count` and `std` are always real, even for complex `values`
assert_type(binned_statistic(_f64_1d, _c128_1d, "count").statistic, onp.Array1D[np.float64])
assert_type(binned_statistic(_f64_1d, _c128_1d, "std").statistic, onp.Array1D[np.float64])
assert_type(binned_statistic(_f64_1d, None, "count").statistic, onp.Array1D[np.float64])

# a list of `values` sequences results in one row of statistics per sequence
assert_type(binned_statistic(_f64_1d, _f64_2d).statistic, onp.Array2D[np.float64])
assert_type(binned_statistic(_f64_1d, [_f64_1d, _f64_1d], "min").statistic, onp.Array2D[np.float64])
assert_type(binned_statistic(_f64_1d, _c128_2d, "median").statistic, onp.Array2D[np.complex128])
assert_type(binned_statistic(_f64_1d, _c128_2d, "std").statistic, onp.Array2D[np.float64])

# callable statistics
assert_type(binned_statistic(_f64_1d, _f64_1d, _to_f64).statistic, onp.Array1D[np.float64])
assert_type(binned_statistic(_f64_1d, _f64_1d, _to_c128).statistic, onp.Array1D[np.complex128])
assert_type(binned_statistic(_f64_1d, _f64_2d, _to_c128).statistic, onp.Array2D[np.complex128])
assert_type(binned_statistic(_f64_1d, _c128_1d, _to_c128_c).statistic, onp.Array1D[np.complex128])

# unknown rank; pyright matches the erased-shape overloads, pyrefly falls through to the fallback
assert_subtype[onp.Array[tuple[int] | tuple[int, int], np.float64 | np.complex128]](
    binned_statistic(_f64_nd, _c128_nd, "sum").statistic
)

# unsupported `values` dtypes fall back to a gradual result; on numpy<2.2 `longdouble` is `floating[Any]`, so it matches `_AsF64`
assert_subtype[onp.ArrayND[np.float64 | np.complex128]](binned_statistic(_f64_1d, _g_1d, "median").statistic)

# binned_statistic_2d

assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d).x_edge, onp.Array1D[np.float64 | Any])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d).y_edge, onp.Array1D[np.float64 | Any])

assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d).binnumber, onp.Array1D[np.intp])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d, expand_binnumbers=False).binnumber, onp.Array1D[np.intp])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d, expand_binnumbers=True).binnumber, onp.Array2D[np.intp])

assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d).statistic, onp.Array2D[np.float64])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _i64_1d, "sum").statistic, onp.Array2D[np.float64])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, [1, 2, 3], "median").statistic, onp.Array2D[np.float64])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _c128_1d).statistic, onp.Array2D[np.complex128])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, [1j, 2j], "max").statistic, onp.Array2D[np.complex128])

assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _c128_1d, "count").statistic, onp.Array2D[np.float64])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _c128_1d, "std").statistic, onp.Array2D[np.float64])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, None, "count").statistic, onp.Array2D[np.float64])

assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_2d).statistic, onp.Array3D[np.float64])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, [_f64_1d, _f64_1d], "min").statistic, onp.Array3D[np.float64])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _c128_2d, "median").statistic, onp.Array3D[np.complex128])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _c128_2d, "std").statistic, onp.Array3D[np.float64])

assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d, _to_f64).statistic, onp.Array2D[np.float64])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d, _to_c128).statistic, onp.Array2D[np.complex128])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_2d, _to_c128).statistic, onp.Array3D[np.complex128])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _c128_1d, _to_c128_c).statistic, onp.Array2D[np.complex128])

assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _c128_2d, "sum", expand_binnumbers=True).statistic, onp.Array3D[np.complex128])

assert_subtype[onp.Array[tuple[int, int] | tuple[int, int, int], np.float64 | np.complex128]](
    binned_statistic_2d(_f64_1d, _f64_1d, _c128_nd, "sum").statistic
)
assert_subtype[onp.ArrayND[np.float64 | np.complex128]](binned_statistic_2d(_f64_1d, _f64_1d, _g_1d, "median").statistic)

# binned_statistic_dd

assert_type(binned_statistic_dd(_f64_2d, _f64_1d).bin_edges, list[onp.Array1D[np.float64 | Any]])

assert_type(binned_statistic_dd(_f64_1d, _f64_1d).binnumber, onp.Array1D[np.intp])
assert_type(binned_statistic_dd(_f64_1d, _f64_1d, expand_binnumbers=False).binnumber, onp.Array1D[np.intp])
assert_type(binned_statistic_dd(_f64_1d, _f64_1d, expand_binnumbers=True).binnumber, onp.Array2D[np.intp])

# the statistic rank is `D` (`+ 1` for 2-d `values`), which isn't known statically
assert_type(binned_statistic_dd(_f64_2d, _f64_1d).statistic, onp.ArrayND[np.float64])
assert_type(binned_statistic_dd(_f64_2d, _i64_1d, "sum").statistic, onp.ArrayND[np.float64])
assert_type(binned_statistic_dd([_f64_1d, _f64_1d], [1, 2, 3], "median").statistic, onp.ArrayND[np.float64])
assert_type(binned_statistic_dd(_f64_2d, _c128_1d).statistic, onp.ArrayND[np.complex128])
assert_type(binned_statistic_dd(_f64_2d, _c128_2d, "max").statistic, onp.ArrayND[np.complex128])

assert_type(binned_statistic_dd(_f64_2d, _c128_1d, "count").statistic, onp.ArrayND[np.float64])
assert_type(binned_statistic_dd(_f64_2d, _c128_2d, "std").statistic, onp.ArrayND[np.float64])
assert_type(binned_statistic_dd(_f64_2d, None, "count").statistic, onp.ArrayND[np.float64])

assert_type(binned_statistic_dd(_f64_2d, _f64_1d, _to_f64).statistic, onp.ArrayND[np.float64])
assert_type(binned_statistic_dd(_f64_2d, _f64_1d, _to_c128).statistic, onp.ArrayND[np.complex128])
assert_type(binned_statistic_dd(_f64_2d, _c128_1d, _to_c128_c).statistic, onp.ArrayND[np.complex128])

assert_type(binned_statistic_dd(_f64_2d, _c128_nd, "sum", expand_binnumbers=True).statistic, onp.ArrayND[np.complex128])

assert_subtype[onp.ArrayND[np.float64 | np.complex128]](binned_statistic_dd(_f64_2d, _g_1d, "median").statistic)

_dd_result = binned_statistic_dd(_f64_2d, _f64_1d)
assert_type(binned_statistic_dd(_f64_2d, _c128_1d, binned_statistic_result=_dd_result).statistic, onp.ArrayND[np.complex128])
