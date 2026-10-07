# type-tests for `linalg/lapack.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import get_lapack_funcs
from scipy.linalg.blas import _FortranFunction
from scipy.linalg.lapack import dgelsd, dgeqrf, dgetrf

###

_f_2d: list[list[float]]
_i64_2d: onp.Array2D[np.int64]

###
# get_lapack_funcs

assert_type(get_lapack_funcs("getrf"), _FortranFunction)
assert_type(get_lapack_funcs(["getrf", "getrs"]), list[_FortranFunction])

###
# dgelsd

assert_type(dgelsd(_f_2d, _f_2d, 10_000, 10_000), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], int, int])

###
# dgeqrf

assert_type(dgeqrf(_f_2d), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64], int])

###
# dgetrf

assert_type(dgetrf(_i64_2d), tuple[onp.Array2D[np.float64], onp.Array1D[np.int32], int])
