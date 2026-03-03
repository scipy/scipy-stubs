# type-tests for `linalg/blas.pyi`

from typing import Literal as L, assert_type

import numpy as np

from scipy.linalg import find_best_blas_type, get_blas_funcs
from scipy.linalg.blas import _FortranFunction

###
# find_best_blas_type

assert_type(
    find_best_blas_type(),
    (
        tuple[L["s"], np.dtype[np.float32], bool]
        | tuple[L["f"], np.dtype[np.float64], bool]
        | tuple[L["c"], np.dtype[np.complex64], bool]
        | tuple[L["z"], np.dtype[np.complex128], bool]
    ),
)

###
# get_blas_funcs

assert_type(get_blas_funcs("gemm"), _FortranFunction)
assert_type(get_blas_funcs(["gemm", "larf"]), list[_FortranFunction] | _FortranFunction)
assert_type(get_blas_funcs(iter(["gemm"])), list[_FortranFunction] | _FortranFunction)
