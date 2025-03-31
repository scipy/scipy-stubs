from typing import TypeAlias

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt

_Vector_i8: TypeAlias = onp.Array2D[np.int64]
_Vector_f8: TypeAlias = onp.Array1D[np.float64]
_Matrix_f8: TypeAlias = onp.Array2D[np.float64]

def _cy_wrapper_centered_discrepancy(sample: _Matrix_f8, iterative: op.CanBool, workers: opt.AnyInt) -> float | int | bool: ...
def _cy_wrapper_wrap_around_discrepancy(sample: _Matrix_f8, iterative: op.CanBool, workers: opt.AnyInt) -> float | int | bool: ...
def _cy_wrapper_mixture_discrepancy(sample: _Matrix_f8, iterative: op.CanBool, workers: opt.AnyInt) -> float | int | bool: ...
def _cy_wrapper_l2_star_discrepancy(sample: _Matrix_f8, iterative: op.CanBool, workers: opt.AnyInt) -> float | int | bool: ...
def _cy_wrapper_update_discrepancy(
    x_new_view: _Vector_f8, sample_view: _Matrix_f8, initial_disc: opt.AnyFloat
) -> float | int | bool: ...
def _cy_van_der_corput(n: opt.AnyInt, base: opt.AnyInt, start_index: opt.AnyInt, workers: opt.AnyInt) -> _Vector_f8: ...
def _cy_van_der_corput_scrambled(
    n: opt.AnyInt,
    base: opt.AnyInt,
    start_index: opt.AnyInt,
    permutations: _Vector_i8,
    workers: opt.AnyInt,
) -> _Vector_f8: ...
