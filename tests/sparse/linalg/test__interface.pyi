from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator

_type_int: type[int]
_type_float: type[float]
_type_complex: type[complex]

_2d: tuple[int, int]
_3d: tuple[int, int, int]

_2d_bool: onp.Array2D[np.bool]
_2d_i64: onp.Array2D[np.int64]
_2d_f64: onp.Array2D[np.float64]
_2d_c128: onp.Array2D[np.complex128]

_3d_i64: onp.Array3D[np.int64]
_3d_f64: onp.Array3D[np.float64]
_3d_c128: onp.Array3D[np.complex128]

_sp_i64: csr_array[np.int64]
_sp_f64: csr_array[np.float64]
_sp_c128: csr_array[np.complex128]

_op_f64: LinearOperator[np.float64, tuple[int, int]]

class _MatVec:
    shape: tuple[int, int]
    def matvec(self, x: onp.ArrayND[np.float64], /) -> onp.ArrayND[np.float64]: ...

class _MatVecDType(_MatVec):
    dtype: np.dtype[np.float64]

class _Diag(LinearOperator[np.float64]):
    def __init__(self, /, n: int) -> None: ...

_diag: _Diag

###
# LinearOperator.__new__

def mv(v: npt.NDArray[np.float64 | np.complex128]) -> npt.NDArray[np.float64 | np.complex128]: ...

assert_type(LinearOperator(_2d, matvec=mv, dtype=np.int16), _CustomLinearOperator[np.int16, tuple[int, int]])
assert_type(LinearOperator(_2d, matvec=mv, dtype=int), _CustomLinearOperator[np.int_, tuple[int, int]])
assert_type(LinearOperator(_2d, matvec=mv, dtype=float), _CustomLinearOperator[np.float64, tuple[int, int]])
assert_type(LinearOperator(_2d, matvec=mv, dtype=complex), _CustomLinearOperator[np.complex128, tuple[int, int]])
assert_type(LinearOperator(_2d, matvec=mv), _CustomLinearOperator[np.int8 | Any, tuple[int, int]])

assert_type(LinearOperator(_3d, matvec=mv, dtype=np.int16), _CustomLinearOperator[np.int16, tuple[int, int, int]])
assert_type(LinearOperator(_3d, matvec=mv, dtype=int), _CustomLinearOperator[np.int_, tuple[int, int, int]])
assert_type(LinearOperator(_3d, matvec=mv, dtype=float), _CustomLinearOperator[np.float64, tuple[int, int, int]])
assert_type(LinearOperator(_3d, matvec=mv, dtype=complex), _CustomLinearOperator[np.complex128, tuple[int, int, int]])
assert_type(LinearOperator(_3d, matvec=mv), _CustomLinearOperator[np.int8 | Any, tuple[int, int, int]])

###
# LinearOperator.__init__

assert_type(super(_Diag, _diag).__init__(np.float64, _2d), None)

###
# aslinearoperator

assert_type(aslinearoperator(_2d_f64), MatrixLinearOperator[np.float64, tuple[int, int]])
assert_type(aslinearoperator(_2d_c128), MatrixLinearOperator[np.complex128, tuple[int, int]])
assert_type(aslinearoperator(_2d_i64), MatrixLinearOperator[np.int64, tuple[int, int]])
assert_type(aslinearoperator(_2d_bool), MatrixLinearOperator[np.bool, tuple[int, int]])

assert_type(aslinearoperator(_3d_f64), MatrixLinearOperator[np.float64, tuple[int, int, int]])
assert_type(aslinearoperator(_3d_c128), MatrixLinearOperator[np.complex128, tuple[int, int, int]])
assert_type(aslinearoperator(_3d_i64), MatrixLinearOperator[np.int64, tuple[int, int, int]])

assert_type(aslinearoperator(_sp_i64), MatrixLinearOperator[np.int64, tuple[int, int]])
assert_type(aslinearoperator(_sp_f64), MatrixLinearOperator[np.float64, tuple[int, int]])
assert_type(aslinearoperator(_sp_c128), MatrixLinearOperator[np.complex128, tuple[int, int]])

assert_type(aslinearoperator(_op_f64), LinearOperator[np.float64, tuple[int, int]])
assert_type(aslinearoperator(_MatVecDType()), _CustomLinearOperator[np.float64, tuple[int, int]])
assert_type(aslinearoperator(_MatVec()), _CustomLinearOperator[np.float64, tuple[int, int]])
