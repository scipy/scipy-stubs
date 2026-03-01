from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator

int_type: type[int]
float_type: type[float]
complex_type: type[complex]

###

def mv(v: npt.NDArray[np.float64 | np.complex128]) -> npt.NDArray[np.float64 | np.complex128]: ...

assert_type(LinearOperator((2, 2), matvec=mv, dtype=np.int16), _CustomLinearOperator[np.int16])
assert_type(LinearOperator((2, 2), matvec=mv, dtype=int), _CustomLinearOperator[np.int_])
assert_type(LinearOperator((2, 2), matvec=mv, dtype=float), _CustomLinearOperator[np.float64])
assert_type(LinearOperator((2, 2), matvec=mv, dtype=complex), _CustomLinearOperator[np.complex128])
assert_type(LinearOperator((2, 2), matvec=mv), _CustomLinearOperator[np.int8 | Any])

# aslinearoperator
a_f64: onp.Array2D[np.float64]
a_c128: onp.Array2D[np.complex128]
a_i64: onp.Array2D[np.int64]
sp_f64: csr_array[np.float64]
sp_c128: csr_array[np.complex128]

assert_type(aslinearoperator(a_f64), MatrixLinearOperator[np.float64])
assert_type(aslinearoperator(a_c128), MatrixLinearOperator[np.complex128])
assert_type(aslinearoperator(a_i64), MatrixLinearOperator[np.float64])
assert_type(aslinearoperator(sp_f64), MatrixLinearOperator[np.float64])
assert_type(aslinearoperator(sp_c128), MatrixLinearOperator[np.complex128])
