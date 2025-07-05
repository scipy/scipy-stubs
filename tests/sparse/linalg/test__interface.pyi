from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg._interface import _CustomLinearOperator

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

# TODO(jorenham): add more tests for `LinearOperator`
# TODO(jorenham): add tests for `aslinearoperator`
