# type-tests for `linalg/_solvers.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import (
    solve_continuous_are,
    solve_continuous_lyapunov,
    solve_discrete_are,
    solve_discrete_lyapunov,
    solve_sylvester,
)

_FloatND: TypeAlias = onp.ArrayND[np.float32 | np.float64]
_ComplexND: TypeAlias = onp.ArrayND[np.complex64 | np.complex128]

###

f64_nd: onp.ArrayND[np.float64]
c128_nd: onp.ArrayND[np.complex128]

###
# solve_sylvester

assert_type(solve_sylvester(f64_nd, f64_nd, f64_nd), _FloatND)
assert_type(solve_sylvester(c128_nd, f64_nd, f64_nd), _ComplexND)
assert_type(solve_sylvester(f64_nd, c128_nd, f64_nd), _ComplexND)
assert_type(solve_sylvester(f64_nd, f64_nd, c128_nd), _ComplexND)

###
# solve_continuous_lyapunov / solve_discrete_lyapunov

assert_type(solve_continuous_lyapunov(f64_nd, f64_nd), _FloatND)
assert_type(solve_continuous_lyapunov(c128_nd, f64_nd), _ComplexND)
assert_type(solve_continuous_lyapunov(f64_nd, c128_nd), _ComplexND)

assert_type(solve_discrete_lyapunov(f64_nd, f64_nd), _FloatND)
assert_type(solve_discrete_lyapunov(c128_nd, f64_nd), _ComplexND)
assert_type(solve_discrete_lyapunov(f64_nd, c128_nd), _ComplexND)

###
# solve_continuous_are

assert_type(solve_continuous_are(f64_nd, f64_nd, f64_nd, f64_nd), _FloatND)
assert_type(solve_continuous_are(c128_nd, f64_nd, f64_nd, f64_nd), _ComplexND)
assert_type(solve_continuous_are(f64_nd, c128_nd, f64_nd, f64_nd), _ComplexND)
assert_type(solve_continuous_are(f64_nd, f64_nd, c128_nd, f64_nd), _ComplexND)
assert_type(solve_continuous_are(f64_nd, f64_nd, f64_nd, c128_nd), _ComplexND)
assert_type(solve_continuous_are(f64_nd, f64_nd, f64_nd, f64_nd, c128_nd), _ComplexND)
assert_type(solve_continuous_are(f64_nd, f64_nd, f64_nd, f64_nd, s=c128_nd), _ComplexND)

###
# solve_discrete_are

assert_type(solve_discrete_are(f64_nd, f64_nd, f64_nd, f64_nd), _FloatND)
assert_type(solve_discrete_are(c128_nd, f64_nd, f64_nd, f64_nd), _ComplexND)
assert_type(solve_discrete_are(f64_nd, c128_nd, f64_nd, f64_nd), _ComplexND)
assert_type(solve_discrete_are(f64_nd, f64_nd, c128_nd, f64_nd), _ComplexND)
assert_type(solve_discrete_are(f64_nd, f64_nd, f64_nd, c128_nd), _ComplexND)
assert_type(solve_discrete_are(f64_nd, f64_nd, f64_nd, f64_nd, c128_nd), _ComplexND)
assert_type(solve_discrete_are(f64_nd, f64_nd, f64_nd, f64_nd, s=c128_nd), _ComplexND)
