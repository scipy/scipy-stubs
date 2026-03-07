from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import root

###

def _fn_f_1_1(x: onp.Array1D[np.float64]) -> list[float]: ...
def _fn_f_n_1(x: onp.ArrayND[np.float64]) -> list[float]: ...
def _fn_c_n_1(x: onp.ArrayND[np.complex128]) -> list[complex]: ...
def _fn_f_n_2(x: onp.ArrayND[np.float64]) -> list[list[float]]: ...
def _fn_f_1_12(x: onp.Array1D[np.float64]) -> tuple[list[float], list[list[float]]]: ...
def _fn_f_2_23(x: onp.Array2D[np.float64]) -> tuple[list[list[float]], list[list[list[float]]]]: ...
def _fn_f_n_12(x: onp.ArrayND[np.float64]) -> tuple[list[float], list[list[float]]]: ...
def _fn_c_n_12(x: onp.ArrayND[np.complex128]) -> tuple[list[complex], list[list[complex]]]: ...

###
# root

_x0_f: list[float]
_x0_c: list[complex]

# hybr

assert_type(root(_fn_f_n_1, _x0_f).x, onp.Array1D[np.float64])
assert_type(root(_fn_f_n_1, _x0_f).fun, onp.Array1D[np.float64])
assert_type(root(_fn_f_n_1, _x0_f).success, bool)
assert_type(root(_fn_f_n_1, _x0_f).message, str)
assert_type(root(_fn_f_n_1, _x0_f).nfev, int)
assert_type(root(_fn_f_1_12, _x0_f, jac=True).x, onp.Array1D[np.float64])

assert_type(root(_fn_f_1_1, _x0_f, method="hybr").x, onp.Array1D[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="hybr").x, onp.Array1D[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="hybr", jac=_fn_f_n_2).x, onp.Array1D[np.float64])
assert_type(root(_fn_f_1_12, _x0_f, method="hybr", jac=True).x, onp.Array1D[np.float64])

# lm

assert_type(root(_fn_f_1_1, _x0_f, method="lm").x, onp.Array1D[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="lm").x, onp.Array1D[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="lm", jac=_fn_f_n_2).x, onp.Array1D[np.float64])
assert_type(root(_fn_f_1_12, _x0_f, method="lm", jac=True).x, onp.Array1D[np.float64])

# df-sane

assert_type(root(_fn_f_1_1, _x0_f, method="df-sane").x, onp.Array1D[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="df-sane").x, onp.ArrayND[np.float64])
assert_type(root(_fn_c_n_1, _x0_c, method="df-sane").x, onp.ArrayND[np.complex128])

# nonlin

assert_type(root(_fn_f_n_1, _x0_f, method="broyden1").x, onp.ArrayND[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="broyden2").x, onp.ArrayND[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="anderson").x, onp.ArrayND[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="linearmixing").x, onp.ArrayND[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="diagbroyden").x, onp.ArrayND[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="excitingmixing").x, onp.ArrayND[np.float64])
assert_type(root(_fn_f_n_1, _x0_f, method="krylov").x, onp.ArrayND[np.float64])

assert_type(root(_fn_c_n_1, _x0_c, method="broyden1").x, onp.ArrayND[np.complex128])
assert_type(root(_fn_c_n_1, _x0_c, method="broyden2").x, onp.ArrayND[np.complex128])
assert_type(root(_fn_c_n_1, _x0_c, method="anderson").x, onp.ArrayND[np.complex128])
assert_type(root(_fn_c_n_1, _x0_c, method="linearmixing").x, onp.ArrayND[np.complex128])
assert_type(root(_fn_c_n_1, _x0_c, method="diagbroyden").x, onp.ArrayND[np.complex128])
assert_type(root(_fn_c_n_1, _x0_c, method="excitingmixing").x, onp.ArrayND[np.complex128])
assert_type(root(_fn_c_n_1, _x0_c, method="krylov").x, onp.ArrayND[np.complex128])

assert_type(root(_fn_f_1_12, _x0_f, method="broyden1", jac=True).x, onp.Array1D[np.float64])
assert_type(root(_fn_f_n_12, _x0_f, method="broyden1", jac=True).x, onp.ArrayND[np.float64])
assert_type(root(_fn_c_n_12, _x0_c, method="broyden1", jac=True).x, onp.ArrayND[np.complex128])
