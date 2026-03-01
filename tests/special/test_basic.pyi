from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import comb, factorial, factorial2, factorialk, perm, polygamma, sinc, stirling2, zeta

f_arr: onp.ArrayND[np.float64]
i_arr: onp.ArrayND[np.intp]
c_arr: onp.ArrayND[np.complex128]

# sinc
assert_type(sinc(np.float32(1.0)), np.float32)
assert_type(sinc(1.0), np.float64)
assert_type(sinc(1), np.float64)
assert_type(sinc(1j), np.complex128)
assert_type(sinc(i_arr), onp.ArrayND[np.float64])

# comb
assert_type(comb(5, 2, exact=True), int)
assert_type(comb(5.0, 2.0), np.float32 | np.float64)
assert_type(comb(5.0, f_arr), onp.ArrayND[np.float32 | np.float64])
assert_type(comb(f_arr, 2.0), onp.ArrayND[np.float32 | np.float64])

# perm
assert_type(perm(5, 2, True), int)
assert_type(perm(5.0, 2.0), np.float32 | np.float64)
assert_type(perm(5.0, f_arr), onp.ArrayND[np.float32 | np.float64])
assert_type(perm(f_arr, 2.0), onp.ArrayND[np.float32 | np.float64])

# factorial
assert_type(factorial(5, True), int)
assert_type(factorial(i_arr, True), onp.ArrayND[np.int32 | np.int64])
assert_type(factorial(5), np.float64)
assert_type(factorial(f_arr), onp.ArrayND[np.float64])
assert_type(factorial(1j, False, "complex"), np.float64 | np.complex128)
assert_type(factorial(c_arr, False, "complex"), onp.ArrayND[np.float64 | np.complex128])

# factorial2
assert_type(factorial2(5, True), int)
assert_type(factorial2(np.uint8(5), True), np.uint8)
assert_type(factorial2(i_arr, True), onp.ArrayND[np.int32 | np.int64])
assert_type(factorial2(5), np.float64)
assert_type(factorial2(f_arr), onp.ArrayND[np.float64])
assert_type(factorial2(1j, False, "complex"), np.float64 | np.complex128)
assert_type(factorial2(c_arr, False, "complex"), onp.ArrayND[np.float64 | np.complex128])

# factorialk
assert_type(factorialk(5, 2, True), int)
assert_type(factorialk(i_arr, 2, True), onp.ArrayND[np.int32 | np.int64])
assert_type(factorialk(5.0, 2), np.float64)
assert_type(factorialk(f_arr, 2), onp.ArrayND[np.float64])
assert_type(factorialk(1j, 2, False, "complex"), np.float64 | np.complex128)
assert_type(factorialk(c_arr, 2, False, "complex"), onp.ArrayND[np.float64 | np.complex128])

# stirling2
assert_type(stirling2(5, 2, exact=True), int)
assert_type(stirling2(5, i_arr, exact=True), onp.ArrayND[np.object_])
assert_type(stirling2(i_arr, 2, exact=True), onp.ArrayND[np.object_])
assert_type(stirling2(5, 2), np.float64)
assert_type(stirling2(5, i_arr), onp.ArrayND[np.float64])
assert_type(stirling2(i_arr, 2), onp.ArrayND[np.float64])

# polygamma
assert_type(polygamma(1, 1.0), np.float64)
assert_type(polygamma(1, f_arr), onp.ArrayND[np.float64])
assert_type(polygamma(i_arr, 1.0), onp.ArrayND[np.float64])

# zeta
assert_type(zeta(2.0), np.float64)
assert_type(zeta(2.0, f_arr), onp.ArrayND[np.float64])
assert_type(zeta(f_arr), onp.ArrayND[np.float64])
assert_type(zeta(2j), np.complex128)
assert_type(zeta(2j, f_arr), onp.ArrayND[np.complex128])
assert_type(zeta(c_arr), onp.ArrayND[np.complex128])
