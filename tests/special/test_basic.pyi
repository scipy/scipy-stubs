from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import (
    ai_zeros,
    bei_zeros,
    beip_zeros,
    ber_zeros,
    bernoulli,
    berp_zeros,
    bi_zeros,
    comb,
    erf_zeros,
    euler,
    factorial,
    factorial2,
    factorialk,
    fresnel_zeros,
    fresnelc_zeros,
    fresnels_zeros,
    kei_zeros,
    keip_zeros,
    kelvin_zeros,
    kerp_zeros,
    perm,
    polygamma,
    sinc,
    stirling2,
    y0_zeros,
    y1_zeros,
    y1p_zeros,
    yn_zeros,
    ynp_zeros,
    zeta,
)

f_arr: onp.ArrayND[np.float64]
i_arr: onp.ArrayND[np.intp]
c_arr: onp.ArrayND[np.complex128]

# sinc
assert_type(sinc(np.float32(1.0)), np.float32)
assert_type(sinc(1.0), np.float64)
assert_type(sinc(1), np.float64)
assert_type(sinc(1j), np.complex128)
assert_type(sinc(i_arr), onp.ArrayND[np.float64])

# yn_zeros
assert_type(yn_zeros(1, 5), onp.Array1D[np.float64])

# ynp_zeros
assert_type(ynp_zeros(1, 5), onp.Array1D[np.float64])

# y0_zeros
assert_type(y0_zeros(5, False), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])
assert_type(y0_zeros(5, True), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# y1_zeros
assert_type(y1_zeros(5, False), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])
assert_type(y1_zeros(5, True), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# y1p_zeros
assert_type(y1p_zeros(5, False), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])
assert_type(y1p_zeros(5, True), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# erf_zeros
assert_type(erf_zeros(5), onp.Array1D[np.complex128])

# fresnelc_zeros
assert_type(fresnelc_zeros(5), onp.Array1D[np.complex128])

# fresnels_zeros
assert_type(fresnels_zeros(5), onp.Array1D[np.complex128])

# fresnel_zeros
assert_type(fresnel_zeros(5), onp.Array1D[np.complex128])

# polygamma
assert_type(polygamma(1, 1.0), np.float64)
assert_type(polygamma(1, f_arr), onp.ArrayND[np.float64])
assert_type(polygamma(i_arr, 1.0), onp.ArrayND[np.float64])

# bernoulli
assert_type(bernoulli(5), onp.Array1D[np.float64])
assert_type(bernoulli(5.0), onp.Array1D[np.float64])

# euler
assert_type(euler(5), onp.Array1D[np.float64])
assert_type(euler(5.0), onp.Array1D[np.float64])

# ai_zeros
assert_type(
    ai_zeros(5), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]]
)

# bi_zeros
assert_type(
    bi_zeros(5), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]]
)

# ber_zeros
assert_type(ber_zeros(5), onp.Array1D[np.float64])

# bei_zeros
assert_type(bei_zeros(5), onp.Array1D[np.float64])

# kei_zeros
assert_type(kei_zeros(5), onp.Array1D[np.float64])

# berp_zeros
assert_type(berp_zeros(5), onp.Array1D[np.float64])

# beip_zeros
assert_type(beip_zeros(5), onp.Array1D[np.float64])

# kerp_zeros
assert_type(kerp_zeros(5), onp.Array1D[np.float64])

# keip_zeros
assert_type(keip_zeros(5), onp.Array1D[np.float64])

# kelvin_zeros
assert_type(
    kelvin_zeros(5),
    tuple[
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
    ],
)

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

# zeta
assert_type(zeta(2.0), np.float64)
assert_type(zeta(2.0, f_arr), onp.ArrayND[np.float64])
assert_type(zeta(f_arr), onp.ArrayND[np.float64])
assert_type(zeta(2j), np.complex128)
assert_type(zeta(2j, f_arr), onp.ArrayND[np.complex128])
assert_type(zeta(c_arr), onp.ArrayND[np.complex128])
