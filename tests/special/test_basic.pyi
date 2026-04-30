from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.special import (
    ai_zeros,
    assoc_laguerre,
    bei_zeros,
    beip_zeros,
    ber_zeros,
    bernoulli,
    berp_zeros,
    bi_zeros,
    comb,
    diric,
    erf_zeros,
    euler,
    factorial,
    factorial2,
    factorialk,
    fresnel_zeros,
    fresnelc_zeros,
    fresnels_zeros,
    h1vp,
    h2vp,
    ivp,
    jn_zeros,
    jnjnp_zeros,
    jnp_zeros,
    jnyn_zeros,
    jvp,
    kei_zeros,
    keip_zeros,
    kelvin_zeros,
    ker_zeros,
    kerp_zeros,
    kvp,
    lmbda,
    lqmn,
    lqn,
    mathieu_even_coef,
    mathieu_odd_coef,
    obl_cv_seq,
    pbdn_seq,
    pbdv_seq,
    pbvv_seq,
    perm,
    polygamma,
    pro_cv_seq,
    riccati_jn,
    riccati_yn,
    sinc,
    softplus,
    stirling2,
    y0_zeros,
    y1_zeros,
    y1p_zeros,
    yn_zeros,
    ynp_zeros,
    yvp,
    zeta,
)

f_arr: onp.ArrayND[np.float64]
f32_1d: onp.Array1D[np.float32]
f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
i_arr: onp.ArrayND[np.intp]
c_arr: onp.ArrayND[np.complex128]
where: onp.ArrayND[np.bool_]

# sinc
assert_type(sinc(np.float32(1.0)), np.float32)
assert_type(sinc(1.0), np.float64)
assert_type(sinc(1), np.float64)
assert_type(sinc(1j), np.complex128)
assert_type(sinc(i_arr), onp.ArrayND[np.float64])

# diric
assert_type(diric(1.0, 3), onp.Array0D[np.float64])
assert_type(diric(1.0, np.uint8(3)), onp.Array0D[npc.floating])
assert_type(diric(f_arr, 3), onp.ArrayND[npc.floating])

# jnjnp_zeros
assert_type(jnjnp_zeros(2), tuple[onp.Array1D[np.float64], onp.Array1D[np.int32], onp.Array1D[np.int32], onp.Array1D[np.int32]])

# jnyn_zeros
assert_type(
    jnyn_zeros(2, 3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]]
)

# jn_zeros
assert_type(jn_zeros(1, 2), onp.Array1D[np.float64])

# jnp_zeros
assert_type(jnp_zeros(1, 2), onp.Array1D[np.float64])

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

# jvp
assert_type(jvp(1.0, 1.0), np.float32 | np.float64)
assert_type(jvp(1.0, 1j), np.complex64 | np.complex128)
assert_type(jvp(1.0, f_arr), onp.ArrayND[np.float32 | np.float64])
assert_type(jvp(f_arr, 1.0), onp.ArrayND[np.float32 | np.float64])
assert_type(jvp(1.0, c_arr), onp.ArrayND[np.complex64 | np.complex128])
assert_type(jvp(f_arr, 1j), onp.ArrayND[np.complex64 | np.complex128])

# yvp
assert_type(yvp(1.0, 1.0), np.float32 | np.float64)
assert_type(yvp(1.0, 1j), np.complex64 | np.complex128)
assert_type(yvp(1.0, f_arr), onp.ArrayND[np.float32 | np.float64])
assert_type(yvp(f_arr, 1.0), onp.ArrayND[np.float32 | np.float64])
assert_type(yvp(1.0, c_arr), onp.ArrayND[np.complex64 | np.complex128])
assert_type(yvp(f_arr, 1j), onp.ArrayND[np.complex64 | np.complex128])

# kvp
assert_type(kvp(1.0, 1.0), np.float32 | np.float64)
assert_type(kvp(1.0, 1j), np.complex64 | np.complex128)
assert_type(kvp(1.0, f_arr), onp.ArrayND[np.float32 | np.float64])
assert_type(kvp(f_arr, 1.0), onp.ArrayND[np.float32 | np.float64])
assert_type(kvp(1.0, c_arr), onp.ArrayND[np.complex64 | np.complex128])
assert_type(kvp(f_arr, 1j), onp.ArrayND[np.complex64 | np.complex128])

# ivp
assert_type(ivp(1.0, 1.0), np.float32 | np.float64)
assert_type(ivp(1.0, 1j), np.complex64 | np.complex128)
assert_type(ivp(1.0, f_arr), onp.ArrayND[np.float32 | np.float64])
assert_type(ivp(f_arr, 1.0), onp.ArrayND[np.float32 | np.float64])
assert_type(ivp(1.0, c_arr), onp.ArrayND[np.complex64 | np.complex128])
assert_type(ivp(f_arr, 1j), onp.ArrayND[np.complex64 | np.complex128])

# h1vp
assert_type(h1vp(1.0, 1.0), np.complex64 | np.complex128)
assert_type(h1vp(1.0, 1j), np.complex64 | np.complex128)
assert_type(h1vp(1.0, c_arr), np.complex64 | np.complex128)
assert_type(h1vp(f_arr, 1j), np.complex64 | np.complex128)

# h2vp
assert_type(h2vp(1.0, 1.0), np.complex64 | np.complex128)
assert_type(h2vp(1.0, 1j), np.complex64 | np.complex128)
assert_type(h2vp(1.0, c_arr), np.complex64 | np.complex128)
assert_type(h2vp(f_arr, 1j), np.complex64 | np.complex128)

# riccati_jn
assert_type(riccati_jn(1, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# riccati_yn
assert_type(riccati_yn(1, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# erf_zeros
assert_type(erf_zeros(5), onp.Array1D[np.complex128])

# fresnelc_zeros
assert_type(fresnelc_zeros(5), onp.Array1D[np.complex128])

# fresnels_zeros
assert_type(fresnels_zeros(5), onp.Array1D[np.complex128])

# fresnel_zeros
assert_type(fresnel_zeros(5), onp.Array1D[np.complex128])

# assoc_laguerre
assert_type(assoc_laguerre(1.0, 3), np.float32 | np.float64)
assert_type(assoc_laguerre(1j, 3), np.complex64 | np.complex128)
assert_type(assoc_laguerre(1.0, 3, f_arr), onp.ArrayND[np.float32 | np.float64])
assert_type(assoc_laguerre(1.0, i_arr, 0.0), onp.ArrayND[np.float32 | np.float64])
assert_type(assoc_laguerre(f_arr, 3, 0.0), onp.ArrayND[np.float32 | np.float64])
assert_type(assoc_laguerre(1j, 3, f_arr), onp.ArrayND[np.complex64 | np.complex128])
assert_type(assoc_laguerre(1j, i_arr, 0.0), onp.ArrayND[np.complex64 | np.complex128])
assert_type(assoc_laguerre(c_arr, 3, 0.0), onp.ArrayND[np.complex64 | np.complex128])

# polygamma
assert_type(polygamma(1, 1.0), np.float64)
assert_type(polygamma(1, f_arr), onp.ArrayND[np.float64])
assert_type(polygamma(i_arr, 1.0), onp.ArrayND[np.float64])

# mathieu_even_coef
assert_type(mathieu_even_coef(1, 1.0), onp.Array1D[np.float64])

# mathieu_odd_coef
assert_type(mathieu_odd_coef(1, 1.0), onp.Array1D[np.float64])

# lqn
assert_type(lqn(2, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(lqn(2, f_arr), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lqn(2, 1j), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])
assert_type(lqn(2, c_arr), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

# lqmn
assert_type(lqmn(1, 2, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(lqmn(1, 2, f_arr), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lqmn(1, 2, 1j), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])
assert_type(lqmn(1, 2, c_arr), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

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

# lmbda
assert_type(lmbda(1.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# pbdv_seq
assert_type(pbdv_seq(1.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# pbvv_seq
assert_type(pbvv_seq(1.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# pbdn_seq
assert_type(pbdn_seq(1, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# ber_zeros
assert_type(ber_zeros(5), onp.Array1D[np.float64])

# bei_zeros
assert_type(bei_zeros(5), onp.Array1D[np.float64])

# ker_zeros
assert_type(ker_zeros(2), onp.Array1D[np.float64])

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

# pro_cv_seq
assert_type(pro_cv_seq(1, 2, 1.0), onp.Array1D[np.float64])

# obl_cv_seq
assert_type(obl_cv_seq(1, 2, 1.0), onp.Array1D[np.float64])

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

# softplus
assert_type(softplus(1.0), np.float64)
assert_type(softplus(1, out=None), np.float64)
assert_type(softplus(np.float32(1.0)), np.float32)
assert_type(softplus(np.float32(1.0), dtype=np.float32), np.float32)
assert_type(softplus(1.0, where=where), np.float64)
assert_type(softplus(i_arr, out=None), onp.ArrayND[np.float32 | np.float64])
assert_type(softplus(f64_1d), onp.Array1D[np.float64])
assert_type(softplus(f64_2d, dtype=np.float64), onp.Array2D[np.float64])
assert_type(softplus(f_arr, out=f_arr), onp.ArrayND[np.float64])
assert_type(softplus(f32_1d, out=f_arr), onp.ArrayND[np.float64])
