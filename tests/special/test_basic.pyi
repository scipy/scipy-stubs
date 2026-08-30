from typing import Any, assert_type

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

_bool_nd: onp.ArrayND[np.bool]
_i64_nd: onp.ArrayND[np.int64]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_c128_nd: onp.ArrayND[np.complex128]

# sinc
assert_type(sinc(np.float32(1.0)), np.float32)
assert_type(sinc(1.0), np.float64)
assert_type(sinc(1), np.float64)
assert_type(sinc(1j), np.complex128)
assert_type(sinc(_f32_1d), onp.Array1D[np.float32])
assert_type(sinc(_i64_nd), onp.ArrayND[np.float64])
assert_type(sinc(_c128_nd), onp.ArrayND[np.complex128])

# diric
assert_type(diric(1.0, 3), onp.Array0D[np.float64])
assert_type(diric(1.0, np.uint8(3)), onp.Array0D[npc.floating])
assert_type(diric(_f64_nd, 3), onp.ArrayND[npc.floating])

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
assert_type(jvp(1.0, 1.0), np.float64)
assert_type(jvp(1.0, 1j), np.complex128)
assert_type(jvp(1.0, _f64_nd), onp.ArrayND[np.float64])
assert_type(jvp(_f64_nd, 1.0), onp.ArrayND[np.float64])
assert_type(jvp(1.0, _c128_nd), onp.ArrayND[np.complex128])
assert_type(jvp(_f64_nd, 1j), onp.ArrayND[np.complex128])

# yvp
assert_type(yvp(1.0, 1.0), np.float64)
assert_type(yvp(1.0, 1j), np.complex128)
assert_type(yvp(1.0, _f64_nd), onp.ArrayND[np.float64])
assert_type(yvp(_f64_nd, 1.0), onp.ArrayND[np.float64])
assert_type(yvp(1.0, _c128_nd), onp.ArrayND[np.complex128])
assert_type(yvp(_f64_nd, 1j), onp.ArrayND[np.complex128])

# kvp
assert_type(kvp(1.0, 1.0), np.float64)
assert_type(kvp(1.0, 1j), np.complex128)
assert_type(kvp(1.0, _f64_nd), onp.ArrayND[np.float64])
assert_type(kvp(_f64_nd, 1.0), onp.ArrayND[np.float64])
assert_type(kvp(1.0, _c128_nd), onp.ArrayND[np.complex128])
assert_type(kvp(_f64_nd, 1j), onp.ArrayND[np.complex128])

# ivp
assert_type(ivp(1.0, 1.0), np.float64)
assert_type(ivp(1.0, 1j), np.complex128)
assert_type(ivp(1.0, _f64_nd), onp.ArrayND[np.float64])
assert_type(ivp(_f64_nd, 1.0), onp.ArrayND[np.float64])
assert_type(ivp(1.0, _c128_nd), onp.ArrayND[np.complex128])
assert_type(ivp(_f64_nd, 1j), onp.ArrayND[np.complex128])

# h1vp
assert_type(h1vp(1.0, 1.0), np.complex128)
assert_type(h1vp(1.0, 1j), np.complex128)
assert_type(h1vp(1.0, _c128_nd), onp.ArrayND[np.complex128])
assert_type(h1vp(_f64_nd, 1j), onp.ArrayND[np.complex128])

# h2vp
assert_type(h2vp(1.0, 1.0), np.complex128)
assert_type(h2vp(1.0, 1j), np.complex128)
assert_type(h2vp(1.0, _c128_nd), onp.ArrayND[np.complex128])
assert_type(h2vp(_f64_nd, 1j), onp.ArrayND[np.complex128])

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
assert_type(assoc_laguerre(1.0, 3), np.float64)
assert_type(assoc_laguerre(1j, 3), np.complex128)
assert_type(assoc_laguerre(1.0, 3, _f64_nd), onp.ArrayND[np.float64])
assert_type(assoc_laguerre(1.0, _i64_nd, 0.0), onp.ArrayND[np.float64])
assert_type(assoc_laguerre(_f64_nd, 3, 0.0), onp.ArrayND[np.float64])
assert_type(assoc_laguerre(1j, 3, _f64_nd), onp.ArrayND[np.complex128])
assert_type(assoc_laguerre(1j, _i64_nd, 0.0), onp.ArrayND[np.complex128])
assert_type(assoc_laguerre(_c128_nd, 3, 0.0), onp.ArrayND[np.complex128])

# polygamma
assert_type(polygamma(1, 1.0), onp.Array0D[np.float64])
assert_type(polygamma(1, _f64_nd), onp.ArrayND[np.float64])
assert_type(polygamma(_i64_nd, 1.0), onp.ArrayND[np.float64])

# mathieu_even_coef
assert_type(mathieu_even_coef(1, 1.0), onp.Array1D[np.float64])

# mathieu_odd_coef
assert_type(mathieu_odd_coef(1, 1.0), onp.Array1D[np.float64])

# lqn
assert_type(lqn(2, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(lqn(2, _f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lqn(2, 1j), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])
assert_type(lqn(2, _c128_nd), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

# lqmn
assert_type(lqmn(1, 2, 1.0), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(lqmn(1, 2, _f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lqmn(1, 2, 1j), tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128]])
assert_type(lqmn(1, 2, _c128_nd), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

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
assert_type(comb(5.0, 2.0), np.float64)
assert_type(comb(5.0, _f64_nd), onp.ArrayND[np.float64])
assert_type(comb(_f64_nd, 2.0), onp.ArrayND[np.float64])

# perm
assert_type(perm(5, 2, True), int)
assert_type(perm(5.0, 2.0), np.float64)
assert_type(perm(5.0, _f64_nd), onp.ArrayND[np.float64])
assert_type(perm(_f64_nd, 2.0), onp.ArrayND[np.float64])

# factorial
assert_type(factorial(5), np.float64)
assert_type(factorial(5, exact=True), int)
assert_type(factorial(_i64_nd), onp.ArrayND[np.float64])
assert_type(factorial(_i64_nd, exact=True), onp.ArrayND[np.int_ | np.object_])
assert_type(factorial(_f64_nd), onp.ArrayND[np.float64])
assert_type(factorial(1j, extend="complex"), np.complex128)
assert_type(factorial(_c128_nd, extend="complex"), onp.ArrayND[np.complex128])

# factorial2
assert_type(factorial2(5), np.float64)
assert_type(factorial2(5, exact=True), int)
assert_type(factorial2(np.uint8(5), True), int)
assert_type(factorial2(_i64_nd, exact=True), onp.ArrayND[np.int_ | np.object_])
assert_type(factorial2(_f64_nd), onp.ArrayND[np.float64])
assert_type(factorial2(1j, extend="complex"), np.complex128)
assert_type(factorial2(_c128_nd, extend="complex"), onp.ArrayND[np.complex128])

# factorialk
assert_type(factorialk(5, 2), np.float64)
assert_type(factorialk(5, 2, exact=True), int)
assert_type(factorialk(5.0, 2), np.float64)
assert_type(factorialk(_i64_nd, 2, exact=True), onp.ArrayND[np.int_ | np.object_])
assert_type(factorialk(_f64_nd, 2), onp.ArrayND[np.float64])
assert_type(factorialk(1j, 2, extend="complex"), np.complex128)
assert_type(factorialk(_c128_nd, 2, extend="complex"), onp.ArrayND[np.complex128])

# stirling2
assert_type(stirling2(5, 2, exact=True), int)
assert_type(stirling2(5, _i64_nd, exact=True), onp.ArrayND[np.object_])
assert_type(stirling2(_i64_nd, 2, exact=True), onp.ArrayND[np.object_])
assert_type(stirling2(5, 2), np.float64)
assert_type(stirling2(5, _i64_nd), onp.ArrayND[np.float64])
assert_type(stirling2(_i64_nd, 2), onp.ArrayND[np.float64])

# zeta
assert_type(zeta(2.0), np.float64)
assert_type(zeta(2.0, _f64_nd), onp.ArrayND[np.float64])
assert_type(zeta(_f64_nd), onp.ArrayND[np.float64])
assert_type(zeta(2j), np.complex128)
assert_type(zeta(2j, _f64_nd), onp.ArrayND[np.complex128])
assert_type(zeta(_c128_nd), onp.ArrayND[np.complex128])
assert_type(zeta(np.float16(2)), np.float32)
assert_type(zeta(np.float32(2)), np.float32)
assert_type(zeta(np.float32(2), 1.0), np.float32)
assert_type(zeta(2.0, np.float32(1)), np.float32)
assert_type(zeta(np.float32(2), np.float64(1)), np.float64)
assert_type(zeta(_f32_1d), onp.Array1D[np.float32])
assert_type(zeta(np.complex64(2)), np.complex64)
assert_type(zeta(np.complex64(2), np.float32(1)), np.complex64)
assert_type(zeta(2j, np.float32(1)), np.complex64)
assert_type(zeta(2j, np.float64(1)), np.complex128)
assert_type(zeta(np.complex128(2), np.float32(1)), np.complex128)

# softplus
assert_type(softplus(1), np.float64)
assert_type(softplus(1.0), np.float64)
assert_type(softplus(np.int32(1)), np.float64 | Any)
assert_type(softplus(np.float32(1.0)), np.float32)
assert_type(softplus(_i64_nd), onp.ArrayND[np.float64 | Any])
assert_type(softplus(_f32_1d), onp.Array1D[np.float32])
assert_type(softplus(_f64_1d), onp.Array1D[np.float64])
assert_type(softplus(np.float32(1.0), dtype=np.float32), np.float32)
assert_type(softplus(_f64_2d, dtype=np.float64), onp.Array2D[np.float64])
assert_type(softplus(1.0, where=_bool_nd), np.float64)
assert_type(softplus(_f64_nd, out=_f64_nd), onp.ArrayND[np.float64])
assert_type(softplus(_f32_1d, out=_f64_nd), onp.ArrayND[np.float64])
