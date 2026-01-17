from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import (
    c_roots,
    cg_roots,
    chebyc,
    chebys,
    chebyt,
    chebyu,
    gegenbauer,
    genlaguerre,
    h_roots,
    he_roots,
    hermite,
    hermitenorm,
    j_roots,
    jacobi,
    js_roots,
    l_roots,
    la_roots,
    laguerre,
    legendre,
    p_roots,
    ps_roots,
    roots_chebyc,
    roots_chebys,
    roots_chebyt,
    roots_chebyu,
    roots_gegenbauer,
    roots_genlaguerre,
    roots_hermite,
    roots_hermitenorm,
    roots_jacobi,
    roots_laguerre,
    roots_legendre,
    roots_sh_chebyt,
    roots_sh_chebyu,
    roots_sh_jacobi,
    roots_sh_legendre,
    s_roots,
    sh_chebyt,
    sh_chebyu,
    sh_jacobi,
    sh_legendre,
    t_roots,
    ts_roots,
    u_roots,
    us_roots,
)
from scipy.special._orthogonal import orthopoly1d

###

_py_f_1d: list[float]
_py_f_2d: list[list[float]]

_py_c_1d: list[complex]
_py_c_2d: list[list[complex]]

_f32: np.float32
_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_c64: np.complex64
_c64_1d: onp.Array1D[np.complex64]
_c64_2d: onp.Array2D[np.complex64]
_c64_nd: onp.ArrayND[np.complex64]

_orthopoly: orthopoly1d

###
# orthopoly1d

assert_type(legendre(3), orthopoly1d)
assert_type(chebyt(3), orthopoly1d)
assert_type(chebyu(3), orthopoly1d)
assert_type(chebyc(3), orthopoly1d)
assert_type(chebys(3), orthopoly1d)
assert_type(jacobi(3, 2, 1), orthopoly1d)
assert_type(laguerre(3), orthopoly1d)
assert_type(genlaguerre(3, 2), orthopoly1d)
assert_type(hermite(3), orthopoly1d)
assert_type(hermitenorm(3), orthopoly1d)
assert_type(gegenbauer(3, 2), orthopoly1d)
assert_type(sh_legendre(3), orthopoly1d)
assert_type(sh_chebyt(3), orthopoly1d)
assert_type(sh_chebyu(3), orthopoly1d)
assert_type(sh_jacobi(3, 2, 1), orthopoly1d)

assert_type(_orthopoly.limits, tuple[float, float])
assert_type(_orthopoly.weights, onp.Array2D[np.float64])
assert_type(_orthopoly.weight_func(1), float)
assert_type(_orthopoly.normcoef, float)

assert_type(_orthopoly(_orthopoly), np.poly1d)
assert_type(_orthopoly(1), np.float64)
assert_type(_orthopoly(_py_f_1d), onp.Array1D[np.float64])
assert_type(_orthopoly(_py_f_2d), onp.Array2D[np.float64])
assert_type(_orthopoly(1j), np.complex128)
assert_type(_orthopoly(_py_c_1d), onp.Array1D[np.complex128])
assert_type(_orthopoly(_py_c_2d), onp.Array2D[np.complex128])
assert_type(_orthopoly(_f32), np.float64)
assert_type(_orthopoly(_f32_1d), onp.Array1D[np.float64])
assert_type(_orthopoly(_f32_2d), onp.Array2D[np.float64])
assert_type(_orthopoly(_f32_nd), onp.ArrayND[np.float64])
assert_type(_orthopoly(_c64), np.complex128)
assert_type(_orthopoly(_c64_1d), onp.Array1D[np.complex128])
assert_type(_orthopoly(_c64_2d), onp.Array2D[np.complex128])
assert_type(_orthopoly(_c64_nd), onp.ArrayND[np.complex128])

###
# roots

# jacobi
assert_type(roots_jacobi(3, 2, 1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_jacobi(3, 2, 1, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(j_roots(3, 2, 1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(j_roots(3, 2, 1, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# "sh_jacobi"
assert_type(roots_sh_jacobi(3, 2, 1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_sh_jacobi(3, 2, 1, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(js_roots(3, 2, 1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(js_roots(3, 2, 1, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# genlaguerre
assert_type(roots_genlaguerre(3, 2), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_genlaguerre(3, 2, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(la_roots(3, 2), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(la_roots(3, 2, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# laguerre
assert_type(roots_laguerre(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_laguerre(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(l_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(l_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# hermite
assert_type(roots_hermite(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_hermite(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(h_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(h_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# hermitenorm
assert_type(roots_hermitenorm(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_hermitenorm(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(he_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(he_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# gegegenbauer
assert_type(roots_gegenbauer(3, 2), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_gegenbauer(3, 2, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(cg_roots(3, 2), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(cg_roots(3, 2, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# chebyt
assert_type(roots_chebyt(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_chebyt(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(t_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(t_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# chebyu
assert_type(roots_chebyu(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_chebyu(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(u_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(u_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# chebyc
assert_type(roots_chebyc(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_chebyc(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(c_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(c_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# chebys
assert_type(roots_chebys(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_chebys(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(s_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(s_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# sh_chebyt
assert_type(roots_sh_chebyt(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_sh_chebyt(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(ts_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(ts_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# sh_chebyu
assert_type(roots_sh_chebyu(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_sh_chebyu(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(us_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(us_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# legendre
assert_type(roots_legendre(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_legendre(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(p_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(p_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

# sh_legendre
assert_type(roots_sh_legendre(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(roots_sh_legendre(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(ps_roots(3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(ps_roots(3, mu=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])

###
