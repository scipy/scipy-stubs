from typing import Any, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import (
    assoc_legendre_p,
    assoc_legendre_p_all,
    legendre_p,
    legendre_p_all,
    sph_harm_y,
    sph_harm_y_all,
    sph_legendre_p,
    sph_legendre_p_all,
)

_Float1_D: TypeAlias = onp.Array[onp.AtLeast1D[Any], np.float64]
_Float2_D: TypeAlias = onp.Array[onp.AtLeast2D[Any], np.float64]
_Float3_D: TypeAlias = onp.Array[onp.AtLeast3D[Any], np.float64]
_Complex0D: TypeAlias = onp.Array0D[np.complex128]
_Complex2D: TypeAlias = onp.Array2D[np.complex128]
_Complex1_D: TypeAlias = onp.Array[onp.AtLeast1D[Any], np.complex128]
_Complex3_D: TypeAlias = onp.Array[onp.AtLeast3D[Any], np.complex128]

_i64_1d: onp.Array1D[np.int64]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

# legendre_p
assert_type(legendre_p(1, 1.0), onp.Array1D[np.float64])
assert_type(legendre_p(1, np.float32(1.0)), onp.Array1D[np.float64])
assert_type(legendre_p(1, _f64_2d), _Float2_D)  # type: ignore[assert-type,arg-type]  # pyright: ignore[reportAssertTypeFailure, reportArgumentType]  # pyrefly: ignore [assert-type, bad-argument-type]  # TODO: fix MultiUFunc array overloads
assert_type(legendre_p(1, 1.0, diff_n=True), onp.Array1D[np.float64])
assert_type(legendre_p(1, 1.0, diff_n=1), onp.Array1D[np.float64])

# legendre_p_all
assert_type(legendre_p_all(3, 1.0), _Float3_D)
assert_type(legendre_p_all(n=3, z=np.float32(1.0)), _Float3_D)
assert_type(legendre_p_all(3, _f64_1d), _Float3_D)
assert_type(legendre_p_all(3, _f64_2d), _Float3_D)
assert_type(legendre_p_all(3, 1.0, diff_n=True), _Float3_D)
assert_type(legendre_p_all(3, 1.0, diff_n=2), _Float3_D)

# assoc_legendre_p
assert_type(assoc_legendre_p(3, 2, 1.0), _Float1_D)
assert_type(assoc_legendre_p(n=3, m=2, z=np.float32(1.0)), _Float1_D)
assert_type(assoc_legendre_p(_i64_1d, 2, 1.0), _Float1_D)
assert_type(assoc_legendre_p(3, _i64_1d, _f64_1d), _Float1_D)
assert_type(assoc_legendre_p(3, 2, _f64_2d), _Float1_D)
assert_type(assoc_legendre_p(3, 2, 1.0, branch_cut=3, norm=True, diff_n=1), _Float1_D)
assert_type(assoc_legendre_p(3, 2, _f64_1d, branch_cut=_i64_1d, diff_n=2), _Float1_D)

# assoc_legendre_p_all
assert_type(assoc_legendre_p_all(3, 2, 1.0), _Float3_D)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]  # pyrefly: ignore [assert-type]  # TODO: fix MultiUFunc array overloads
assert_type(assoc_legendre_p_all(n=3, m=2, z=np.float32(1.0)), _Float3_D)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]  # pyrefly: ignore [assert-type]  # TODO: fix MultiUFunc array overloads
assert_type(assoc_legendre_p_all(n=3, m=2, z=_f64_1d), _Float3_D)  # type: ignore[assert-type,arg-type]  # pyright: ignore[reportAssertTypeFailure, reportArgumentType]  # pyrefly: ignore [assert-type, bad-argument-type]  # TODO: fix MultiUFunc array overloads
assert_type(assoc_legendre_p_all(3, 2, 1.0, branch_cut=3, norm=True, diff_n=1), _Float3_D)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]  # pyrefly: ignore [assert-type]  # TODO: fix MultiUFunc array overloads
assert_type(assoc_legendre_p_all(3, 2, np.float64(1.0), branch_cut=2, diff_n=2), _Float3_D)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]  # pyrefly: ignore [assert-type]  # TODO: fix MultiUFunc array overloads

# sph_legendre_p
assert_type(sph_legendre_p(3, 2, 1.0), onp.Array1D[np.float64])
assert_type(sph_legendre_p(n=3, m=2, theta=np.float32(1.0)), onp.Array1D[np.float64])
assert_type(sph_legendre_p(n=3, m=2, theta=_f64_1d), _Float2_D)  # type: ignore[assert-type,arg-type]  # pyright: ignore[reportAssertTypeFailure, reportArgumentType]  # pyrefly: ignore [assert-type, bad-argument-type]  # TODO: fix MultiUFunc array overloads
assert_type(sph_legendre_p(3, 2, 1.0, diff_n=True), onp.Array1D[np.float64])
assert_type(sph_legendre_p(3, 2, 1.0, diff_n=2), onp.Array1D[np.float64])

# sph_legendre_p_all
assert_type(sph_legendre_p_all(3, 2, 1.0), onp.Array3D[np.float64])
assert_type(sph_legendre_p_all(n=3, m=2, theta=np.float32(1.0)), onp.Array3D[np.float64])
assert_type(sph_legendre_p_all(n=3, m=2, theta=_f64_1d), _Float3_D)  # type: ignore[assert-type,arg-type]  # pyright: ignore[reportAssertTypeFailure, reportArgumentType]  # pyrefly: ignore [assert-type, bad-argument-type]  # TODO: fix MultiUFunc array overloads
assert_type(sph_legendre_p_all(3, 2, 1.0, diff_n=True), onp.Array3D[np.float64])
assert_type(sph_legendre_p_all(3, 2, 1.0, diff_n=2), onp.Array3D[np.float64])

# sph_harm_y
assert_type(sph_harm_y(3, 2, 1.0, 2.0), _Complex0D)
assert_type(sph_harm_y(n=3, m=2, theta=np.float32(1.0), phi=np.float32(2.0)), _Complex0D)
assert_type(sph_harm_y(3, 2, 1.0, _f64_1d), _Complex1_D)  # type: ignore[assert-type,arg-type]  # pyright: ignore[reportAssertTypeFailure, reportArgumentType]  # pyrefly: ignore [assert-type, bad-argument-type]  # TODO: fix MultiUFunc array overloads
assert_type(sph_harm_y(3, 2, 1.0, 2.0, diff_n=False), _Complex0D)
assert_type(sph_harm_y(3, 2, 1.0, 2.0, diff_n=0), _Complex0D)

# sph_harm_y_all
assert_type(sph_harm_y_all(3, 2, 1.0, 2.0), _Complex2D)
assert_type(sph_harm_y_all(n=3, m=2, theta=np.float32(1.0), phi=np.float32(2.0)), _Complex2D)
assert_type(sph_harm_y_all(3, 2, 1.0, _f64_1d), _Complex3_D)  # type: ignore[assert-type,arg-type]  # pyright: ignore[reportAssertTypeFailure, reportArgumentType]  # pyrefly: ignore [assert-type, bad-argument-type]  # TODO: fix MultiUFunc array overloads
assert_type(sph_harm_y_all(3, 2, 1.0, 2.0, diff_n=False), _Complex2D)
assert_type(sph_harm_y_all(3, 2, 1.0, 2.0, diff_n=0), _Complex2D)
