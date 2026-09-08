from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse.linalg import lobpcg

###

_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

###

assert_type(lobpcg(_f64_2d, _f64_2d), tuple[onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(
    lobpcg(_f64_2d, _f64_2d, retLambdaHistory=True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.float64], list[onp.Array1D[np.float64]]],
)
assert_type(
    lobpcg(_f64_2d, _f64_2d, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.float64], list[onp.Array1D[np.float64]]],
)
assert_type(
    lobpcg(_f64_2d, _f64_2d, retLambdaHistory=True, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.float64], list[onp.Array1D[np.float64]], list[onp.Array1D[np.float64]]],
)

assert_type(lobpcg(_c128_2d, _f64_2d), tuple[onp.Array1D[np.float64], onp.Array2D[np.complex128]])
assert_type(
    lobpcg(_c128_2d, _f64_2d, retLambdaHistory=True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.complex128], list[onp.Array1D[np.float64]]],
)
assert_type(
    lobpcg(_c128_2d, _f64_2d, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.complex128], list[onp.Array1D[np.float64]]],
)
assert_type(
    lobpcg(_c128_2d, _f64_2d, retLambdaHistory=True, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.complex128], list[onp.Array1D[np.float64]], list[onp.Array1D[np.float64]]],
)

assert_type(lobpcg(_f32_2d, _f32_2d), tuple[onp.Array1D[np.float32], onp.Array2D[np.float32]])
assert_type(
    lobpcg(_f32_2d, _f32_2d, retLambdaHistory=True),
    tuple[onp.Array1D[np.float32], onp.Array2D[np.float32], list[onp.Array1D[np.float32]]],
)
assert_type(
    lobpcg(_f32_2d, _f32_2d, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float32], onp.Array2D[np.float32], list[onp.Array1D[np.float32]]],
)
assert_type(
    lobpcg(_f32_2d, _f32_2d, retLambdaHistory=True, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float32], onp.Array2D[np.float32], list[onp.Array1D[np.float32]], list[onp.Array1D[np.float32]]],
)

assert_type(lobpcg(_f32_2d, _c64_2d), tuple[onp.Array1D[np.float32], onp.Array2D[np.complex64]])
assert_type(
    lobpcg(_f32_2d, _c64_2d, retLambdaHistory=True),
    tuple[onp.Array1D[np.float32], onp.Array2D[np.complex64], list[onp.Array1D[np.complex64]]],
)
assert_type(
    lobpcg(_f32_2d, _c64_2d, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float32], onp.Array2D[np.complex64], list[onp.Array1D[np.complex64]]],
)
assert_type(
    lobpcg(_f32_2d, _c64_2d, retLambdaHistory=True, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float32], onp.Array2D[np.complex64], list[onp.Array1D[np.complex64]], list[onp.Array1D[np.complex64]]],
)

assert_type(lobpcg(_f64_2d, _f16_2d), tuple[onp.Array1D[np.float64 | Any], onp.Array2D[np.float64 | Any]])
assert_type(
    lobpcg(_f64_2d, _f16_2d, retLambdaHistory=True),
    tuple[onp.Array1D[np.float64 | Any], onp.Array2D[np.float64 | Any], list[onp.Array1D[np.float16]]],
)
assert_type(
    lobpcg(_f64_2d, _f16_2d, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float64 | Any], onp.Array2D[np.float64 | Any], list[onp.Array1D[np.float16]]],
)
assert_type(
    lobpcg(_f64_2d, _f16_2d, retLambdaHistory=True, retResidualNormsHistory=True),
    tuple[
        onp.Array1D[np.float64 | Any], onp.Array2D[np.float64 | Any], list[onp.Array1D[np.float16]], list[onp.Array1D[np.float16]]
    ],
)
