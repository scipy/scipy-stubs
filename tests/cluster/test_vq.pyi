from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.cluster import vq

i64_2d: np.ndarray[tuple[int, int], np.dtype[np.int64]]
f32_2d: onp.Array2D[np.float32]
f64_2d: onp.Array2D[np.float64]
floating_2d: onp.Array2D[np.float32 | np.float64]
c128_2d: np.ndarray[tuple[int, int], np.dtype[np.complex128]]

###

# whiten
assert_type(vq.whiten(i64_2d), onp.Array2D[np.float64])
assert_type(vq.whiten(f64_2d), onp.Array2D[np.float64])
assert_type(vq.whiten(c128_2d), onp.Array2D[np.complex128])

# vq
vq.vq(i64_2d, i64_2d)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(vq.vq(f32_2d, f32_2d), tuple[onp.Array1D[np.int32], onp.Array1D[np.float32]])
assert_type(vq.vq(f64_2d, f64_2d), tuple[onp.Array1D[np.int32], onp.Array1D[np.float64]])
assert_type(vq.vq(floating_2d, floating_2d), tuple[onp.Array1D[np.int32], onp.Array1D[np.floating[Any]]])
vq.vq(c128_2d, f64_2d)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]

# py_vq
assert_type(vq.py_vq(i64_2d, i64_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])
assert_type(vq.py_vq(f32_2d, f32_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])
assert_type(vq.py_vq(f64_2d, f64_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])
assert_type(vq.py_vq(floating_2d, floating_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])
vq.py_vq(c128_2d, f64_2d)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]

# kmeans
vq.kmeans(i64_2d, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(vq.kmeans(f32_2d, 2), tuple[onp.Array2D[np.float32], np.float32])
assert_type(vq.kmeans(f64_2d, 2), tuple[onp.Array2D[np.float64], np.float64])
assert_type(vq.kmeans(floating_2d, 2), tuple[onp.Array2D[np.floating[Any]], np.floating[Any]])
vq.kmeans(c128_2d, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]

# kmeans2
vq.kmeans2(i64_2d, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(vq.kmeans2(f32_2d, 2), tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]])
assert_type(vq.kmeans2(f64_2d, 2), tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]])
assert_type(vq.kmeans2(floating_2d, 2), tuple[onp.Array2D[np.floating[Any]], onp.Array1D[np.int32]])
vq.kmeans2(c128_2d, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
