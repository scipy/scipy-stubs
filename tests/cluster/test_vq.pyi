from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.cluster.vq import kmeans, kmeans2, py_vq, vq, whiten

i64_2d: onp.Array2D[np.int64]
f32_2d: onp.Array2D[np.float32]
f64_2d: onp.Array2D[np.float64]
floating_2d: onp.Array2D[np.float32 | np.float64]
c128_2d: onp.Array2D[np.complex128]

###

# whiten
assert_type(whiten(i64_2d), onp.Array2D[np.float64])
assert_type(whiten(f64_2d), onp.Array2D[np.float64])
assert_type(whiten(c128_2d), onp.Array2D[np.complex128])

# vq
vq(i64_2d, i64_2d)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(vq(f32_2d, f32_2d), tuple[onp.Array1D[np.int32], onp.Array1D[np.float32]])
assert_type(vq(f64_2d, f64_2d), tuple[onp.Array1D[np.int32], onp.Array1D[np.float64]])
assert_type(vq(floating_2d, floating_2d), tuple[onp.Array1D[np.int32], onp.Array1D[npc.floating]])
vq(c128_2d, f64_2d)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]

# py_vq
assert_type(py_vq(i64_2d, i64_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])
assert_type(py_vq(f32_2d, f32_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])
assert_type(py_vq(f64_2d, f64_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])
assert_type(py_vq(floating_2d, floating_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])
py_vq(c128_2d, f64_2d)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]

# kmeans
kmeans(i64_2d, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(kmeans(f32_2d, 2), tuple[onp.Array2D[np.float32], np.float32])
assert_type(kmeans(f64_2d, 2), tuple[onp.Array2D[np.float64], np.float64])
assert_type(kmeans(floating_2d, 2), tuple[onp.Array2D[npc.floating], npc.floating])
kmeans(c128_2d, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]

# kmeans2
kmeans2(i64_2d, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(kmeans2(f32_2d, 2), tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]])
assert_type(kmeans2(f64_2d, 2), tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]])
assert_type(kmeans2(floating_2d, 2), tuple[onp.Array2D[npc.floating], onp.Array1D[np.int32]])
kmeans2(c128_2d, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
