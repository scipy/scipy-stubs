from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy.compat as npc

from scipy.special import logsumexp, softmax

py_f_0d: float
py_c_0d: complex
py_f_1d: list[float]
py_c_1d: list[complex]

f16_0d: np.float16
f16_1d: np.ndarray[tuple[int], np.dtype[np.float16]]
c64_0d: np.complex64
c64_1d: np.ndarray[tuple[int], np.dtype[np.complex64]]

###
# logsumexp

assert_type(logsumexp(py_f_0d), np.float64)
assert_type(logsumexp(py_f_1d), np.float64)
assert_type(logsumexp(py_c_0d), np.complex128)
assert_type(logsumexp(py_c_1d), np.complex128)
assert_type(logsumexp(f16_0d), np.float16)
assert_type(logsumexp(f16_1d), np.float16)
assert_type(logsumexp(c64_0d), np.complex64)
assert_type(logsumexp(c64_1d), np.complex64)

assert_type(logsumexp(py_f_0d, keepdims=True), npt.NDArray[np.float64])
assert_type(logsumexp(py_f_1d, keepdims=True), npt.NDArray[np.float64])
assert_type(logsumexp(py_c_0d, keepdims=True), npt.NDArray[np.complex128])
assert_type(logsumexp(py_c_1d, keepdims=True), npt.NDArray[np.complex128])
assert_type(logsumexp(f16_0d, keepdims=True), npt.NDArray[np.float16])
assert_type(logsumexp(f16_1d, keepdims=True), npt.NDArray[np.float16])
assert_type(logsumexp(c64_0d, keepdims=True), npt.NDArray[np.complex64])
assert_type(logsumexp(c64_1d, keepdims=True), npt.NDArray[np.complex64])

assert_type(logsumexp(py_f_0d, axis=0), npt.NDArray[np.float64] | Any)
assert_type(logsumexp(py_f_1d, axis=0), npt.NDArray[np.float64] | Any)
assert_type(logsumexp(py_c_0d, axis=0), npt.NDArray[np.complex128] | Any)
assert_type(logsumexp(py_c_1d, axis=0), npt.NDArray[np.complex128] | Any)
assert_type(logsumexp(f16_0d, axis=0), npt.NDArray[np.float16] | Any)
assert_type(logsumexp(f16_1d, axis=0), npt.NDArray[np.float16] | Any)
assert_type(logsumexp(c64_0d, axis=0), npt.NDArray[np.complex64] | Any)
assert_type(logsumexp(c64_1d, axis=0), npt.NDArray[np.complex64] | Any)

assert_type(logsumexp(py_f_0d, return_sign=True), tuple[np.float64, np.float64])
assert_type(logsumexp(py_f_1d, return_sign=True), tuple[np.float64, np.float64])
assert_type(logsumexp(py_c_0d, return_sign=True), tuple[np.float64, np.complex128])
assert_type(logsumexp(py_c_1d, return_sign=True), tuple[np.float64, np.complex128])
assert_type(logsumexp(f16_0d, return_sign=True), tuple[np.float16, np.float16])
assert_type(logsumexp(f16_1d, return_sign=True), tuple[np.float16, np.float16])
assert_type(logsumexp(c64_0d, return_sign=True), tuple[npc.floating, np.complex64])
assert_type(logsumexp(c64_1d, return_sign=True), tuple[npc.floating, np.complex64])

assert_type(logsumexp(py_f_0d, keepdims=True, return_sign=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(logsumexp(py_f_1d, keepdims=True, return_sign=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(logsumexp(py_c_0d, keepdims=True, return_sign=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]])
assert_type(logsumexp(py_c_1d, keepdims=True, return_sign=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]])
assert_type(logsumexp(f16_0d, keepdims=True, return_sign=True), tuple[npt.NDArray[np.float16], npt.NDArray[np.float16]])
assert_type(logsumexp(f16_1d, keepdims=True, return_sign=True), tuple[npt.NDArray[np.float16], npt.NDArray[np.float16]])
assert_type(logsumexp(c64_0d, keepdims=True, return_sign=True), tuple[npt.NDArray[npc.floating], npt.NDArray[np.complex64]])
assert_type(logsumexp(c64_1d, keepdims=True, return_sign=True), tuple[npt.NDArray[npc.floating], npt.NDArray[np.complex64]])

assert_type(logsumexp(py_f_0d, axis=0, return_sign=True), tuple[npt.NDArray[np.float64] | Any, npt.NDArray[np.float64] | Any])
assert_type(logsumexp(py_f_1d, axis=0, return_sign=True), tuple[npt.NDArray[np.float64] | Any, npt.NDArray[np.float64] | Any])
assert_type(logsumexp(py_c_0d, axis=0, return_sign=True), tuple[npt.NDArray[np.float64] | Any, npt.NDArray[np.complex128] | Any])
assert_type(logsumexp(py_c_1d, axis=0, return_sign=True), tuple[npt.NDArray[np.float64] | Any, npt.NDArray[np.complex128] | Any])
assert_type(logsumexp(f16_0d, axis=0, return_sign=True), tuple[npt.NDArray[np.float16] | Any, npt.NDArray[np.float16] | Any])
assert_type(logsumexp(f16_1d, axis=0, return_sign=True), tuple[npt.NDArray[np.float16] | Any, npt.NDArray[np.float16] | Any])
assert_type(logsumexp(c64_0d, axis=0, return_sign=True), tuple[npt.NDArray[npc.floating] | Any, npt.NDArray[np.complex64] | Any])
assert_type(logsumexp(c64_1d, axis=0, return_sign=True), tuple[npt.NDArray[npc.floating] | Any, npt.NDArray[np.complex64] | Any])

###
# softmax (equiv log_softmax)

assert_type(softmax(py_f_0d), np.float64)
assert_type(softmax(py_c_0d), np.complex128)
assert_type(softmax(f16_0d), np.float16)
assert_type(softmax(c64_0d), np.complex64)

assert_type(softmax(py_f_1d), npt.NDArray[np.float64])
assert_type(softmax(py_c_1d), npt.NDArray[np.complex128])
assert_type(softmax(f16_1d), np.ndarray[tuple[int], np.dtype[np.float16]])
assert_type(softmax(c64_1d), np.ndarray[tuple[int], np.dtype[np.complex64]])
