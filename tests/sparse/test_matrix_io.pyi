import io
from pathlib import Path
from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from ._types import (
    bsr_arr,
    bsr_mat,
    coo_arr,
    coo_mat,
    csc_arr,
    csc_mat,
    csr_arr,
    csr_mat,
    dia_arr,
    dia_mat,
    dok_arr,
    dok_mat,
    lil_arr,
    lil_mat,
)
from scipy import sparse

type _Loaded = (
    sparse.bsr_array[Any]
    | sparse.coo_array[Any]
    | sparse.csc_array[Any]
    | sparse.csr_array[Any, tuple[Any, ...]]
    | sparse.dia_array[Any]
    | sparse.bsr_matrix[Any]
    | sparse.coo_matrix[Any]
    | sparse.csc_matrix[Any]
    | sparse.csr_matrix[Any]
    | sparse.dia_matrix[Any]
    | Any
)

###
# save_npz

assert_type(sparse.save_npz("", bsr_mat), None)
assert_type(sparse.save_npz("", coo_mat), None)
assert_type(sparse.save_npz("", csc_mat), None)
assert_type(sparse.save_npz("", csr_mat), None)
assert_type(sparse.save_npz("", dia_mat), None)
# pyrefly: ignore [bad-argument-type]
sparse.save_npz("", dok_mat)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
# pyrefly: ignore [bad-argument-type]
sparse.save_npz("", lil_mat)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

assert_type(sparse.save_npz("", bsr_arr), None)
assert_type(sparse.save_npz("", coo_arr), None)
assert_type(sparse.save_npz("", csc_arr), None)
assert_type(sparse.save_npz("", csr_arr), None)
assert_type(sparse.save_npz("", dia_arr), None)
# pyrefly: ignore [bad-argument-type]
sparse.save_npz("", dok_arr)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
# pyrefly: ignore [bad-argument-type]
sparse.save_npz("", lil_arr)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

# pyrefly: ignore [bad-argument-type]
sparse.save_npz(b"", coo_arr)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
assert_type(sparse.save_npz(Path(), coo_arr), None)
assert_type(sparse.save_npz(io.BytesIO(), coo_arr), None)
assert_type(sparse.save_npz("", coo_arr, False), None)
assert_type(sparse.save_npz("", coo_arr, False), None)
assert_type(sparse.save_npz("", coo_arr, True), None)
assert_type(sparse.save_npz("", coo_arr, compressed=False), None)
assert_type(sparse.save_npz("", matrix=coo_arr, compressed=True), None)
assert_type(sparse.save_npz(file="", matrix=coo_arr, compressed=True), None)

###
# load_npz

assert_type(sparse.load_npz(""), _Loaded)
assert_type(sparse.load_npz(b""), _Loaded)
assert_type(sparse.load_npz(Path()), _Loaded)
assert_type(sparse.load_npz(io.BytesIO()), _Loaded)
assert_type(
    sparse.load_npz("").tocsr(),
    sparse.csr_array[Any, tuple[int, int]] | sparse.csr_array[Any, tuple[Any, ...]] | sparse.csr_matrix[Any] | Any,
)
assert_type(
    sparse.load_npz("").tocoo(), sparse.coo_array[Any, tuple[int, int]] | sparse.coo_array[Any] | sparse.coo_matrix[Any] | Any
)
assert_type(sparse.load_npz("").sum(axis=0), onp.Array1D[Any] | np.matrix[tuple[int, int], np.dtype[Any]] | Any)
