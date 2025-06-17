import io
from pathlib import Path
from typing import Any, assert_type

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
from scipy.sparse._data import _data_matrix

###
# save_npz

assert_type(sparse.save_npz("", bsr_mat), None)
assert_type(sparse.save_npz("", coo_mat), None)
assert_type(sparse.save_npz("", csc_mat), None)
assert_type(sparse.save_npz("", csr_mat), None)
assert_type(sparse.save_npz("", dia_mat), None)
sparse.save_npz("", dok_mat)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
sparse.save_npz("", lil_mat)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

assert_type(sparse.save_npz("", bsr_arr), None)
assert_type(sparse.save_npz("", coo_arr), None)
assert_type(sparse.save_npz("", csc_arr), None)
assert_type(sparse.save_npz("", csr_arr), None)
assert_type(sparse.save_npz("", dia_arr), None)
sparse.save_npz("", dok_arr)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
sparse.save_npz("", lil_arr)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

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

assert_type(sparse.load_npz(""), _data_matrix | Any)
assert_type(sparse.load_npz(b""), _data_matrix | Any)
assert_type(sparse.load_npz(Path()), _data_matrix | Any)
assert_type(sparse.load_npz(io.BytesIO()), _data_matrix | Any)
