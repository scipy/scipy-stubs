# scipy/linalg/_decomp_interpolative.pyx

from typing import TypeAlias
from typing_extensions import TypeAliasType

import numpy as np
import optype.numpy as onp

from scipy.sparse.linalg import LinearOperator

__all__ = [  # noqa: RUF022
    "idd_estrank",
    "idd_reconid",
    "iddp_aid",
    "iddp_asvd",
    "iddp_id",
    "iddp_qrpiv",
    "iddp_svd",
    "iddr_aid",
    "iddr_asvd",
    "iddr_id",
    "iddr_qrpiv",
    "iddr_svd",
    "idz_estrank",
    "idz_reconid",
    "idzp_aid",
    "idzp_asvd",
    "idzp_id",
    "idzp_qrpiv",
    "idzp_svd",
    "idzr_aid",
    "idzr_asvd",
    "idzr_id",
    "idzr_qrpiv",
    "idzr_svd",
    "idd_id2svd",
    "idz_id2svd",
    "idd_findrank",
    "iddp_rid",
    "iddp_rsvd",
    "iddr_rid",
    "iddr_rsvd",
    "idz_findrank",
    "idzp_rid",
    "idzp_rsvd",
    "idzr_rid",
    "idzr_rsvd",
    "idd_snorm",
    "idz_snorm",
    "idd_diffsnorm",
    "idz_diffsnorm",
]

###
# ruff: noqa: PYI042

_rng: TypeAlias = onp.random.ToRNG

# we use `TypeAliasType` instead `TypeAlias` to reduce the amount of type-checker spaghetti output
_f64 = TypeAliasType("_f64", np.float64)
_i64_1d = TypeAliasType("_i64_1d", onp.Array1D[np.int64])
_f64_1d = TypeAliasType("_f64_1d", onp.Array1D[np.float64])
_c128_1d = TypeAliasType("_c128_1d", onp.Array1D[np.complex128])
_f64_2d = TypeAliasType("_f64_2d", onp.Array2D[np.float64])
_c128_2d = TypeAliasType("_c128_2d", onp.Array2D[np.complex128])

###

#
def idd_diffsnorm(A: LinearOperator, B: LinearOperator, *, rng: _rng, its: int = 20) -> _f64: ...  # undocumented
def idz_diffsnorm(A: LinearOperator, B: LinearOperator, *, rng: _rng, its: int = 20) -> _f64: ...  # undocumented
def idd_snorm(A: LinearOperator, *, rng: _rng, its: int = 20) -> _f64: ...  # undocumented
def idz_snorm(A: LinearOperator, *, rng: _rng, its: int = 20) -> _f64: ...  # undocumented

#
def idd_estrank(a: _f64_2d, eps: float, *, rng: _rng) -> tuple[int, _f64_2d]: ...  # undocumented
def idz_estrank(a: _c128_2d, eps: float, *, rng: _rng) -> tuple[int, _c128_2d]: ...  # undocumented
def idd_findrank(A: LinearOperator, eps: float, *, rng: _rng) -> tuple[int, _f64_2d]: ...  # undocumented
def idz_findrank(A: LinearOperator, eps: float, *, rng: _rng) -> tuple[int, _c128_2d]: ...  # undocumented

#
def idd_reconid(B: _f64_2d, idx: _i64_1d, proj: _f64_1d | _f64_2d) -> _f64_2d: ...  # undocumented
def idz_reconid(B: _c128_2d, idx: _i64_1d, proj: _f64_1d | _f64_2d | _c128_1d | _c128_2d) -> _c128_2d: ...  # undocumented
def iddp_aid(a: _f64_2d, eps: float, *, rng: _rng) -> tuple[int, _i64_1d, _f64_2d]: ...  # undocumented
def idzp_aid(a: _c128_2d, eps: float, *, rng: _rng) -> tuple[int, _i64_1d, _c128_2d]: ...  # undocumented
def iddr_aid(a: _f64_2d, krank: int, *, rng: _rng) -> tuple[int, _i64_1d, _f64_2d]: ...  # undocumented
def idzr_aid(a: _c128_2d, krank: int, *, rng: _rng) -> tuple[_i64_1d, _f64_2d]: ...  # undocumented
def iddp_rid(A: LinearOperator, eps: float, *, rng: _rng) -> tuple[int, _i64_1d, _f64_2d]: ...  # undocumented
def idzp_rid(A: LinearOperator, eps: float, *, rng: _rng) -> tuple[int, _i64_1d, _c128_2d]: ...  # undocumented
def iddr_rid(A: LinearOperator, krank: int, *, rng: _rng) -> tuple[_i64_1d, _f64_2d]: ...  # undocumented
def idzr_rid(A: LinearOperator, krank: int, *, rng: _rng) -> tuple[_i64_1d, _c128_2d]: ...  # undocumented
def iddp_id(a: _f64_2d, eps: float) -> tuple[int, _i64_1d, _f64_2d]: ...  # undocumented
def idzp_id(a: _c128_2d, eps: float) -> tuple[int, _i64_1d, _c128_2d]: ...  # undocumented
def iddr_id(a: _f64_2d, krank: int) -> tuple[_i64_1d, _f64_2d]: ...  # undocumented
def idzr_id(a: _c128_2d, krank: int) -> tuple[_i64_1d, _c128_2d]: ...  # undocumented

#
def iddp_qrpiv(a: _f64_2d, eps: float) -> tuple[int, _f64_1d, _i64_1d]: ...  # undocumented
def idzp_qrpiv(a: _c128_2d, eps: float) -> tuple[int, _c128_1d, _i64_1d]: ...  # undocumented
def iddr_qrpiv(a: _f64_2d, krank: int) -> tuple[_i64_1d, _f64_1d]: ...  # undocumented
def idzr_qrpiv(a: _c128_2d, krank: int) -> tuple[_i64_1d, _c128_1d]: ...  # undocumented

#
def idd_id2svd(cols: _f64_2d, perms: _i64_1d, proj: _f64_2d) -> tuple[_f64_2d, _f64_1d, _f64_2d]: ...  # undocumented
def idz_id2svd(cols: _c128_2d, perms: _i64_1d, proj: _c128_2d) -> tuple[_c128_2d, _f64_1d, _c128_2d]: ...  # undocumented
def iddp_asvd(a: _f64_2d, eps: float, *, rng: _rng) -> tuple[_f64_2d, _f64_1d, _f64_2d]: ...  # undocumented
def idzp_asvd(a: _c128_2d, eps: float, *, rng: _rng) -> tuple[_c128_2d, _f64_1d, _c128_2d]: ...  # undocumented
def iddr_asvd(a: _f64_2d, krank: int, *, rng: _rng) -> tuple[_f64_2d, _f64_1d, _f64_2d]: ...  # undocumented
def idzr_asvd(a: _c128_2d, krank: int, *, rng: _rng) -> tuple[_c128_2d, _f64_1d, _c128_2d]: ...  # undocumented
def iddp_rsvd(A: LinearOperator, eps: float, *, rng: _rng) -> tuple[_f64_2d, _f64_1d, _f64_2d] | None: ...  # undocumented
def idzp_rsvd(A: LinearOperator, eps: float, *, rng: _rng) -> tuple[_c128_2d, _f64_1d, _c128_2d] | None: ...  # undocumented
def iddr_rsvd(A: LinearOperator, krank: int, *, rng: _rng) -> tuple[_f64_2d, _f64_1d, _f64_2d]: ...  # undocumented
def idzr_rsvd(A: LinearOperator, krank: int, *, rng: _rng) -> tuple[_c128_2d, _f64_1d, _c128_2d]: ...  # undocumented
def iddp_svd(a: _f64_2d, eps: float) -> tuple[_f64_2d, _f64_1d, _f64_2d]: ...  # undocumented
def idzp_svd(a: _c128_2d, eps: float) -> tuple[_c128_2d, _f64_1d, _c128_2d]: ...  # undocumented
def iddr_svd(a: _f64_2d, krank: int) -> tuple[_f64_2d, _f64_1d, _f64_2d]: ...  # undocumented
def idzr_svd(a: _c128_2d, krank: int) -> tuple[_c128_2d, _f64_1d, _c128_2d]: ...  # undocumented
