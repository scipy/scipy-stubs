# ruff: file-ignore[snake-case-type-alias]

from typing import Final, Protocol, type_check_only

import numpy as np
import optype.numpy as onp

###

__f2py_numpy_version__: Final[str] = ...  # undocumented

###
# level 1

# (x, [n, offx, incx]) -> s
@type_check_only
class _function_asum[ST: np.generic, T](Protocol):
    def __call__(
        self,
        /,
        x: onp.Array1D[ST],
        *,
        n: int = ...,  # = (len(x) - offx) / abs(incx)
        offx: int = 0,
        incx: int = 1,
    ) -> T: ...

sasum: _function_asum[np.float32, float] = ...
dasum: _function_asum[np.float64, float] = ...
scasum: _function_asum[np.complex64, float] = ...
dzasum: _function_asum[np.complex128, float] = ...

# (x, [n, offx, incx]) -> k
type _function_amax[ST: np.generic] = _function_asum[ST, int]

isamax: _function_amax[np.float32] = ...
idamax: _function_amax[np.float64] = ...
icamax: _function_amax[np.complex64] = ...
izamax: _function_amax[np.complex128] = ...

# (x, [n, offx, incx]) -> n2
type _function_nrm2[ST: np.generic] = _function_asum[ST, float]

snrm2: _function_nrm2[np.float32] = ...
dnrm2: _function_nrm2[np.float64] = ...
scnrm2: _function_nrm2[np.complex64] = ...
dznrm2: _function_nrm2[np.complex128] = ...

# (x, y, [n, offx, incx, offy, incy]) -> xy
@type_check_only
class _function_dot[ST: np.generic, T](Protocol):
    def __call__(
        self,
        /,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        *,
        n: int = ...,  # = (len(x) - offx) / abs(incx)
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
    ) -> T: ...

sdot: _function_dot[np.float32, float] = ...
ddot: _function_dot[np.float64, float] = ...
cdotc: _function_dot[np.complex64, complex] = ...
zdotc: _function_dot[np.complex128, complex] = ...
cdotu: _function_dot[np.complex64, complex] = ...
zdotu: _function_dot[np.complex128, complex] = ...

# (a, b) -> (c, s)
@type_check_only
class _function_rotg[T](Protocol):
    def __call__(self, /, a: T, b: T) -> tuple[T, T]: ...

srotg: _function_rotg[float] = ...
drotg: _function_rotg[float] = ...
crotg: _function_rotg[complex] = ...
zrotg: _function_rotg[complex] = ...

# (d1, d2, x1, y1) -> param
@type_check_only
class _function_rotmg[ST: np.generic](Protocol):
    def __call__(self, /, d1: float, d2: float, x1: float, y1: float) -> onp.Array1D[ST]: ...

srotmg: _function_rotmg[np.float32] = ...
drotmg: _function_rotmg[np.float64] = ...

# (x, y, param, [n, offx, incx, offy, incy, overwrite_x, overwrite_y]) -> (x, y)
@type_check_only
class _function_rotm[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        param: onp.Array1D[ST],
        *,
        n: int = ...,  # = (len(x) - offx) / abs(incx)
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
        overwrite_x: int = 0,
        overwrite_y: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST]]: ...

srotm: _function_rotm[np.float32] = ...
drotm: _function_rotm[np.float64] = ...

# (x, y, c, s, [n, offx, incx, offy, incy, overwrite_x, overwrite_y]) -> (x, y)
@type_check_only
class _function_rot[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        c: float,
        s: float,
        *,
        n: int = ...,  # = (len(x) - 1 - offx) / abs(incx) + 1
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
        overwrite_x: int = 0,
        overwrite_y: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST]]: ...

srot: _function_rot[np.float32] = ...
drot: _function_rot[np.float64] = ...
csrot: _function_rot[np.complex64] = ...
zdrot: _function_rot[np.complex128] = ...

# (x, y, [n, offx, incx, offy, incy]) -> (x, y)
@type_check_only
class _function_swap[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        *,
        n: int = ...,  # = (len(x) - offx) / abs(incx)
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST]]: ...

sswap: _function_swap[np.float32] = ...
dswap: _function_swap[np.float64] = ...
cswap: _function_swap[np.complex64] = ...
zswap: _function_swap[np.complex128] = ...

# (a, x, [n, offx, incx]) -> x
@type_check_only
class _function_scal[ST: np.generic, T](Protocol):
    def __call__(
        self,
        /,
        a: T,
        x: onp.Array1D[ST],
        *,
        n: int = ...,  # = (len(x) - offx) / abs(incx)
        offx: int = 0,
        incx: int = 1,
    ) -> onp.Array1D[ST]: ...

sscal: _function_scal[np.float32, float] = ...
dscal: _function_scal[np.float64, float] = ...
cscal: _function_scal[np.complex64, complex] = ...
zscal: _function_scal[np.complex128, complex] = ...

# (a, x, [n, offx, incx, overwrite_x]) -> x
@type_check_only
class _function_scal_cszd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: float,
        x: onp.Array1D[ST],
        *,
        n: int = ...,  # = (len(x) - offx) / abs(incx)
        offx: int = 0,
        incx: int = 1,
        overwrite_x: int = 0,
    ) -> onp.Array1D[ST]: ...

csscal: _function_scal_cszd[np.complex64] = ...
zdscal: _function_scal_cszd[np.complex128] = ...

# (x, y, [n, offx, incx, offy, incy]) -> y
@type_check_only
class _function_copy[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        *,
        n: int = ...,  # = (len(x) - offx) / abs(incx)
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
    ) -> onp.Array1D[ST]: ...

scopy: _function_copy[np.float32] = ...
dcopy: _function_copy[np.float64] = ...
ccopy: _function_copy[np.complex64] = ...
zcopy: _function_copy[np.complex128] = ...

# (x, y, [n, a, offx, incx, offy, incy]) -> z
@type_check_only
class _function_axpy[ST: np.generic, T](Protocol):
    def __call__(
        self,
        /,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        *,
        n: int = ...,  # = (len(x) - offx) / abs(incx)
        a: T = ...,  # = 1
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
    ) -> onp.Array1D[ST]: ...

saxpy: _function_axpy[np.float32, float] = ...
daxpy: _function_axpy[np.float64, float] = ...
caxpy: _function_axpy[np.complex64, complex] = ...
zaxpy: _function_axpy[np.complex128, complex] = ...

###
# level 2

# (a, x, [offx, incx, lower, trans, diag, overwrite_x]) -> x
@type_check_only
class _function_trmv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        x: onp.Array1D[ST],
        *,
        offx: int = 0,
        incx: int = 1,
        lower: int = 0,
        trans: int = 0,
        diag: int = 0,
        overwrite_x: int = 0,
    ) -> onp.Array1D[ST]: ...

strmv: _function_trmv[np.float32] = ...
dtrmv: _function_trmv[np.float64] = ...
ctrmv: _function_trmv[np.complex64] = ...
ztrmv: _function_trmv[np.complex128] = ...

# (a, x, [incx, offx, lower, trans, diag, overwrite_x]) -> xout
type _function_trsv[ST: np.generic] = _function_trmv[ST]

strsv: _function_trsv[np.float32] = ...
dtrsv: _function_trsv[np.float64] = ...
ctrsv: _function_trsv[np.complex64] = ...
ztrsv: _function_trsv[np.complex128] = ...

# (alpha, a, x, [beta, y, offx, incx, offy, incy, trans, overwrite_y]) -> y
@type_check_only
class _function_gemv[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        x: onp.Array1D[ST],
        *,
        beta: AlphaT = ...,  # = 0
        y: onp.Array1D[ST] | None = None,
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
        trans: int = 0,
        overwrite_y: int = 0,
    ) -> onp.Array1D[ST]: ...

sgemv: _function_gemv[np.float32, float] = ...
dgemv: _function_gemv[np.float64, float] = ...
cgemv: _function_gemv[np.complex64, complex] = ...
zgemv: _function_gemv[np.complex128, complex] = ...

@type_check_only
class _function_mv[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        x: onp.Array1D[ST],
        *,
        beta: AlphaT = ...,  # = 0
        y: onp.Array1D[ST] | None = None,
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
        lower: int = 0,
        overwrite_y: int = 0,
    ) -> onp.Array1D[ST]: ...

ssymv: _function_mv[np.float32, float] = ...
dsymv: _function_mv[np.float64, float] = ...
chemv: _function_mv[np.complex64, complex] = ...
zhemv: _function_mv[np.complex128, complex] = ...

# (alpha, x, [lower, incx, offx, n, a, overwrite_a]) -> a
@type_check_only
class _function_r[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        x: onp.Array1D[ST],
        *,
        lower: int = 0,
        incx: int = 1,
        offx: int = 0,
        n: int = ...,  # = (len(x) - 1 - offx) / abs(incx) + 1
        a: onp.Array2D[ST] | None = None,
        overwrite_a: int = 0,
    ) -> onp.Array2D[ST]: ...

ssyr: _function_r[np.float32, float] = ...
dsyr: _function_r[np.float64, float] = ...
csyr: _function_r[np.complex64, complex] = ...
zsyr: _function_r[np.complex128, complex] = ...
cher: _function_r[np.complex64, complex] = ...
zher: _function_r[np.complex128, complex] = ...

# (k, a, x, [incx, offx, lower, trans, diag, overwrite_x]) -> xout
@type_check_only
class _function_tbsv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        k: int,
        a: onp.Array2D[ST],
        x: onp.Array1D[ST],
        *,
        incx: int = 1,
        offx: int = 0,
        lower: int = 0,
        trans: int = 0,
        diag: int = 0,
        overwrite_x: int = 0,
    ) -> onp.Array1D[ST]: ...

stbsv: _function_tbsv[np.float32] = ...
dtbsv: _function_tbsv[np.float64] = ...
ctbsv: _function_tbsv[np.complex64] = ...
ztbsv: _function_tbsv[np.complex128] = ...

# (n, ap, x, [incx, offx, lower, trans, diag, overwrite_x]) -> xout
@type_check_only
class _function_tpsv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        n: int,
        ap: onp.Array1D[ST],
        x: onp.Array1D[ST],
        *,
        incx: int = 1,
        offx: int = 0,
        lower: int = 0,
        trans: int = 0,
        diag: int = 0,
        overwrite_x: int = 0,
    ) -> onp.Array1D[ST]: ...

stpsv: _function_tpsv[np.float32] = ...
dtpsv: _function_tpsv[np.float64] = ...
ctpsv: _function_tpsv[np.complex64] = ...
ztpsv: _function_tpsv[np.complex128] = ...

# (k, a, x, [incx, offx, lower, trans, diag, overwrite_x]) -> xout
type _function_tbmv[ST: np.generic] = _function_tbsv[ST]

stbmv: _function_tbmv[np.float32] = ...
dtbmv: _function_tbmv[np.float64] = ...
ctbmv: _function_tbmv[np.complex64] = ...
ztbmv: _function_tbmv[np.complex128] = ...

# (n, ap, x, [incx, offx, lower, trans, diag, overwrite_x]) -> xout
type _function_tpmv[ST: np.generic] = _function_tpsv[ST]

stpmv: _function_tpmv[np.float32] = ...
dtpmv: _function_tpmv[np.float64] = ...
ctpmv: _function_tpmv[np.complex64] = ...
ztpmv: _function_tpmv[np.complex128] = ...

# (alpha, x, y, [incx, incy, a, overwrite_x, overwrite_y, overwrite_a]) -> a
@type_check_only
class _function_ger[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        *,
        incx: int = 1,
        incy: int = 1,
        a: onp.Array2D[ST] | None = None,
        overwrite_x: int = 1,
        overwrite_y: int = 1,
        overwrite_a: int = 0,
    ) -> onp.Array2D[ST]: ...

sger: _function_ger[np.float32, float] = ...
dger: _function_ger[np.float64, float] = ...
cgerc: _function_ger[np.complex64, complex] = ...
zgerc: _function_ger[np.complex128, complex] = ...
cgeru: _function_ger[np.complex64, complex] = ...
zgeru: _function_ger[np.complex128, complex] = ...

# (alpha, x, y, [lower, incx, offx, incy, offy, n, a, overwrite_a]) -> a
@type_check_only
class _function_r2[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        *,
        lower: int = 0,
        incx: int = 1,
        offx: int = 0,
        incy: int = 1,
        offy: int = 0,
        n: int = ...,
        a: onp.Array2D[ST] | None = None,
        overwrite_a: int = 0,
    ) -> onp.Array2D[ST]: ...

ssyr2: _function_r2[np.float32, float] = ...
dsyr2: _function_r2[np.float64, float] = ...
cher2: _function_r2[np.complex64, complex] = ...
zher2: _function_r2[np.complex128, complex] = ...

# (n, alpha, x, ap, [incx, offx, lower, overwrite_ap]) -> apu
@type_check_only
class _function_spr[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        n: int,
        alpha: AlphaT,
        x: onp.Array1D[ST],
        ap: onp.Array1D[ST],
        *,
        incx: int = 1,
        offx: int = 0,
        lower: int = 0,
        overwrite_ap: int = 0,
    ) -> onp.Array1D[ST]: ...

sspr: _function_spr[np.float32, float] = ...
dspr: _function_spr[np.float64, float] = ...
cspr: _function_spr[np.complex64, complex] = ...
zspr: _function_spr[np.complex128, complex] = ...

type _function_hpr[ST: np.generic] = _function_spr[ST, float]

chpr: _function_hpr[np.complex64] = ...
zhpr: _function_hpr[np.complex128] = ...

# (n, alpha, x, y, ap, [incx, offx, incy, offy, lower, overwrite_ap]) -> apu
@type_check_only
class _function_pr2[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        n: int,
        alpha: AlphaT,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        ap: onp.Array1D[ST],
        *,
        incx: int = 1,
        offx: int = 0,
        incy: int = 1,
        offy: int = 0,
        lower: int = 0,
        overwrite_ap: int = 0,
    ) -> onp.Array1D[ST]: ...

sspr2: _function_pr2[np.float32, float] = ...
dspr2: _function_pr2[np.float64, float] = ...
chpr2: _function_pr2[np.complex64, complex] = ...
zhpr2: _function_pr2[np.complex128, complex] = ...

# (k, alpha, a, x, [incx, offx, beta, y, incy, offy, lower, overwrite_y]) -> yout
@type_check_only
class _function_bmv[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        k: int,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        x: onp.Array1D[ST],
        *,
        incx: int = 1,
        offx: int = 0,
        beta: AlphaT = ...,  # = 0
        y: onp.Array1D[ST] | None = None,
        incy: int = 1,
        offy: int = 0,
        lower: int = 0,
        overwrite_y: int = 0,
    ) -> onp.Array1D[ST]: ...

ssbmv: _function_bmv[np.float32, float] = ...
dsbmv: _function_bmv[np.float64, float] = ...
chbmv: _function_bmv[np.complex64, complex] = ...
zhbmv: _function_bmv[np.complex128, complex] = ...

# (m, n, kl, ku, alpha, a, x, [incx, offx, beta, y, incy, offy, trans, overwrite_y]) -> yout
@type_check_only
class _function_gbmv[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        m: int,
        n: int,
        kl: int,
        ku: int,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        x: onp.Array1D[ST],
        *,
        incx: int = 1,
        offx: int = 0,
        beta: AlphaT = ...,  # = 0
        y: onp.Array1D[ST] | None = None,
        incy: int = 1,
        offy: int = 0,
        trans: int = 0,
        overwrite_y: int = 0,
    ) -> onp.Array1D[ST]: ...

sgbmv: _function_gbmv[np.float32, float] = ...
dgbmv: _function_gbmv[np.float64, float] = ...
cgbmv: _function_gbmv[np.complex64, complex] = ...
zgbmv: _function_gbmv[np.complex128, complex] = ...

# (n, alpha, ap, x, [incx, offx, beta, y, incy, offy, lower, overwrite_y]) -> yout
@type_check_only
class _function_pmv[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        n: int,
        alpha: AlphaT,
        ap: onp.Array1D[ST],
        x: onp.Array1D[ST],
        *,
        incx: int = 1,
        offx: int = 0,
        beta: AlphaT = ...,  # = 0
        y: onp.Array1D[ST] | None = None,
        incy: int = 1,
        offy: int = 0,
        lower: int = 0,
        overwrite_y: int = 0,
    ) -> onp.Array1D[ST]: ...

sspmv: _function_pmv[np.float32, float] = ...
dspmv: _function_pmv[np.float64, float] = ...
cspmv: _function_pmv[np.complex64, complex] = ...
zspmv: _function_pmv[np.complex128, complex] = ...
chpmv: _function_pmv[np.complex64, complex] = ...
zhpmv: _function_pmv[np.complex128, complex] = ...

###
# level 3

# (alpha, a, [beta, c, trans, lower, overwrite_c]) -> c
@type_check_only
class _function_rk[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        *,
        beta: AlphaT = ...,  # = 0
        c: onp.Array2D[ST] | None = None,
        trans: int = 0,
        lower: int = 0,
        overwrite_c: int = 0,
    ) -> onp.Array2D[ST]: ...

ssyrk: _function_rk[np.float32, float] = ...
dsyrk: _function_rk[np.float64, float] = ...
csyrk: _function_rk[np.complex64, complex] = ...
zsyrk: _function_rk[np.complex128, complex] = ...
cherk: _function_rk[np.complex64, complex] = ...
zherk: _function_rk[np.complex128, complex] = ...

# (alpha, a, b, [beta, c, trans, lower, overwrite_c]) -> c
@type_check_only
class _function_r2k[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        beta: AlphaT = ...,  # = 0
        c: onp.Array2D[ST] | None = None,
        trans: int = 0,
        lower: int = 0,
        overwrite_c: int = 0,
    ) -> onp.Array2D[ST]: ...

ssyr2k: _function_r2k[np.float32, float] = ...
dsyr2k: _function_r2k[np.float64, float] = ...
csyr2k: _function_r2k[np.complex64, complex] = ...
zsyr2k: _function_r2k[np.complex128, complex] = ...
cher2k: _function_r2k[np.complex64, complex] = ...
zher2k: _function_r2k[np.complex128, complex] = ...

# (alpha, a, b, [beta, c, side, lower, overwrite_c]) -> c
@type_check_only
class _function_mm[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        beta: AlphaT = ...,  # = 0
        c: onp.Array2D[ST] | None = None,
        side: int = 0,
        lower: int = 0,
        overwrite_c: int = 0,
    ) -> onp.Array2D[ST]: ...

ssymm: _function_mm[np.float32, float] = ...
dsymm: _function_mm[np.float64, float] = ...
csymm: _function_mm[np.complex64, complex] = ...
zsymm: _function_mm[np.complex128, complex] = ...
chemm: _function_mm[np.complex64, complex] = ...
zhemm: _function_mm[np.complex128, complex] = ...

# (alpha, a, b, [beta, c, trans_a, trans_b, overwrite_c]) -> c
@type_check_only
class _function_gemm[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        beta: AlphaT = ...,  # = 0
        c: onp.Array2D[ST] | None = None,
        trans_a: int = 0,
        trans_b: int = 0,
        overwrite_c: int = 0,
    ) -> onp.Array2D[ST]: ...

sgemm: _function_gemm[np.float32, float] = ...
dgemm: _function_gemm[np.float64, float] = ...
cgemm: _function_gemm[np.complex64, complex] = ...
zgemm: _function_gemm[np.complex128, complex] = ...

# (alpha, a, b, [side, lower, trans_a, diag, overwrite_b]) -> b
@type_check_only
class _function_trmm[ST: np.generic, AlphaT](Protocol):
    def __call__(
        self,
        /,
        alpha: AlphaT,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        side: int = 0,
        lower: int = 0,
        trans_a: int = 0,
        diag: int = 0,
        overwrite_b: int = 0,
    ) -> onp.Array2D[ST]: ...

strmm: _function_trmm[np.float32, float] = ...
dtrmm: _function_trmm[np.float64, float] = ...
ctrmm: _function_trmm[np.complex64, complex] = ...
ztrmm: _function_trmm[np.complex128, complex] = ...

# (alpha, a, b, [side, lower, trans_a, diag, overwrite_b]) -> x
type _function_trsm[ST: np.generic, AlphaT] = _function_trmm[ST, AlphaT]

strsm: _function_trsm[np.float32, float] = ...
dtrsm: _function_trsm[np.float64, float] = ...
ctrsm: _function_trsm[np.complex64, complex] = ...
ztrsm: _function_trsm[np.complex128, complex] = ...
