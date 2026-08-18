from collections.abc import Callable
from typing import Final, Protocol, type_check_only

import numpy as np
import optype.numpy as onp

###

type _rcond = float  # ruff: ignore[snake-case-type-alias]
type _info = int  # ruff: ignore[snake-case-type-alias]

###

__f2py_numpy_version__: Final[str] = ...  # undocumented

###

# (kl, ku, ab, ipiv, anorm, [norm, ldab]) -> (rcond, info)
@type_check_only
class _function_gbcon[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        kl: int,
        ku: int,
        ab: onp.Array2D[ST],
        ipiv: onp.Array1D[np.int32],
        anorm: float,
        *,
        norm: str | bytes = "1",
        ldab: int = ...,  # = 2 * kl + ku + 1
    ) -> tuple[_rcond, _info]: ...

sgbcon: _function_gbcon[np.float32] = ...
dgbcon: _function_gbcon[np.float64] = ...
cgbcon: _function_gbcon[np.complex64] = ...
zgbcon: _function_gbcon[np.complex128] = ...

# (a, anorm, [norm]) -> (rcond, info)
@type_check_only
class _function_gecon[ST: np.generic](Protocol):
    def __call__(self, /, a: onp.Array2D[ST], anorm: float, *, norm: str | bytes = "1") -> tuple[_rcond, _info]: ...

sgecon: _function_gecon[np.float32] = ...
dgecon: _function_gecon[np.float64] = ...
cgecon: _function_gecon[np.complex64] = ...
zgecon: _function_gecon[np.complex128] = ...

# (kl, ku, ab, b, [overwrite_ab, overwrite_b]) -> (lub, piv, x, info)
@type_check_only
class _function_gbsv[ST: np.generic](Protocol):
    def __call__(
        self, /, kl: int, ku: int, ab: onp.Array2D[ST], b: onp.Array2D[ST], *, overwrite_ab: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], onp.Array2D[ST], _info]: ...

sgbsv: _function_gbsv[np.float32] = ...
dgbsv: _function_gbsv[np.float64] = ...
cgbsv: _function_gbsv[np.complex64] = ...
zgbsv: _function_gbsv[np.complex128] = ...

# (ab,kl,ku,[m,n,ldab,overwrite_ab]) -> (lu, ipiv, info)
@type_check_only
class _function_gbtrf[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        kl: int,
        ku: int,
        m: int = ...,  # = ab.shape[1]
        n: int = ...,  # = ab.shape[1]
        *,
        overwrite_ab: int = 0,
        ldab: int = ...,  # = max(ab.shape[0], 1)
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], _info]: ...

sgbtrf: _function_gbtrf[np.float32] = ...
dgbtrf: _function_gbtrf[np.float64] = ...
cgbtrf: _function_gbtrf[np.complex64] = ...
zgbtrf: _function_gbtrf[np.complex128] = ...

# (ab, kl, ku, b, ipiv, [trans, n, ldab, ldb, overwrite_b]) -> (x, info)
@type_check_only
class _function_gbtrs[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        kl: int,
        ku: int,
        b: onp.Array2D[ST],
        ipiv: onp.Array1D[np.int32],
        *,
        trans: int = 0,
        n: int = ...,  # = ab.shape[1]
        ldab: int = ...,  # = ab.shape[0]
        ldb: int = ...,  # = b.shape[0]
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

sgbtrs: _function_gbtrs[np.float32] = ...
dgbtrs: _function_gbtrs[np.float64] = ...
cgbtrs: _function_gbtrs[np.complex64] = ...
zgbtrs: _function_gbtrs[np.complex128] = ...

# (a, [scale, permute, overwrite_a]) -> (ba, lo, hi, pivscale, info)
@type_check_only
class _function_gebal[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, scale: int = 0, permute: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], int, int, onp.Array1D[RT], _info]: ...

sgebal: _function_gebal[np.float32, np.float32] = ...
dgebal: _function_gebal[np.float64, np.float64] = ...
cgebal: _function_gebal[np.complex64, np.float32] = ...
zgebal: _function_gebal[np.complex128, np.float64] = ...

# (a) -> (r, c, rowcnd, colcnd, amax, info)
@type_check_only
class _function_geequ[ST: np.generic, RT: np.generic](Protocol):
    def __call__(self, /, a: onp.Array2D[ST]) -> tuple[onp.Array1D[RT], onp.Array1D[RT], float, float, float, _info]: ...

sgeequ: _function_geequ[np.float32, np.float32] = ...
dgeequ: _function_geequ[np.float64, np.float64] = ...
cgeequ: _function_geequ[np.complex64, np.float32] = ...
zgeequ: _function_geequ[np.complex128, np.float64] = ...

sgeequb: _function_geequ[np.float32, np.float32] = ...
dgeequb: _function_geequ[np.float64, np.float64] = ...
cgeequb: _function_geequ[np.complex64, np.float32] = ...
zgeequb: _function_geequ[np.complex128, np.float64] = ...

# (sselect, a, [compute_v, sort_t, lwork, sselect_extra_args, overwrite_a]) -> (t, sdim, wr, wi, vs, work, info)
@type_check_only
class _function_gees_s(Protocol):
    def __call__(
        self,
        /,
        sselect: Callable[..., int],
        a: onp.Array2D[np.float32],
        *,
        compute_v: int = 1,
        sort_t: int = 0,
        lwork: int = ...,  # = max(3 * n, 1)
        sselect_extra_args: tuple[object, ...] = (),
        overwrite_a: int = 0,
    ) -> tuple[
        onp.Array2D[np.float32],
        int,
        onp.Array1D[np.float32],
        onp.Array1D[np.float32],
        onp.Array2D[np.float32],
        onp.Array1D[np.float32],
        _info,
    ]: ...

sgees: _function_gees_s = ...

@type_check_only
class _function_gees_d(Protocol):
    def __call__(
        self,
        /,
        dselect: Callable[..., int],
        a: onp.Array2D[np.float64],
        *,
        compute_v: int = 1,
        sort_t: int = 0,
        lwork: int = ...,  # = max(3 * n, 1)
        dselect_extra_args: tuple[object, ...] = (),
        overwrite_a: int = 0,
    ) -> tuple[
        onp.Array2D[np.float64],
        int,
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array2D[np.float64],
        onp.Array1D[np.float64],
        _info,
    ]: ...

dgees: _function_gees_d = ...

# (cselect, a, [compute_v, sort_t, lwork, cselect_extra_args, overwrite_a]) -> (t, sdim, w, vs, work, info)
@type_check_only
class _function_gees_c(Protocol):
    def __call__(
        self,
        /,
        cselect: Callable[..., int],
        a: onp.Array2D[np.complex64],
        *,
        compute_v: int = 1,
        sort_t: int = 0,
        lwork: int = ...,  # = max(3 * n, 1)
        cselect_extra_args: tuple[object, ...] = (),
        overwrite_a: int = 0,
    ) -> tuple[
        onp.Array2D[np.complex64], int, onp.Array1D[np.complex64], onp.Array2D[np.complex64], onp.Array1D[np.complex64], _info
    ]: ...

cgees: _function_gees_c = ...

@type_check_only
class _function_gees_z(Protocol):
    def __call__(
        self,
        /,
        zselect: Callable[..., int],
        a: onp.Array2D[np.complex128],
        *,
        compute_v: int = 1,
        sort_t: int = 0,
        lwork: int = ...,  # = max(3 * n, 1)
        zselect_extra_args: tuple[object, ...] = (),
        overwrite_a: int = 0,
    ) -> tuple[
        onp.Array2D[np.complex128], int, onp.Array1D[np.complex128], onp.Array2D[np.complex128], onp.Array1D[np.complex128], _info
    ]: ...

zgees: _function_gees_z = ...

# (a, [compute_vl, compute_vr, lwork, overwrite_a]) -> (wr, wi, vl, vr, info)
@type_check_only
class _function_geev_sd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        compute_vl: int = 1,
        compute_vr: int = 1,
        lwork: int = ...,  # = max(4 * n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST], onp.Array2D[ST], onp.Array2D[ST], _info]: ...

sgeev: _function_geev_sd[np.float32] = ...
dgeev: _function_geev_sd[np.float64] = ...

# (a, [compute_vl, compute_vr, lwork, overwrite_a]) -> (w, vl, vr, info)
@type_check_only
class _function_geev_cz[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        compute_vl: int = 1,
        compute_vr: int = 1,
        lwork: int = ...,  # = max(2 * n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], onp.Array2D[ST], _info]: ...

cgeev: _function_geev_cz[np.complex64] = ...
zgeev: _function_geev_cz[np.complex128] = ...

# (n, [compute_vl, compute_vr]) -> (work, info)
@type_check_only
class _function_geev_lwork[WorkT](Protocol):
    def __call__(self, /, n: int, *, compute_vl: int = 1, compute_vr: int = 1) -> tuple[WorkT, _info]: ...

sgeev_lwork: _function_geev_lwork[float] = ...
dgeev_lwork: _function_geev_lwork[float] = ...
cgeev_lwork: _function_geev_lwork[complex] = ...
zgeev_lwork: _function_geev_lwork[complex] = ...

# (a, [lo, hi, lwork, overwrite_a]) -> (ht, tau, info)
@type_check_only
class _function_gehrd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lo: int = 0,
        hi: int = ...,  # = n - 1
        lwork: int = ...,  # = max(n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

sgehrd: _function_gehrd[np.float32] = ...
dgehrd: _function_gehrd[np.float64] = ...
cgehrd: _function_gehrd[np.complex64] = ...
zgehrd: _function_gehrd[np.complex128] = ...

# (n, [lo, hi]) -> (work, info)
@type_check_only
class _function_gehrd_lwork[WorkT](Protocol):
    def __call__(
        self,
        /,
        n: int,
        *,
        lo: int = 0,
        hi: int = ...,  # = n - 1
    ) -> tuple[WorkT, _info]: ...

sgehrd_lwork: _function_gehrd_lwork[float] = ...
dgehrd_lwork: _function_gehrd_lwork[float] = ...
cgehrd_lwork: _function_gehrd_lwork[complex] = ...
zgehrd_lwork: _function_gehrd_lwork[complex] = ...

# (a, [joba, jobu, jobv, jobr, jobt, jobp, lwork, overwrite_a]) -> (sva, u, v, workout, iworkout, info)
@type_check_only
class _function_gejsv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        joba: int = 4,
        jobu: int = 0,
        jobv: int = 0,
        jobr: int = 1,
        jobt: int = 0,
        jobp: int = 1,
        lwork: int = ...,  # = max(6 * n + 2 * n * n, max(2 * m + n, max(4 * n + n * n, max(2 * n + n * n + 6, 7))))
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[ST], onp.Array1D[np.int32], _info]: ...

sgejsv: _function_gejsv[np.float32] = ...
dgejsv: _function_gejsv[np.float64] = ...

# (a, b, [trans, lwork, overwrite_a, overwrite_b]) -> (lqr, x, info)
@type_check_only
class _function_gels[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        trans: str | bytes = "N",
        lwork: int = ...,  # = max(min(m, n) + max(min(m, n), nrhs), 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], _info]: ...

sgels: _function_gels[np.float32] = ...
dgels: _function_gels[np.float64] = ...
cgels: _function_gels[np.complex64] = ...
zgels: _function_gels[np.complex128] = ...

# (m, n, nrhs, [trans]) -> (work, info)
@type_check_only
class _function_gels_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int, nrhs: int, *, trans: str | bytes = "N") -> tuple[WorkT, _info]: ...

sgels_lwork: _function_gels_lwork[float] = ...
dgels_lwork: _function_gels_lwork[float] = ...
cgels_lwork: _function_gels_lwork[complex] = ...
zgels_lwork: _function_gels_lwork[complex] = ...

# (a, b, lwork, size_iwork, [cond, overwrite_a, overwrite_b]) -> (x, s, rank, info)
@type_check_only
class _function_gelsd_sd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        lwork: int,
        size_iwork: int,
        *,
        cond: float = -1.0,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], int, _info]: ...

sgelsd: _function_gelsd_sd[np.float32] = ...
dgelsd: _function_gelsd_sd[np.float64] = ...

# (a, b, lwork, size_rwork, size_iwork, [cond, overwrite_a, overwrite_b]) -> (x, s, rank, info)
@type_check_only
class _function_gelsd_cz[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        b: onp.Array2D[CT],
        lwork: int,
        size_rwork: int,
        size_iwork: int,
        *,
        cond: float = -1.0,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[CT], onp.Array1D[RT], int, _info]: ...

cgelsd: _function_gelsd_cz[np.complex64, np.float32] = ...
zgelsd: _function_gelsd_cz[np.complex128, np.float64] = ...

# (m, n, nrhs, [cond, lwork]) -> (work, iwork, info)
@type_check_only
class _function_gelsd_lwork_sd(Protocol):
    def __call__(self, /, m: int, n: int, nrhs: int, *, cond: float = -1.0, lwork: int = -1) -> tuple[float, int, _info]: ...

sgelsd_lwork: _function_gelsd_lwork_sd = ...
dgelsd_lwork: _function_gelsd_lwork_sd = ...

# (m, n, nrhs, [cond, lwork]) -> (work, rwork, iwork, info)
@type_check_only
class _function_gelsd_lwork_cz(Protocol):
    def __call__(
        self, /, m: int, n: int, nrhs: int, *, cond: float = -1.0, lwork: int = -1
    ) -> tuple[complex, float, int, _info]: ...

cgelsd_lwork: _function_gelsd_lwork_cz = ...
zgelsd_lwork: _function_gelsd_lwork_cz = ...

# (a, b, [cond, lwork, overwrite_a, overwrite_b]) -> (v, x, s, rank, work, info)
@type_check_only
class _function_gelss_sd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        cond: float = -1.0,
        lwork: int = ...,  # = max(3 * minmn + max(2 * minmn, max(maxmn, nrhs)), 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[ST], int, onp.Array1D[ST], _info]: ...

sgelss: _function_gelss_sd[np.float32] = ...
dgelss: _function_gelss_sd[np.float64] = ...

@type_check_only
class _function_gelss_cz[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        b: onp.Array2D[CT],
        *,
        cond: float = -1.0,
        lwork: int = ...,  # = max(2 * minmn + max(maxmn, nrhs), 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[CT], onp.Array2D[CT], onp.Array1D[RT], int, onp.Array1D[CT], _info]: ...

cgelss: _function_gelss_cz[np.complex64, np.float32] = ...
zgelss: _function_gelss_cz[np.complex128, np.float64] = ...

# (m, n, nrhs, [cond, lwork]) -> (work, info)
@type_check_only
class _function_gelss_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int, nrhs: int, *, cond: float = -1.0, lwork: int = -1) -> tuple[WorkT, _info]: ...

sgelss_lwork: _function_gelss_lwork[float] = ...
dgelss_lwork: _function_gelss_lwork[float] = ...
cgelss_lwork: _function_gelss_lwork[complex] = ...
zgelss_lwork: _function_gelss_lwork[complex] = ...

# (a, b, jptv, cond, lwork, [overwrite_a, overwrite_b]) -> (v, x, j, rank, info)
@type_check_only
class _function_gelsy[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        jptv: onp.Array1D[np.int32],
        cond: float,
        lwork: int,
        *,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[np.int32], int, _info]: ...

sgelsy: _function_gelsy[np.float32] = ...
dgelsy: _function_gelsy[np.float64] = ...
cgelsy: _function_gelsy[np.complex64] = ...
zgelsy: _function_gelsy[np.complex128] = ...

# (m, n, nrhs, cond, [lwork]) -> (work, info)
@type_check_only
class _function_gelsy_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int, nrhs: int, cond: float, *, lwork: int = -1) -> tuple[WorkT, _info]: ...

sgelsy_lwork: _function_gelsy_lwork[float] = ...
dgelsy_lwork: _function_gelsy_lwork[float] = ...
cgelsy_lwork: _function_gelsy_lwork[complex] = ...
zgelsy_lwork: _function_gelsy_lwork[complex] = ...

# (v, t, c, [side, trans, overwrite_c]) -> (c, info)
@type_check_only
class _function_gemqrt[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        v: onp.Array2D[ST],
        t: onp.Array2D[ST],
        c: onp.Array2D[ST],
        *,
        side: str | bytes = "L",
        trans: str | bytes = "N",
        overwrite_c: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

sgemqrt: _function_gemqrt[np.float32] = ...
dgemqrt: _function_gemqrt[np.float64] = ...
cgemqrt: _function_gemqrt[np.complex64] = ...
zgemqrt: _function_gemqrt[np.complex128] = ...

# (a, [lwork, overwrite_a]) -> (qr, jpvt, tau, work, info)
@type_check_only
class _function_geqp3[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lwork: int = ...,  # = max(3 * (n + 1), 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], onp.Array1D[ST], onp.Array1D[ST], _info]: ...

sgeqp3: _function_geqp3[np.float32] = ...
dgeqp3: _function_geqp3[np.float64] = ...
cgeqp3: _function_geqp3[np.complex64] = ...
zgeqp3: _function_geqp3[np.complex128] = ...

# (a, [lwork, overwrite_a]) -> (qr, tau, work, info)
@type_check_only
class _function_geqrf[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lwork: int = ...,  # = max(3 * n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], onp.Array1D[ST], _info]: ...

sgeqrf: _function_geqrf[np.float32] = ...
dgeqrf: _function_geqrf[np.float64] = ...
cgeqrf: _function_geqrf[np.complex64] = ...
zgeqrf: _function_geqrf[np.complex128] = ...

# (m, n) -> (work, info)
@type_check_only
class _function_geqrf_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int) -> tuple[WorkT, _info]: ...

sgeqrf_lwork: _function_geqrf_lwork[float] = ...
dgeqrf_lwork: _function_geqrf_lwork[float] = ...
cgeqrf_lwork: _function_geqrf_lwork[complex] = ...
zgeqrf_lwork: _function_geqrf_lwork[complex] = ...

# (a, [lwork, overwrite_a]) -> (qr, tau, info)
@type_check_only
class _function_geqrfp[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lwork: int = ...,  # = max(1, n)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

sgeqrfp: _function_geqrfp[np.float32] = ...
dgeqrfp: _function_geqrfp[np.float64] = ...
cgeqrfp: _function_geqrfp[np.complex64] = ...
zgeqrfp: _function_geqrfp[np.complex128] = ...

# (m, n) -> (work, info)
@type_check_only
class _function_geqrfp_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int) -> tuple[WorkT, _info]: ...

sgeqrfp_lwork: _function_geqrfp_lwork[float] = ...
dgeqrfp_lwork: _function_geqrfp_lwork[float] = ...
cgeqrfp_lwork: _function_geqrfp_lwork[complex] = ...
zgeqrfp_lwork: _function_geqrfp_lwork[complex] = ...

# (nb, a, [overwrite_a]) -> (a, t, info)
@type_check_only
class _function_geqrt[ST: np.generic](Protocol):
    def __call__(
        self, /, nb: int, a: onp.Array2D[ST], *, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], _info]: ...

sgeqrt: _function_geqrt[np.float32] = ...
dgeqrt: _function_geqrt[np.float64] = ...
cgeqrt: _function_geqrt[np.complex64] = ...
zgeqrt: _function_geqrt[np.complex128] = ...

# (a, [lwork, overwrite_a]) -> (qr, tau, work, info)
@type_check_only
class _function_gerqf[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lwork: int = ...,  # = max(3 * m, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], onp.Array1D[ST], _info]: ...

sgerqf: _function_gerqf[np.float32] = ...
dgerqf: _function_gerqf[np.float64] = ...
cgerqf: _function_gerqf[np.complex64] = ...
zgerqf: _function_gerqf[np.complex128] = ...

# (lu, rhs, ipiv, jpiv, [overwrite_rhs]) -> (x, scale)
@type_check_only
class _function_gesc2[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        lu: onp.Array2D[ST],
        rhs: onp.Array1D[ST],
        ipiv: onp.Array1D[np.int32],
        jpiv: onp.Array1D[np.int32],
        *,
        overwrite_rhs: int = 0,
    ) -> tuple[onp.Array1D[ST], float]: ...

sgesc2: _function_gesc2[np.float32] = ...
dgesc2: _function_gesc2[np.float64] = ...
cgesc2: _function_gesc2[np.complex64] = ...
zgesc2: _function_gesc2[np.complex128] = ...

# (a, [compute_uv, full_matrices, lwork, overwrite_a]) -> (u, s, vt, info)
@type_check_only
class _function_gesdd_sd[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, compute_uv: int = 1, full_matrices: int = 1, lwork: int = ..., overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], onp.Array2D[ST], _info]: ...

sgesdd: _function_gesdd_sd[np.float32] = ...
dgesdd: _function_gesdd_sd[np.float64] = ...

@type_check_only
class _function_gesdd_cz[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        *,
        compute_uv: int = 1,
        full_matrices: int = 1,
        lwork: int = ...,  # = max((2 * minmn * minmn + max(m, n) + 2 * minmn if compute_uv else 2 * minmn + max(m, n)), 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[CT], onp.Array1D[RT], onp.Array2D[CT], _info]: ...

cgesdd: _function_gesdd_cz[np.complex64, np.float32] = ...
zgesdd: _function_gesdd_cz[np.complex128, np.float64] = ...

# (m, n, [compute_uv, full_matrices]) -> (work, info)
@type_check_only
class _function_gesdd_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int, *, compute_uv: int = 1, full_matrices: int = 1) -> tuple[WorkT, _info]: ...

sgesdd_lwork: _function_gesdd_lwork[float] = ...
dgesdd_lwork: _function_gesdd_lwork[float] = ...
cgesdd_lwork: _function_gesdd_lwork[complex] = ...
zgesdd_lwork: _function_gesdd_lwork[complex] = ...

# (a, b, [overwrite_a, overwrite_b]) -> (lu, piv, x, info)
@type_check_only
class _function_gesv[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], b: onp.Array2D[ST], *, overwrite_a: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], onp.Array2D[ST], _info]: ...

sgesv: _function_gesv[np.float32] = ...
dgesv: _function_gesv[np.float64] = ...
cgesv: _function_gesv[np.complex64] = ...
zgesv: _function_gesv[np.complex128] = ...

# (a, [compute_uv, full_matrices, lwork, overwrite_a]) -> (u, s, vt, info)
@type_check_only
class _function_gesvd_sd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        compute_uv: int = 1,
        full_matrices: int = 1,
        lwork: int = ...,  # = max(max(3 * minmn + max(m, n), 5 * minmn), 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], onp.Array2D[ST], _info]: ...

sgesvd: _function_gesvd_sd[np.float32] = ...
dgesvd: _function_gesvd_sd[np.float64] = ...

@type_check_only
class _function_gesvd_cz[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        *,
        compute_uv: int = 1,
        full_matrices: int = 1,
        lwork: int = ...,  # = max(2 * minmn + max(m, n), 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[CT], onp.Array1D[RT], onp.Array2D[CT], _info]: ...

cgesvd: _function_gesvd_cz[np.complex64, np.float32] = ...
zgesvd: _function_gesvd_cz[np.complex128, np.float64] = ...

# (m, n, [compute_uv, full_matrices]) -> (work, info)
@type_check_only
class _function_gesvd_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int, *, compute_uv: int = 1, full_matrices: int = 1) -> tuple[WorkT, _info]: ...

sgesvd_lwork: _function_gesvd_lwork[float] = ...
dgesvd_lwork: _function_gesvd_lwork[float] = ...
cgesvd_lwork: _function_gesvd_lwork[complex] = ...
zgesvd_lwork: _function_gesvd_lwork[complex] = ...

# (a, b, [fact, trans, af, ipiv, equed, r, c, ...]) -> (as, lu, ipiv, equed, rs, cs, bs, x, rcond, ferr, berr, info)
@type_check_only
class _function_gesvx[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        fact: str | bytes = "E",
        trans: str | bytes = "N",
        af: onp.Array2D[ST] | None = None,
        ipiv: onp.Array1D[np.int32] | None = None,
        equed: str | bytes = "B",
        r: onp.Array1D[RT] | None = None,
        c: onp.Array1D[RT] | None = None,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[
        onp.Array2D[ST],
        onp.Array2D[ST],
        onp.Array1D[np.int32],
        bytes,
        onp.Array1D[RT],
        onp.Array1D[RT],
        onp.Array2D[ST],
        onp.Array2D[ST],
        _rcond,
        onp.Array1D[RT],
        onp.Array1D[RT],
        _info,
    ]: ...

sgesvx: _function_gesvx[np.float32, np.float32] = ...
dgesvx: _function_gesvx[np.float64, np.float64] = ...
cgesvx: _function_gesvx[np.complex64, np.float32] = ...
zgesvx: _function_gesvx[np.complex128, np.float64] = ...

# (a, [overwrite_a]) -> (lu, ipiv, jpiv, info)
@type_check_only
class _function_getc2[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], onp.Array1D[np.int32], _info]: ...

sgetc2: _function_getc2[np.float32] = ...
dgetc2: _function_getc2[np.float64] = ...
cgetc2: _function_getc2[np.complex64] = ...
zgetc2: _function_getc2[np.complex128] = ...

# (a, [overwrite_a]) -> (lu, piv, info)
@type_check_only
class _function_getrf[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], _info]: ...

sgetrf: _function_getrf[np.float32] = ...
dgetrf: _function_getrf[np.float64] = ...
cgetrf: _function_getrf[np.complex64] = ...
zgetrf: _function_getrf[np.complex128] = ...

# (lu, piv, [lwork, overwrite_lu]) -> (inv_a, info)
@type_check_only
class _function_getri[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        lu: onp.Array2D[ST],
        piv: onp.Array1D[np.int32],
        *,
        lwork: int = ...,  # = max(3 * n, 1)
        overwrite_lu: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

sgetri: _function_getri[np.float32] = ...
dgetri: _function_getri[np.float64] = ...
cgetri: _function_getri[np.complex64] = ...
zgetri: _function_getri[np.complex128] = ...

# (n) -> (work, info)
@type_check_only
class _function_getri_lwork[WorkT](Protocol):
    def __call__(self, /, n: int) -> tuple[WorkT, _info]: ...

sgetri_lwork: _function_getri_lwork[float] = ...
dgetri_lwork: _function_getri_lwork[float] = ...
cgetri_lwork: _function_getri_lwork[complex] = ...
zgetri_lwork: _function_getri_lwork[complex] = ...

# (lu, piv, b, [trans, overwrite_b]) -> (x, info)
@type_check_only
class _function_getrs[ST: np.generic](Protocol):
    def __call__(
        self, /, lu: onp.Array2D[ST], piv: onp.Array1D[np.int32], b: onp.Array2D[ST], *, trans: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

sgetrs: _function_getrs[np.float32] = ...
dgetrs: _function_getrs[np.float64] = ...
cgetrs: _function_getrs[np.complex64] = ...
zgetrs: _function_getrs[np.complex128] = ...

# (sselect, a, b, [jobvsl, jobvsr, sort_t, ldvsl, ldvsr, lwork, ...]) -> (a, b, sdim, alphar, alphai, beta, vsl, vsr, work, info)
@type_check_only
class _function_gges_s(Protocol):
    def __call__(
        self,
        /,
        sselect: Callable[..., int],
        a: onp.Array2D[np.float32],
        b: onp.Array2D[np.float32],
        *,
        jobvsl: int = 1,
        jobvsr: int = 1,
        sort_t: int = 0,
        ldvsl: int = ...,  # = (n if (jobvsl == 1) else 1)
        ldvsr: int = ...,  # = (n if (jobvsr == 1) else 1)
        lwork: int = ...,  # = max(8 * n + 16, 1)
        sselect_extra_args: tuple[object, ...] = (),
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[
        onp.Array2D[np.float32],
        onp.Array2D[np.float32],
        int,
        onp.Array1D[np.float32],
        onp.Array1D[np.float32],
        onp.Array1D[np.float32],
        onp.Array2D[np.float32],
        onp.Array2D[np.float32],
        onp.Array1D[np.float32],
        _info,
    ]: ...

sgges: _function_gges_s = ...

@type_check_only
class _function_gges_d(Protocol):
    def __call__(
        self,
        /,
        dselect: Callable[..., int],
        a: onp.Array2D[np.float64],
        b: onp.Array2D[np.float64],
        *,
        jobvsl: int = 1,
        jobvsr: int = 1,
        sort_t: int = 0,
        ldvsl: int = ...,  # = (n if (jobvsl == 1) else 1)
        ldvsr: int = ...,  # = (n if (jobvsr == 1) else 1)
        lwork: int = ...,  # = max(8 * n + 16, 1)
        dselect_extra_args: tuple[object, ...] = (),
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[
        onp.Array2D[np.float64],
        onp.Array2D[np.float64],
        int,
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array1D[np.float64],
        onp.Array2D[np.float64],
        onp.Array2D[np.float64],
        onp.Array1D[np.float64],
        _info,
    ]: ...

dgges: _function_gges_d = ...

# (cselect, a, b, [jobvsl, jobvsr, sort_t, ldvsl, ldvsr, lwork, ...]) -> (a, b, sdim, alpha, beta, vsl, vsr, work, info)
@type_check_only
class _function_gges_c(Protocol):
    def __call__(
        self,
        /,
        cselect: Callable[..., int],
        a: onp.Array2D[np.complex64],
        b: onp.Array2D[np.complex64],
        *,
        jobvsl: int = 1,
        jobvsr: int = 1,
        sort_t: int = 0,
        ldvsl: int = ...,  # = (n if (jobvsl == 1) else 1)
        ldvsr: int = ...,  # = (n if (jobvsr == 1) else 1)
        lwork: int = ...,  # = max(2 * n, 1)
        cselect_extra_args: tuple[object, ...] = (),
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[
        onp.Array2D[np.complex64],
        onp.Array2D[np.complex64],
        int,
        onp.Array1D[np.complex64],
        onp.Array1D[np.complex64],
        onp.Array2D[np.complex64],
        onp.Array2D[np.complex64],
        onp.Array1D[np.complex64],
        _info,
    ]: ...

cgges: _function_gges_c = ...

@type_check_only
class _function_gges_z(Protocol):
    def __call__(
        self,
        /,
        zselect: Callable[..., int],
        a: onp.Array2D[np.complex128],
        b: onp.Array2D[np.complex128],
        *,
        jobvsl: int = 1,
        jobvsr: int = 1,
        sort_t: int = 0,
        ldvsl: int = ...,  # = (n if (jobvsl == 1) else 1)
        ldvsr: int = ...,  # = (n if (jobvsr == 1) else 1)
        lwork: int = ...,  # = max(2 * n, 1)
        zselect_extra_args: tuple[object, ...] = (),
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[
        onp.Array2D[np.complex128],
        onp.Array2D[np.complex128],
        int,
        onp.Array1D[np.complex128],
        onp.Array1D[np.complex128],
        onp.Array2D[np.complex128],
        onp.Array2D[np.complex128],
        onp.Array1D[np.complex128],
        _info,
    ]: ...

zgges: _function_gges_z = ...

# (a, b, [compute_vl, compute_vr, lwork, overwrite_a, overwrite_b]) -> (alphar, alphai, beta, vl, vr, work, info)
@type_check_only
class _function_ggev_sd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        compute_vl: int = 1,
        compute_vr: int = 1,
        lwork: int = ...,  # = max(8 * n, 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST], onp.Array1D[ST], onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[ST], _info]: ...

sggev: _function_ggev_sd[np.float32] = ...
dggev: _function_ggev_sd[np.float64] = ...

# (a, b, [compute_vl, compute_vr, lwork, overwrite_a, overwrite_b]) -> (alpha, beta, vl, vr, work, info)
@type_check_only
class _function_ggev_cz[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        compute_vl: int = 1,
        compute_vr: int = 1,
        lwork: int = ...,  # = max(2 * n, 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST], onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[ST], _info]: ...

cggev: _function_ggev_cz[np.complex64] = ...
zggev: _function_ggev_cz[np.complex128] = ...

# (a, b, c, d, [lwork, overwrite_a, overwrite_b, overwrite_c, overwrite_d]) -> (t, r, res, x, info)
@type_check_only
class _function_gglse[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        c: onp.Array1D[ST],
        d: onp.Array1D[ST],
        *,
        lwork: int = ...,  # = max(m + n + p, 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
        overwrite_c: int = 0,
        overwrite_d: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[ST], onp.Array1D[ST], _info]: ...

sgglse: _function_gglse[np.float32] = ...
dgglse: _function_gglse[np.float64] = ...
cgglse: _function_gglse[np.complex64] = ...
zgglse: _function_gglse[np.complex128] = ...

# (m, n, p) -> (work, info)
@type_check_only
class _function_gglse_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int, p: int) -> tuple[WorkT, _info]: ...

sgglse_lwork: _function_gglse_lwork[float] = ...
dgglse_lwork: _function_gglse_lwork[float] = ...
cgglse_lwork: _function_gglse_lwork[complex] = ...
zgglse_lwork: _function_gglse_lwork[complex] = ...

# (dl, d, du, du2, ipiv, anorm, [norm]) -> (rcond, info)
@type_check_only
class _function_gtcon[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        dl: onp.Array1D[ST],
        d: onp.Array1D[ST],
        du: onp.Array1D[ST],
        du2: onp.Array1D[ST],
        ipiv: onp.Array1D[np.int32],
        anorm: float,
        *,
        norm: str | bytes = "1",
    ) -> tuple[_rcond, _info]: ...

sgtcon: _function_gtcon[np.float32] = ...
dgtcon: _function_gtcon[np.float64] = ...
cgtcon: _function_gtcon[np.complex64] = ...
zgtcon: _function_gtcon[np.complex128] = ...

# (dl, d, du, b, [overwrite_dl, overwrite_d, overwrite_du, overwrite_b]) -> (du2, d, du, x, info)
@type_check_only
class _function_gtsv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        dl: onp.Array1D[ST],
        d: onp.Array1D[ST],
        du: onp.Array1D[ST],
        b: onp.Array2D[ST],
        *,
        overwrite_dl: int = 0,
        overwrite_d: int = 0,
        overwrite_du: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST], onp.Array1D[ST], onp.Array2D[ST], _info]: ...

sgtsv: _function_gtsv[np.float32] = ...
dgtsv: _function_gtsv[np.float64] = ...
cgtsv: _function_gtsv[np.complex64] = ...
zgtsv: _function_gtsv[np.complex128] = ...

# (dl, d, du, b, [fact, trans, dlf, df, duf, du2, ipiv]) -> (dlf, df, duf, du2, ipiv, x, rcond, ferr, berr, info)
@type_check_only
class _function_gtsvx[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        dl: onp.Array1D[ST],
        d: onp.Array1D[ST],
        du: onp.Array1D[ST],
        b: onp.Array2D[ST],
        *,
        fact: str | bytes = "N",
        trans: str | bytes = "N",
        dlf: onp.Array1D[ST] | None = None,
        df: onp.Array1D[ST] | None = None,
        duf: onp.Array1D[ST] | None = None,
        du2: onp.Array1D[ST] | None = None,
        ipiv: onp.Array1D[np.int32] | None = None,
    ) -> tuple[
        onp.Array1D[ST],
        onp.Array1D[ST],
        onp.Array1D[ST],
        onp.Array1D[ST],
        onp.Array1D[np.int32],
        onp.Array2D[ST],
        _rcond,
        onp.Array1D[RT],
        onp.Array1D[RT],
        _info,
    ]: ...

sgtsvx: _function_gtsvx[np.float32, np.float32] = ...
dgtsvx: _function_gtsvx[np.float64, np.float64] = ...
cgtsvx: _function_gtsvx[np.complex64, np.float32] = ...
zgtsvx: _function_gtsvx[np.complex128, np.float64] = ...

# (dl, d, du, [overwrite_dl, overwrite_d, overwrite_du]) -> (dl, d, du, du2, ipiv, info)
@type_check_only
class _function_gttrf[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        dl: onp.Array1D[ST],
        d: onp.Array1D[ST],
        du: onp.Array1D[ST],
        *,
        overwrite_dl: int = 0,
        overwrite_d: int = 0,
        overwrite_du: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST], onp.Array1D[ST], onp.Array1D[ST], onp.Array1D[np.int32], _info]: ...

sgttrf: _function_gttrf[np.float32] = ...
dgttrf: _function_gttrf[np.float64] = ...
cgttrf: _function_gttrf[np.complex64] = ...
zgttrf: _function_gttrf[np.complex128] = ...

# (dl, d, du, du2, ipiv, b, [trans, overwrite_b]) -> (x, info)
@type_check_only
class _function_gttrs[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        dl: onp.Array1D[ST],
        d: onp.Array1D[ST],
        du: onp.Array1D[ST],
        du2: onp.Array1D[ST],
        ipiv: onp.Array1D[np.int32],
        b: onp.Array2D[ST],
        *,
        trans: str | bytes = "N",
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

sgttrs: _function_gttrs[np.float32] = ...
dgttrs: _function_gttrs[np.float64] = ...
cgttrs: _function_gttrs[np.complex64] = ...
zgttrs: _function_gttrs[np.complex128] = ...

# (ab, [compute_v, lower, ldab, lrwork, liwork, overwrite_ab]) -> (w, z, info)
@type_check_only
class _function_hbevd[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[CT],
        *,
        compute_v: int = 1,
        lower: int = 0,
        ldab: int = ...,  # = ab.shape[0]
        lrwork: int = ...,  # = (1 + 5 * n + 2 * n * n if compute_v else n)
        liwork: int = ...,  # = (3 + 5 * n if compute_v else 1)
        overwrite_ab: int = 1,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], _info]: ...

chbevd: _function_hbevd[np.complex64, np.float32] = ...
zhbevd: _function_hbevd[np.complex128, np.float64] = ...

# (ab, vl, vu, il, iu, [ldab, compute_v, range, lower, abstol, mmax, overwrite_ab]) -> (w, z, m, ifail, info)
@type_check_only
class _function_hbevx[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[CT],
        vl: float,
        vu: float,
        il: int,
        iu: int,
        *,
        ldab: int = ...,  # = ab.shape[0]
        compute_v: int = 1,
        range: int = 0,
        lower: int = 0,
        abstol: float = 0.0,
        mmax: int = ...,  # = (((iu - il + 1) if range == 2 else n) if compute_v else 1)
        overwrite_ab: int = 1,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], int, onp.Array1D[np.int32], _info]: ...

chbevx: _function_hbevx[np.complex64, np.float32] = ...
zhbevx: _function_hbevx[np.complex128, np.float64] = ...

# (a, ipiv, anorm, [lower]) -> (rcond, info)
@type_check_only
class _function_hecon[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], ipiv: onp.Array1D[np.int32], anorm: float, *, lower: int = 0
    ) -> tuple[_rcond, _info]: ...

checon: _function_hecon[np.complex64] = ...
zhecon: _function_hecon[np.complex128] = ...

# (a, [lower]) -> (s, scond, amax, info)
@type_check_only
class _function_heequb[CT: np.generic, RT: np.generic](Protocol):
    def __call__(self, /, a: onp.Array2D[CT], *, lower: int = 0) -> tuple[onp.Array1D[RT], float, float, _info]: ...

cheequb: _function_heequb[np.complex64, np.float32] = ...
zheequb: _function_heequb[np.complex128, np.float64] = ...

# (a, [compute_v, lower, lwork, overwrite_a]) -> (w, v, info)
@type_check_only
class _function_heev[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        *,
        compute_v: int = 1,
        lower: int = 0,
        lwork: int = ...,  # = max(2 * n - 1, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], _info]: ...

cheev: _function_heev[np.complex64, np.float32] = ...
zheev: _function_heev[np.complex128, np.float64] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_heev_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[complex, _info]: ...

cheev_lwork: _function_heev_lwork = ...
zheev_lwork: _function_heev_lwork = ...

# (a, [compute_v, lower, lwork, liwork, lrwork, overwrite_a]) -> (w, v, info)
@type_check_only
class _function_heevd[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        *,
        compute_v: int = 1,
        lower: int = 0,
        lwork: int = ...,  # = max((2 * n + n * n if compute_v else n + 1), 1)
        liwork: int = ...,  # = (3 + 5 * n if compute_v else 1)
        lrwork: int = ...,  # = (1 + 5 * n + 2 * n * n if compute_v else n)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], _info]: ...

cheevd: _function_heevd[np.complex64, np.float32] = ...
zheevd: _function_heevd[np.complex128, np.float64] = ...

# (n, [compute_v, lower]) -> (work, iwork, rwork, info)
@type_check_only
class _function_heevd_lwork(Protocol):
    def __call__(self, /, n: int, *, compute_v: int = 1, lower: int = 0) -> tuple[complex, int, float, _info]: ...

cheevd_lwork: _function_heevd_lwork = ...
zheevd_lwork: _function_heevd_lwork = ...

# (a, [compute_v, range, lower, vl, vu, il, iu, abstol, lwork, lrwork, liwork, overwrite_a]) -> (w, z, m, isuppz, info)
@type_check_only
class _function_heevr[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        *,
        compute_v: int = 1,
        range: str | bytes = "A",
        lower: int = 0,
        vl: float = 0.0,
        vu: float = 1.0,
        il: int = 1,
        iu: int = ...,  # = n
        abstol: float = 0.0,
        lwork: int = ...,  # = max(2 * n, 1)
        lrwork: int = ...,  # = max(24 * n, 1)
        liwork: int = ...,  # = max(1, 10 * n)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], int, onp.Array1D[np.int32], _info]: ...

cheevr: _function_heevr[np.complex64, np.float32] = ...
zheevr: _function_heevr[np.complex128, np.float64] = ...

# (n, [lower]) -> (work, rwork, iwork, info)
@type_check_only
class _function_heevr_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[complex, float, int, _info]: ...

cheevr_lwork: _function_heevr_lwork = ...
zheevr_lwork: _function_heevr_lwork = ...

# (a, [compute_v, range, lower, vl, vu, il, iu, abstol, lwork, overwrite_a]) -> (w, z, m, ifail, info)
@type_check_only
class _function_heevx[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        *,
        compute_v: int = 1,
        range: str | bytes = "A",
        lower: int = 0,
        vl: float = 0.0,
        vu: float = 1.0,
        il: int = 1,
        iu: int = ...,  # = n
        abstol: float = 0.0,
        lwork: int = ...,  # = max(2 * n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], int, onp.Array1D[np.int32], _info]: ...

cheevx: _function_heevx[np.complex64, np.float32] = ...
zheevx: _function_heevx[np.complex128, np.float64] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_heevx_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[complex, _info]: ...

cheevx_lwork: _function_heevx_lwork = ...
zheevx_lwork: _function_heevx_lwork = ...

# (a, b, [itype, lower, overwrite_a]) -> (c, info)
@type_check_only
class _function_hegst[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], b: onp.Array2D[ST], *, itype: int = 1, lower: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

chegst: _function_hegst[np.complex64] = ...
zhegst: _function_hegst[np.complex128] = ...

# (a, b, [itype, jobz, uplo, lwork, overwrite_a, overwrite_b]) -> (w, v, info)
@type_check_only
class _function_hegv[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        b: onp.Array2D[CT],
        *,
        itype: int = 1,
        jobz: str | bytes = "V",
        uplo: str | bytes = "L",
        lwork: int = ...,  # = max(2 * n - 1, 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], _info]: ...

chegv: _function_hegv[np.complex64, np.float32] = ...
zhegv: _function_hegv[np.complex128, np.float64] = ...

# (n, [uplo]) -> (work, info)
@type_check_only
class _function_hegv_lwork(Protocol):
    def __call__(self, /, n: int, *, uplo: str | bytes = "L") -> tuple[complex, _info]: ...

chegv_lwork: _function_hegv_lwork = ...
zhegv_lwork: _function_hegv_lwork = ...

# (a, b, [itype, jobz, uplo, lwork, lrwork, liwork, overwrite_a, overwrite_b]) -> (w, v, info)
@type_check_only
class _function_hegvd[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        b: onp.Array2D[CT],
        *,
        itype: int = 1,
        jobz: str | bytes = "V",
        uplo: str | bytes = "L",
        lwork: int = ...,  # = (n + 1 if jobz == "N" else n * (n + 2))
        lrwork: int = ...,  # = max((n if jobz == "N" else 2 * n * n + 5 * n + 1), 1)
        liwork: int = ...,  # = (1 if jobz == "N" else 5 * n + 3)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], _info]: ...

chegvd: _function_hegvd[np.complex64, np.float32] = ...
zhegvd: _function_hegvd[np.complex128, np.float64] = ...

# (a, b, [itype, jobz, range, uplo, vl, vu, il, iu, abstol, lwork, overwrite_a, overwrite_b]) -> (w, z, m, ifail, info)
@type_check_only
class _function_hegvx[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        b: onp.Array2D[CT],
        *,
        itype: int = 1,
        jobz: str | bytes = "V",
        range: str | bytes = "A",
        uplo: str | bytes = "L",
        vl: float = 0.0,
        vu: float = 1.0,
        il: int = 1,
        iu: int = ...,  # = n
        abstol: float = 0.0,
        lwork: int = ...,  # = max(2 * n, 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array2D[CT], int, onp.Array1D[np.int32], _info]: ...

chegvx: _function_hegvx[np.complex64, np.float32] = ...
zhegvx: _function_hegvx[np.complex128, np.float64] = ...

# (n, [uplo]) -> (work, info)
@type_check_only
class _function_hegvx_lwork(Protocol):
    def __call__(self, /, n: int, *, uplo: str | bytes = "L") -> tuple[complex, _info]: ...

chegvx_lwork: _function_hegvx_lwork = ...
zhegvx_lwork: _function_hegvx_lwork = ...

# (a, b, [lwork, lower, overwrite_a, overwrite_b]) -> (uduh, ipiv, x, info)
@type_check_only
class _function_hesv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        lwork: int = ...,  # = max(n, 1)
        lower: int = 0,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], onp.Array2D[ST], _info]: ...

chesv: _function_hesv[np.complex64] = ...
zhesv: _function_hesv[np.complex128] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_hesv_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[complex, _info]: ...

chesv_lwork: _function_hesv_lwork = ...
zhesv_lwork: _function_hesv_lwork = ...

# (a, b, [af, ipiv, lwork, factored, lower, overwrite_a, overwrite_b]) -> (uduh, ipiv, x, rcond, ferr, berr, info)
@type_check_only
class _function_hesvx[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        b: onp.Array2D[CT],
        *,
        af: onp.Array2D[CT] | None = None,
        ipiv: onp.Array1D[np.int32] | None = None,
        lwork: int = ...,  # = max(2 * n, 1)
        factored: int = 0,
        lower: int = 0,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[CT], onp.Array1D[np.int32], onp.Array2D[CT], _rcond, onp.Array1D[RT], onp.Array1D[RT], _info]: ...

chesvx: _function_hesvx[np.complex64, np.float32] = ...
zhesvx: _function_hesvx[np.complex128, np.float64] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_hesvx_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[complex, _info]: ...

chesvx_lwork: _function_hesvx_lwork = ...
zhesvx_lwork: _function_hesvx_lwork = ...

# (a, [lower, lwork, overwrite_a]) -> (c, d, e, tau, info)
@type_check_only
class _function_hetrd[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[CT],
        *,
        lower: int = 0,
        lwork: int = ...,  # = max(n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[CT], onp.Array1D[RT], onp.Array1D[RT], onp.Array1D[CT], _info]: ...

chetrd: _function_hetrd[np.complex64, np.float32] = ...
zhetrd: _function_hetrd[np.complex128, np.float64] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_hetrd_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[complex, _info]: ...

chetrd_lwork: _function_hetrd_lwork = ...
zhetrd_lwork: _function_hetrd_lwork = ...

# (a, [lower, lwork, overwrite_a]) -> (ldu, ipiv, info)
@type_check_only
class _function_hetrf[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lower: int = 0,
        lwork: int = ...,  # = max(n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], _info]: ...

chetrf: _function_hetrf[np.complex64] = ...
zhetrf: _function_hetrf[np.complex128] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_hetrf_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[complex, _info]: ...

chetrf_lwork: _function_hetrf_lwork = ...
zhetrf_lwork: _function_hetrf_lwork = ...

# (a, ipiv, [lower, overwrite_a]) -> (inv_a, info)
@type_check_only
class _function_hetri[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], ipiv: onp.Array1D[np.int32], *, lower: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

chetri: _function_hetri[np.complex64] = ...
zhetri: _function_hetri[np.complex128] = ...

# (a, ipiv, b, [lower, overwrite_b]) -> (x, info)
@type_check_only
class _function_hetrs[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], ipiv: onp.Array1D[np.int32], b: onp.Array2D[ST], *, lower: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

chetrs: _function_hetrs[np.complex64] = ...
zhetrs: _function_hetrs[np.complex128] = ...

# (n, k, alpha, a, beta, c, [transr, uplo, trans, overwrite_c]) -> cout
@type_check_only
class _function_hfrk[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        n: int,
        k: int,
        alpha: float,
        a: onp.Array2D[ST],
        beta: float,
        c: onp.Array1D[ST],
        *,
        transr: str | bytes = "N",
        uplo: str | bytes = "U",
        trans: str | bytes = "N",
        overwrite_c: int = 0,
    ) -> onp.Array1D[ST]: ...

chfrk: _function_hfrk[np.complex64] = ...
zhfrk: _function_hfrk[np.complex128] = ...

# (cmach) -> x
@type_check_only
class _function_lamch(Protocol):
    def __call__(self, /, cmach: str | bytes) -> float: ...

slamch: _function_lamch = ...
dlamch: _function_lamch = ...

# (norm, kl, ku, ab, [ldab]) -> n2
@type_check_only
class _function_langb[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        norm: str | bytes,
        kl: int,
        ku: int,
        ab: onp.Array2D[ST],
        *,
        ldab: int = ...,  # = kl + ku + 1
    ) -> float: ...

slangb: _function_langb[np.float32] = ...
dlangb: _function_langb[np.float64] = ...
clangb: _function_langb[np.complex64] = ...
zlangb: _function_langb[np.complex128] = ...

# (norm, a) -> n2
@type_check_only
class _function_lange[ST: np.generic](Protocol):
    def __call__(self, /, norm: str | bytes, a: onp.Array2D[ST]) -> float: ...

slange: _function_lange[np.float32] = ...
dlange: _function_lange[np.float64] = ...
clange: _function_lange[np.complex64] = ...
zlange: _function_lange[np.complex128] = ...

# (norm, a, [uplo, diag]) -> n2
@type_check_only
class _function_lantr[ST: np.generic](Protocol):
    def __call__(
        self, /, norm: str | bytes, a: onp.Array2D[ST], *, uplo: str | bytes = "U", diag: str | bytes = "N"
    ) -> float: ...

slantr: _function_lantr[np.float32] = ...
dlantr: _function_lantr[np.float64] = ...
clantr: _function_lantr[np.complex64] = ...
zlantr: _function_lantr[np.complex128] = ...

# (v, tau, c, work, [side, incv, overwrite_c]) -> c
@type_check_only
class _function_larf[ST: np.generic, WorkT](Protocol):
    def __call__(
        self,
        /,
        v: onp.Array1D[ST],
        tau: WorkT,
        c: onp.Array2D[ST],
        work: onp.Array1D[ST],
        *,
        side: str | bytes = "L",
        incv: int = 1,
        overwrite_c: int = 0,
    ) -> onp.Array2D[ST]: ...

slarf: _function_larf[np.float32, float] = ...
dlarf: _function_larf[np.float64, float] = ...
clarf: _function_larf[np.complex64, complex] = ...
zlarf: _function_larf[np.complex128, complex] = ...

# (n, alpha, x, [incx, overwrite_x]) -> (alpha, x, tau)
@type_check_only
class _function_larfg[ST: np.generic, WorkT](Protocol):
    def __call__(
        self, /, n: int, alpha: WorkT, x: onp.Array1D[ST], *, incx: int = 1, overwrite_x: int = 0
    ) -> tuple[WorkT, onp.Array1D[ST], WorkT]: ...

slarfg: _function_larfg[np.float32, float] = ...
dlarfg: _function_larfg[np.float64, float] = ...
clarfg: _function_larfg[np.complex64, complex] = ...
zlarfg: _function_larfg[np.complex128, complex] = ...

# (f, g) -> (cs, sn, r)
@type_check_only
class _function_lartg[WorkT](Protocol):
    def __call__(self, /, f: WorkT, g: WorkT) -> tuple[float, WorkT, WorkT]: ...

slartg: _function_lartg[float] = ...
dlartg: _function_lartg[float] = ...
clartg: _function_lartg[complex] = ...
zlartg: _function_lartg[complex] = ...

# (i, d, z, [rho]) -> (delta, sigma, work, info)
@type_check_only
class _function_lasd4[ST: np.generic](Protocol):
    def __call__(
        self, /, i: int, d: onp.Array1D[ST], z: onp.Array1D[ST], *, rho: float = 1.0
    ) -> tuple[onp.Array1D[ST], float, onp.Array1D[ST], _info]: ...

slasd4: _function_lasd4[np.float32] = ...
dlasd4: _function_lasd4[np.float64] = ...

# (a, piv, [k1, k2, off, inc, overwrite_a]) -> a
@type_check_only
class _function_laswp[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        piv: onp.Array1D[np.int32],
        *,
        k1: int = 0,
        k2: int = ...,  # = npiv - 1
        off: int = 0,
        inc: int = 1,
        overwrite_a: int = 0,
    ) -> onp.Array2D[ST]: ...

slaswp: _function_laswp[np.float32] = ...
dlaswp: _function_laswp[np.float64] = ...
claswp: _function_laswp[np.complex64] = ...
zlaswp: _function_laswp[np.complex128] = ...

# (c, [lower, overwrite_c]) -> (a, info)
@type_check_only
class _function_lauum[ST: np.generic](Protocol):
    def __call__(self, /, c: onp.Array2D[ST], *, lower: int = 0, overwrite_c: int = 0) -> tuple[onp.Array2D[ST], _info]: ...

slauum: _function_lauum[np.float32] = ...
dlauum: _function_lauum[np.float64] = ...
clauum: _function_lauum[np.complex64] = ...
zlauum: _function_lauum[np.complex128] = ...

# (x11, x12, x21, x22, [...]) -> (cs11, cs12, cs21, cs22, theta, u1, u2, v1t, v2t, info)
@type_check_only
class _function_orcsd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        x11: onp.Array2D[ST],
        x12: onp.Array2D[ST],
        x21: onp.Array2D[ST],
        x22: onp.Array2D[ST],
        *,
        compute_u1: int = 1,
        compute_u2: int = 1,
        compute_v1t: int = 1,
        compute_v2t: int = 1,
        trans: int = 0,
        signs: int = 0,
        lwork: int = ...,  # = 2 + 2 * m + 5 * max(1, q - 1) + 4 * max(1, q) + 8 * q
        overwrite_x11: int = 0,
        overwrite_x12: int = 0,
        overwrite_x21: int = 0,
        overwrite_x22: int = 0,
    ) -> tuple[
        onp.Array2D[ST],
        onp.Array2D[ST],
        onp.Array2D[ST],
        onp.Array2D[ST],
        onp.Array1D[ST],
        onp.Array2D[ST],
        onp.Array2D[ST],
        onp.Array2D[ST],
        onp.Array2D[ST],
        _info,
    ]: ...

sorcsd: _function_orcsd[np.float32] = ...
dorcsd: _function_orcsd[np.float64] = ...

# (m, p, q) -> (work, info)
@type_check_only
class _function_orcsd_lwork(Protocol):
    def __call__(self, /, m: int, p: int, q: int) -> tuple[float, _info]: ...

sorcsd_lwork: _function_orcsd_lwork = ...
dorcsd_lwork: _function_orcsd_lwork = ...

# (a, tau, [lo, hi, lwork, overwrite_a]) -> (ht, info)
@type_check_only
class _function_orghr[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        *,
        lo: int = 0,
        hi: int = ...,  # = n - 1
        lwork: int = ...,  # = max(hi - lo, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

sorghr: _function_orghr[np.float32] = ...
dorghr: _function_orghr[np.float64] = ...

# (n, [lo, hi]) -> (work, info)
@type_check_only
class _function_orghr_lwork(Protocol):
    def __call__(
        self,
        /,
        n: int,
        *,
        lo: int = 0,
        hi: int = ...,  # = n - 1
    ) -> tuple[float, _info]: ...

sorghr_lwork: _function_orghr_lwork = ...
dorghr_lwork: _function_orghr_lwork = ...

# (a, tau, [lwork, overwrite_a]) -> (q, work, info)
@type_check_only
class _function_orgqr[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        *,
        lwork: int = ...,  # = max(3 * n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

sorgqr: _function_orgqr[np.float32] = ...
dorgqr: _function_orgqr[np.float64] = ...

@type_check_only
class _function_orgrq[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        *,
        lwork: int = ...,  # = max(3 * m, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

sorgrq: _function_orgrq[np.float32] = ...
dorgrq: _function_orgrq[np.float64] = ...

# (side, trans, a, tau, c, lwork, [overwrite_c]) -> (cq, work, info)
@type_check_only
class _function_ormqr[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        side: str | bytes,
        trans: str | bytes,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        c: onp.Array2D[ST],
        lwork: int,
        *,
        overwrite_c: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

sormqr: _function_ormqr[np.float32] = ...
dormqr: _function_ormqr[np.float64] = ...

# (a, tau, c, [side, trans, lwork, overwrite_c]) -> (cq, info)
@type_check_only
class _function_ormrz[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        c: onp.Array2D[ST],
        *,
        side: str | bytes = "L",
        trans: str | bytes = "N",
        lwork: int = ...,  # = max((n if side == "L" else m), 1)
        overwrite_c: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

sormrz: _function_ormrz[np.float32] = ...
dormrz: _function_ormrz[np.float64] = ...

# (m, n, [side, trans]) -> (work, info)
@type_check_only
class _function_ormrz_lwork(Protocol):
    def __call__(self, /, m: int, n: int, *, side: str | bytes = "L", trans: str | bytes = "N") -> tuple[float, _info]: ...

sormrz_lwork: _function_ormrz_lwork = ...
dormrz_lwork: _function_ormrz_lwork = ...

# (ab, b, [lower, ldab, overwrite_ab, overwrite_b]) -> (c, x, info)
@type_check_only
class _function_pbsv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        lower: int = 0,
        ldab: int = ...,  # = ab.shape[0]
        overwrite_ab: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], _info]: ...

spbsv: _function_pbsv[np.float32] = ...
dpbsv: _function_pbsv[np.float64] = ...
cpbsv: _function_pbsv[np.complex64] = ...
zpbsv: _function_pbsv[np.complex128] = ...

# (ab, [lower, ldab, overwrite_ab]) -> (c, info)
@type_check_only
class _function_pbtrf[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        *,
        lower: int = 0,
        ldab: int = ...,  # = ab.shape[0]
        overwrite_ab: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

spbtrf: _function_pbtrf[np.float32] = ...
dpbtrf: _function_pbtrf[np.float64] = ...
cpbtrf: _function_pbtrf[np.complex64] = ...
zpbtrf: _function_pbtrf[np.complex128] = ...

# (ab, b, [lower, ldab, overwrite_b]) -> (x, info)
@type_check_only
class _function_pbtrs[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        lower: int = 0,
        ldab: int = ...,  # = ab.shape[0]
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

spbtrs: _function_pbtrs[np.float32] = ...
dpbtrs: _function_pbtrs[np.float64] = ...
cpbtrs: _function_pbtrs[np.complex64] = ...
zpbtrs: _function_pbtrs[np.complex128] = ...

# (n, a, [transr, uplo, overwrite_a]) -> (achol, info)
@type_check_only
class _function_pftrf[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, a: onp.Array1D[ST], *, transr: str | bytes = "N", uplo: str | bytes = "U", overwrite_a: int = 0
    ) -> tuple[onp.Array1D[ST], _info]: ...

spftrf: _function_pftrf[np.float32] = ...
dpftrf: _function_pftrf[np.float64] = ...
cpftrf: _function_pftrf[np.complex64] = ...
zpftrf: _function_pftrf[np.complex128] = ...

# (n, a, [transr, uplo, overwrite_a]) -> (ainv, info)
@type_check_only
class _function_pftri[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, a: onp.Array1D[ST], *, transr: str | bytes = "N", uplo: str | bytes = "U", overwrite_a: int = 0
    ) -> tuple[onp.Array1D[ST], _info]: ...

spftri: _function_pftri[np.float32] = ...
dpftri: _function_pftri[np.float64] = ...
cpftri: _function_pftri[np.complex64] = ...
zpftri: _function_pftri[np.complex128] = ...

# (n, a, b, [transr, uplo, overwrite_b]) -> (x, info)
@type_check_only
class _function_pftrs[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        n: int,
        a: onp.Array1D[ST],
        b: onp.Array2D[ST],
        *,
        transr: str | bytes = "N",
        uplo: str | bytes = "U",
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

spftrs: _function_pftrs[np.float32] = ...
dpftrs: _function_pftrs[np.float64] = ...
cpftrs: _function_pftrs[np.complex64] = ...
zpftrs: _function_pftrs[np.complex128] = ...

# (a, anorm, [uplo]) -> (rcond, info)
@type_check_only
class _function_pocon[ST: np.generic](Protocol):
    def __call__(self, /, a: onp.Array2D[ST], anorm: float, *, uplo: str | bytes = "U") -> tuple[_rcond, _info]: ...

spocon: _function_pocon[np.float32] = ...
dpocon: _function_pocon[np.float64] = ...
cpocon: _function_pocon[np.complex64] = ...
zpocon: _function_pocon[np.complex128] = ...

# (a, b, [lower, overwrite_a, overwrite_b]) -> (c, x, info)
@type_check_only
class _function_posv[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], b: onp.Array2D[ST], *, lower: int = 0, overwrite_a: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], _info]: ...

sposv: _function_posv[np.float32] = ...
dposv: _function_posv[np.float64] = ...
cposv: _function_posv[np.complex64] = ...
zposv: _function_posv[np.complex128] = ...

# (a, b, [fact, af, equed, s, lower, overwrite_a, overwrite_b]) -> (a_s, lu, equed, s, b_s, x, rcond, ferr, berr, info)
@type_check_only
class _function_posvx[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        fact: str | bytes = "E",
        af: onp.Array2D[ST] | None = None,
        equed: str | bytes = "Y",
        s: onp.Array1D[RT] | None = None,
        lower: int = 0,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[
        onp.Array2D[ST],
        onp.Array2D[ST],
        bytes,
        onp.Array1D[RT],
        onp.Array2D[ST],
        onp.Array2D[ST],
        _rcond,
        onp.Array1D[RT],
        onp.Array1D[RT],
        _info,
    ]: ...

sposvx: _function_posvx[np.float32, np.float32] = ...
dposvx: _function_posvx[np.float64, np.float64] = ...
cposvx: _function_posvx[np.complex64, np.float32] = ...
zposvx: _function_posvx[np.complex128, np.float64] = ...

# (a, [lower, clean, overwrite_a]) -> (c, info)
@type_check_only
class _function_potrf[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, lower: int = 0, clean: int = 1, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

spotrf: _function_potrf[np.float32] = ...
dpotrf: _function_potrf[np.float64] = ...
cpotrf: _function_potrf[np.complex64] = ...
zpotrf: _function_potrf[np.complex128] = ...

# (c, [lower, overwrite_c]) -> (inv_a, info)
@type_check_only
class _function_potri[ST: np.generic](Protocol):
    def __call__(self, /, c: onp.Array2D[ST], *, lower: int = 0, overwrite_c: int = 0) -> tuple[onp.Array2D[ST], _info]: ...

spotri: _function_potri[np.float32] = ...
dpotri: _function_potri[np.float64] = ...
cpotri: _function_potri[np.complex64] = ...
zpotri: _function_potri[np.complex128] = ...

# (c, b, [lower, overwrite_b]) -> (x, info)
@type_check_only
class _function_potrs[ST: np.generic](Protocol):
    def __call__(
        self, /, c: onp.Array2D[ST], b: onp.Array2D[ST], *, lower: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

spotrs: _function_potrs[np.float32] = ...
dpotrs: _function_potrs[np.float64] = ...
cpotrs: _function_potrs[np.complex64] = ...
zpotrs: _function_potrs[np.complex128] = ...

# (n, ap, anorm, [lower]) -> (rcond, info)
@type_check_only
class _function_ppcon[ST: np.generic](Protocol):
    def __call__(self, /, n: int, ap: onp.Array1D[ST], anorm: float, *, lower: int = 0) -> tuple[_rcond, _info]: ...

sppcon: _function_ppcon[np.float32] = ...
dppcon: _function_ppcon[np.float64] = ...
cppcon: _function_ppcon[np.complex64] = ...
zppcon: _function_ppcon[np.complex128] = ...

# (n, ap, b, [lower, overwrite_b]) -> (x, info)
@type_check_only
class _function_ppsv[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, ap: onp.Array1D[ST], b: onp.Array2D[ST], *, lower: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

sppsv: _function_ppsv[np.float32] = ...
dppsv: _function_ppsv[np.float64] = ...
cppsv: _function_ppsv[np.complex64] = ...
zppsv: _function_ppsv[np.complex128] = ...

# (n, ap, [lower, overwrite_ap]) -> (ul, info)
@type_check_only
class _function_pptrf[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, ap: onp.Array1D[ST], *, lower: int = 0, overwrite_ap: int = 0
    ) -> tuple[onp.Array1D[ST], _info]: ...

spptrf: _function_pptrf[np.float32] = ...
dpptrf: _function_pptrf[np.float64] = ...
cpptrf: _function_pptrf[np.complex64] = ...
zpptrf: _function_pptrf[np.complex128] = ...

# (n, ap, [lower, overwrite_ap]) -> (uli, info)
@type_check_only
class _function_pptri[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, ap: onp.Array1D[ST], *, lower: int = 0, overwrite_ap: int = 0
    ) -> tuple[onp.Array1D[ST], _info]: ...

spptri: _function_pptri[np.float32] = ...
dpptri: _function_pptri[np.float64] = ...
cpptri: _function_pptri[np.complex64] = ...
zpptri: _function_pptri[np.complex128] = ...

# (n, ap, b, [lower, overwrite_b]) -> (x, info)
@type_check_only
class _function_pptrs[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, ap: onp.Array1D[ST], b: onp.Array2D[ST], *, lower: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

spptrs: _function_pptrs[np.float32] = ...
dpptrs: _function_pptrs[np.float64] = ...
cpptrs: _function_pptrs[np.complex64] = ...
zpptrs: _function_pptrs[np.complex128] = ...

# (a, [tol, lower, overwrite_a]) -> (c, piv, rank_c, info)
@type_check_only
class _function_pstf2[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, tol: float = -1.0, lower: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], int, _info]: ...

spstf2: _function_pstf2[np.float32] = ...
dpstf2: _function_pstf2[np.float64] = ...
cpstf2: _function_pstf2[np.complex64] = ...
zpstf2: _function_pstf2[np.complex128] = ...

@type_check_only
class _function_pstrf[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, tol: float = -1.0, lower: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], int, _info]: ...

spstrf: _function_pstrf[np.float32] = ...
dpstrf: _function_pstrf[np.float64] = ...
cpstrf: _function_pstrf[np.complex64] = ...
zpstrf: _function_pstrf[np.complex128] = ...

# (d, e, z, [compute_z, overwrite_d, overwrite_e, overwrite_z]) -> (d, e, z, info)
@type_check_only
class _function_pteqr[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        d: onp.Array1D[RT],
        e: onp.Array1D[RT],
        z: onp.Array2D[ST],
        *,
        compute_z: int = 0,
        overwrite_d: int = 0,
        overwrite_e: int = 0,
        overwrite_z: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array1D[RT], onp.Array2D[ST], _info]: ...

spteqr: _function_pteqr[np.float32, np.float32] = ...
dpteqr: _function_pteqr[np.float64, np.float64] = ...
cpteqr: _function_pteqr[np.complex64, np.float32] = ...
zpteqr: _function_pteqr[np.complex128, np.float64] = ...

# (d, e, b, [overwrite_d, overwrite_e, overwrite_b]) -> (d, du, x, info)
@type_check_only
class _function_ptsv[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        d: onp.Array1D[RT],
        e: onp.Array1D[ST],
        b: onp.Array2D[ST],
        *,
        overwrite_d: int = 0,
        overwrite_e: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[RT], onp.Array1D[ST], onp.Array2D[ST], _info]: ...

sptsv: _function_ptsv[np.float32, np.float32] = ...
dptsv: _function_ptsv[np.float64, np.float64] = ...
cptsv: _function_ptsv[np.complex64, np.float32] = ...
zptsv: _function_ptsv[np.complex128, np.float64] = ...

# (d, e, b, [fact, df, ef]) -> (df, ef, x, rcond, ferr, berr, info)
@type_check_only
class _function_ptsvx[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        d: onp.Array1D[RT],
        e: onp.Array1D[ST],
        b: onp.Array2D[ST],
        *,
        fact: str | bytes = "N",
        df: onp.Array1D[RT] | None = None,
        ef: onp.Array1D[ST] | None = None,
    ) -> tuple[onp.Array1D[RT], onp.Array1D[ST], onp.Array2D[ST], _rcond, onp.Array1D[RT], onp.Array1D[RT], _info]: ...

sptsvx: _function_ptsvx[np.float32, np.float32] = ...
dptsvx: _function_ptsvx[np.float64, np.float64] = ...
cptsvx: _function_ptsvx[np.complex64, np.float32] = ...
zptsvx: _function_ptsvx[np.complex128, np.float64] = ...

# (d, e, [overwrite_d, overwrite_e]) -> (d, e, info)
@type_check_only
class _function_pttrf[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self, /, d: onp.Array1D[RT], e: onp.Array1D[ST], *, overwrite_d: int = 0, overwrite_e: int = 0
    ) -> tuple[onp.Array1D[RT], onp.Array1D[ST], _info]: ...

spttrf: _function_pttrf[np.float32, np.float32] = ...
dpttrf: _function_pttrf[np.float64, np.float64] = ...
cpttrf: _function_pttrf[np.complex64, np.float32] = ...
zpttrf: _function_pttrf[np.complex128, np.float64] = ...

# (d, e, b, [overwrite_b]) -> (x, info)
@type_check_only
class _function_pttrs_sd[ST: np.generic](Protocol):
    def __call__(
        self, /, d: onp.Array1D[ST], e: onp.Array1D[ST], b: onp.Array2D[ST], *, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

spttrs: _function_pttrs_sd[np.float32] = ...
dpttrs: _function_pttrs_sd[np.float64] = ...

# (d, e, b, [lower, overwrite_b]) -> (x, info)
@type_check_only
class _function_pttrs_cz[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self, /, d: onp.Array1D[RT], e: onp.Array1D[CT], b: onp.Array2D[CT], *, lower: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[CT], _info]: ...

cpttrs: _function_pttrs_cz[np.complex64, np.float32] = ...
zpttrs: _function_pttrs_cz[np.complex128, np.float64] = ...

# (x, y, c, s, [n, offx, incx, offy, incy, overwrite_x, overwrite_y]) -> (x, y)
@type_check_only
class _function_rot[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        x: onp.Array1D[ST],
        y: onp.Array1D[ST],
        c: float,
        s: complex,
        *,
        n: int = ...,  # = (lx - 1 - offx) / abs(incx) + 1
        offx: int = 0,
        incx: int = 1,
        offy: int = 0,
        incy: int = 1,
        overwrite_x: int = 0,
        overwrite_y: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array1D[ST]]: ...

crot: _function_rot[np.complex64] = ...
zrot: _function_rot[np.complex128] = ...

# (ab, [compute_v, lower, ldab, overwrite_ab]) -> (w, z, info)
@type_check_only
class _function_sbev[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        *,
        compute_v: int = 1,
        lower: int = 0,
        ldab: int = ...,  # = ab.shape[0]
        overwrite_ab: int = 1,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], _info]: ...

ssbev: _function_sbev[np.float32] = ...
dsbev: _function_sbev[np.float64] = ...

# (ab, [compute_v, lower, ldab, liwork, overwrite_ab]) -> (w, z, info)
@type_check_only
class _function_sbevd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        *,
        compute_v: int = 1,
        lower: int = 0,
        ldab: int = ...,  # = ab.shape[0]
        liwork: int = ...,  # = (3 + 5 * n if compute_v else 1)
        overwrite_ab: int = 1,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], _info]: ...

ssbevd: _function_sbevd[np.float32] = ...
dsbevd: _function_sbevd[np.float64] = ...

# (ab, vl, vu, il, iu, [ldab, compute_v, range, lower, abstol, mmax, overwrite_ab]) -> (w, z, m, ifail, info)
@type_check_only
class _function_sbevx[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        vl: float,
        vu: float,
        il: int,
        iu: int,
        *,
        ldab: int = ...,  # = ab.shape[0]
        compute_v: int = 1,
        range: int = 0,
        lower: int = 0,
        abstol: float = 0.0,
        mmax: int = ...,  # = (((iu - il + 1) if range == 2 else n) if compute_v else 1)
        overwrite_ab: int = 1,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], int, onp.Array1D[np.int32], _info]: ...

ssbevx: _function_sbevx[np.float32] = ...
dsbevx: _function_sbevx[np.float64] = ...

# (n, k, alpha, a, beta, c, [transr, uplo, trans, overwrite_c]) -> cout
@type_check_only
class _function_sfrk[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        n: int,
        k: int,
        alpha: float,
        a: onp.Array2D[ST],
        beta: float,
        c: onp.Array1D[ST],
        *,
        transr: str | bytes = "N",
        uplo: str | bytes = "U",
        trans: str | bytes = "N",
        overwrite_c: int = 0,
    ) -> onp.Array1D[ST]: ...

ssfrk: _function_sfrk[np.float32] = ...
dsfrk: _function_sfrk[np.float64] = ...

# (d, e, range, vl, vu, il, iu, tol, order) -> (m, w, iblock, isplit, info)
@type_check_only
class _function_stebz[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        d: onp.Array1D[ST],
        e: onp.Array1D[ST],
        range: int,
        vl: float,
        vu: float,
        il: int,
        iu: int,
        tol: float,
        order: str | bytes,
    ) -> tuple[int, onp.Array1D[ST], onp.Array1D[np.int32], onp.Array1D[np.int32], _info]: ...

sstebz: _function_stebz[np.float32] = ...
dstebz: _function_stebz[np.float64] = ...

# (d, e, w, iblock, isplit) -> (z, info)
@type_check_only
class _function_stein[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        d: onp.Array1D[ST],
        e: onp.Array1D[ST],
        w: onp.Array1D[ST],
        iblock: onp.Array1D[np.int32],
        isplit: onp.Array1D[np.int32],
    ) -> tuple[onp.Array2D[ST], _info]: ...

sstein: _function_stein[np.float32] = ...
dstein: _function_stein[np.float64] = ...

# (d, e, range, vl, vu, il, iu, [compute_v, lwork, liwork, overwrite_d]) -> (m, w, z, info)
@type_check_only
class _function_stemr[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        d: onp.Array1D[ST],
        e: onp.Array1D[ST],
        range: int,
        vl: float,
        vu: float,
        il: int,
        iu: int,
        *,
        compute_v: int = 1,
        lwork: int = ...,  # = max((18 * n if compute_v else 12 * n), 1)
        liwork: int = ...,  # = (10 * n if compute_v else 8 * n)
        overwrite_d: int = 0,
    ) -> tuple[int, onp.Array1D[ST], onp.Array2D[ST], _info]: ...

sstemr: _function_stemr[np.float32] = ...
dstemr: _function_stemr[np.float64] = ...

# (d, e, range, vl, vu, il, iu, [compute_v, overwrite_d, overwrite_e]) -> (work, iwork, info)
@type_check_only
class _function_stemr_lwork[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        d: onp.Array1D[ST],
        e: onp.Array1D[ST],
        range: int,
        vl: float,
        vu: float,
        il: int,
        iu: int,
        *,
        compute_v: int = 1,
        overwrite_d: int = 0,
        overwrite_e: int = 0,
    ) -> tuple[float, int, _info]: ...

sstemr_lwork: _function_stemr_lwork[np.float32] = ...
dstemr_lwork: _function_stemr_lwork[np.float64] = ...

# (d, e, [overwrite_d, overwrite_e]) -> (vals, info)
@type_check_only
class _function_sterf[ST: np.generic](Protocol):
    def __call__(
        self, /, d: onp.Array1D[ST], e: onp.Array1D[ST], *, overwrite_d: int = 0, overwrite_e: int = 0
    ) -> tuple[onp.Array1D[ST], _info]: ...

ssterf: _function_sterf[np.float32] = ...
dsterf: _function_sterf[np.float64] = ...

# (d, e, [compute_v, overwrite_d, overwrite_e]) -> (vals, z, info)
@type_check_only
class _function_stev[ST: np.generic](Protocol):
    def __call__(
        self, /, d: onp.Array1D[ST], e: onp.Array1D[ST], *, compute_v: int = 1, overwrite_d: int = 0, overwrite_e: int = 0
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], _info]: ...

sstev: _function_stev[np.float32] = ...
dstev: _function_stev[np.float64] = ...

# (d, e, [compute_v, lwork, liwork, overwrite_d, overwrite_e]) -> (vals, z, info)
@type_check_only
class _function_stevd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        d: onp.Array1D[ST],
        e: onp.Array1D[ST],
        *,
        compute_v: int = 1,
        lwork: int = ...,  # = (1 + 4 * n + n * n if compute_v else 1)
        liwork: int = ...,  # = (3 + 5 * n if compute_v else 1)
        overwrite_d: int = 0,
        overwrite_e: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], _info]: ...

sstevd: _function_stevd[np.float32] = ...
dstevd: _function_stevd[np.float64] = ...

# (a, ipiv, anorm, [lower]) -> (rcond, info)
@type_check_only
class _function_sycon[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], ipiv: onp.Array1D[np.int32], anorm: float, *, lower: int = 0
    ) -> tuple[_rcond, _info]: ...

ssycon: _function_sycon[np.float32] = ...
dsycon: _function_sycon[np.float64] = ...
csycon: _function_sycon[np.complex64] = ...
zsycon: _function_sycon[np.complex128] = ...

# (a, ipiv, [lower, way, overwrite_a]) -> (a, e, info)
@type_check_only
class _function_syconv[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], ipiv: onp.Array1D[np.int32], *, lower: int = 0, way: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

ssyconv: _function_syconv[np.float32] = ...
dsyconv: _function_syconv[np.float64] = ...
csyconv: _function_syconv[np.complex64] = ...
zsyconv: _function_syconv[np.complex128] = ...

# (a, [lower]) -> (s, scond, amax, info)
@type_check_only
class _function_syequb[ST: np.generic, RT: np.generic](Protocol):
    def __call__(self, /, a: onp.Array2D[ST], *, lower: int = 0) -> tuple[onp.Array1D[RT], float, float, _info]: ...

ssyequb: _function_syequb[np.float32, np.float32] = ...
dsyequb: _function_syequb[np.float64, np.float64] = ...
csyequb: _function_syequb[np.complex64, np.float32] = ...
zsyequb: _function_syequb[np.complex128, np.float64] = ...

# (a, [compute_v, lower, lwork, overwrite_a]) -> (w, v, info)
@type_check_only
class _function_syev[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        compute_v: int = 1,
        lower: int = 0,
        lwork: int = ...,  # = max(3 * n - 1, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], _info]: ...

ssyev: _function_syev[np.float32] = ...
dsyev: _function_syev[np.float64] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_syev_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[float, _info]: ...

ssyev_lwork: _function_syev_lwork = ...
dsyev_lwork: _function_syev_lwork = ...

# (a, [compute_v, lower, lwork, liwork, overwrite_a]) -> (w, v, info)
@type_check_only
class _function_syevd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        compute_v: int = 1,
        lower: int = 0,
        lwork: int = ...,  # = max((1 + 6 * n + 2 * n * n if compute_v else 2 * n + 1), 1)
        liwork: int = ...,  # = (3 + 5 * n if compute_v else 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], _info]: ...

ssyevd: _function_syevd[np.float32] = ...
dsyevd: _function_syevd[np.float64] = ...

# (n, [compute_v, lower]) -> (work, iwork, info)
@type_check_only
class _function_syevd_lwork(Protocol):
    def __call__(self, /, n: int, *, compute_v: int = 1, lower: int = 0) -> tuple[float, int, _info]: ...

ssyevd_lwork: _function_syevd_lwork = ...
dsyevd_lwork: _function_syevd_lwork = ...

# (a, [compute_v, range, lower, vl, vu, il, iu, abstol, lwork, liwork, overwrite_a]) -> (w, z, m, isuppz, info)
@type_check_only
class _function_syevr[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        compute_v: int = 1,
        range: str | bytes = "A",
        lower: int = 0,
        vl: float = 0.0,
        vu: float = 1.0,
        il: int = 1,
        iu: int = ...,  # = n
        abstol: float = 0.0,
        lwork: int = ...,  # = max(26 * n, 1)
        liwork: int = ...,  # = max(1, 10 * n)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], int, onp.Array1D[np.int32], _info]: ...

ssyevr: _function_syevr[np.float32] = ...
dsyevr: _function_syevr[np.float64] = ...

# (n, [lower]) -> (work, iwork, info)
@type_check_only
class _function_syevr_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[float, int, _info]: ...

ssyevr_lwork: _function_syevr_lwork = ...
dsyevr_lwork: _function_syevr_lwork = ...

# (a, [compute_v, range, lower, vl, vu, il, iu, abstol, lwork, overwrite_a]) -> (w, z, m, ifail, info)
@type_check_only
class _function_syevx[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        compute_v: int = 1,
        range: str | bytes = "A",
        lower: int = 0,
        vl: float = 0.0,
        vu: float = 1.0,
        il: int = 1,
        iu: int = ...,  # = n
        abstol: float = 0.0,
        lwork: int = ...,  # = max(8 * n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], int, onp.Array1D[np.int32], _info]: ...

ssyevx: _function_syevx[np.float32] = ...
dsyevx: _function_syevx[np.float64] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_syevx_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[float, _info]: ...

ssyevx_lwork: _function_syevx_lwork = ...
dsyevx_lwork: _function_syevx_lwork = ...

# (a, b, [itype, lower, overwrite_a]) -> (c, info)
@type_check_only
class _function_sygst[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], b: onp.Array2D[ST], *, itype: int = 1, lower: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

ssygst: _function_sygst[np.float32] = ...
dsygst: _function_sygst[np.float64] = ...

# (a, b, [itype, jobz, uplo, lwork, overwrite_a, overwrite_b]) -> (w, v, info)
@type_check_only
class _function_sygv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        itype: int = 1,
        jobz: str | bytes = "V",
        uplo: str | bytes = "L",
        lwork: int = ...,  # = max(3 * n - 1, 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], _info]: ...

ssygv: _function_sygv[np.float32] = ...
dsygv: _function_sygv[np.float64] = ...

# (n, [uplo]) -> (work, info)
@type_check_only
class _function_sygv_lwork(Protocol):
    def __call__(self, /, n: int, *, uplo: str | bytes = "L") -> tuple[float, _info]: ...

ssygv_lwork: _function_sygv_lwork = ...
dsygv_lwork: _function_sygv_lwork = ...

# (a, b, [itype, jobz, uplo, lwork, liwork, overwrite_a, overwrite_b]) -> (w, v, info)
@type_check_only
class _function_sygvd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        itype: int = 1,
        jobz: str | bytes = "V",
        uplo: str | bytes = "L",
        lwork: int = ...,  # = (2 * n + 1 if jobz == "N" else 1 + 6 * n + 2 * n * n)
        liwork: int = ...,  # = (1 if jobz == "N" else 5 * n + 3)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], _info]: ...

ssygvd: _function_sygvd[np.float32] = ...
dsygvd: _function_sygvd[np.float64] = ...

# (a, b, [itype, jobz, range, uplo, vl, vu, il, iu, abstol, lwork, overwrite_a, overwrite_b]) -> (w, z, m, ifail, info)
@type_check_only
class _function_sygvx[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        itype: int = 1,
        jobz: str | bytes = "V",
        range: str | bytes = "A",
        uplo: str | bytes = "L",
        vl: float = 0.0,
        vu: float = 1.0,
        il: int = 1,
        iu: int = ...,  # = n
        abstol: float = 0.0,
        lwork: int = ...,  # = max(8 * n, 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array1D[ST], onp.Array2D[ST], int, onp.Array1D[np.int32], _info]: ...

ssygvx: _function_sygvx[np.float32] = ...
dsygvx: _function_sygvx[np.float64] = ...

# (n, [uplo]) -> (work, info)
@type_check_only
class _function_sygvx_lwork(Protocol):
    def __call__(self, /, n: int, *, uplo: str | bytes = "L") -> tuple[float, _info]: ...

ssygvx_lwork: _function_sygvx_lwork = ...
dsygvx_lwork: _function_sygvx_lwork = ...

# (a, b, [lwork, lower, overwrite_a, overwrite_b]) -> (udut, ipiv, x, info)
@type_check_only
class _function_sysv[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        lwork: int = ...,  # = max(n, 1)
        lower: int = 0,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], onp.Array2D[ST], _info]: ...

ssysv: _function_sysv[np.float32] = ...
dsysv: _function_sysv[np.float64] = ...
csysv: _function_sysv[np.complex64] = ...
zsysv: _function_sysv[np.complex128] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_sysv_lwork[WorkT](Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[WorkT, _info]: ...

ssysv_lwork: _function_sysv_lwork[float] = ...
dsysv_lwork: _function_sysv_lwork[float] = ...
csysv_lwork: _function_sysv_lwork[complex] = ...
zsysv_lwork: _function_sysv_lwork[complex] = ...

# (a, b, [af, ipiv, lwork, factored, lower, overwrite_a, overwrite_b]) -> (a_s, udut, ipiv, b_s, x, rcond, ferr, berr, info)
@type_check_only
class _function_sysvx[ST: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        af: onp.Array2D[ST] | None = None,
        ipiv: onp.Array1D[np.int32] | None = None,
        lwork: int = ...,  # = max(3 * n, 1)
        factored: int = 0,
        lower: int = 0,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[
        onp.Array2D[ST],
        onp.Array2D[ST],
        onp.Array1D[np.int32],
        onp.Array2D[ST],
        onp.Array2D[ST],
        _rcond,
        onp.Array1D[RT],
        onp.Array1D[RT],
        _info,
    ]: ...

ssysvx: _function_sysvx[np.float32, np.float32] = ...
dsysvx: _function_sysvx[np.float64, np.float64] = ...
csysvx: _function_sysvx[np.complex64, np.float32] = ...
zsysvx: _function_sysvx[np.complex128, np.float64] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_sysvx_lwork[WorkT](Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[WorkT, _info]: ...

ssysvx_lwork: _function_sysvx_lwork[float] = ...
dsysvx_lwork: _function_sysvx_lwork[float] = ...
csysvx_lwork: _function_sysvx_lwork[complex] = ...
zsysvx_lwork: _function_sysvx_lwork[complex] = ...

# (a, [lower, overwrite_a]) -> (ldu, ipiv, info)
@type_check_only
class _function_sytf2[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, lower: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], _info]: ...

ssytf2: _function_sytf2[np.float32] = ...
dsytf2: _function_sytf2[np.float64] = ...
csytf2: _function_sytf2[np.complex64] = ...
zsytf2: _function_sytf2[np.complex128] = ...

# (a, [lower, lwork, overwrite_a]) -> (c, d, e, tau, info)
@type_check_only
class _function_sytrd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lower: int = 0,
        lwork: int = ...,  # = max(n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], onp.Array1D[ST], onp.Array1D[ST], _info]: ...

ssytrd: _function_sytrd[np.float32] = ...
dsytrd: _function_sytrd[np.float64] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_sytrd_lwork(Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[float, _info]: ...

ssytrd_lwork: _function_sytrd_lwork = ...
dsytrd_lwork: _function_sytrd_lwork = ...

# (a, [lower, lwork, overwrite_a]) -> (ldu, ipiv, info)
@type_check_only
class _function_sytrf[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lower: int = 0,
        lwork: int = ...,  # = max(n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[np.int32], _info]: ...

ssytrf: _function_sytrf[np.float32] = ...
dsytrf: _function_sytrf[np.float64] = ...
csytrf: _function_sytrf[np.complex64] = ...
zsytrf: _function_sytrf[np.complex128] = ...

# (n, [lower]) -> (work, info)
@type_check_only
class _function_sytrf_lwork[WorkT](Protocol):
    def __call__(self, /, n: int, *, lower: int = 0) -> tuple[WorkT, _info]: ...

ssytrf_lwork: _function_sytrf_lwork[float] = ...
dsytrf_lwork: _function_sytrf_lwork[float] = ...
csytrf_lwork: _function_sytrf_lwork[complex] = ...
zsytrf_lwork: _function_sytrf_lwork[complex] = ...

# (a, ipiv, [lower, overwrite_a]) -> (inv_a, info)
@type_check_only
class _function_sytri[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], ipiv: onp.Array1D[np.int32], *, lower: int = 0, overwrite_a: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

ssytri: _function_sytri[np.float32] = ...
dsytri: _function_sytri[np.float64] = ...
csytri: _function_sytri[np.complex64] = ...
zsytri: _function_sytri[np.complex128] = ...

# (a, ipiv, b, [lower, overwrite_b]) -> (x, info)
@type_check_only
class _function_sytrs[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], ipiv: onp.Array1D[np.int32], b: onp.Array2D[ST], *, lower: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

ssytrs: _function_sytrs[np.float32] = ...
dsytrs: _function_sytrs[np.float64] = ...
csytrs: _function_sytrs[np.complex64] = ...
zsytrs: _function_sytrs[np.complex128] = ...

# (ab, b, [uplo, trans, diag, overwrite_b]) -> (x, info)
@type_check_only
class _function_tbtrs[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        ab: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        uplo: str | bytes = "U",
        trans: str | bytes = "N",
        diag: str | bytes = "N",
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

stbtrs: _function_tbtrs[np.float32] = ...
dtbtrs: _function_tbtrs[np.float64] = ...
ctbtrs: _function_tbtrs[np.complex64] = ...
ztbtrs: _function_tbtrs[np.complex128] = ...

# (alpha, a, b, [transr, side, uplo, trans, diag, overwrite_b]) -> x
@type_check_only
class _function_tfsm[ST: np.generic, WorkT](Protocol):
    def __call__(
        self,
        /,
        alpha: WorkT,
        a: onp.Array1D[ST],
        b: onp.Array2D[ST],
        *,
        transr: str | bytes = "N",
        side: str | bytes = "L",
        uplo: str | bytes = "U",
        trans: str | bytes = "N",
        diag: str | bytes = "N",
        overwrite_b: int = 0,
    ) -> onp.Array2D[ST]: ...

stfsm: _function_tfsm[np.float32, float] = ...
dtfsm: _function_tfsm[np.float64, float] = ...
ctfsm: _function_tfsm[np.complex64, complex] = ...
ztfsm: _function_tfsm[np.complex128, complex] = ...

# (n, arf, [transr, uplo]) -> (ap, info)
@type_check_only
class _function_tfttp[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, arf: onp.Array1D[ST], *, transr: str | bytes = "N", uplo: str | bytes = "U"
    ) -> tuple[onp.Array1D[ST], _info]: ...

stfttp: _function_tfttp[np.float32] = ...
dtfttp: _function_tfttp[np.float64] = ...
ctfttp: _function_tfttp[np.complex64] = ...
ztfttp: _function_tfttp[np.complex128] = ...

# (n, arf, [transr, uplo]) -> (a, info)
@type_check_only
class _function_tfttr[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, arf: onp.Array1D[ST], *, transr: str | bytes = "N", uplo: str | bytes = "U"
    ) -> tuple[onp.Array2D[ST], _info]: ...

stfttr: _function_tfttr[np.float32] = ...
dtfttr: _function_tfttr[np.float64] = ...
ctfttr: _function_tfttr[np.complex64] = ...
ztfttr: _function_tfttr[np.complex128] = ...

# (a, b, q, z, ifst, ilst, [wantq, wantz, lwork, overwrite_a, overwrite_b, overwrite_q, overwrite_z]) -> (a, b, q, z, work, info)
@type_check_only
class _function_tgexc_sd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        q: onp.Array2D[ST],
        z: onp.Array2D[ST],
        ifst: int,
        ilst: int,
        *,
        wantq: int = 1,
        wantz: int = 1,
        lwork: int = ...,  # = max(4 * n + 16, 1)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
        overwrite_q: int = 0,
        overwrite_z: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[ST], _info]: ...

stgexc: _function_tgexc_sd[np.float32] = ...
dtgexc: _function_tgexc_sd[np.float64] = ...

# (a, b, q, z, ifst, ilst, [wantq, wantz, overwrite_a, overwrite_b, overwrite_q, overwrite_z]) -> (a, b, q, z, info)
@type_check_only
class _function_tgexc_cz[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        q: onp.Array2D[ST],
        z: onp.Array2D[ST],
        ifst: int,
        ilst: int,
        *,
        wantq: int = 1,
        wantz: int = 1,
        overwrite_a: int = 0,
        overwrite_b: int = 0,
        overwrite_q: int = 0,
        overwrite_z: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], onp.Array2D[ST], onp.Array2D[ST], _info]: ...

ctgexc: _function_tgexc_cz[np.complex64] = ...
ztgexc: _function_tgexc_cz[np.complex128] = ...

# (select, a, b, q, z, [ijob, wantq, wantz, lwork, liwork, ...]) -> (as, bs, alphar, alphai, beta, qs, zs, m, pl, pr, dif, info)
@type_check_only
class _function_tgsen_sd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        select: onp.Array1D[np.int32],
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        q: onp.Array2D[ST],
        z: onp.Array2D[ST],
        *,
        ijob: int = 4,
        wantq: int = 1,
        wantz: int = 1,
        lwork: int = ...,  # = 4 * n + 16
        liwork: int = ...,  # = n + 6
        overwrite_a: int = 0,
        overwrite_b: int = 0,
        overwrite_q: int = 0,
        overwrite_z: int = 0,
    ) -> tuple[
        onp.Array2D[ST],
        onp.Array2D[ST],
        onp.Array1D[ST],
        onp.Array1D[ST],
        onp.Array1D[ST],
        onp.Array2D[ST],
        onp.Array2D[ST],
        int,
        float,
        float,
        onp.Array1D[ST],
        _info,
    ]: ...

stgsen: _function_tgsen_sd[np.float32] = ...
dtgsen: _function_tgsen_sd[np.float64] = ...

# (select, a, b, q, z, [ijob, wantq, wantz, lwork, liwork, ...]) -> (as, bs, alpha, beta, qs, zs, m, pl, pr, dif, info)
@type_check_only
class _function_tgsen_cz[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        select: onp.Array1D[np.int32],
        a: onp.Array2D[CT],
        b: onp.Array2D[CT],
        q: onp.Array2D[CT],
        z: onp.Array2D[CT],
        *,
        ijob: int = 4,
        wantq: int = 1,
        wantz: int = 1,
        lwork: int = ...,  # = (1 if ijob == 0 else n + 2)
        liwork: int = ...,  # = (1 if ijob == 0 else n + 2)
        overwrite_a: int = 0,
        overwrite_b: int = 0,
        overwrite_q: int = 0,
        overwrite_z: int = 0,
    ) -> tuple[
        onp.Array2D[CT],
        onp.Array2D[CT],
        onp.Array1D[CT],
        onp.Array1D[CT],
        onp.Array2D[CT],
        onp.Array2D[CT],
        int,
        float,
        float,
        onp.Array1D[RT],
        _info,
    ]: ...

ctgsen: _function_tgsen_cz[np.complex64, np.float32] = ...
ztgsen: _function_tgsen_cz[np.complex128, np.float64] = ...

# (select, a, [ijob]) -> (work, iwork, info)
@type_check_only
class _function_tgsen_lwork_sd[ST: np.generic](Protocol):
    def __call__(self, /, select: onp.Array1D[np.int32], a: onp.Array2D[ST], *, ijob: int = 4) -> tuple[float, int, _info]: ...

stgsen_lwork: _function_tgsen_lwork_sd[np.float32] = ...
dtgsen_lwork: _function_tgsen_lwork_sd[np.float64] = ...

# (select, a, b, [ijob]) -> (work, iwork, info)
@type_check_only
class _function_tgsen_lwork_cz[ST: np.generic](Protocol):
    def __call__(
        self, /, select: onp.Array1D[np.int32], a: onp.Array2D[ST], b: onp.Array2D[ST], *, ijob: int = 4
    ) -> tuple[complex, int, _info]: ...

ctgsen_lwork: _function_tgsen_lwork_cz[np.complex64] = ...
ztgsen_lwork: _function_tgsen_lwork_cz[np.complex128] = ...

# (a, b, c, d, e, f, [trans, ijob, lwork, overwrite_c, overwrite_f]) -> (r, l, scale, dif, info)
@type_check_only
class _function_tgsyl[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        c: onp.Array2D[ST],
        d: onp.Array2D[ST],
        e: onp.Array2D[ST],
        f: onp.Array2D[ST],
        *,
        trans: str | bytes = "N",
        ijob: int = 0,
        lwork: int = ...,  # = max(1, 2 * m * n)
        overwrite_c: int = 0,
        overwrite_f: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], float, float, _info]: ...

stgsyl: _function_tgsyl[np.float32] = ...
dtgsyl: _function_tgsyl[np.float64] = ...

# (l, v, t, a, b, [side, trans, overwrite_a, overwrite_b]) -> (a, b, info)
@type_check_only
class _function_tpmqrt[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        l: int,
        v: onp.Array2D[ST],
        t: onp.Array2D[ST],
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        side: str | bytes = "L",
        trans: str | bytes = "N",
        overwrite_a: int = 0,
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], _info]: ...

stpmqrt: _function_tpmqrt[np.float32] = ...
dtpmqrt: _function_tpmqrt[np.float64] = ...
ctpmqrt: _function_tpmqrt[np.complex64] = ...
ztpmqrt: _function_tpmqrt[np.complex128] = ...

# (l, nb, a, b, [overwrite_a, overwrite_b]) -> (a, b, t, info)
@type_check_only
class _function_tpqrt[ST: np.generic](Protocol):
    def __call__(
        self, /, l: int, nb: int, a: onp.Array2D[ST], b: onp.Array2D[ST], *, overwrite_a: int = 0, overwrite_b: int = 0
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], onp.Array2D[ST], _info]: ...

stpqrt: _function_tpqrt[np.float32] = ...
dtpqrt: _function_tpqrt[np.float64] = ...
ctpqrt: _function_tpqrt[np.complex64] = ...
ztpqrt: _function_tpqrt[np.complex128] = ...

# (n, ap, [transr, uplo]) -> (arf, info)
@type_check_only
class _function_tpttf[ST: np.generic](Protocol):
    def __call__(
        self, /, n: int, ap: onp.Array1D[ST], *, transr: str | bytes = "N", uplo: str | bytes = "U"
    ) -> tuple[onp.Array1D[ST], _info]: ...

stpttf: _function_tpttf[np.float32] = ...
dtpttf: _function_tpttf[np.float64] = ...
ctpttf: _function_tpttf[np.complex64] = ...
ztpttf: _function_tpttf[np.complex128] = ...

# (n, ap, [uplo]) -> (a, info)
@type_check_only
class _function_tpttr[ST: np.generic](Protocol):
    def __call__(self, /, n: int, ap: onp.Array1D[ST], *, uplo: str | bytes = "U") -> tuple[onp.Array2D[ST], _info]: ...

stpttr: _function_tpttr[np.float32] = ...
dtpttr: _function_tpttr[np.float64] = ...
ctpttr: _function_tpttr[np.complex64] = ...
ztpttr: _function_tpttr[np.complex128] = ...

# (a, [norm, uplo, diag]) -> (rcond, info)
@type_check_only
class _function_trcon[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, norm: str | bytes = "1", uplo: str | bytes = "U", diag: str | bytes = "N"
    ) -> tuple[_rcond, _info]: ...

strcon: _function_trcon[np.float32] = ...
dtrcon: _function_trcon[np.float64] = ...
ctrcon: _function_trcon[np.complex64] = ...
ztrcon: _function_trcon[np.complex128] = ...

# (a, q, ifst, ilst, [wantq, overwrite_a, overwrite_q]) -> (a, q, info)
@type_check_only
class _function_trexc[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        q: onp.Array2D[ST],
        ifst: int,
        ilst: int,
        *,
        wantq: int = 1,
        overwrite_a: int = 0,
        overwrite_q: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], _info]: ...

strexc: _function_trexc[np.float32] = ...
dtrexc: _function_trexc[np.float64] = ...
ctrexc: _function_trexc[np.complex64] = ...
ztrexc: _function_trexc[np.complex128] = ...

# (select, t, q, [job, wantq, lwork, liwork, overwrite_t, overwrite_q]) -> (ts, qs, wr, wi, m, s, sep, info)
@type_check_only
class _function_trsen_sd[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        select: onp.Array1D[np.int32],
        t: onp.Array2D[ST],
        q: onp.Array2D[ST],
        *,
        job: str | bytes = "B",
        wantq: int = 1,
        lwork: int = ...,  # = max(1, n)
        liwork: int = 1,
        overwrite_t: int = 0,
        overwrite_q: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[ST], onp.Array1D[ST], int, float, float, _info]: ...

strsen: _function_trsen_sd[np.float32] = ...
dtrsen: _function_trsen_sd[np.float64] = ...

# (select, t, q, [job, wantq, lwork, overwrite_t, overwrite_q]) -> (ts, qs, w, m, s, sep, info)
@type_check_only
class _function_trsen_cz[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        select: onp.Array1D[np.int32],
        t: onp.Array2D[ST],
        q: onp.Array2D[ST],
        *,
        job: str | bytes = "B",
        wantq: int = 1,
        lwork: int = ...,  # = max(1, n)
        overwrite_t: int = 0,
        overwrite_q: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array2D[ST], onp.Array1D[ST], int, float, float, _info]: ...

ctrsen: _function_trsen_cz[np.complex64] = ...
ztrsen: _function_trsen_cz[np.complex128] = ...

# (select, t, [job]) -> (work, iwork, info)
@type_check_only
class _function_trsen_lwork_sd[ST: np.generic](Protocol):
    def __call__(
        self, /, select: onp.Array1D[np.int32], t: onp.Array2D[ST], *, job: str | bytes = "B"
    ) -> tuple[float, int, _info]: ...

strsen_lwork: _function_trsen_lwork_sd[np.float32] = ...
dtrsen_lwork: _function_trsen_lwork_sd[np.float64] = ...

# (select, t, [job]) -> (work, info)
@type_check_only
class _function_trsen_lwork_cz[ST: np.generic](Protocol):
    def __call__(
        self, /, select: onp.Array1D[np.int32], t: onp.Array2D[ST], *, job: str | bytes = "B"
    ) -> tuple[complex, _info]: ...

ctrsen_lwork: _function_trsen_lwork_cz[np.complex64] = ...
ztrsen_lwork: _function_trsen_lwork_cz[np.complex128] = ...

# (a, b, c, [trana, tranb, isgn, overwrite_c]) -> (x, scale, info)
@type_check_only
class _function_trsyl[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        c: onp.Array2D[ST],
        *,
        trana: str | bytes = "N",
        tranb: str | bytes = "N",
        isgn: int = 1,
        overwrite_c: int = 0,
    ) -> tuple[onp.Array2D[ST], float, _info]: ...

strsyl: _function_trsyl[np.float32] = ...
dtrsyl: _function_trsyl[np.float64] = ...
ctrsyl: _function_trsyl[np.complex64] = ...
ztrsyl: _function_trsyl[np.complex128] = ...

# (c, [lower, unitdiag, overwrite_c]) -> (inv_c, info)
@type_check_only
class _function_trtri[ST: np.generic](Protocol):
    def __call__(
        self, /, c: onp.Array2D[ST], *, lower: int = 0, unitdiag: int = 0, overwrite_c: int = 0
    ) -> tuple[onp.Array2D[ST], _info]: ...

strtri: _function_trtri[np.float32] = ...
dtrtri: _function_trtri[np.float64] = ...
ctrtri: _function_trtri[np.complex64] = ...
ztrtri: _function_trtri[np.complex128] = ...

# (a, b, [lower, trans, unitdiag, lda, overwrite_b]) -> (x, info)
@type_check_only
class _function_trtrs[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        b: onp.Array2D[ST],
        *,
        lower: int = 0,
        trans: int = 0,
        unitdiag: int = 0,
        lda: int = ...,  # = a.shape[0]
        overwrite_b: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

strtrs: _function_trtrs[np.float32] = ...
dtrtrs: _function_trtrs[np.float64] = ...
ctrtrs: _function_trtrs[np.complex64] = ...
ztrtrs: _function_trtrs[np.complex128] = ...

# (a, [transr, uplo]) -> (arf, info)
@type_check_only
class _function_trttf[ST: np.generic](Protocol):
    def __call__(
        self, /, a: onp.Array2D[ST], *, transr: str | bytes = "N", uplo: str | bytes = "U"
    ) -> tuple[onp.Array1D[ST], _info]: ...

strttf: _function_trttf[np.float32] = ...
dtrttf: _function_trttf[np.float64] = ...
ctrttf: _function_trttf[np.complex64] = ...
ztrttf: _function_trttf[np.complex128] = ...

# (a, [uplo]) -> (ap, info)
@type_check_only
class _function_trttp[ST: np.generic](Protocol):
    def __call__(self, /, a: onp.Array2D[ST], *, uplo: str | bytes = "U") -> tuple[onp.Array1D[ST], _info]: ...

strttp: _function_trttp[np.float32] = ...
dtrttp: _function_trttp[np.float64] = ...
ctrttp: _function_trttp[np.complex64] = ...
ztrttp: _function_trttp[np.complex128] = ...

# (a, [lwork, overwrite_a]) -> (rz, tau, info)
@type_check_only
class _function_tzrzf[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        *,
        lwork: int = ...,  # = max(m, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

stzrzf: _function_tzrzf[np.float32] = ...
dtzrzf: _function_tzrzf[np.float64] = ...
ctzrzf: _function_tzrzf[np.complex64] = ...
ztzrzf: _function_tzrzf[np.complex128] = ...

# (m, n) -> (work, info)
@type_check_only
class _function_tzrzf_lwork[WorkT](Protocol):
    def __call__(self, /, m: int, n: int) -> tuple[WorkT, _info]: ...

stzrzf_lwork: _function_tzrzf_lwork[float] = ...
dtzrzf_lwork: _function_tzrzf_lwork[float] = ...
ctzrzf_lwork: _function_tzrzf_lwork[complex] = ...
ztzrzf_lwork: _function_tzrzf_lwork[complex] = ...

# (x11, x12, x21, x22, [...]) -> cs11, cs12, cs21, cs22, theta, u1, u2, v1t, v2t, info
@type_check_only
class _function_uncsd[CT: np.generic, RT: np.generic](Protocol):
    def __call__(
        self,
        /,
        x11: onp.Array2D[CT],
        x12: onp.Array2D[CT],
        x21: onp.Array2D[CT],
        x22: onp.Array2D[CT],
        *,
        compute_u1: int = 1,
        compute_u2: int = 1,
        compute_v1t: int = 1,
        compute_v2t: int = 1,
        trans: int = 0,
        signs: int = 0,
        lwork: int = ...,  # = 2 * m + max(1, max(mmp, mmq)) + 1
        lrwork: int = ...,  # = 5 * max(1, q - 1) + 4 * max(1, q) + 8 * q + 1
        overwrite_x11: int = 0,
        overwrite_x12: int = 0,
        overwrite_x21: int = 0,
        overwrite_x22: int = 0,
    ) -> tuple[
        onp.Array2D[CT],
        onp.Array2D[CT],
        onp.Array2D[CT],
        onp.Array2D[CT],
        onp.Array1D[RT],
        onp.Array2D[CT],
        onp.Array2D[CT],
        onp.Array2D[CT],
        onp.Array2D[CT],
        _info,
    ]: ...

cuncsd: _function_uncsd[np.complex64, np.float32] = ...
zuncsd: _function_uncsd[np.complex128, np.float64] = ...

# (m, p, q) -> (work, rwork, info)
@type_check_only
class _function_uncsd_lwork(Protocol):
    def __call__(self, /, m: int, p: int, q: int) -> tuple[complex, float, _info]: ...

cuncsd_lwork: _function_uncsd_lwork = ...
zuncsd_lwork: _function_uncsd_lwork = ...

# (a, tau, [lo, hi, lwork, overwrite_a]) -> (ht, info)
@type_check_only
class _function_unghr[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        *,
        lo: int = 0,
        hi: int = ...,  # = n - 1
        lwork: int = ...,  # = max(hi - lo, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

cunghr: _function_unghr[np.complex64] = ...
zunghr: _function_unghr[np.complex128] = ...

# (n, [lo, hi]) -> (work, info)
@type_check_only
class _function_unghr_lwork(Protocol):
    def __call__(
        self,
        /,
        n: int,
        *,
        lo: int = 0,
        hi: int = ...,  # = n - 1
    ) -> tuple[complex, _info]: ...

cunghr_lwork: _function_unghr_lwork = ...
zunghr_lwork: _function_unghr_lwork = ...

# (a, tau, [lwork, overwrite_a]) -> (q, work, info)
@type_check_only
class _function_ungqr[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        *,
        lwork: int = ...,  # = max(3 * n, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

cungqr: _function_ungqr[np.complex64] = ...
zungqr: _function_ungqr[np.complex128] = ...

@type_check_only
class _function_ungrq[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        *,
        lwork: int = ...,  # = max(3 * m, 1)
        overwrite_a: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

cungrq: _function_ungrq[np.complex64] = ...
zungrq: _function_ungrq[np.complex128] = ...

# (side, trans, a, tau, c, lwork, [overwrite_c]) -> (cq, work, info)
@type_check_only
class _function_unmqr[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        side: str | bytes,
        trans: str | bytes,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        c: onp.Array2D[ST],
        lwork: int,
        *,
        overwrite_c: int = 0,
    ) -> tuple[onp.Array2D[ST], onp.Array1D[ST], _info]: ...

cunmqr: _function_unmqr[np.complex64] = ...
zunmqr: _function_unmqr[np.complex128] = ...

# (a, tau, c, [side, trans, lwork, overwrite_c]) -> (cq, info)
@type_check_only
class _function_unmrz[ST: np.generic](Protocol):
    def __call__(
        self,
        /,
        a: onp.Array2D[ST],
        tau: onp.Array1D[ST],
        c: onp.Array2D[ST],
        *,
        side: str | bytes = "L",
        trans: str | bytes = "N",
        lwork: int = ...,  # = max((n if side == "L" else m), 1)
        overwrite_c: int = 0,
    ) -> tuple[onp.Array2D[ST], _info]: ...

cunmrz: _function_unmrz[np.complex64] = ...
zunmrz: _function_unmrz[np.complex128] = ...

# (m, n, [side, trans]) -> (work, info)
@type_check_only
class _function_unmrz_lwork(Protocol):
    def __call__(self, /, m: int, n: int, *, side: str | bytes = "L", trans: str | bytes = "N") -> tuple[complex, _info]: ...

cunmrz_lwork: _function_unmrz_lwork = ...
zunmrz_lwork: _function_unmrz_lwork = ...

# () -> major, minor, patch
@type_check_only
class _function_ilaver(Protocol):
    def __call__(self, /) -> tuple[int, int, int]: ...

ilaver: _function_ilaver = ...
