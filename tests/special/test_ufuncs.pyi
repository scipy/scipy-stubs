from typing import Literal as L, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

import scipy.special as sp

_Float32ND: TypeAlias = onp.ArrayND[np.float32]
_Float64ND: TypeAlias = onp.ArrayND[np.float64]
_Complex64ND: TypeAlias = onp.ArrayND[np.complex64]
_Complex128ND: TypeAlias = onp.ArrayND[np.complex128]

_b1: np.bool_
_i: npc.integer
_f: npc.floating
_f2: np.float16
_f4: np.float32
_f8: np.float64
_c8: np.complex64
_c16: np.complex128

_b1_nd: onp.ArrayND[np.bool_]
_i1_nd: onp.ArrayND[np.uint8 | np.int8]
_f2_nd: onp.ArrayND[np.float16]
_f4_nd: _Float32ND
_f8_nd: _Float64ND
_c8_nd: _Complex64ND
_c16_nd: _Complex128ND

# NOTE: `[c]longdouble` can't be tested, because it's typed as `floating` on `numpy<2.2`

# _UFunc
assert_type(sp.cbrt.__name__, L["cbrt"])
assert_type(sp.cbrt.identity, L[0])

# _UFunc11
assert_type(sp.cbrt.nin, L[1])
assert_type(sp.cbrt.nout, L[1])
assert_type(sp.cbrt.nargs, L[2])
assert_type(sp.cbrt.ntypes, L[2])
assert_type(sp.cbrt.types, list[L["f->f", "d->d"]])
assert_type(sp.exprel.identity, None)

# _UFunc11f
assert_type(sp.cbrt(_b1), np.float64)
assert_type(sp.cbrt(_b1_nd), _Float64ND)
assert_type(sp.cbrt(_i), np.float64)
assert_type(sp.cbrt(_i1_nd), _Float64ND)
assert_type(sp.cbrt(_f2), np.float64)
assert_type(sp.cbrt(_f2_nd), _Float64ND)
assert_type(sp.cbrt(_f4), np.float32)
assert_type(sp.cbrt(_f4_nd), _Float32ND)
assert_type(sp.cbrt(_f8), np.float64)
assert_type(sp.cbrt(_f8_nd), _Float64ND)
sp.cbrt(_c16)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
sp.cbrt(_c16_nd)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(sp.cbrt(False), np.float64)
assert_type(sp.cbrt([False]), _Float64ND)
assert_type(sp.cbrt(0), np.float64)
assert_type(sp.cbrt([0]), _Float64ND)
assert_type(sp.cbrt(0.0), np.float64)
assert_type(sp.cbrt([0.0]), _Float64ND)
sp.cbrt(0j)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
sp.cbrt([0j])  # pyright: ignore[reportArgumentType, reportCallIssue]  # pyrefly: ignore[no-matching-overload]
assert_type(sp.cbrt.at(_b1_nd, _i), None)
assert_type(sp.cbrt.at(_f8_nd, _i), None)
sp.cbrt.at(_c16, _i)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType]
assert_type(sp.cbrt.nin, L[1])

# _UFunc11g
assert_type(sp.logit.ntypes, L[3])
assert_type(sp.logit(_b1), np.float64)
assert_type(sp.logit(_b1_nd), _Float64ND)
assert_type(sp.logit(_f4), np.float32)
assert_type(sp.logit(_f4_nd), _Float32ND)
assert_type(sp.logit(_f8), np.float64)
assert_type(sp.logit(_f8_nd), _Float64ND)
sp.logit(_c16)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
sp.logit(_c16_nd)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(sp.logit(0), np.float64)
assert_type(sp.logit([0]), _Float64ND)
assert_type(sp.logit(0.0), np.float64)
assert_type(sp.logit([0.0]), _Float64ND)
assert_type(sp.logit.at(_b1_nd, _i), None)
assert_type(sp.logit.at(_f8_nd, _i), None)
sp.logit.at(_c16, _i)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType]

# _UFunc11c - TODO: wofz
# _UFunc11fc - TODO: erf

# _UFunc12 - TODO
# _UFunc14 - TODO

###

# _UFunc21 - TODO
# _UFunc22 - TODO
# _UFunc24 - TODO

###

# _UFunc31 - TODO
# _UFunc32 - TODO

###

# _UFunc42 - TODO

###

# _UFunc52 - TODO

###

def _assert_ufunc(x: np.ufunc, /) -> None: ...

_assert_ufunc(sp.agm)
_assert_ufunc(sp.airy)
_assert_ufunc(sp.airye)
_assert_ufunc(sp.bdtr)
_assert_ufunc(sp.bdtrc)
_assert_ufunc(sp.bdtri)
_assert_ufunc(sp.bdtrik)
_assert_ufunc(sp.bdtrin)
_assert_ufunc(sp.bei)
_assert_ufunc(sp.beip)
_assert_ufunc(sp.ber)
_assert_ufunc(sp.berp)
_assert_ufunc(sp.besselpoly)
_assert_ufunc(sp.beta)
_assert_ufunc(sp.betainc)
_assert_ufunc(sp.betaincc)
_assert_ufunc(sp.betainccinv)
_assert_ufunc(sp.betaincinv)
_assert_ufunc(sp.betaln)
_assert_ufunc(sp.binom)
_assert_ufunc(sp.boxcox)
_assert_ufunc(sp.boxcox1p)
_assert_ufunc(sp.btdtria)
_assert_ufunc(sp.btdtrib)
_assert_ufunc(sp.cbrt)
_assert_ufunc(sp.chdtr)
_assert_ufunc(sp.chdtrc)
_assert_ufunc(sp.chdtri)
_assert_ufunc(sp.chdtriv)
_assert_ufunc(sp.chndtr)
_assert_ufunc(sp.chndtridf)
_assert_ufunc(sp.chndtrinc)
_assert_ufunc(sp.chndtrix)
_assert_ufunc(sp.cosdg)
_assert_ufunc(sp.cosm1)
_assert_ufunc(sp.cotdg)
_assert_ufunc(sp.dawsn)
_assert_ufunc(sp.ellipe)
_assert_ufunc(sp.ellipeinc)
_assert_ufunc(sp.ellipj)
_assert_ufunc(sp.ellipk)
_assert_ufunc(sp.ellipkinc)
_assert_ufunc(sp.ellipkm1)
_assert_ufunc(sp.elliprc)
_assert_ufunc(sp.elliprd)
_assert_ufunc(sp.elliprf)
_assert_ufunc(sp.elliprg)
_assert_ufunc(sp.elliprj)
_assert_ufunc(sp.entr)
_assert_ufunc(sp.erf)
_assert_ufunc(sp.erfc)
_assert_ufunc(sp.erfcinv)
_assert_ufunc(sp.erfcx)
_assert_ufunc(sp.erfi)
_assert_ufunc(sp.erfinv)
_assert_ufunc(sp.eval_chebyc)
_assert_ufunc(sp.eval_chebys)
_assert_ufunc(sp.eval_chebyt)
_assert_ufunc(sp.eval_chebyu)
_assert_ufunc(sp.eval_gegenbauer)
_assert_ufunc(sp.eval_genlaguerre)
_assert_ufunc(sp.eval_hermite)
_assert_ufunc(sp.eval_hermitenorm)
_assert_ufunc(sp.eval_jacobi)
_assert_ufunc(sp.eval_laguerre)
_assert_ufunc(sp.eval_legendre)
_assert_ufunc(sp.eval_sh_chebyt)
_assert_ufunc(sp.eval_sh_chebyu)
_assert_ufunc(sp.eval_sh_jacobi)
_assert_ufunc(sp.eval_sh_legendre)
_assert_ufunc(sp.exp1)
_assert_ufunc(sp.exp2)
_assert_ufunc(sp.exp10)
_assert_ufunc(sp.expi)
_assert_ufunc(sp.expit)
_assert_ufunc(sp.expm1)
_assert_ufunc(sp.expn)
_assert_ufunc(sp.exprel)
_assert_ufunc(sp.fdtr)
_assert_ufunc(sp.fdtrc)
_assert_ufunc(sp.fdtri)
_assert_ufunc(sp.fdtridfd)
_assert_ufunc(sp.fresnel)
_assert_ufunc(sp.gamma)
_assert_ufunc(sp.gammainc)
_assert_ufunc(sp.gammaincc)
_assert_ufunc(sp.gammainccinv)
_assert_ufunc(sp.gammaincinv)
_assert_ufunc(sp.gammaln)
_assert_ufunc(sp.gammasgn)
_assert_ufunc(sp.gdtr)
_assert_ufunc(sp.gdtrc)
_assert_ufunc(sp.gdtria)
_assert_ufunc(sp.gdtrib)
_assert_ufunc(sp.gdtrix)
_assert_ufunc(sp.hankel1)
_assert_ufunc(sp.hankel1e)
_assert_ufunc(sp.hankel2)
_assert_ufunc(sp.hankel2e)
_assert_ufunc(sp.huber)
_assert_ufunc(sp.hyp0f1)
_assert_ufunc(sp.hyp1f1)
_assert_ufunc(sp.hyp2f1)
_assert_ufunc(sp.hyperu)
_assert_ufunc(sp.i0)
_assert_ufunc(sp.i0e)
_assert_ufunc(sp.i1)
_assert_ufunc(sp.i1e)
_assert_ufunc(sp.inv_boxcox)
_assert_ufunc(sp.inv_boxcox1p)
_assert_ufunc(sp.it2i0k0)
_assert_ufunc(sp.it2j0y0)
_assert_ufunc(sp.it2struve0)
_assert_ufunc(sp.itairy)
_assert_ufunc(sp.iti0k0)
_assert_ufunc(sp.itj0y0)
_assert_ufunc(sp.itmodstruve0)
_assert_ufunc(sp.itstruve0)
_assert_ufunc(sp.iv)
_assert_ufunc(sp.ive)
_assert_ufunc(sp.j0)
_assert_ufunc(sp.j1)
_assert_ufunc(sp.jn)
_assert_ufunc(sp.jv)
_assert_ufunc(sp.jve)
_assert_ufunc(sp.k0)
_assert_ufunc(sp.k0e)
_assert_ufunc(sp.k1)
_assert_ufunc(sp.k1e)
_assert_ufunc(sp.kei)
_assert_ufunc(sp.keip)
_assert_ufunc(sp.kelvin)
_assert_ufunc(sp.ker)
_assert_ufunc(sp.kerp)
_assert_ufunc(sp.kl_div)
_assert_ufunc(sp.kn)
_assert_ufunc(sp.kolmogi)
_assert_ufunc(sp.kolmogorov)
_assert_ufunc(sp.kv)
_assert_ufunc(sp.kve)
_assert_ufunc(sp.log1p)
_assert_ufunc(sp.log_expit)
_assert_ufunc(sp.log_ndtr)
_assert_ufunc(sp.log_wright_bessel)
_assert_ufunc(sp.loggamma)
_assert_ufunc(sp.logit)
_assert_ufunc(sp.lpmv)
_assert_ufunc(sp.mathieu_a)
_assert_ufunc(sp.mathieu_b)
_assert_ufunc(sp.mathieu_cem)
_assert_ufunc(sp.mathieu_modcem1)
_assert_ufunc(sp.mathieu_modcem2)
_assert_ufunc(sp.mathieu_modsem1)
_assert_ufunc(sp.mathieu_modsem2)
_assert_ufunc(sp.mathieu_sem)
_assert_ufunc(sp.modfresnelm)
_assert_ufunc(sp.modfresnelp)
_assert_ufunc(sp.modstruve)
_assert_ufunc(sp.nbdtr)
_assert_ufunc(sp.nbdtrc)
_assert_ufunc(sp.nbdtri)
_assert_ufunc(sp.nbdtrik)
_assert_ufunc(sp.nbdtrin)
_assert_ufunc(sp.ncfdtr)
_assert_ufunc(sp.ncfdtri)
_assert_ufunc(sp.ncfdtridfd)
_assert_ufunc(sp.ncfdtridfn)
_assert_ufunc(sp.ncfdtrinc)
_assert_ufunc(sp.nctdtr)
_assert_ufunc(sp.nctdtridf)
_assert_ufunc(sp.nctdtrinc)
_assert_ufunc(sp.nctdtrit)
_assert_ufunc(sp.ndtr)
_assert_ufunc(sp.ndtri)
_assert_ufunc(sp.ndtri_exp)
_assert_ufunc(sp.nrdtrimn)
_assert_ufunc(sp.nrdtrisd)
_assert_ufunc(sp.obl_ang1)
_assert_ufunc(sp.obl_ang1_cv)
_assert_ufunc(sp.obl_cv)
_assert_ufunc(sp.obl_rad1)
_assert_ufunc(sp.obl_rad1_cv)
_assert_ufunc(sp.obl_rad2)
_assert_ufunc(sp.obl_rad2_cv)
_assert_ufunc(sp.owens_t)
_assert_ufunc(sp.pbdv)
_assert_ufunc(sp.pbvv)
_assert_ufunc(sp.pbwa)
_assert_ufunc(sp.pdtr)
_assert_ufunc(sp.pdtrc)
_assert_ufunc(sp.pdtri)
_assert_ufunc(sp.pdtrik)
_assert_ufunc(sp.poch)
_assert_ufunc(sp.powm1)
_assert_ufunc(sp.pro_ang1)
_assert_ufunc(sp.pro_ang1_cv)
_assert_ufunc(sp.pro_cv)
_assert_ufunc(sp.pro_rad1)
_assert_ufunc(sp.pro_rad1_cv)
_assert_ufunc(sp.pro_rad2)
_assert_ufunc(sp.pro_rad2_cv)
_assert_ufunc(sp.pseudo_huber)
_assert_ufunc(sp.psi)
_assert_ufunc(sp.radian)
_assert_ufunc(sp.rel_entr)
_assert_ufunc(sp.rgamma)
_assert_ufunc(sp.round)
_assert_ufunc(sp.shichi)
_assert_ufunc(sp.sici)
_assert_ufunc(sp.sindg)
_assert_ufunc(sp.smirnov)
_assert_ufunc(sp.smirnovi)
_assert_ufunc(sp.spence)
_assert_ufunc(sp.stdtr)
_assert_ufunc(sp.stdtridf)
_assert_ufunc(sp.stdtrit)
_assert_ufunc(sp.struve)
_assert_ufunc(sp.tandg)
_assert_ufunc(sp.tklmbda)
_assert_ufunc(sp.voigt_profile)
_assert_ufunc(sp.wofz)
_assert_ufunc(sp.wright_bessel)
_assert_ufunc(sp.wrightomega)
_assert_ufunc(sp.xlog1py)
_assert_ufunc(sp.xlogy)
_assert_ufunc(sp.y0)
_assert_ufunc(sp.y1)
_assert_ufunc(sp.yn)
_assert_ufunc(sp.yv)
_assert_ufunc(sp.yve)
_assert_ufunc(sp.zetac)
