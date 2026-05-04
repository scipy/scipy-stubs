from typing import Literal as L, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc
from optype.test import assert_subtype

import scipy.special as sp

_Float32ND: TypeAlias = onp.ArrayND[np.float32]
_Float64ND: TypeAlias = onp.ArrayND[np.float64]
_Complex64ND: TypeAlias = onp.ArrayND[np.complex64]
_Complex128ND: TypeAlias = onp.ArrayND[np.complex128]
_ErrOption: TypeAlias = L["ignored", "warn", "raise"]

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
assert_type(sp.geterr()["singular"], _ErrOption)
assert_type(sp.geterr()["underflow"], _ErrOption)
assert_type(sp.seterr()["overflow"], _ErrOption)
assert_type(sp.seterr(all="warn", singular="raise", underflow="raise")["singular"], _ErrOption)
assert_type(sp.errstate(all="warn", singular="raise"), sp.errstate)

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
# pyrefly: ignore [no-matching-overload]
sp.cbrt(_c16)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
sp.cbrt(_c16_nd)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(sp.cbrt(False), np.float64)
assert_type(sp.cbrt([False]), _Float64ND)
assert_type(sp.cbrt(0), np.float64)
assert_type(sp.cbrt([0]), _Float64ND)
assert_type(sp.cbrt(0.0), np.float64)
assert_type(sp.cbrt([0.0]), _Float64ND)
# pyrefly: ignore [no-matching-overload]
sp.cbrt(0j)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
sp.cbrt([0j])  # pyright: ignore[reportArgumentType, reportCallIssue]  # pyrefly: ignore[no-matching-overload]
assert_type(sp.cbrt.at(_b1_nd, _i), None)
assert_type(sp.cbrt.at(_f8_nd, _i), None)
# pyrefly: ignore [bad-argument-type]
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
# pyrefly: ignore [no-matching-overload]
sp.logit(_c16)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
sp.logit(_c16_nd)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(sp.logit(0), np.float64)
assert_type(sp.logit([0]), _Float64ND)
assert_type(sp.logit(0.0), np.float64)
assert_type(sp.logit([0.0]), _Float64ND)
assert_type(sp.logit.at(_b1_nd, _i), None)
assert_type(sp.logit.at(_f8_nd, _i), None)
# pyrefly: ignore [bad-argument-type]
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

assert_subtype[np.ufunc](sp.agm)
assert_subtype[np.ufunc](sp.airy)
assert_subtype[np.ufunc](sp.airye)
assert_subtype[np.ufunc](sp.bdtr)
assert_subtype[np.ufunc](sp.bdtrc)
assert_subtype[np.ufunc](sp.bdtri)
assert_subtype[np.ufunc](sp.bdtrik)
assert_subtype[np.ufunc](sp.bdtrin)
assert_subtype[np.ufunc](sp.bei)
assert_subtype[np.ufunc](sp.beip)
assert_subtype[np.ufunc](sp.ber)
assert_subtype[np.ufunc](sp.berp)
assert_subtype[np.ufunc](sp.besselpoly)
assert_subtype[np.ufunc](sp.beta)
assert_subtype[np.ufunc](sp.betainc)
assert_subtype[np.ufunc](sp.betaincc)
assert_subtype[np.ufunc](sp.betainccinv)
assert_subtype[np.ufunc](sp.betaincinv)
assert_subtype[np.ufunc](sp.betaln)
assert_subtype[np.ufunc](sp.binom)
assert_subtype[np.ufunc](sp.boxcox)
assert_subtype[np.ufunc](sp.boxcox1p)
assert_subtype[np.ufunc](sp.btdtria)
assert_subtype[np.ufunc](sp.btdtrib)
assert_subtype[np.ufunc](sp.cbrt)
assert_subtype[np.ufunc](sp.chdtr)
assert_subtype[np.ufunc](sp.chdtrc)
assert_subtype[np.ufunc](sp.chdtri)
assert_subtype[np.ufunc](sp.chdtriv)
assert_subtype[np.ufunc](sp.chndtr)
assert_subtype[np.ufunc](sp.chndtridf)
assert_subtype[np.ufunc](sp.chndtrinc)
assert_subtype[np.ufunc](sp.chndtrix)
assert_subtype[np.ufunc](sp.cosdg)
assert_subtype[np.ufunc](sp.cosm1)
assert_subtype[np.ufunc](sp.cotdg)
assert_subtype[np.ufunc](sp.dawsn)
assert_subtype[np.ufunc](sp.ellipe)
assert_subtype[np.ufunc](sp.ellipeinc)
assert_subtype[np.ufunc](sp.ellipj)
assert_subtype[np.ufunc](sp.ellipk)
assert_subtype[np.ufunc](sp.ellipkinc)
assert_subtype[np.ufunc](sp.ellipkm1)
assert_subtype[np.ufunc](sp.elliprc)
assert_subtype[np.ufunc](sp.elliprd)
assert_subtype[np.ufunc](sp.elliprf)
assert_subtype[np.ufunc](sp.elliprg)
assert_subtype[np.ufunc](sp.elliprj)
assert_subtype[np.ufunc](sp.entr)
assert_subtype[np.ufunc](sp.erf)
assert_subtype[np.ufunc](sp.erfc)
assert_subtype[np.ufunc](sp.erfcinv)
assert_subtype[np.ufunc](sp.erfcx)
assert_subtype[np.ufunc](sp.erfi)
assert_subtype[np.ufunc](sp.erfinv)
assert_subtype[np.ufunc](sp.eval_chebyc)
assert_subtype[np.ufunc](sp.eval_chebys)
assert_subtype[np.ufunc](sp.eval_chebyt)
assert_subtype[np.ufunc](sp.eval_chebyu)
assert_subtype[np.ufunc](sp.eval_gegenbauer)
assert_subtype[np.ufunc](sp.eval_genlaguerre)
assert_subtype[np.ufunc](sp.eval_hermite)
assert_subtype[np.ufunc](sp.eval_hermitenorm)
assert_subtype[np.ufunc](sp.eval_jacobi)
assert_subtype[np.ufunc](sp.eval_laguerre)
assert_subtype[np.ufunc](sp.eval_legendre)
assert_subtype[np.ufunc](sp.eval_sh_chebyt)
assert_subtype[np.ufunc](sp.eval_sh_chebyu)
assert_subtype[np.ufunc](sp.eval_sh_jacobi)
assert_subtype[np.ufunc](sp.eval_sh_legendre)
assert_subtype[np.ufunc](sp.exp1)
assert_subtype[np.ufunc](sp.exp2)
assert_subtype[np.ufunc](sp.exp10)
assert_subtype[np.ufunc](sp.expi)
assert_subtype[np.ufunc](sp.expit)
assert_subtype[np.ufunc](sp.expm1)
assert_subtype[np.ufunc](sp.expn)
assert_subtype[np.ufunc](sp.exprel)
assert_subtype[np.ufunc](sp.fdtr)
assert_subtype[np.ufunc](sp.fdtrc)
assert_subtype[np.ufunc](sp.fdtri)
assert_subtype[np.ufunc](sp.fdtridfd)
assert_subtype[np.ufunc](sp.fresnel)
assert_subtype[np.ufunc](sp.gamma)
assert_subtype[np.ufunc](sp.gammainc)
assert_subtype[np.ufunc](sp.gammaincc)
assert_subtype[np.ufunc](sp.gammainccinv)
assert_subtype[np.ufunc](sp.gammaincinv)
assert_subtype[np.ufunc](sp.gammaln)
assert_subtype[np.ufunc](sp.gammasgn)
assert_subtype[np.ufunc](sp.gdtr)
assert_subtype[np.ufunc](sp.gdtrc)
assert_subtype[np.ufunc](sp.gdtria)
assert_subtype[np.ufunc](sp.gdtrib)
assert_subtype[np.ufunc](sp.gdtrix)
assert_subtype[np.ufunc](sp.hankel1)
assert_subtype[np.ufunc](sp.hankel1e)
assert_subtype[np.ufunc](sp.hankel2)
assert_subtype[np.ufunc](sp.hankel2e)
assert_subtype[np.ufunc](sp.huber)
assert_subtype[np.ufunc](sp.hyp0f1)
assert_subtype[np.ufunc](sp.hyp1f1)
assert_subtype[np.ufunc](sp.hyp2f1)
assert_subtype[np.ufunc](sp.hyperu)
assert_subtype[np.ufunc](sp.i0)
assert_subtype[np.ufunc](sp.i0e)
assert_subtype[np.ufunc](sp.i1)
assert_subtype[np.ufunc](sp.i1e)
assert_subtype[np.ufunc](sp.inv_boxcox)
assert_subtype[np.ufunc](sp.inv_boxcox1p)
assert_subtype[np.ufunc](sp.it2i0k0)
assert_subtype[np.ufunc](sp.it2j0y0)
assert_subtype[np.ufunc](sp.it2struve0)
assert_subtype[np.ufunc](sp.itairy)
assert_subtype[np.ufunc](sp.iti0k0)
assert_subtype[np.ufunc](sp.itj0y0)
assert_subtype[np.ufunc](sp.itmodstruve0)
assert_subtype[np.ufunc](sp.itstruve0)
assert_subtype[np.ufunc](sp.iv)
assert_subtype[np.ufunc](sp.ive)
assert_subtype[np.ufunc](sp.j0)
assert_subtype[np.ufunc](sp.j1)
assert_subtype[np.ufunc](sp.jn)
assert_subtype[np.ufunc](sp.jv)
assert_subtype[np.ufunc](sp.jve)
assert_subtype[np.ufunc](sp.k0)
assert_subtype[np.ufunc](sp.k0e)
assert_subtype[np.ufunc](sp.k1)
assert_subtype[np.ufunc](sp.k1e)
assert_subtype[np.ufunc](sp.kei)
assert_subtype[np.ufunc](sp.keip)
assert_subtype[np.ufunc](sp.kelvin)
assert_subtype[np.ufunc](sp.ker)
assert_subtype[np.ufunc](sp.kerp)
assert_subtype[np.ufunc](sp.kl_div)
assert_subtype[np.ufunc](sp.kn)
assert_subtype[np.ufunc](sp.kolmogi)
assert_subtype[np.ufunc](sp.kolmogorov)
assert_subtype[np.ufunc](sp.kv)
assert_subtype[np.ufunc](sp.kve)
assert_subtype[np.ufunc](sp.log1p)
assert_subtype[np.ufunc](sp.log_expit)
assert_subtype[np.ufunc](sp.log_ndtr)
assert_subtype[np.ufunc](sp.log_wright_bessel)
assert_subtype[np.ufunc](sp.loggamma)
assert_subtype[np.ufunc](sp.logit)
assert_subtype[np.ufunc](sp.lpmv)
assert_subtype[np.ufunc](sp.mathieu_a)
assert_subtype[np.ufunc](sp.mathieu_b)
assert_subtype[np.ufunc](sp.mathieu_cem)
assert_subtype[np.ufunc](sp.mathieu_modcem1)
assert_subtype[np.ufunc](sp.mathieu_modcem2)
assert_subtype[np.ufunc](sp.mathieu_modsem1)
assert_subtype[np.ufunc](sp.mathieu_modsem2)
assert_subtype[np.ufunc](sp.mathieu_sem)
assert_subtype[np.ufunc](sp.modfresnelm)
assert_subtype[np.ufunc](sp.modfresnelp)
assert_subtype[np.ufunc](sp.modstruve)
assert_subtype[np.ufunc](sp.nbdtr)
assert_subtype[np.ufunc](sp.nbdtrc)
assert_subtype[np.ufunc](sp.nbdtri)
assert_subtype[np.ufunc](sp.nbdtrik)
assert_subtype[np.ufunc](sp.nbdtrin)
assert_subtype[np.ufunc](sp.ncfdtr)
assert_subtype[np.ufunc](sp.ncfdtri)
assert_subtype[np.ufunc](sp.ncfdtridfd)
assert_subtype[np.ufunc](sp.ncfdtridfn)
assert_subtype[np.ufunc](sp.ncfdtrinc)
assert_subtype[np.ufunc](sp.nctdtr)
assert_subtype[np.ufunc](sp.nctdtridf)
assert_subtype[np.ufunc](sp.nctdtrinc)
assert_subtype[np.ufunc](sp.nctdtrit)
assert_subtype[np.ufunc](sp.ndtr)
assert_subtype[np.ufunc](sp.ndtri)
assert_subtype[np.ufunc](sp.ndtri_exp)
assert_subtype[np.ufunc](sp.nrdtrimn)
assert_subtype[np.ufunc](sp.nrdtrisd)
assert_subtype[np.ufunc](sp.obl_ang1)
assert_subtype[np.ufunc](sp.obl_ang1_cv)
assert_subtype[np.ufunc](sp.obl_cv)
assert_subtype[np.ufunc](sp.obl_rad1)
assert_subtype[np.ufunc](sp.obl_rad1_cv)
assert_subtype[np.ufunc](sp.obl_rad2)
assert_subtype[np.ufunc](sp.obl_rad2_cv)
assert_subtype[np.ufunc](sp.owens_t)
assert_subtype[np.ufunc](sp.pbdv)
assert_subtype[np.ufunc](sp.pbvv)
assert_subtype[np.ufunc](sp.pbwa)
assert_subtype[np.ufunc](sp.pdtr)
assert_subtype[np.ufunc](sp.pdtrc)
assert_subtype[np.ufunc](sp.pdtri)
assert_subtype[np.ufunc](sp.pdtrik)
assert_subtype[np.ufunc](sp.poch)
assert_subtype[np.ufunc](sp.powm1)
assert_subtype[np.ufunc](sp.pro_ang1)
assert_subtype[np.ufunc](sp.pro_ang1_cv)
assert_subtype[np.ufunc](sp.pro_cv)
assert_subtype[np.ufunc](sp.pro_rad1)
assert_subtype[np.ufunc](sp.pro_rad1_cv)
assert_subtype[np.ufunc](sp.pro_rad2)
assert_subtype[np.ufunc](sp.pro_rad2_cv)
assert_subtype[np.ufunc](sp.pseudo_huber)
assert_subtype[np.ufunc](sp.psi)
assert_subtype[np.ufunc](sp.radian)
assert_subtype[np.ufunc](sp.rel_entr)
assert_subtype[np.ufunc](sp.rgamma)
assert_subtype[np.ufunc](sp.round)
assert_subtype[np.ufunc](sp.shichi)
assert_subtype[np.ufunc](sp.sici)
assert_subtype[np.ufunc](sp.sindg)
assert_subtype[np.ufunc](sp.smirnov)
assert_subtype[np.ufunc](sp.smirnovi)
assert_subtype[np.ufunc](sp.spence)
assert_subtype[np.ufunc](sp.stdtr)
assert_subtype[np.ufunc](sp.stdtridf)
assert_subtype[np.ufunc](sp.stdtrit)
assert_subtype[np.ufunc](sp.struve)
assert_subtype[np.ufunc](sp.tandg)
assert_subtype[np.ufunc](sp.tklmbda)
assert_subtype[np.ufunc](sp.voigt_profile)
assert_subtype[np.ufunc](sp.wofz)
assert_subtype[np.ufunc](sp.wright_bessel)
assert_subtype[np.ufunc](sp.wrightomega)
assert_subtype[np.ufunc](sp.xlog1py)
assert_subtype[np.ufunc](sp.xlogy)
assert_subtype[np.ufunc](sp.y0)
assert_subtype[np.ufunc](sp.y1)
assert_subtype[np.ufunc](sp.yn)
assert_subtype[np.ufunc](sp.yv)
assert_subtype[np.ufunc](sp.yve)
assert_subtype[np.ufunc](sp.zetac)
assert_subtype[np.ufunc]
