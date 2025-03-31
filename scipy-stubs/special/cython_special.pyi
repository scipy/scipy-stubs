from types import CodeType
from typing import Any, Final, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import CapsuleType, LiteralString, ReadOnly

import numpy as np

_X_b: TypeAlias = bool | np.bool_ | Literal[0, 1]
_X_i: TypeAlias = int | bool | np.intp
_X_f: TypeAlias = float | int | bool | np.float64
_X_c: TypeAlias = complex | float | int | bool | np.complex128
_X_if: TypeAlias = float | int | bool | np.intp | np.float64
_X_fc: TypeAlias = complex | float | int | bool | np.float64 | np.complex128

@type_check_only
class _BaseCythonFunctionOrMethod(Protocol):
    __name__: LiteralString
    __qualname__: str  # cannot be `LiteralString` (blame typeshed)
    __module__: str  # cannot be `Literal["scipy.special.cython_special"]` (blame typeshed)

    __annotations__: dict[str, Any]
    __defaults__: tuple[()] | tuple[Literal[0]] | None
    __kwdefaults__: None  # kw-only params aren't used

    __closure__: None
    __code__: CodeType

    _is_coroutine: Literal[False]
    func_defaults: tuple[()] | tuple[Literal[0]] | None
    # like `r'See the documentation for scipy.special.{self.__name__}'`
    func_doc: str | None

    # NOTE: __call__ should be defined in the subtype

@type_check_only
class _CythonFunctionOrMethod_1f(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f) -> float | int | bool: ...

@type_check_only
class _CythonFunctionOrMethod_1c(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_1fc(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, x0: _X_f) -> float | int | bool: ...
    @overload
    def __call__(self, /, x0: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_2f(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f, x1: _X_f) -> float | int | bool: ...

@type_check_only
class _CythonFunctionOrMethod_2fc(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_fc, x1: _X_fc) -> float | int | bool | complex: ...

@type_check_only
class _CythonFunctionOrMethod_2_poly(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, x0: _X_if, x1: _X_f) -> float | int | bool: ...
    @overload
    def __call__(self, /, x0: _X_f, x1: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_2_hankel(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f, x1: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_2_spherical(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, n: _X_i, z: _X_f, derivative: _X_b = 0) -> float | int | bool: ...
    @overload
    def __call__(self, /, n: _X_i, z: _X_c, derivative: _X_b = 0) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_3f(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f, x1: _X_f, x2: _X_f) -> float | int | bool: ...

@type_check_only
class _CythonFunctionOrMethod_3fc(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_fc, x1: _X_fc, x2: _X_fc) -> float | int | bool | complex: ...

@type_check_only
class _CythonFunctionOrMethod_3_poly(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, x0: _X_if, x1: _X_f, x2: _X_f) -> float | int | bool: ...
    @overload
    def __call__(self, /, x0: _X_f, x1: _X_f, x2: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_4f(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f, x1: _X_f, x2: _X_f, x3: _X_f) -> float | int | bool: ...

@type_check_only
class _CythonFunctionOrMethod_4fc(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_fc, x1: _X_fc, x2: _X_fc, x3: _X_fc) -> float | int | bool | complex: ...

@type_check_only
class _CythonFunctionOrMethod_4_poly(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, x0: _X_if, x1: _X_f, x2: _X_f, x3: _X_f) -> float | int | bool: ...
    @overload
    def __call__(self, /, x0: _X_f, x1: _X_f, x2: _X_c, x3: _X_fc) -> complex: ...

@type_check_only
class _CApiDict(TypedDict):
    agm: ReadOnly[CapsuleType]
    bdtrik: ReadOnly[CapsuleType]
    bdtrin: ReadOnly[CapsuleType]
    bei: ReadOnly[CapsuleType]
    beip: ReadOnly[CapsuleType]
    ber: ReadOnly[CapsuleType]
    berp: ReadOnly[CapsuleType]
    besselpoly: ReadOnly[CapsuleType]
    beta: ReadOnly[CapsuleType]
    betaln: ReadOnly[CapsuleType]
    binom: ReadOnly[CapsuleType]
    boxcox: ReadOnly[CapsuleType]
    boxcox1p: ReadOnly[CapsuleType]
    btdtria: ReadOnly[CapsuleType]
    btdtrib: ReadOnly[CapsuleType]
    cbrt: ReadOnly[CapsuleType]
    chdtr: ReadOnly[CapsuleType]
    chdtrc: ReadOnly[CapsuleType]
    chdtri: ReadOnly[CapsuleType]
    chdtriv: ReadOnly[CapsuleType]
    chndtr: ReadOnly[CapsuleType]
    chndtridf: ReadOnly[CapsuleType]
    chndtrinc: ReadOnly[CapsuleType]
    chndtrix: ReadOnly[CapsuleType]
    cosdg: ReadOnly[CapsuleType]
    cosm1: ReadOnly[CapsuleType]
    cotdg: ReadOnly[CapsuleType]
    ellipe: ReadOnly[CapsuleType]
    ellipeinc: ReadOnly[CapsuleType]
    ellipj: ReadOnly[CapsuleType]
    ellipkinc: ReadOnly[CapsuleType]
    ellipkm1: ReadOnly[CapsuleType]
    ellipk: ReadOnly[CapsuleType]
    entr: ReadOnly[CapsuleType]
    erfcinv: ReadOnly[CapsuleType]
    eval_hermite: ReadOnly[CapsuleType]
    eval_hermitenorm: ReadOnly[CapsuleType]
    exp10: ReadOnly[CapsuleType]
    exp2: ReadOnly[CapsuleType]
    exprel: ReadOnly[CapsuleType]
    fdtr: ReadOnly[CapsuleType]
    fdtrc: ReadOnly[CapsuleType]
    fdtri: ReadOnly[CapsuleType]
    fdtridfd: ReadOnly[CapsuleType]
    gammainc: ReadOnly[CapsuleType]
    gammaincc: ReadOnly[CapsuleType]
    gammainccinv: ReadOnly[CapsuleType]
    gammaincinv: ReadOnly[CapsuleType]
    gammaln: ReadOnly[CapsuleType]
    gammasgn: ReadOnly[CapsuleType]
    gdtr: ReadOnly[CapsuleType]
    gdtrc: ReadOnly[CapsuleType]
    gdtria: ReadOnly[CapsuleType]
    gdtrib: ReadOnly[CapsuleType]
    gdtrix: ReadOnly[CapsuleType]
    hankel1: ReadOnly[CapsuleType]
    hankel1e: ReadOnly[CapsuleType]
    hankel2: ReadOnly[CapsuleType]
    hankel2e: ReadOnly[CapsuleType]
    huber: ReadOnly[CapsuleType]
    hyperu: ReadOnly[CapsuleType]
    i0: ReadOnly[CapsuleType]
    i0e: ReadOnly[CapsuleType]
    i1: ReadOnly[CapsuleType]
    i1e: ReadOnly[CapsuleType]
    inv_boxcox: ReadOnly[CapsuleType]
    inv_boxcox1p: ReadOnly[CapsuleType]
    it2i0k0: ReadOnly[CapsuleType]
    it2j0y0: ReadOnly[CapsuleType]
    it2struve0: ReadOnly[CapsuleType]
    itairy: ReadOnly[CapsuleType]
    iti0k0: ReadOnly[CapsuleType]
    itj0y0: ReadOnly[CapsuleType]
    itmodstruve0: ReadOnly[CapsuleType]
    itstruve0: ReadOnly[CapsuleType]
    j0: ReadOnly[CapsuleType]
    j1: ReadOnly[CapsuleType]
    k0: ReadOnly[CapsuleType]
    k0e: ReadOnly[CapsuleType]
    k1: ReadOnly[CapsuleType]
    k1e: ReadOnly[CapsuleType]
    kei: ReadOnly[CapsuleType]
    keip: ReadOnly[CapsuleType]
    kelvin: ReadOnly[CapsuleType]
    ker: ReadOnly[CapsuleType]
    kerp: ReadOnly[CapsuleType]
    kl_div: ReadOnly[CapsuleType]
    kolmogi: ReadOnly[CapsuleType]
    kolmogorov: ReadOnly[CapsuleType]
    lpmv: ReadOnly[CapsuleType]
    mathieu_a: ReadOnly[CapsuleType]
    mathieu_b: ReadOnly[CapsuleType]
    mathieu_cem: ReadOnly[CapsuleType]
    mathieu_modcem1: ReadOnly[CapsuleType]
    mathieu_modcem2: ReadOnly[CapsuleType]
    mathieu_modsem1: ReadOnly[CapsuleType]
    mathieu_modsem2: ReadOnly[CapsuleType]
    mathieu_sem: ReadOnly[CapsuleType]
    modfresnelm: ReadOnly[CapsuleType]
    modfresnelp: ReadOnly[CapsuleType]
    modstruve: ReadOnly[CapsuleType]
    nbdtrik: ReadOnly[CapsuleType]
    nbdtrin: ReadOnly[CapsuleType]
    ncfdtr: ReadOnly[CapsuleType]
    ncfdtri: ReadOnly[CapsuleType]
    ncfdtridfd: ReadOnly[CapsuleType]
    ncfdtridfn: ReadOnly[CapsuleType]
    ncfdtrinc: ReadOnly[CapsuleType]
    nctdtr: ReadOnly[CapsuleType]
    nctdtridf: ReadOnly[CapsuleType]
    nctdtrinc: ReadOnly[CapsuleType]
    nctdtrit: ReadOnly[CapsuleType]
    ndtri: ReadOnly[CapsuleType]
    nrdtrimn: ReadOnly[CapsuleType]
    nrdtrisd: ReadOnly[CapsuleType]
    obl_ang1: ReadOnly[CapsuleType]
    obl_ang1_cv: ReadOnly[CapsuleType]
    obl_cv: ReadOnly[CapsuleType]
    obl_rad1: ReadOnly[CapsuleType]
    obl_rad1_cv: ReadOnly[CapsuleType]
    obl_rad2: ReadOnly[CapsuleType]
    obl_rad2_cv: ReadOnly[CapsuleType]
    owens_t: ReadOnly[CapsuleType]
    pbdv: ReadOnly[CapsuleType]
    pbvv: ReadOnly[CapsuleType]
    pbwa: ReadOnly[CapsuleType]
    pdtr: ReadOnly[CapsuleType]
    pdtrc: ReadOnly[CapsuleType]
    pdtrik: ReadOnly[CapsuleType]
    poch: ReadOnly[CapsuleType]
    pro_ang1: ReadOnly[CapsuleType]
    pro_ang1_cv: ReadOnly[CapsuleType]
    pro_cv: ReadOnly[CapsuleType]
    pro_rad1: ReadOnly[CapsuleType]
    pro_rad1_cv: ReadOnly[CapsuleType]
    pro_rad2: ReadOnly[CapsuleType]
    pro_rad2_cv: ReadOnly[CapsuleType]
    pseudo_huber: ReadOnly[CapsuleType]
    radian: ReadOnly[CapsuleType]
    rel_entr: ReadOnly[CapsuleType]
    round: ReadOnly[CapsuleType]
    sindg: ReadOnly[CapsuleType]
    stdtr: ReadOnly[CapsuleType]
    stdtridf: ReadOnly[CapsuleType]
    stdtrit: ReadOnly[CapsuleType]
    struve: ReadOnly[CapsuleType]
    tandg: ReadOnly[CapsuleType]
    tklmbda: ReadOnly[CapsuleType]
    voigt_profile: ReadOnly[CapsuleType]
    wofz: ReadOnly[CapsuleType]
    y0: ReadOnly[CapsuleType]
    y1: ReadOnly[CapsuleType]
    zetac: ReadOnly[CapsuleType]
    wright_bessel: ReadOnly[CapsuleType]
    log_wright_bessel: ReadOnly[CapsuleType]
    ndtri_exp: ReadOnly[CapsuleType]
    __pyx_fuse_0spherical_jn: ReadOnly[CapsuleType]
    __pyx_fuse_1spherical_jn: ReadOnly[CapsuleType]
    __pyx_fuse_0spherical_yn: ReadOnly[CapsuleType]
    __pyx_fuse_1spherical_yn: ReadOnly[CapsuleType]
    __pyx_fuse_0spherical_in: ReadOnly[CapsuleType]
    __pyx_fuse_1spherical_in: ReadOnly[CapsuleType]
    __pyx_fuse_0spherical_kn: ReadOnly[CapsuleType]
    __pyx_fuse_1spherical_kn: ReadOnly[CapsuleType]
    __pyx_fuse_0airy: ReadOnly[CapsuleType]
    __pyx_fuse_1airy: ReadOnly[CapsuleType]
    __pyx_fuse_0airye: ReadOnly[CapsuleType]
    __pyx_fuse_1airye: ReadOnly[CapsuleType]
    __pyx_fuse_0bdtr: ReadOnly[CapsuleType]
    __pyx_fuse_1bdtr: ReadOnly[CapsuleType]
    __pyx_fuse_2bdtr: ReadOnly[CapsuleType]
    __pyx_fuse_0bdtrc: ReadOnly[CapsuleType]
    __pyx_fuse_1bdtrc: ReadOnly[CapsuleType]
    __pyx_fuse_2bdtrc: ReadOnly[CapsuleType]
    __pyx_fuse_0bdtri: ReadOnly[CapsuleType]
    __pyx_fuse_1bdtri: ReadOnly[CapsuleType]
    __pyx_fuse_2bdtri: ReadOnly[CapsuleType]
    __pyx_fuse_0betainc: ReadOnly[CapsuleType]
    __pyx_fuse_1betainc: ReadOnly[CapsuleType]
    __pyx_fuse_0betaincc: ReadOnly[CapsuleType]
    __pyx_fuse_1betaincc: ReadOnly[CapsuleType]
    __pyx_fuse_0betaincinv: ReadOnly[CapsuleType]
    __pyx_fuse_1betaincinv: ReadOnly[CapsuleType]
    __pyx_fuse_0betainccinv: ReadOnly[CapsuleType]
    __pyx_fuse_1betainccinv: ReadOnly[CapsuleType]
    __pyx_fuse_0dawsn: ReadOnly[CapsuleType]
    __pyx_fuse_1dawsn: ReadOnly[CapsuleType]
    __pyx_fuse_0elliprc: ReadOnly[CapsuleType]
    __pyx_fuse_1elliprc: ReadOnly[CapsuleType]
    __pyx_fuse_0elliprd: ReadOnly[CapsuleType]
    __pyx_fuse_1elliprd: ReadOnly[CapsuleType]
    __pyx_fuse_0elliprf: ReadOnly[CapsuleType]
    __pyx_fuse_1elliprf: ReadOnly[CapsuleType]
    __pyx_fuse_0elliprg: ReadOnly[CapsuleType]
    __pyx_fuse_1elliprg: ReadOnly[CapsuleType]
    __pyx_fuse_0elliprj: ReadOnly[CapsuleType]
    __pyx_fuse_1elliprj: ReadOnly[CapsuleType]
    __pyx_fuse_0erf: ReadOnly[CapsuleType]
    __pyx_fuse_1erf: ReadOnly[CapsuleType]
    __pyx_fuse_0erfc: ReadOnly[CapsuleType]
    __pyx_fuse_1erfc: ReadOnly[CapsuleType]
    __pyx_fuse_0erfcx: ReadOnly[CapsuleType]
    __pyx_fuse_1erfcx: ReadOnly[CapsuleType]
    __pyx_fuse_0erfi: ReadOnly[CapsuleType]
    __pyx_fuse_1erfi: ReadOnly[CapsuleType]
    __pyx_fuse_0erfinv: ReadOnly[CapsuleType]
    __pyx_fuse_1erfinv: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_chebyc: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_chebyc: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_chebyc: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_chebyc: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_chebyc: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_chebyc: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_chebys: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_chebys: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_chebys: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_chebys: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_chebys: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_chebys: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_gegenbauer: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_gegenbauer: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_gegenbauer: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_gegenbauer: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_gegenbauer: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_gegenbauer: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_genlaguerre: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_genlaguerre: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_genlaguerre: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_genlaguerre: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_genlaguerre: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_genlaguerre: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_laguerre: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_laguerre: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_laguerre: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_laguerre: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_laguerre: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_laguerre: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_sh_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_sh_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_sh_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_sh_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_sh_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_sh_chebyt: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_sh_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_sh_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_sh_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_sh_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_sh_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_sh_chebyu: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_sh_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_sh_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_sh_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_sh_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_sh_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_sh_jacobi: ReadOnly[CapsuleType]
    __pyx_fuse_0_0eval_sh_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_0_1eval_sh_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_1_0eval_sh_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_1_1eval_sh_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_2_0eval_sh_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_2_1eval_sh_legendre: ReadOnly[CapsuleType]
    __pyx_fuse_0exp1: ReadOnly[CapsuleType]
    __pyx_fuse_1exp1: ReadOnly[CapsuleType]
    __pyx_fuse_0expi: ReadOnly[CapsuleType]
    __pyx_fuse_1expi: ReadOnly[CapsuleType]
    __pyx_fuse_0expit: ReadOnly[CapsuleType]
    __pyx_fuse_1expit: ReadOnly[CapsuleType]
    __pyx_fuse_2expit: ReadOnly[CapsuleType]
    __pyx_fuse_0expm1: ReadOnly[CapsuleType]
    __pyx_fuse_1expm1: ReadOnly[CapsuleType]
    __pyx_fuse_0expn: ReadOnly[CapsuleType]
    __pyx_fuse_1expn: ReadOnly[CapsuleType]
    __pyx_fuse_2expn: ReadOnly[CapsuleType]
    __pyx_fuse_0fresnel: ReadOnly[CapsuleType]
    __pyx_fuse_1fresnel: ReadOnly[CapsuleType]
    __pyx_fuse_0gamma: ReadOnly[CapsuleType]
    __pyx_fuse_1gamma: ReadOnly[CapsuleType]
    __pyx_fuse_0hyp0f1: ReadOnly[CapsuleType]
    __pyx_fuse_1hyp0f1: ReadOnly[CapsuleType]
    __pyx_fuse_0hyp1f1: ReadOnly[CapsuleType]
    __pyx_fuse_1hyp1f1: ReadOnly[CapsuleType]
    __pyx_fuse_0hyp2f1: ReadOnly[CapsuleType]
    __pyx_fuse_1hyp2f1: ReadOnly[CapsuleType]
    __pyx_fuse_0iv: ReadOnly[CapsuleType]
    __pyx_fuse_1iv: ReadOnly[CapsuleType]
    __pyx_fuse_0ive: ReadOnly[CapsuleType]
    __pyx_fuse_1ive: ReadOnly[CapsuleType]
    __pyx_fuse_0jv: ReadOnly[CapsuleType]
    __pyx_fuse_1jv: ReadOnly[CapsuleType]
    __pyx_fuse_0jve: ReadOnly[CapsuleType]
    __pyx_fuse_1jve: ReadOnly[CapsuleType]
    __pyx_fuse_0kn: ReadOnly[CapsuleType]
    __pyx_fuse_1kn: ReadOnly[CapsuleType]
    __pyx_fuse_2kn: ReadOnly[CapsuleType]
    __pyx_fuse_0kv: ReadOnly[CapsuleType]
    __pyx_fuse_1kv: ReadOnly[CapsuleType]
    __pyx_fuse_0kve: ReadOnly[CapsuleType]
    __pyx_fuse_1kve: ReadOnly[CapsuleType]
    __pyx_fuse_0log1p: ReadOnly[CapsuleType]
    __pyx_fuse_1log1p: ReadOnly[CapsuleType]
    __pyx_fuse_0log_expit: ReadOnly[CapsuleType]
    __pyx_fuse_1log_expit: ReadOnly[CapsuleType]
    __pyx_fuse_2log_expit: ReadOnly[CapsuleType]
    __pyx_fuse_0log_ndtr: ReadOnly[CapsuleType]
    __pyx_fuse_1log_ndtr: ReadOnly[CapsuleType]
    __pyx_fuse_0loggamma: ReadOnly[CapsuleType]
    __pyx_fuse_1loggamma: ReadOnly[CapsuleType]
    __pyx_fuse_0logit: ReadOnly[CapsuleType]
    __pyx_fuse_1logit: ReadOnly[CapsuleType]
    __pyx_fuse_2logit: ReadOnly[CapsuleType]
    __pyx_fuse_0nbdtr: ReadOnly[CapsuleType]
    __pyx_fuse_1nbdtr: ReadOnly[CapsuleType]
    __pyx_fuse_2nbdtr: ReadOnly[CapsuleType]
    __pyx_fuse_0nbdtrc: ReadOnly[CapsuleType]
    __pyx_fuse_1nbdtrc: ReadOnly[CapsuleType]
    __pyx_fuse_2nbdtrc: ReadOnly[CapsuleType]
    __pyx_fuse_0nbdtri: ReadOnly[CapsuleType]
    __pyx_fuse_1nbdtri: ReadOnly[CapsuleType]
    __pyx_fuse_2nbdtri: ReadOnly[CapsuleType]
    __pyx_fuse_0ndtr: ReadOnly[CapsuleType]
    __pyx_fuse_1ndtr: ReadOnly[CapsuleType]
    __pyx_fuse_0pdtri: ReadOnly[CapsuleType]
    __pyx_fuse_1pdtri: ReadOnly[CapsuleType]
    __pyx_fuse_2pdtri: ReadOnly[CapsuleType]
    __pyx_fuse_0powm1: ReadOnly[CapsuleType]
    __pyx_fuse_1powm1: ReadOnly[CapsuleType]
    __pyx_fuse_0psi: ReadOnly[CapsuleType]
    __pyx_fuse_1psi: ReadOnly[CapsuleType]
    __pyx_fuse_0rgamma: ReadOnly[CapsuleType]
    __pyx_fuse_1rgamma: ReadOnly[CapsuleType]
    __pyx_fuse_0shichi: ReadOnly[CapsuleType]
    __pyx_fuse_1shichi: ReadOnly[CapsuleType]
    __pyx_fuse_0sici: ReadOnly[CapsuleType]
    __pyx_fuse_1sici: ReadOnly[CapsuleType]
    __pyx_fuse_0smirnov: ReadOnly[CapsuleType]
    __pyx_fuse_1smirnov: ReadOnly[CapsuleType]
    __pyx_fuse_2smirnov: ReadOnly[CapsuleType]
    __pyx_fuse_0smirnovi: ReadOnly[CapsuleType]
    __pyx_fuse_1smirnovi: ReadOnly[CapsuleType]
    __pyx_fuse_2smirnovi: ReadOnly[CapsuleType]
    __pyx_fuse_0spence: ReadOnly[CapsuleType]
    __pyx_fuse_1spence: ReadOnly[CapsuleType]
    __pyx_fuse_0sph_harm: ReadOnly[CapsuleType]
    __pyx_fuse_1sph_harm: ReadOnly[CapsuleType]
    __pyx_fuse_2sph_harm: ReadOnly[CapsuleType]
    __pyx_fuse_0wrightomega: ReadOnly[CapsuleType]
    __pyx_fuse_1wrightomega: ReadOnly[CapsuleType]
    __pyx_fuse_0xlog1py: ReadOnly[CapsuleType]
    __pyx_fuse_1xlog1py: ReadOnly[CapsuleType]
    __pyx_fuse_0xlogy: ReadOnly[CapsuleType]
    __pyx_fuse_1xlogy: ReadOnly[CapsuleType]
    __pyx_fuse_0yn: ReadOnly[CapsuleType]
    __pyx_fuse_1yn: ReadOnly[CapsuleType]
    __pyx_fuse_2yn: ReadOnly[CapsuleType]
    __pyx_fuse_0yv: ReadOnly[CapsuleType]
    __pyx_fuse_1yv: ReadOnly[CapsuleType]
    __pyx_fuse_0yve: ReadOnly[CapsuleType]
    __pyx_fuse_1yve: ReadOnly[CapsuleType]

class _TestDict(TypedDict): ...

__pyx_capi__: Final[_CApiDict]
__test__: Final[_TestDict]

agm: Final[_CythonFunctionOrMethod_2f]
bdtr: Final[_CythonFunctionOrMethod_3f]
bdtrc: Final[_CythonFunctionOrMethod_3f]
bdtri: Final[_CythonFunctionOrMethod_3f]
bdtrik: Final[_CythonFunctionOrMethod_3f]
bdtrin: Final[_CythonFunctionOrMethod_3f]
bei: Final[_CythonFunctionOrMethod_1f]
beip: Final[_CythonFunctionOrMethod_1f]
ber: Final[_CythonFunctionOrMethod_1f]
berp: Final[_CythonFunctionOrMethod_1f]
besselpoly: Final[_CythonFunctionOrMethod_3f]
beta: Final[_CythonFunctionOrMethod_2f]
betainc: Final[_CythonFunctionOrMethod_3f]
betaincc: Final[_CythonFunctionOrMethod_3f]
betainccinv: Final[_CythonFunctionOrMethod_3f]
betaincinv: Final[_CythonFunctionOrMethod_3f]
betaln: Final[_CythonFunctionOrMethod_2f]
binom: Final[_CythonFunctionOrMethod_2f]
boxcox: Final[_CythonFunctionOrMethod_2f]
boxcox1p: Final[_CythonFunctionOrMethod_2f]
btdtria: Final[_CythonFunctionOrMethod_3f]
btdtrib: Final[_CythonFunctionOrMethod_3f]
cbrt: Final[_CythonFunctionOrMethod_1f]
chdtr: Final[_CythonFunctionOrMethod_2f]
chdtrc: Final[_CythonFunctionOrMethod_2f]
chdtri: Final[_CythonFunctionOrMethod_2f]
chdtriv: Final[_CythonFunctionOrMethod_2f]
chndtr: Final[_CythonFunctionOrMethod_3f]
chndtridf: Final[_CythonFunctionOrMethod_3f]
chndtrinc: Final[_CythonFunctionOrMethod_3f]
chndtrix: Final[_CythonFunctionOrMethod_3f]
cosdg: Final[_CythonFunctionOrMethod_1f]
cosm1: Final[_CythonFunctionOrMethod_1f]
cotdg: Final[_CythonFunctionOrMethod_1f]
dawsn: Final[_CythonFunctionOrMethod_1fc]
ellipe: Final[_CythonFunctionOrMethod_1f]
ellipeinc: Final[_CythonFunctionOrMethod_2f]
ellipk: Final[_CythonFunctionOrMethod_1f]
ellipkinc: Final[_CythonFunctionOrMethod_2f]
ellipkm1: Final[_CythonFunctionOrMethod_1f]
elliprc: Final[_CythonFunctionOrMethod_2fc]
elliprd: Final[_CythonFunctionOrMethod_3fc]
elliprf: Final[_CythonFunctionOrMethod_3fc]
elliprg: Final[_CythonFunctionOrMethod_3fc]
elliprj: Final[_CythonFunctionOrMethod_4fc]
entr: Final[_CythonFunctionOrMethod_1f]
erf: Final[_CythonFunctionOrMethod_1fc]
erfc: Final[_CythonFunctionOrMethod_1fc]
erfcinv: Final[_CythonFunctionOrMethod_1f]
erfcx: Final[_CythonFunctionOrMethod_1fc]
erfi: Final[_CythonFunctionOrMethod_1fc]
erfinv: Final[_CythonFunctionOrMethod_1f]
eval_chebyc: Final[_CythonFunctionOrMethod_2_poly]
eval_chebys: Final[_CythonFunctionOrMethod_2_poly]
eval_chebyt: Final[_CythonFunctionOrMethod_2_poly]
eval_chebyu: Final[_CythonFunctionOrMethod_2_poly]
eval_gegenbauer: Final[_CythonFunctionOrMethod_3_poly]
eval_genlaguerre: Final[_CythonFunctionOrMethod_3_poly]
eval_hermite: Final[_CythonFunctionOrMethod_2_poly]
eval_hermitenorm: Final[_CythonFunctionOrMethod_2_poly]
eval_jacobi: Final[_CythonFunctionOrMethod_4_poly]
eval_laguerre: Final[_CythonFunctionOrMethod_2_poly]
eval_legendre: Final[_CythonFunctionOrMethod_2_poly]
eval_sh_chebyt: Final[_CythonFunctionOrMethod_2_poly]
eval_sh_chebyu: Final[_CythonFunctionOrMethod_2_poly]
eval_sh_jacobi: Final[_CythonFunctionOrMethod_4_poly]
eval_sh_legendre: Final[_CythonFunctionOrMethod_2_poly]
exp1: Final[_CythonFunctionOrMethod_1fc]
exp2: Final[_CythonFunctionOrMethod_1f]
exp10: Final[_CythonFunctionOrMethod_1f]
expi: Final[_CythonFunctionOrMethod_1fc]
expit: Final[_CythonFunctionOrMethod_1f]
expm1: Final[_CythonFunctionOrMethod_1fc]
expn: Final[_CythonFunctionOrMethod_2f]
exprel: Final[_CythonFunctionOrMethod_1f]
fdtr: Final[_CythonFunctionOrMethod_3f]
fdtrc: Final[_CythonFunctionOrMethod_3f]
fdtri: Final[_CythonFunctionOrMethod_3f]
fdtridfd: Final[_CythonFunctionOrMethod_3f]
gamma: Final[_CythonFunctionOrMethod_1fc]
gammainc: Final[_CythonFunctionOrMethod_2f]
gammaincc: Final[_CythonFunctionOrMethod_2f]
gammainccinv: Final[_CythonFunctionOrMethod_2f]
gammaincinv: Final[_CythonFunctionOrMethod_2f]
gammaln: Final[_CythonFunctionOrMethod_1f]
gammasgn: Final[_CythonFunctionOrMethod_1f]
gdtr: Final[_CythonFunctionOrMethod_3f]
gdtrc: Final[_CythonFunctionOrMethod_3f]
gdtria: Final[_CythonFunctionOrMethod_3f]
gdtrib: Final[_CythonFunctionOrMethod_3f]
gdtrix: Final[_CythonFunctionOrMethod_3f]
hankel1: Final[_CythonFunctionOrMethod_2_hankel]
hankel1e: Final[_CythonFunctionOrMethod_2_hankel]
hankel2: Final[_CythonFunctionOrMethod_2_hankel]
hankel2e: Final[_CythonFunctionOrMethod_2_hankel]
huber: Final[_CythonFunctionOrMethod_2f]
hyp0f1: Final[_CythonFunctionOrMethod_2fc]
hyp1f1: Final[_CythonFunctionOrMethod_3fc]
hyp2f1: Final[_CythonFunctionOrMethod_4fc]
hyperu: Final[_CythonFunctionOrMethod_3f]
i0: Final[_CythonFunctionOrMethod_1f]
i0e: Final[_CythonFunctionOrMethod_1f]
i1: Final[_CythonFunctionOrMethod_1f]
i1e: Final[_CythonFunctionOrMethod_1f]
inv_boxcox: Final[_CythonFunctionOrMethod_2f]
inv_boxcox1p: Final[_CythonFunctionOrMethod_2f]
it2struve0: Final[_CythonFunctionOrMethod_1f]
itmodstruve0: Final[_CythonFunctionOrMethod_1f]
itstruve0: Final[_CythonFunctionOrMethod_1f]
iv: Final[_CythonFunctionOrMethod_2fc]
ive: Final[_CythonFunctionOrMethod_2fc]
j0: Final[_CythonFunctionOrMethod_1f]
j1: Final[_CythonFunctionOrMethod_1f]
jv: Final[_CythonFunctionOrMethod_2fc]
jve: Final[_CythonFunctionOrMethod_2fc]
k0: Final[_CythonFunctionOrMethod_1f]
k0e: Final[_CythonFunctionOrMethod_1f]
k1: Final[_CythonFunctionOrMethod_1f]
k1e: Final[_CythonFunctionOrMethod_1f]
kei: Final[_CythonFunctionOrMethod_1f]
keip: Final[_CythonFunctionOrMethod_1f]
ker: Final[_CythonFunctionOrMethod_1f]
kerp: Final[_CythonFunctionOrMethod_1f]
kl_div: Final[_CythonFunctionOrMethod_2f]
kn: Final[_CythonFunctionOrMethod_2f]
kolmogi: Final[_CythonFunctionOrMethod_1f]
kolmogorov: Final[_CythonFunctionOrMethod_1f]
kv: Final[_CythonFunctionOrMethod_2fc]
kve: Final[_CythonFunctionOrMethod_2fc]
log1p: Final[_CythonFunctionOrMethod_1fc]
log_expit: Final[_CythonFunctionOrMethod_1f]
log_ndtr: Final[_CythonFunctionOrMethod_1fc]
log_wright_bessel: Final[_CythonFunctionOrMethod_3f]
loggamma: Final[_CythonFunctionOrMethod_1fc]
logit: Final[_CythonFunctionOrMethod_1f]
lpmv: Final[_CythonFunctionOrMethod_3f]
mathieu_a: Final[_CythonFunctionOrMethod_2f]
mathieu_b: Final[_CythonFunctionOrMethod_2f]
modstruve: Final[_CythonFunctionOrMethod_2f]
nbdtr: Final[_CythonFunctionOrMethod_3f]
nbdtrc: Final[_CythonFunctionOrMethod_3f]
nbdtri: Final[_CythonFunctionOrMethod_3f]
nbdtrik: Final[_CythonFunctionOrMethod_3f]
nbdtrin: Final[_CythonFunctionOrMethod_3f]
ncfdtr: Final[_CythonFunctionOrMethod_4f]
ncfdtri: Final[_CythonFunctionOrMethod_4f]
ncfdtridfd: Final[_CythonFunctionOrMethod_4f]
ncfdtridfn: Final[_CythonFunctionOrMethod_4f]
ncfdtrinc: Final[_CythonFunctionOrMethod_4f]
nctdtr: Final[_CythonFunctionOrMethod_3f]
nctdtridf: Final[_CythonFunctionOrMethod_3f]
nctdtrinc: Final[_CythonFunctionOrMethod_3f]
nctdtrit: Final[_CythonFunctionOrMethod_3f]
ndtr: Final[_CythonFunctionOrMethod_1fc]
ndtri: Final[_CythonFunctionOrMethod_1f]
ndtri_exp: Final[_CythonFunctionOrMethod_1f]
nrdtrimn: Final[_CythonFunctionOrMethod_3f]
nrdtrisd: Final[_CythonFunctionOrMethod_3f]
obl_cv: Final[_CythonFunctionOrMethod_3f]
owens_t: Final[_CythonFunctionOrMethod_2f]
pdtr: Final[_CythonFunctionOrMethod_2f]
pdtrc: Final[_CythonFunctionOrMethod_2f]
pdtri: Final[_CythonFunctionOrMethod_2f]
pdtrik: Final[_CythonFunctionOrMethod_2f]
poch: Final[_CythonFunctionOrMethod_2f]
powm1: Final[_CythonFunctionOrMethod_2f]
pro_cv: Final[_CythonFunctionOrMethod_3f]
pseudo_huber: Final[_CythonFunctionOrMethod_2f]
psi: Final[_CythonFunctionOrMethod_1fc]
radian: Final[_CythonFunctionOrMethod_3f]
rel_entr: Final[_CythonFunctionOrMethod_2f]
rgamma: Final[_CythonFunctionOrMethod_1fc]
round: Final[_CythonFunctionOrMethod_1f]
sindg: Final[_CythonFunctionOrMethod_1f]
smirnov: Final[_CythonFunctionOrMethod_2f]
smirnovi: Final[_CythonFunctionOrMethod_2f]
spence: Final[_CythonFunctionOrMethod_1fc]
sph_harm: Final[_CythonFunctionOrMethod_4fc]
stdtr: Final[_CythonFunctionOrMethod_2f]
stdtridf: Final[_CythonFunctionOrMethod_2f]
stdtrit: Final[_CythonFunctionOrMethod_2f]
struve: Final[_CythonFunctionOrMethod_2f]
tandg: Final[_CythonFunctionOrMethod_1f]
tklmbda: Final[_CythonFunctionOrMethod_2f]
voigt_profile: Final[_CythonFunctionOrMethod_3f]
wofz: Final[_CythonFunctionOrMethod_1c]
wright_bessel: Final[_CythonFunctionOrMethod_3f]
wrightomega: Final[_CythonFunctionOrMethod_1fc]
xlog1py: Final[_CythonFunctionOrMethod_2fc]
xlogy: Final[_CythonFunctionOrMethod_2fc]
y0: Final[_CythonFunctionOrMethod_1f]
y1: Final[_CythonFunctionOrMethod_1f]
yn: Final[_CythonFunctionOrMethod_2f]
yv: Final[_CythonFunctionOrMethod_2fc]
yve: Final[_CythonFunctionOrMethod_2fc]
zetac: Final[_CythonFunctionOrMethod_1f]

spherical_in: Final[_CythonFunctionOrMethod_2_spherical]
spherical_jn: Final[_CythonFunctionOrMethod_2_spherical]
spherical_kn: Final[_CythonFunctionOrMethod_2_spherical]
spherical_yn: Final[_CythonFunctionOrMethod_2_spherical]
