from typing import Any, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import (
    BroydenFirst,
    InverseJacobian,
    KrylovJacobian,
    NoConvergence,
    anderson,
    broyden1,
    broyden2,
    diagbroyden,
    excitingmixing,
    linearmixing,
    newton_krylov,
)

_Inexact: TypeAlias = np.float32 | np.float64 | np.complex64 | np.complex128
_InexactND: TypeAlias = onp.ArrayND[_Inexact]

def _F(x: onp.ArrayND[np.float64]) -> float: ...

###
# NoConvergence

_exc: NoConvergence
assert_type(_exc, NoConvergence)
_: Exception = _exc

###
# BroydenFirst

_bf: BroydenFirst[Any]
assert_type(BroydenFirst(), BroydenFirst[_Inexact])
assert_type(_bf.todense(), onp.Array2D[Any])

###
# InverseJacobian

_ij: InverseJacobian[np.float64]
assert_type(_ij.shape, tuple[int, int])
assert_type(_ij.dtype, np.dtype[np.float64])

###
# KrylovJacobian

_kj: KrylovJacobian[Any]
assert_type(KrylovJacobian(), KrylovJacobian[_Inexact])

###
# broyden1, broyden2, anderson, linearmixing, diagbroyden, excitingmixing, newton_krylov

assert_type(broyden1(_F, [1.0, 2.0]), _InexactND)
assert_type(broyden2(_F, [1.0, 2.0]), _InexactND)
assert_type(anderson(_F, [1.0, 2.0]), _InexactND)
assert_type(linearmixing(_F, [1.0, 2.0]), _InexactND)
assert_type(diagbroyden(_F, [1.0, 2.0]), _InexactND)
assert_type(excitingmixing(_F, [1.0, 2.0]), _InexactND)
assert_type(newton_krylov(_F, [1.0, 2.0]), _InexactND)
