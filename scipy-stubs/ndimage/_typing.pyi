from typing import Any, TypeAlias

import numpy as np
import optype.numpy as onp

__all__ = "_ComplexArrayOut", "_FloatArrayOut", "_ScalarArrayOut"

_FloatArrayOut: TypeAlias = onp.ArrayND[np.float64 | np.float32]
_ComplexArrayOut: TypeAlias = onp.ArrayND[np.complex128 | np.float64 | np.complex64 | np.float32]
_ScalarArrayOut: TypeAlias = onp.ArrayND[np.number[Any] | np.bool_]
