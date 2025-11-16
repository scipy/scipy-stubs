from typing import Any, Final, TypeVar, type_check_only

import numpy as np
import optype.numpy as onp

from ._odrpack import Model

__all__ = ["Model", "exponential", "multilinear", "polynomial", "quadratic", "unilinear"]

_ScalarType = TypeVar("_ScalarType", bound=np.generic)

@type_check_only
class _NamedModel(Model[_ScalarType]):
    name: Final[str]
    equ: Final[str]
    TeXequ: Final[str]

@type_check_only
class _SimpleModel(_NamedModel[_ScalarType]):
    def __init__(self, /) -> None: ...

###

class _MultilinearModel(_SimpleModel[_ScalarType]): ...
class _ExponentialModel(_SimpleModel[_ScalarType]): ...
class _UnilinearModel(_SimpleModel[_ScalarType]): ...
class _QuadraticModel(_SimpleModel[_ScalarType]): ...

def polynomial(order: onp.ToInt | onp.ToInt1D) -> _NamedModel[np.floating[Any]]: ...

multilinear: Final[_MultilinearModel[np.floating[Any]]] = ...
exponential: Final[_ExponentialModel[np.floating[Any]]] = ...
unilinear: Final[_UnilinearModel[np.floating[Any]]] = ...
quadratic: Final[_QuadraticModel[np.floating[Any]]] = ...
