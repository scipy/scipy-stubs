from typing import Final, type_check_only

import optype.numpy as onp

from ._odrpack import Model

__all__ = ["Model", "exponential", "multilinear", "polynomial", "quadratic", "unilinear"]

@type_check_only
class _NamedModel(Model):
    name: Final[str]
    equ: Final[str]
    TeXequ: Final[str]

@type_check_only
class _SimpleModel(_NamedModel):
    def __init__(self, /) -> None: ...

###

class _MultilinearModel(_SimpleModel): ...
class _ExponentialModel(_SimpleModel): ...
class _UnilinearModel(_SimpleModel): ...
class _QuadraticModel(_SimpleModel): ...

def polynomial(order: onp.ToInt | onp.ToInt1D) -> _NamedModel: ...

multilinear: Final[_MultilinearModel] = ...
exponential: Final[_ExponentialModel] = ...
unilinear: Final[_UnilinearModel] = ...
quadratic: Final[_QuadraticModel] = ...
