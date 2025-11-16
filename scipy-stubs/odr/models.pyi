# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing import Any, TypeVar
from typing_extensions import deprecated

import numpy as np

from ._models import exponential, multilinear, quadratic, unilinear
from ._odrpack import Model as _Model

__all__ = ["Model", "exponential", "multilinear", "polynomial", "quadratic", "unilinear"]

_ScalarType = TypeVar("_ScalarType", bound=np.generic)

@deprecated("will be removed in SciPy v2.0.0")
class Model(_Model[_ScalarType]): ...

@deprecated("will be removed in SciPy v2.0.0")
def polynomial(order: object) -> Model[np.floating[Any]]: ...  # pyright: ignore[reportDeprecated]
