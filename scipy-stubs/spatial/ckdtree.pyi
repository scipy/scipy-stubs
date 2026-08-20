# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.
from typing_extensions import deprecated, disjoint_base

from ._ckdtree import cKDTree as _cKDTree

__all__ = ["cKDTree"]

@deprecated("will be removed in SciPy v2.0.0")
@disjoint_base
class cKDTree(_cKDTree): ...
