from typing_extensions import deprecated

from . import _rotation

@deprecated("will be removed in SciPy v2.0.0")
class Rotation(_rotation.Rotation): ...

@deprecated("will be removed in SciPy v2.0.0")
class Slerp(_rotation.Slerp): ...
