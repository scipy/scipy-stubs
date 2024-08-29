from typing import TypeAlias

import numpy as np

import numpy.typing as npt

__all__ = ["spherical_in", "spherical_jn", "spherical_kn", "spherical_yn"]

_Scalar_fc: TypeAlias = np.float64 | np.complex128

def spherical_jn(n: npt.ArrayLike, z: npt.ArrayLike, derivative: bool = ...) -> _Scalar_fc | npt.NDArray[_Scalar_fc]: ...
def spherical_yn(n: npt.ArrayLike, z: npt.ArrayLike, derivative: bool = ...) -> _Scalar_fc | npt.NDArray[_Scalar_fc]: ...
def spherical_in(n: npt.ArrayLike, z: npt.ArrayLike, derivative: bool = ...) -> _Scalar_fc | npt.NDArray[_Scalar_fc]: ...
def spherical_kn(n: npt.ArrayLike, z: npt.ArrayLike, derivative: bool = ...) -> _Scalar_fc | npt.NDArray[_Scalar_fc]: ...
