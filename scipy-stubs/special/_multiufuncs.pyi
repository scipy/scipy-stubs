# pyright: reportIncompatibleMethodOverride=false

# mypy reports false positive `overload-overlap` errors on `numpy<2.5`
# mypy: disable-error-code=overload-overlap

from collections.abc import Callable, Iterable
from typing import Any, Final, Literal as L, overload, override, type_check_only

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = [
    "assoc_legendre_p",
    "assoc_legendre_p_all",
    "legendre_p",
    "legendre_p_all",
    "sph_harm_y",
    "sph_harm_y_all",
    "sph_legendre_p",
    "sph_legendre_p_all",
]

###

type _AsF64 = float | npc.floating64 | npc.integer | np.bool
type _AsF64ND = onp.ToArrayND[_AsF64, npc.floating64 | npc.integer | np.bool]
type _AsF64_D = _AsF64 | _AsF64ND

type _ToJustFloat32_D = onp.ToJustFloat32 | onp.ToJustFloat32_ND

type _ToInt_D = onp.ToInt | onp.ToIntND
type _ToFloat_D = onp.ToFloat | onp.ToFloatND
type _ToComplex_D = onp.ToComplex | onp.ToComplexND

type _ArrayMin1D[ScalarT: np.generic] = onp.ArrayND[ScalarT, tuple[int, *tuple[Any, ...]]]
type _ArrayMin2D[ScalarT: np.generic] = onp.ArrayND[ScalarT, tuple[int, int, *tuple[Any, ...]]]
type _ArrayMin3D[ScalarT: np.generic] = onp.ArrayND[ScalarT, tuple[int, int, int, *tuple[Any, ...]]]

type _Branch = L[2, 3]
type _Branch_D = _Branch | onp.SequenceND[_Branch] | onp.CanArrayND[npc.integer]

###

class MultiUFunc:  # undocumented
    @property
    @override
    # pyrefly: ignore[bad-override]
    def __doc__(self, /) -> str | None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]

    #
    def __init__(
        self,
        /,
        ufunc_or_ufuncs: Callable[..., object] | Iterable[Callable[..., object]],
        name: str | None = None,
        doc: str | None = None,
        *,
        force_complex_output: bool = False,
        **default_kwargs: object,
    ) -> None: ...
    def __call__(self, /, *args: Any, **kwargs: Any) -> Any: ...

@type_check_only
class _LegendreP(MultiUFunc):
    @overload  # 0d, 0d +f64
    def __call__(self, /, n: int, z: _AsF64, *, diff_n: int = 0) -> onp.Array1D[np.float64]: ...
    @overload  # 0d, 0d ~f32
    def __call__(self, /, n: int, z: onp.ToJustFloat32, *, diff_n: int = 0) -> onp.Array1D[np.float32]: ...
    @overload  # 0d, >0d +f64
    def __call__(self, /, n: int, z: _AsF64ND, *, diff_n: int = 0) -> _ArrayMin2D[np.float64]: ...
    @overload  # 0d, >0d ~f32
    def __call__(self, /, n: int, z: onp.ToJustFloat32_ND, *, diff_n: int = 0) -> _ArrayMin2D[np.float32]: ...
    @overload  # >0d, >=0d +f64
    def __call__(self, /, n: onp.ToIntND, z: _AsF64_D, *, diff_n: int = 0) -> _ArrayMin2D[np.float64]: ...
    @overload  # >0d, >=0d ~f32
    def __call__(self, /, n: onp.ToIntND, z: _ToJustFloat32_D, *, diff_n: int = 0) -> _ArrayMin2D[np.float32]: ...
    @overload  # fallback
    def __call__(self, /, n: _ToInt_D, z: _ToFloat_D, *, diff_n: int = 0) -> onp.ArrayND[Any]: ...

@type_check_only
class _LegendrePAll(MultiUFunc):
    @overload  # 0d +f64
    def __call__(self, /, n: int, z: _AsF64, *, diff_n: int = 0) -> onp.Array2D[np.float64]: ...
    @overload  # 0d ~c128
    def __call__(self, /, n: int, z: onp.ToJustComplex128, *, diff_n: int = 0) -> onp.Array2D[np.complex128]: ...
    @overload  # 0d T:f32|c64
    def __call__[InexactT: npc.inexact32](self, /, n: int, z: InexactT, *, diff_n: int = 0) -> onp.Array2D[InexactT]: ...
    @overload  # >0d +f64
    def __call__(self, /, n: int, z: _AsF64ND, *, diff_n: int = 0) -> _ArrayMin3D[np.float64]: ...
    @overload  # >0d ~c128
    def __call__(self, /, n: int, z: onp.ToJustComplex128_ND, *, diff_n: int = 0) -> _ArrayMin3D[np.complex128]: ...
    @overload  # >0d T:f32|c64
    def __call__[InexactT: npc.inexact32](
        self, /, n: int, z: onp.ToArrayND[InexactT, InexactT], *, diff_n: int = 0
    ) -> _ArrayMin3D[InexactT]: ...
    @overload  # fallback
    def __call__(self, /, n: int, z: _ToComplex_D, *, diff_n: int = 0) -> onp.ArrayND[Any]: ...

@type_check_only
class _AssocLegendreP(MultiUFunc):
    @overload  # +f64
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, z: _AsF64_D, *, branch_cut: _Branch_D = 2, norm: bool = False, diff_n: int = 0
    ) -> _ArrayMin1D[np.float64]: ...
    @overload  # ~c128
    def __call__(
        self,
        /,
        n: _ToInt_D,
        m: _ToInt_D,
        z: onp.ToJustComplex128 | onp.ToJustComplex128_ND,
        *,
        branch_cut: _Branch_D = 2,
        norm: bool = False,
        diff_n: int = 0,
    ) -> _ArrayMin1D[np.complex128]: ...
    @overload  # T:f32|c64
    def __call__[InexactT: npc.inexact32](
        self,
        /,
        n: _ToInt_D,
        m: _ToInt_D,
        z: InexactT | onp.ToArrayND[InexactT, InexactT],
        *,
        branch_cut: _Branch_D = 2,
        norm: bool = False,
        diff_n: int = 0,
    ) -> _ArrayMin1D[InexactT]: ...
    @overload  # fallback
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, z: _ToComplex_D, *, branch_cut: _Branch_D = 2, norm: bool = False, diff_n: int = 0
    ) -> onp.ArrayND[Any]: ...

@type_check_only
class _AssocLegendrePAll(MultiUFunc):
    @overload  # 0d +f64
    def __call__(
        self, /, n: int, m: int, z: _AsF64, *, branch_cut: _Branch = 2, norm: bool = False, diff_n: int = 0
    ) -> onp.Array3D[np.float64]: ...
    @overload  # 0d ~c128
    def __call__(
        self, /, n: int, m: int, z: onp.ToJustComplex128, *, branch_cut: _Branch = 2, norm: bool = False, diff_n: int = 0
    ) -> onp.Array3D[np.complex128]: ...
    @overload  # 0d T:f32|c64
    def __call__[InexactT: npc.inexact32](
        self, /, n: int, m: int, z: InexactT, *, branch_cut: _Branch = 2, norm: bool = False, diff_n: int = 0
    ) -> onp.Array3D[InexactT]: ...
    @overload  # >=0d +f64
    def __call__(
        self, /, n: int, m: int, z: _AsF64_D, *, branch_cut: _Branch_D = 2, norm: bool = False, diff_n: int = 0
    ) -> _ArrayMin3D[np.float64]: ...
    @overload  # >=0d ~c128
    def __call__(
        self, /, n: int, m: int, z: onp.ToJustComplex128_ND, *, branch_cut: _Branch_D = 2, norm: bool = False, diff_n: int = 0
    ) -> _ArrayMin3D[np.complex128]: ...
    @overload  # >=0d T:f32|c64
    def __call__[InexactT: npc.inexact32](
        self,
        /,
        n: int,
        m: int,
        z: InexactT | onp.ToArrayND[InexactT, InexactT],
        *,
        branch_cut: _Branch_D = 2,
        norm: bool = False,
        diff_n: int = 0,
    ) -> _ArrayMin3D[InexactT]: ...
    @overload  # fallback
    def __call__(
        self, /, n: int, m: int, z: _ToComplex_D, *, branch_cut: _Branch_D = 2, norm: bool = False, diff_n: int = 0
    ) -> onp.ArrayND[Any]: ...

@type_check_only
class _SphLegendreP(MultiUFunc):
    @overload  # 0d, 0d, 0d
    def __call__(self, /, n: int, m: int, theta: onp.ToFloat, *, diff_n: int = 0) -> onp.Array1D[np.float64]: ...
    @overload  # >=0d, >=0d, >0d
    def __call__(self, /, n: _ToInt_D, m: _ToInt_D, theta: onp.ToFloatND, *, diff_n: int = 0) -> _ArrayMin2D[np.float64]: ...
    @overload  # >=0d, >0d, >=0d
    def __call__(self, /, n: _ToInt_D, m: onp.ToIntND, theta: _ToFloat_D, *, diff_n: int = 0) -> _ArrayMin2D[np.float64]: ...
    @overload  # >0d, >=0d, >=0d
    def __call__(self, /, n: onp.ToIntND, m: _ToInt_D, theta: _ToFloat_D, *, diff_n: int = 0) -> _ArrayMin2D[np.float64]: ...

@type_check_only
class _SphLegendrePAll(MultiUFunc):
    @overload  # 0d +f64
    def __call__(self, /, n: int, m: int, theta: _AsF64, *, diff_n: int = 0) -> onp.Array3D[np.float64]: ...
    @overload  # 0d ~f32
    def __call__(self, /, n: int, m: int, theta: onp.ToJustFloat32, *, diff_n: int = 0) -> onp.Array3D[np.float32]: ...
    @overload  # >=0d +f64
    def __call__(self, /, n: int, m: int, theta: _AsF64_D, *, diff_n: int = 0) -> _ArrayMin3D[np.float64]: ...
    @overload  # >=0d ~f32
    def __call__(self, /, n: int, m: int, theta: _ToJustFloat32_D, *, diff_n: int = 0) -> _ArrayMin3D[np.float32]: ...
    @overload  # fallback
    def __call__(self, /, n: int, m: int, theta: _ToFloat_D, *, diff_n: int = 0) -> onp.ArrayND[Any]: ...

@type_check_only
class _SphHarmY(MultiUFunc):
    @overload  # 0d, 0d, 0d +f64, 0d +f64, diff_n=0
    def __call__(self, /, n: int, m: int, theta: _AsF64, phi: _AsF64, *, diff_n: L[0] = 0) -> onp.Array0D[np.complex128]: ...
    @overload  # 0d, 0d, 0d +f64, 0d +f64, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: _AsF64, phi: _AsF64, *, diff_n: L[1]
    ) -> tuple[onp.Array0D[np.complex128], onp.Array1D[np.complex128]]: ...
    @overload  # 0d, 0d, 0d +f64, 0d +f64, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: _AsF64, phi: _AsF64, *, diff_n: L[2]
    ) -> tuple[onp.Array0D[np.complex128], onp.Array1D[np.complex128], onp.Array2D[np.complex128]]: ...
    @overload  # 0d, 0d, 0d ~f32, 0d ~f32, diff_n=0
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32, phi: onp.ToJustFloat32, *, diff_n: L[0] = 0
    ) -> onp.Array0D[np.complex64]: ...
    @overload  # 0d, 0d, 0d ~f32, 0d ~f32, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32, phi: onp.ToJustFloat32, *, diff_n: L[1]
    ) -> tuple[onp.Array0D[np.complex64], onp.Array1D[np.complex64]]: ...
    @overload  # 0d, 0d, 0d ~f32, 0d ~f32, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32, phi: onp.ToJustFloat32, *, diff_n: L[2]
    ) -> tuple[onp.Array0D[np.complex64], onp.Array1D[np.complex64], onp.Array2D[np.complex64]]: ...
    @overload  # >=0d, >=0d, >=0d +f64, >0d +f64, diff_n=0
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _AsF64_D, phi: _AsF64ND, *, diff_n: L[0] = 0
    ) -> _ArrayMin1D[np.complex128]: ...
    @overload  # >=0d, >=0d, >=0d +f64, >0d +f64, diff_n=1
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _AsF64_D, phi: _AsF64ND, *, diff_n: L[1]
    ) -> tuple[_ArrayMin1D[np.complex128], _ArrayMin2D[np.complex128]]: ...
    @overload  # >=0d, >=0d, >=0d +f64, >0d +f64, diff_n=2
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _AsF64_D, phi: _AsF64ND, *, diff_n: L[2]
    ) -> tuple[_ArrayMin1D[np.complex128], _ArrayMin2D[np.complex128], _ArrayMin3D[np.complex128]]: ...
    @overload  # >=0d, >=0d, >=0d ~f32, >0d ~f32, diff_n=0
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _ToJustFloat32_D, phi: onp.ToJustFloat32_ND, *, diff_n: L[0] = 0
    ) -> _ArrayMin1D[np.complex64]: ...
    @overload  # >=0d, >=0d, >=0d ~f32, >0d ~f32, diff_n=1
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _ToJustFloat32_D, phi: onp.ToJustFloat32_ND, *, diff_n: L[1]
    ) -> tuple[_ArrayMin1D[np.complex64], _ArrayMin2D[np.complex64]]: ...
    @overload  # >=0d, >=0d, >=0d ~f32, >0d ~f32, diff_n=2
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _ToJustFloat32_D, phi: onp.ToJustFloat32_ND, *, diff_n: L[2]
    ) -> tuple[_ArrayMin1D[np.complex64], _ArrayMin2D[np.complex64], _ArrayMin3D[np.complex64]]: ...
    @overload  # >=0d, >=0d, >0d +f64, >=0d +f64, diff_n=0
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _AsF64ND, phi: _AsF64_D, *, diff_n: L[0] = 0
    ) -> _ArrayMin1D[np.complex128]: ...
    @overload  # >=0d, >=0d, >0d +f64, >=0d +f64, diff_n=1
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _AsF64ND, phi: _AsF64_D, *, diff_n: L[1]
    ) -> tuple[_ArrayMin1D[np.complex128], _ArrayMin2D[np.complex128]]: ...
    @overload  # >=0d, >=0d, >0d +f64, >=0d +f64, diff_n=2
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _AsF64ND, phi: _AsF64_D, *, diff_n: L[2]
    ) -> tuple[_ArrayMin1D[np.complex128], _ArrayMin2D[np.complex128], _ArrayMin3D[np.complex128]]: ...
    @overload  # >=0d, >=0d, >0d ~f32, >=0d ~f32, diff_n=0
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: onp.ToJustFloat32_ND, phi: _ToJustFloat32_D, *, diff_n: L[0] = 0
    ) -> _ArrayMin1D[np.complex64]: ...
    @overload  # >=0d, >=0d, >0d ~f32, >=0d ~f32, diff_n=1
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: onp.ToJustFloat32_ND, phi: _ToJustFloat32_D, *, diff_n: L[1]
    ) -> tuple[_ArrayMin1D[np.complex64], _ArrayMin2D[np.complex64]]: ...
    @overload  # >=0d, >=0d, >0d ~f32, >=0d ~f32, diff_n=2
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: onp.ToJustFloat32_ND, phi: _ToJustFloat32_D, *, diff_n: L[2]
    ) -> tuple[_ArrayMin1D[np.complex64], _ArrayMin2D[np.complex64], _ArrayMin3D[np.complex64]]: ...
    @overload  # >=0d, >0d, >=0d +f64, >=0d +f64, diff_n=0
    def __call__(
        self, /, n: _ToInt_D, m: onp.ToIntND, theta: _AsF64_D, phi: _AsF64_D, *, diff_n: L[0] = 0
    ) -> _ArrayMin1D[np.complex128]: ...
    @overload  # >=0d, >0d, >=0d +f64, >=0d +f64, diff_n=1
    def __call__(
        self, /, n: _ToInt_D, m: onp.ToIntND, theta: _AsF64_D, phi: _AsF64_D, *, diff_n: L[1]
    ) -> tuple[_ArrayMin1D[np.complex128], _ArrayMin2D[np.complex128]]: ...
    @overload  # >=0d, >0d, >=0d +f64, >=0d +f64, diff_n=2
    def __call__(
        self, /, n: _ToInt_D, m: onp.ToIntND, theta: _AsF64_D, phi: _AsF64_D, *, diff_n: L[2]
    ) -> tuple[_ArrayMin1D[np.complex128], _ArrayMin2D[np.complex128], _ArrayMin3D[np.complex128]]: ...
    @overload  # >=0d, >0d, >=0d ~f32, >=0d ~f32, diff_n=0
    def __call__(
        self, /, n: _ToInt_D, m: onp.ToIntND, theta: _ToJustFloat32_D, phi: _ToJustFloat32_D, *, diff_n: L[0] = 0
    ) -> _ArrayMin1D[np.complex64]: ...
    @overload  # >=0d, >0d, >=0d ~f32, >=0d ~f32, diff_n=1
    def __call__(
        self, /, n: _ToInt_D, m: onp.ToIntND, theta: _ToJustFloat32_D, phi: _ToJustFloat32_D, *, diff_n: L[1]
    ) -> tuple[_ArrayMin1D[np.complex64], _ArrayMin2D[np.complex64]]: ...
    @overload  # >=0d, >0d, >=0d ~f32, >=0d ~f32, diff_n=2
    def __call__(
        self, /, n: _ToInt_D, m: onp.ToIntND, theta: _ToJustFloat32_D, phi: _ToJustFloat32_D, *, diff_n: L[2]
    ) -> tuple[_ArrayMin1D[np.complex64], _ArrayMin2D[np.complex64], _ArrayMin3D[np.complex64]]: ...
    @overload  # >0d, >=0d, >=0d +f64, >=0d +f64, diff_n=0
    def __call__(
        self, /, n: onp.ToIntND, m: _ToInt_D, theta: _AsF64_D, phi: _AsF64_D, *, diff_n: L[0] = 0
    ) -> _ArrayMin1D[np.complex128]: ...
    @overload  # >0d, >=0d, >=0d +f64, >=0d +f64, diff_n=1
    def __call__(
        self, /, n: onp.ToIntND, m: _ToInt_D, theta: _AsF64_D, phi: _AsF64_D, *, diff_n: L[1]
    ) -> tuple[_ArrayMin1D[np.complex128], _ArrayMin2D[np.complex128]]: ...
    @overload  # >0d, >=0d, >=0d +f64, >=0d +f64, diff_n=2
    def __call__(
        self, /, n: onp.ToIntND, m: _ToInt_D, theta: _AsF64_D, phi: _AsF64_D, *, diff_n: L[2]
    ) -> tuple[_ArrayMin1D[np.complex128], _ArrayMin2D[np.complex128], _ArrayMin3D[np.complex128]]: ...
    @overload  # >0d, >=0d, >=0d ~f32, >=0d ~f32, diff_n=0
    def __call__(
        self, /, n: onp.ToIntND, m: _ToInt_D, theta: _ToJustFloat32_D, phi: _ToJustFloat32_D, *, diff_n: L[0] = 0
    ) -> _ArrayMin1D[np.complex64]: ...
    @overload  # >0d, >=0d, >=0d ~f32, >=0d ~f32, diff_n=1
    def __call__(
        self, /, n: onp.ToIntND, m: _ToInt_D, theta: _ToJustFloat32_D, phi: _ToJustFloat32_D, *, diff_n: L[1]
    ) -> tuple[_ArrayMin1D[np.complex64], _ArrayMin2D[np.complex64]]: ...
    @overload  # >0d, >=0d, >=0d ~f32, >=0d ~f32, diff_n=2
    def __call__(
        self, /, n: onp.ToIntND, m: _ToInt_D, theta: _ToJustFloat32_D, phi: _ToJustFloat32_D, *, diff_n: L[2]
    ) -> tuple[_ArrayMin1D[np.complex64], _ArrayMin2D[np.complex64], _ArrayMin3D[np.complex64]]: ...
    @overload  # fallback, diff_n=0
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _ToFloat_D, phi: _ToFloat_D, *, diff_n: L[0] = 0
    ) -> onp.ArrayND[Any]: ...
    @overload  # fallback, diff_n=1
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _ToFloat_D, phi: _ToFloat_D, *, diff_n: L[1]
    ) -> tuple[onp.ArrayND[Any], onp.ArrayND[Any]]: ...
    @overload  # fallback, diff_n=2
    def __call__(
        self, /, n: _ToInt_D, m: _ToInt_D, theta: _ToFloat_D, phi: _ToFloat_D, *, diff_n: L[2]
    ) -> tuple[onp.ArrayND[Any], onp.ArrayND[Any], onp.ArrayND[Any]]: ...

@type_check_only
class _SphHarmYAll(MultiUFunc):
    @overload  # 0d +f64, 0d +f64, diff_n=0
    def __call__(self, /, n: int, m: int, theta: _AsF64, phi: _AsF64, *, diff_n: L[0] = 0) -> onp.Array2D[np.complex128]: ...
    @overload  # 0d +f64, 0d +f64, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: _AsF64, phi: _AsF64, *, diff_n: L[1]
    ) -> tuple[onp.Array2D[np.complex128], onp.Array3D[np.complex128]]: ...
    @overload  # 0d +f64, 0d +f64, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: _AsF64, phi: _AsF64, *, diff_n: L[2]
    ) -> tuple[onp.Array2D[np.complex128], onp.Array3D[np.complex128], onp.ArrayND[np.complex128, tuple[int, int, int, int]]]: ...
    @overload  # 0d ~f32, 0d ~f32, diff_n=0
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32, phi: onp.ToJustFloat32, *, diff_n: L[0] = 0
    ) -> onp.Array2D[np.complex64]: ...
    @overload  # 0d ~f32, 0d ~f32, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32, phi: onp.ToJustFloat32, *, diff_n: L[1]
    ) -> tuple[onp.Array2D[np.complex64], onp.Array3D[np.complex64]]: ...
    @overload  # 0d ~f32, 0d ~f32, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32, phi: onp.ToJustFloat32, *, diff_n: L[2]
    ) -> tuple[onp.Array2D[np.complex64], onp.Array3D[np.complex64], onp.ArrayND[np.complex64, tuple[int, int, int, int]]]: ...
    @overload  # >=0d +f64, >0d +f64, diff_n=0
    def __call__(self, /, n: int, m: int, theta: _AsF64_D, phi: _AsF64ND, *, diff_n: L[0] = 0) -> _ArrayMin3D[np.complex128]: ...
    @overload  # >=0d +f64, >0d +f64, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: _AsF64_D, phi: _AsF64ND, *, diff_n: L[1]
    ) -> tuple[_ArrayMin3D[np.complex128], _ArrayMin3D[np.complex128]]: ...
    @overload  # >=0d +f64, >0d +f64, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: _AsF64_D, phi: _AsF64ND, *, diff_n: L[2]
    ) -> tuple[_ArrayMin3D[np.complex128], _ArrayMin3D[np.complex128], _ArrayMin3D[np.complex128]]: ...
    @overload  # >=0d ~f32, >0d ~f32, diff_n=0
    def __call__(
        self, /, n: int, m: int, theta: _ToJustFloat32_D, phi: onp.ToJustFloat32_ND, *, diff_n: L[0] = 0
    ) -> _ArrayMin3D[np.complex64]: ...
    @overload  # >=0d ~f32, >0d ~f32, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: _ToJustFloat32_D, phi: onp.ToJustFloat32_ND, *, diff_n: L[1]
    ) -> tuple[_ArrayMin3D[np.complex64], _ArrayMin3D[np.complex64]]: ...
    @overload  # >=0d ~f32, >0d ~f32, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: _ToJustFloat32_D, phi: onp.ToJustFloat32_ND, *, diff_n: L[2]
    ) -> tuple[_ArrayMin3D[np.complex64], _ArrayMin3D[np.complex64], _ArrayMin3D[np.complex64]]: ...
    @overload  # >0d +f64, >=0d +f64, diff_n=0
    def __call__(self, /, n: int, m: int, theta: _AsF64ND, phi: _AsF64_D, *, diff_n: L[0] = 0) -> _ArrayMin3D[np.complex128]: ...
    @overload  # >0d +f64, >=0d +f64, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: _AsF64ND, phi: _AsF64_D, *, diff_n: L[1]
    ) -> tuple[_ArrayMin3D[np.complex128], _ArrayMin3D[np.complex128]]: ...
    @overload  # >0d +f64, >=0d +f64, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: _AsF64ND, phi: _AsF64_D, *, diff_n: L[2]
    ) -> tuple[_ArrayMin3D[np.complex128], _ArrayMin3D[np.complex128], _ArrayMin3D[np.complex128]]: ...
    @overload  # >0d ~f32, >=0d ~f32, diff_n=0
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32_ND, phi: _ToJustFloat32_D, *, diff_n: L[0] = 0
    ) -> _ArrayMin3D[np.complex64]: ...
    @overload  # >0d ~f32, >=0d ~f32, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32_ND, phi: _ToJustFloat32_D, *, diff_n: L[1]
    ) -> tuple[_ArrayMin3D[np.complex64], _ArrayMin3D[np.complex64]]: ...
    @overload  # >0d ~f32, >=0d ~f32, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: onp.ToJustFloat32_ND, phi: _ToJustFloat32_D, *, diff_n: L[2]
    ) -> tuple[_ArrayMin3D[np.complex64], _ArrayMin3D[np.complex64], _ArrayMin3D[np.complex64]]: ...
    @overload  # fallback, diff_n=0
    def __call__(self, /, n: int, m: int, theta: _ToFloat_D, phi: _ToFloat_D, *, diff_n: L[0] = 0) -> onp.ArrayND[Any]: ...
    @overload  # fallback, diff_n=1
    def __call__(
        self, /, n: int, m: int, theta: _ToFloat_D, phi: _ToFloat_D, *, diff_n: L[1]
    ) -> tuple[onp.ArrayND[Any], onp.ArrayND[Any]]: ...
    @overload  # fallback, diff_n=2
    def __call__(
        self, /, n: int, m: int, theta: _ToFloat_D, phi: _ToFloat_D, *, diff_n: L[2]
    ) -> tuple[onp.ArrayND[Any], onp.ArrayND[Any], onp.ArrayND[Any]]: ...

###

legendre_p: Final[_LegendreP] = ...
legendre_p_all: Final[_LegendrePAll] = ...

assoc_legendre_p: Final[_AssocLegendreP] = ...
assoc_legendre_p_all: Final[_AssocLegendrePAll] = ...

sph_legendre_p: Final[_SphLegendreP] = ...
sph_legendre_p_all: Final[_SphLegendrePAll] = ...

sph_harm_y: Final[_SphHarmY] = ...
sph_harm_y_all: Final[_SphHarmYAll] = ...
