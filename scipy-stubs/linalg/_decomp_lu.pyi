from typing import Literal, Never, TypeAlias, overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["lu", "lu_factor", "lu_solve"]

###

_as_f32: TypeAlias = np.float32 | np.float16 | npc.integer16 | npc.integer8 | np.bool_  # noqa: PYI042
_as_f64: TypeAlias = npc.floating64 | npc.floating80 | npc.integer64 | npc.integer32  # noqa: PYI042
_as_c64: TypeAlias = np.complex64  # noqa: PYI042
_as_c128: TypeAlias = npc.complexfloating160 | npc.complexfloating128  # noqa: PYI042

_ISizeND: TypeAlias = onp.ArrayND[np.intp]
_Float2D: TypeAlias = onp.Array2D[npc.floating]
_FloatND: TypeAlias = onp.ArrayND[npc.floating]
_Complex2D: TypeAlias = onp.Array2D[npc.complexfloating]
_ComplexND: TypeAlias = onp.ArrayND[npc.complexfloating]
_InexactND: TypeAlias = onp.ArrayND[npc.inexact]

_Trans: TypeAlias = Literal[0, 1, 2]

# workaround for mypy & pyright's failure to conform to the overload typing specification
_JustAnyShape: TypeAlias = tuple[Never, Never, Never]

###

# NOTE: The ignored mypy `overload-overlap` errors are false positives

@overload  # ?d f64
def lu_factor(  # type: ignore[overload-overlap]
    a: onp.ArrayND[_as_f64, _JustAnyShape], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]]: ...
@overload  # ?d f32
def lu_factor(
    a: onp.ArrayND[_as_f32, _JustAnyShape], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float32], onp.ArrayND[np.int32]]: ...
@overload  # ?d c128
def lu_factor(
    a: onp.ArrayND[_as_c128, _JustAnyShape], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.int32]]: ...
@overload  # ?d c64
def lu_factor(
    a: onp.ArrayND[_as_c64, _JustAnyShape], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.int32]]: ...
@overload  # 2d f64
def lu_factor(  # type: ignore[overload-overlap]
    a: onp.ToArrayStrict2D[float, _as_f64], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]]: ...
@overload  # 2d f32
def lu_factor(
    a: onp.ToArrayStrict2D[np.float32, _as_f32], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]]: ...
@overload  # 2d c128
def lu_factor(
    a: onp.ToArrayStrict2D[op.JustComplex, _as_c128], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array2D[np.complex128], onp.Array1D[np.int32]]: ...
@overload  # 2d c64
def lu_factor(
    a: onp.ToArrayStrict2D[np.complex64, _as_c64], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array2D[np.complex64], onp.Array1D[np.int32]]: ...
@overload  # 3d f64
def lu_factor(  # type: ignore[overload-overlap]
    a: onp.ToArrayStrict3D[float, _as_f64], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array3D[np.float64], onp.Array2D[np.int32]]: ...
@overload  # 3d f32
def lu_factor(
    a: onp.ToArrayStrict3D[np.float32, _as_f32], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array3D[np.float32], onp.Array2D[np.int32]]: ...
@overload  # 3d c128
def lu_factor(
    a: onp.ToArrayStrict3D[op.JustComplex, _as_c128], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array3D[np.complex128], onp.Array2D[np.int32]]: ...
@overload  # 3d c64
def lu_factor(
    a: onp.ToArrayStrict3D[np.complex64, _as_c64], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array3D[np.complex64], onp.Array2D[np.int32]]: ...
@overload  # nd f64
def lu_factor(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[float, _as_f64], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]]: ...
@overload  # nd f32
def lu_factor(
    a: onp.ToArrayND[np.float32, _as_f32], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float32], onp.ArrayND[np.int32]]: ...
@overload  # nd c128
def lu_factor(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[op.JustComplex, _as_c128], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.int32]]: ...
@overload  # nd c64
def lu_factor(
    a: onp.ToArrayND[np.complex64, _as_c64], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.int32]]: ...

#
@overload  # (float[:, :], float[:]) -> float[:, :]
def lu_solve(
    lu_and_piv: tuple[onp.ToFloatStrict2D, onp.ToFloatStrict1D],
    b: onp.ToFloat1D,
    trans: _Trans = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Float2D: ...
@overload  # (float[:, :, ...], float[:, ...]) -> float[:, :, ...]
def lu_solve(
    lu_and_piv: tuple[onp.ToFloatND, onp.ToFloatND],
    b: onp.ToFloatND,
    trans: _Trans = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _FloatND: ...
@overload  # (complex[:, :,], complex[:]) -> complex[:, :]
def lu_solve(
    lu_and_piv: tuple[onp.ToJustComplexStrict2D, onp.ToJustComplexStrict1D],
    b: onp.ToJustComplex1D,
    trans: _Trans = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Complex2D: ...
@overload  # (complex[:, :, ...], complex[:, ...]) -> complex[:, :, ...]
def lu_solve(
    lu_and_piv: tuple[onp.ToJustComplexND, onp.ToJustComplexND],
    b: onp.ToJustComplexND,
    trans: _Trans = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _ComplexND: ...
@overload  # fallback
def lu_solve(
    lu_and_piv: tuple[onp.ToComplexND, onp.ToComplexND],
    b: onp.ToComplexND,
    trans: _Trans = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _InexactND: ...

#
@overload  # (float[:, :], permute_l=False, p_indices=False) -> (float[...], float[...], float[...])
def lu(
    a: onp.ToFloatND,
    permute_l: onp.ToFalse = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    p_indices: onp.ToFalse = False,
) -> tuple[_FloatND, _FloatND, _FloatND]: ...
@overload  # (float[:, :], permute_l=False, p_indices=True, /) -> (intp[...], float[...], float[...])
def lu(
    a: onp.ToFloatND, permute_l: onp.ToFalse, overwrite_a: bool, check_finite: bool, p_indices: onp.ToTrue
) -> tuple[_ISizeND, _FloatND, _FloatND]: ...
@overload  # (float[:, :], permute_l=False, *, p_indices=True) -> (intp[...], float[...], float[...])
def lu(
    a: onp.ToFloatND,
    permute_l: onp.ToFalse = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    *,
    p_indices: onp.ToTrue,
) -> tuple[_ISizeND, _FloatND, _FloatND]: ...
@overload  # (float[:, :], permute_l=True, p_indices=False) -> (intp[...], float[...], float[...])
def lu(
    a: onp.ToFloatND, permute_l: onp.ToTrue, overwrite_a: bool = False, check_finite: bool = True, p_indices: bool = False
) -> tuple[_FloatND, _FloatND]: ...
@overload  # (complex[:, :], permute_l=False, p_indices=False) -> (complex[...], complex[...], complex[...])
def lu(
    a: onp.ToJustComplexND,
    permute_l: onp.ToFalse = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    p_indices: onp.ToFalse = False,
) -> tuple[_ComplexND, _ComplexND, _ComplexND]: ...
@overload  # (complex[:, :], permute_l=False, p_indices=True, /) -> (intp[...], complex[...], complex[...])
def lu(
    a: onp.ToJustComplexND, permute_l: onp.ToFalse, overwrite_a: bool, check_finite: bool, p_indices: onp.ToTrue
) -> tuple[_ISizeND, _ComplexND, _ComplexND]: ...
@overload  # (complex[:, :], permute_l=False, *, p_indices=True) -> (intp[...], complex[...], complex[...])
def lu(
    a: onp.ToJustComplexND,
    permute_l: onp.ToFalse = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    *,
    p_indices: onp.ToTrue,
) -> tuple[_ISizeND, _ComplexND, _ComplexND]: ...
@overload  # (complex[:, :], permute_l=True, p_indices=False) -> (intp[...], complex[...], complex[...])
def lu(
    a: onp.ToJustComplexND, permute_l: onp.ToTrue, overwrite_a: bool = False, check_finite: bool = True, p_indices: bool = False
) -> tuple[_ComplexND, _ComplexND]: ...
@overload  # fallback, permute_l=False, p_indices=False
def lu(
    a: onp.ToComplexND,
    permute_l: onp.ToFalse = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    p_indices: onp.ToFalse = False,
) -> tuple[_InexactND, _InexactND, _InexactND]: ...
@overload  # fallback, permute_l=False, p_indices=True
def lu(
    a: onp.ToComplexND, permute_l: onp.ToFalse, overwrite_a: bool, check_finite: bool, p_indices: onp.ToTrue
) -> tuple[_ISizeND, _InexactND, _InexactND]: ...
@overload  # fallback, permute_l=False, *, p_indices=True
def lu(
    a: onp.ToComplexND,
    permute_l: onp.ToFalse = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    *,
    p_indices: onp.ToTrue,
) -> tuple[_ISizeND, _InexactND, _InexactND]: ...
@overload  # fallback, permute_l=True, p_indices=False
def lu(
    a: onp.ToComplexND, permute_l: onp.ToTrue, overwrite_a: bool = False, check_finite: bool = True, p_indices: bool = False
) -> tuple[_InexactND, _InexactND]: ...
