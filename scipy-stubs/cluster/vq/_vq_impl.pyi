from collections.abc import Callable
from types import ModuleType
from typing import Any, Final, Literal, Never, overload
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["ClusterError", "kmeans", "kmeans2", "py_vq", "vq", "whiten"]

###

type _InitMethod = Literal["random", "points", "++", "matrix"]
type _MissingMethod = Literal["warn", "raise"]

type _ToFloat32_2D = onp.ToArray2D[int, np.float32 | np.float16 | npc.integer16 | npc.integer8]
type _AsFloat64_2D = onp.ToArray2D[float, npc.floating64 | npc.integer]

# workaround for https://github.com/microsoft/pyright/issues/10232
type _JustAnyShape = tuple[Never, Never, Never]

###

class ClusterError(Exception): ...

#
@overload  # ?d, +f64
def whiten(
    obs: onp.ArrayND[npc.integer | np.bool, _JustAnyShape], check_finite: bool | None = None
) -> onp.ArrayND[np.float64]: ...
@overload  # ?d, ~inexact
def whiten[InexactT: npc.inexact](
    obs: onp.ArrayND[InexactT, _JustAnyShape], check_finite: bool | None = None
) -> onp.ArrayND[InexactT]: ...
@overload  # 1d, +f64
def whiten(obs: onp.Array1D[npc.integer | np.bool], check_finite: bool | None = None) -> onp.Array1D[np.float64]: ...
@overload  # 1d, ~inexact
def whiten[InexactT: npc.inexact](obs: onp.Array1D[InexactT], check_finite: bool | None = None) -> onp.Array1D[InexactT]: ...
@overload  # 2d, +f64
def whiten(obs: onp.Array2D[npc.integer | np.bool], check_finite: bool | None = None) -> onp.Array2D[np.float64]: ...
@overload  # 2d, ~inexact
def whiten[InexactT: npc.inexact](obs: onp.Array2D[InexactT], check_finite: bool | None = None) -> onp.Array2D[InexactT]: ...
@overload  # nd, +f64
def whiten(obs: onp.ArrayND[npc.integer | np.bool], check_finite: bool | None = None) -> onp.ArrayND[np.float64]: ...
@overload  # nd, ~inexact
def whiten[InexactT: npc.inexact](obs: onp.ArrayND[InexactT], check_finite: bool | None = None) -> onp.ArrayND[InexactT]: ...

#
@overload  # float32
def vq(
    obs: onp.CanArrayND[np.float32], code_book: _ToFloat32_2D, check_finite: bool = True
) -> tuple[onp.Array1D[np.int32], onp.Array1D[np.float32]]: ...
@overload  # float64
def vq(
    obs: onp.ToJustFloat64_2D, code_book: _AsFloat64_2D, check_finite: bool = True
) -> tuple[onp.Array1D[np.int32], onp.Array1D[np.float64]]: ...
@overload  # floating
def vq(
    obs: onp.ToJustFloat2D, code_book: onp.ToFloat2D, check_finite: bool = True
) -> tuple[onp.Array1D[np.int32], onp.Array1D[npc.floating]]: ...

#
@overload  # float64
@deprecated(
    "`scipy.cluster.vq.py_vq` was unintentionally public, and will be removed in SciPy 2.1.0, use `scipy.cluster.vq.vq` instead."
)
def py_vq(
    obs: onp.ToFloat64_2D, code_book: onp.ToFloat64_2D, check_finite: bool = True
) -> tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]]: ...
@overload  # floating
@deprecated(
    "`scipy.cluster.vq.py_vq` was unintentionally public, and will be removed in SciPy 2.1.0, use `scipy.cluster.vq.vq` instead."
)
def py_vq(
    obs: onp.ToFloat2D, code_book: onp.ToFloat2D, check_finite: bool = True
) -> tuple[onp.Array1D[np.intp], onp.Array1D[npc.floating]]: ...

#
@overload  # ?d float32
def kmeans(
    obs: onp.ArrayND[np.float32, _JustAnyShape],
    k_or_guess: int | _ToFloat32_2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.ArrayND[np.float32], np.float32]: ...
@overload  # ?d float64
def kmeans(
    obs: onp.ArrayND[np.float64, _JustAnyShape],
    k_or_guess: int | _AsFloat64_2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.ArrayND[np.float64], np.float64]: ...
@overload  # ?d floating
def kmeans(
    obs: onp.ArrayND[npc.floating, _JustAnyShape],
    k_or_guess: int | onp.ToFloat2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.ArrayND[np.float64 | Any], np.float64 | Any]: ...
@overload  # 1d float32
def kmeans(
    obs: onp.ToJustFloat32Strict1D,
    k_or_guess: int | _ToFloat32_2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float32], np.float32]: ...
@overload  # 1d float64
def kmeans(
    obs: onp.ToJustFloat64Strict1D,
    k_or_guess: int | _AsFloat64_2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float64], np.float64]: ...
@overload  # 1d floating
def kmeans(
    obs: onp.ToJustFloatStrict1D,
    k_or_guess: int | onp.ToFloat2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float64 | Any], np.float64 | Any]: ...
@overload  # 2d float32
def kmeans(
    obs: onp.ToJustFloat32Strict2D,
    k_or_guess: int | _ToFloat32_2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array2D[np.float32], np.float32]: ...
@overload  # 2d float64
def kmeans(
    obs: onp.ToJustFloat64Strict2D,
    k_or_guess: int | _AsFloat64_2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array2D[np.float64], np.float64]: ...
@overload  # 2d floating
def kmeans(
    obs: onp.ToJustFloatStrict2D,
    k_or_guess: int | onp.ToFloat2D,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array2D[np.float64 | Any], np.float64 | Any]: ...

#
def _kpoints[InexactT: npc.inexact](
    data: onp.ArrayND[InexactT], k: int, rng: onp.random.ToRNG, xp: ModuleType
) -> onp.Array2D[InexactT]: ...  # undocumented
def _krandinit(
    data: onp.ArrayND[npc.inexact], k: int, rng: onp.random.ToRNG, xp: ModuleType
) -> onp.Array2D[np.float64]: ...  # undocumented
def _kpp(
    data: onp.ArrayND[npc.inexact], k: int, rng: onp.random.ToRNG, xp: ModuleType
) -> onp.Array2D[np.float64]: ...  # undocumented

_valid_init_meth: Final[
    dict[str, Callable[[onp.ArrayND[npc.inexact], int, onp.random.ToRNG, ModuleType], onp.Array2D[npc.inexact]]]
] = ...  # undocumented

def _missing_warn() -> None: ...  # undocumented
def _missing_raise() -> None: ...  # undocumented

_valid_miss_meth: Final[dict[str, Callable[[], None]]] = ...  # undocumented

# NOTE: There is a false positive `overload-overlap` mypy error that only occurs with `numpy<2.2`
# mypy: disable-error-code=overload-overlap

#
@overload  # ?d float32
def kmeans2(
    data: onp.ArrayND[np.float32, _JustAnyShape],
    k: int | _ToFloat32_2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.ArrayND[np.float32], onp.Array1D[np.int32]]: ...
@overload  # ?d float64
def kmeans2(
    data: onp.ArrayND[np.float64, _JustAnyShape],
    k: int | _AsFloat64_2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.ArrayND[np.float64], onp.Array1D[np.int32]]: ...
@overload  # ?d floating
def kmeans2(
    data: onp.ArrayND[npc.floating, _JustAnyShape],
    k: int | onp.ToFloat2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.Array1D[np.int32]]: ...
@overload  # 1d float32
def kmeans2(
    data: onp.ToJustFloat32Strict1D,
    k: int | _ToFloat32_2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float32], onp.Array1D[np.int32]]: ...
@overload  # 1d float64
def kmeans2(
    data: onp.ToJustFloat64Strict1D,
    k: int | _AsFloat64_2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.int32]]: ...
@overload  # 1d floating
def kmeans2(
    data: onp.ToJustFloatStrict1D,
    k: int | onp.ToFloat2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float64 | Any], onp.Array1D[np.int32]]: ...
@overload  # 2d float32
def kmeans2(
    data: onp.ToJustFloat32Strict2D,
    k: int | _ToFloat32_2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]]: ...
@overload  # 2d float64
def kmeans2(
    data: onp.ToJustFloat64Strict2D,
    k: int | _AsFloat64_2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]]: ...
@overload  # 2d floating
def kmeans2(
    data: onp.ToJustFloatStrict2D,
    k: int | onp.ToFloat2D,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    seed: onp.random.ToRNG | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array2D[np.float64 | Any], onp.Array1D[np.int32]]: ...
