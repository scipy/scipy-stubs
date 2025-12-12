"""
Prints the names of all SciPy modules that are not stubbed.
"""

# ruff: noqa: T201, S101
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import scipy

STUBS_PATH = Path(__file__).parent.parent / "scipy-stubs"
BUNDLED = (
    "scipy._lib.array_api_compat",
    "scipy._lib.array_api_extra",
    "scipy.fft._pocketfft",
    "scipy.optimize._highspy",
    "scipy.sparse.linalg._eigen.arpack",
    "scipy.sparse.linalg._propack",
)
# TODO(@jorenham): remove when stubs are added for these new SciPy 1.17 modules
TODO_1_17 = (
    "scipy.spatial.transform._rigid_transform_xp",
    "scipy.spatial.transform._rotation_xp",
)


def _check_stubs_path() -> None:
    # sanity check
    assert STUBS_PATH.is_dir()
    assert (STUBS_PATH / "__init__.pyi").exists()


def modules(
    mod: types.ModuleType, _seen: set[types.ModuleType] | None = None
) -> list[str]:
    seen = _seen or set()
    out: list[str] = []

    assert _seen is not None or mod.__spec__

    # the `dir` + `getattr` ensures that lazily loaded modules are included in `vars`
    mod_vars: dict[str, Any] = {}
    for k in dir(mod):
        try:
            mod_vars[k] = getattr(mod, k)
        except ModuleNotFoundError as e:
            # workaround for https://github.com/scipy/scipy/issues/24131
            if e.name == "scipy.integrate._lsoda":
                continue
            raise

    mod_vars |= vars(mod)

    for k, v in mod_vars.items():
        if (
            isinstance(v, types.ModuleType)
            and v not in seen
            and v.__name__.startswith("scipy")
        ):
            seen.add(v)
            fname = v.__spec__.name if v.__spec__ else k
            if "." in fname:
                out.append(fname)
                out.extend(modules(v, _seen=seen))
    return out


def module_to_path(mod: str) -> Path | None:
    _, *submods = mod.split(".")
    if (path := STUBS_PATH.joinpath(*submods, "__init__.pyi")).is_file():
        return path

    # https://github.com/facebook/pyrefly/issues/913#issuecomment-3367579203
    assert submods, path  # pyrefly: ignore[unbound-name]

    if (path := STUBS_PATH.joinpath(*submods[:-1], f"{submods[-1]}.pyi")).is_file():
        return path

    return None


def is_stubbed(mod: str) -> bool:
    return mod.startswith("scipy.") and module_to_path(mod) is not None


def main() -> int:
    _check_stubs_path()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        module_list = modules(scipy)
    module_list.sort()

    exit_code = 0
    for name in module_list:
        if any(map(name.startswith, BUNDLED + TODO_1_17)):
            continue

        if not is_stubbed(name):
            print(name, file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
