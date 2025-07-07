"""
Prints the names of all SciPy modules that are not stubbed.
"""

# ruff: noqa: T201, S101

import types
import warnings
from pathlib import Path
from typing import Any

import scipy

BUNDLED = (
    "scipy._lib.array_api_compat",
    "scipy._lib.array_api_extra",
    "scipy.fft._pocketfft",
    "scipy.optimize._highspy",
    "scipy.sparse.linalg._eigen.arpack",
    "scipy.sparse.linalg._propack",
)


def modules(
    mod: types.ModuleType, _seen: set[types.ModuleType] | None = None
) -> list[str]:
    seen = _seen or set()
    out: list[str] = []

    assert _seen is not None or mod.__spec__

    # the `dir` + `getattr` ensures that lazily loaded modules are included in `vars`
    mod_vars: dict[str, Any] = {}
    for k in dir(mod):
        mod_vars[k] = getattr(mod, k)

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


def is_stubbed(mod: str) -> bool:
    if not mod.startswith("scipy."):
        return False

    stubs_path = Path(__file__).parent.parent / "scipy-stubs"
    if not stubs_path.is_dir():
        raise FileNotFoundError(stubs_path)

    _, *submods = mod.split(".")
    if not submods:
        return (stubs_path / "__init__.pyi").is_file()

    *subpackages, submod = submods
    subpackage_path = stubs_path.joinpath(*subpackages)
    return (subpackage_path / f"{submod}.pyi").is_file() or (
        subpackage_path / submod / "__init__.pyi"
    ).is_file()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        module_list = modules(scipy)

    module_list.sort()
    for name in module_list:
        if not any(map(name.startswith, BUNDLED)) and not is_stubbed(name):
            print(name)
