"""
Prints the names of all SciPy modules that are not stubbed.
"""

# ruff: noqa: T201, S101
import contextlib
import sys
import warnings
from collections.abc import Iterator
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES, SOURCE_SUFFIXES
from pathlib import Path

import scipy

STUBS_PATH = Path(__file__).parent.parent / "scipy-stubs"
_INIT_FILES = tuple(f"__init__{s}" for s in (*SOURCE_SUFFIXES, *EXTENSION_SUFFIXES))

IGNORED = (
    # bundled
    "scipy._lib._uarray",
    "scipy.fft._duccfft",
    "scipy.io._fast_matrix_market._fmm_core",
    "scipy.optimize._highspy",
    "scipy.sparse.linalg._eigen.arpack",
    "scipy.sparse.linalg._propack",
    # internal testing
    "scipy._lib._test_ccallback",
    "scipy._lib._test_deprecation_call",
    "scipy._lib._test_deprecation_def",
    "scipy.integrate._test_multivariate",
    "scipy.ndimage._ctest",
    "scipy.ndimage._cytest",
    "scipy.optimize._tstutils",
    "scipy.special._mptestutils",
    # internal non-functional utilities
    "scipy.optimize.cython_optimize._zeros",
    "scipy.sparse.linalg._eigen._svds_doc",
    "scipy.special._precompute",
    # private array-api modules (TODO, maybe?)
    "scipy.ndimage._delegators",
    "scipy.ndimage._ndimage_api",
    "scipy.ndimage._support_alternative_backends",
    "scipy.signal._support_alternative_backends",
    # definitely TODO
    "scipy._lib._fpumode",
    "scipy.interpolate._regrid",
)


def _check_stubs_path() -> None:
    # sanity check
    assert STUBS_PATH.is_dir()
    assert (STUBS_PATH / "__init__.pyi").exists()


def _walk(pkg_dir: Path, pkg_name: str) -> Iterator[str]:
    for entry in sorted(pkg_dir.iterdir()):
        if (name := entry.name).startswith((".", "__")):
            continue

        if entry.is_dir():
            if name == "tests" or not any((entry / f).exists() for f in _INIT_FILES):
                continue

            yield (sub := f"{pkg_name}.{name}")
            yield from _walk(entry, sub)

        elif entry.suffix in {".py", ".so", ".pyd"} and name != "conftest.py":
            yield f"{pkg_name}.{name.split('.', 1)[0]}"


def modules() -> Iterator[str]:
    root = Path(scipy.__path__[0])
    for name in dict.fromkeys(_walk(root, "scipy")):
        with contextlib.suppress(ImportError):
            _ = import_module(name)
            yield name


def module_to_path(mod: str) -> Path | None:
    _, *submods = mod.split(".")
    if (path := STUBS_PATH.joinpath(*submods, "__init__.pyi")).is_file():
        return path

    assert submods, path

    if (path := STUBS_PATH.joinpath(*submods[:-1], f"{submods[-1]}.pyi")).is_file():
        return path

    return None


def main() -> int:
    _check_stubs_path()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        module_list = sorted(modules())

    unused_ignores = set(IGNORED)
    stubbed = 0
    unstubbed = 0
    ignored = 0
    exit_code = 0
    for name in module_list:
        if ignore := next((p for p in IGNORED if name.startswith(p)), None):
            unused_ignores.discard(ignore)
            ignored += 1
            continue

        if module_to_path(name):
            stubbed += 1
            continue

        print(name, file=sys.stderr)
        unstubbed += 1
        exit_code = 1

    for ignore in sorted(unused_ignores):
        print(f"unused IGNORED entry: {ignore}", file=sys.stderr)
        exit_code = 1

    print(f"{stubbed} stubbed, {unstubbed} unstubbed modules ({ignored} ignored)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
