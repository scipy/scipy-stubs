"""
Prints the names of all SciPy modules that are not stubbed.
"""

# ruff: noqa: T201, S101
import sys
import warnings
from collections.abc import Iterator, Sequence
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES, SOURCE_SUFFIXES
from importlib.metadata import distribution
from pathlib import Path

STUBS_PATH = Path(__file__).parent.parent / "scipy-stubs"

_MODULE_SUFFIXES = *SOURCE_SUFFIXES, *EXTENSION_SUFFIXES
_INIT_LEAVES = frozenset(f"__init__{s}" for s in _MODULE_SUFFIXES)
_TYPECHECK_ONLY_STUBS = ("._typing",)

IGNORED = (
    # bundled
    "scipy._external",
    "scipy._lib._uarray",
    "scipy.fft._duccfft",
    "scipy.io._fast_matrix_market._fmm_core",
    "scipy.odr.__odrpack",
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
)


def _check_stubs_path() -> None:
    # sanity check
    assert STUBS_PATH.is_dir()
    assert (STUBS_PATH / "__init__.pyi").exists()


def _path_to_module(parts: tuple[str, ...]) -> str | None:
    name_parts: Sequence[str] = []
    if parts and parts[0] == "scipy" and "tests" not in parts:
        *parent, leaf = parts
        if leaf in _INIT_LEAVES:
            name_parts = parent
        elif leaf != "conftest.py" and leaf.endswith(_MODULE_SUFFIXES):
            name_parts = *parent, leaf.split(".", 1)[0]
    return ".".join(name_parts) or None


def _candidates() -> Iterator[str]:
    files = distribution("scipy").files
    assert files is not None, "scipy was installed without a RECORD"
    return (name for f in files if (name := _path_to_module(f.parts)) is not None)


def modules() -> Iterator[str]:
    for name in dict.fromkeys(_candidates()):
        try:
            _ = import_module(name)
        except ModuleNotFoundError as e:
            if e.name == name:
                continue
        except ImportError:
            pass
        yield name


def _walk_stubs(stubs_dir: Path, pkg_name: str) -> Iterator[str]:
    for entry in sorted(stubs_dir.iterdir()):
        if (name := entry.name).startswith("."):
            continue

        if entry.is_dir():
            if name.startswith("__") or not (entry / "__init__.pyi").is_file():
                continue

            yield (sub := f"{pkg_name}.{name}")
            yield from _walk_stubs(entry, sub)

        elif entry.suffix == ".pyi" and name != "__init__.pyi":
            yield f"{pkg_name}.{entry.stem}"


def stubs() -> Iterator[str]:
    yield from _walk_stubs(STUBS_PATH, "scipy")


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

    orphans = sorted(
        name
        for name in set(stubs()) - set(module_list)
        if not name.endswith(_TYPECHECK_ONLY_STUBS)
    )
    for orphan in orphans:
        print(f"orphaned stub: {orphan}", file=sys.stderr)
        exit_code = 1

    print(
        f"{stubbed} stubbed, {unstubbed} unstubbed, {len(orphans)} orphaned "
        f"({ignored} ignored)"
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
