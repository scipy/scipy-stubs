"""Walk through tests/ and collect the qualnames of accessed scipy.* names."""

# ruff: noqa: S101, T201

import ast
import importlib
import sys
from pathlib import Path
from typing import Final

_SCIPY: Final = "scipy"
_IGNORED_SUFFIXES: Final = {"__class__"}

_PACKAGES_PUBLIC: Final = (
    "cluster",
    "cluster.hierarchy",
    "cluster.vq",
    "constants",
    "datasets",
    "differentiate",
    "fft",
    "integrate",
    "interpolate",
    "io",
    "io.arff",
    "io.matlab",
    "linalg",
    "linalg.blas",
    "linalg.cython_blas",
    "linalg.cython_lapack",
    "linalg.interpolative",
    "linalg.lapack",
    "ndimage",
    "optimize",
    "optimize.elementwise",
    "signal",
    "signal.windows",
    "sparse",
    "sparse.csgraph",
    "sparse.linalg",
    "spatial",
    "spatial.transform",
    "special",
    "special.cython_special",
    "stats",
    "stats.contingency",
    "stats.mstats",
    "stats.qmc",
    "stats.sampling",
    "version",
)
_PACKAGES_DEPRECATED: Final = (
    "constants.codata",
    "constants.constants",
    "integrate.dop",
    "integrate.lsoda",
    "integrate.odepack",
    "integrate.quadpack",
    "integrate.vode",
    "interpolate.dfitpack",
    "interpolate.fitpack",
    "interpolate.fitpack2",
    "interpolate.interpnd",
    "interpolate.interpolate",
    "interpolate.ndgriddata",
    "interpolate.polyint",
    "interpolate.rbf",
    "io.arff.arffread",
    "io.harwell_boeing",
    "io.idl",
    "io.matlab.byteordercodes",
    "io.matlab.mio_utils",
    "io.matlab.mio",
    "io.matlab.mio4",
    "io.matlab.mio5",
    "io.matlab.mio5_params",
    "io.matlab.mio5_utils",
    "io.matlab.miobase",
    "io.matlab.streams",
    "io.mmio",
    "io.netcdf",
    "linalg.basic",
    "linalg.decomp",
    "linalg.decomp_cholesky",
    "linalg.decomp_lu",
    "linalg.decomp_qr",
    "linalg.decomp_schur",
    "linalg.decomp_svd",
    "linalg.matfuncs",
    "linalg.misc",
    "linalg.special_matrices",
    "misc",
    "misc.common",
    "misc.doccer",
    "ndimage.filters",
    "ndimage.fourier",
    "ndimage.interpolation",
    "ndimage.measurements",
    "ndimage.morphology",
    "odr",
    "odr.models",
    "odr.odrpack",
    "optimize.cobyla",
    "optimize.lbfgsb",
    "optimize.linesearch",
    "optimize.minpack",
    "optimize.minpack2",
    "optimize.moduleTNC",
    "optimize.nonlin",
    "optimize.optimize",
    "optimize.slsqp",
    "optimize.tnc",
    "optimize.zeros",
    "signal.bsplines",
    "signal.filter_design",
    "signal.fir_filter_design",
    "signal.lti_conversion",
    "signal.ltisys",
    "signal.signaltools",
    "signal.spectral",
    "signal.spline",
    "signal.waveforms",
    "signal.wavelets",
    "signal.windows.windows",
    "sparse.base",
    "sparse.compressed",
    "sparse.construct",
    "sparse.coo",
    "sparse.csc",
    "sparse.csr",
    "sparse.data",
    "sparse.dia",
    "sparse.dok",
    "sparse.extract",
    "sparse.lil",
    "sparse.linalg.dsolve",
    "sparse.linalg.eigen",
    "sparse.linalg.interface",
    "sparse.linalg.isolve",
    "sparse.linalg.matfuncs",
    "sparse.sparsetools",
    "sparse.spfuncs",
    "sparse.sputils",
    "spatial.ckdtree",
    "spatial.kdtree",
    "spatial.qhull",
    "spatial.transform.rotation",
    "special.add_newdocs",
    "special.basic",
    "special.orthogonal",
    "special.sf_error",
    "special.specfun",
    "special.spfun_stats",
    "stats.biasedurn",
    "stats.distributions",  # not deprecated, but only re-exports from `scipy.stats`
    "stats.kde",
    "stats.morestats",
    "stats.mstats_basic",
    "stats.mstats_extras",
    "stats.mvn",
    "stats.stats",
)


def _extract_attribute_chain(node: ast.Attribute) -> list[str]:
    """Extract the full attribute chain from an ast.Attribute node.

    For example, `scipy.stats.pearsonr` returns `["scipy", "stats", "pearsonr"]`.
    """
    parts: list[str] = []
    current: ast.expr = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()

    if parts and parts[-1] in _IGNORED_SUFFIXES:
        _ = parts.pop()

    return parts


def _resolve_to_qualname(parts: list[str], imports: dict[str, str]) -> str | None:
    """Resolve an attribute chain to a fully qualified scipy name."""
    if not parts:
        return None

    # direct scipy.* access
    if parts[0] == _SCIPY:
        return ".".join(parts)

    # resolve through imports
    if parts[0] in imports:
        return f"{imports[parts[0]]}.{'.'.join(parts[1:])}"

    return None


def _parse_scipy_imports(tree: ast.AST) -> dict[str, str]:
    """Extract scipy imports, returning a mapping of local name -> qualified name."""
    imports: dict[str, str] = {}

    for node in ast.walk(tree):
        match node:
            case ast.ImportFrom(module=str(m), names=aliases) if m.startswith(_SCIPY):
                for alias in aliases:
                    imports[alias.asname or alias.name] = f"{m}.{alias.name}"

            case ast.Import(names=aliases):
                for alias in aliases:
                    if alias.name.startswith(_SCIPY):
                        imports[alias.asname or alias.name] = alias.name

            case _:
                pass

    return imports


def _should_ignore(qualname: str) -> bool:
    """Check if a qualified name should be ignored (private or bare package)."""
    parts = qualname.split(".")
    if any(part.startswith("_") for part in parts[1:]):
        return True  # private
    if len(parts) < 3 or qualname.endswith(_PACKAGES_PUBLIC):  # noqa: PLR2004
        return True  # bare package
    for deprecated_pkg in _PACKAGES_DEPRECATED:
        if (
            qualname.startswith(f"scipy.{deprecated_pkg}.")
            or qualname == f"scipy.{deprecated_pkg}"
        ):
            return True  # deprecated
    return False


def _find_scipy_names_in_tree(tree: ast.AST, imports: dict[str, str]) -> set[str]:
    """Find all scipy names that are actually used in the AST."""
    used: set[str] = set()

    for node in ast.walk(tree):
        qualname: str | None = None
        if isinstance(node, ast.Name) and node.id in imports:
            qualname = imports[node.id]
        elif isinstance(node, ast.Attribute):
            qualname = _resolve_to_qualname(_extract_attribute_chain(node), imports)
        if qualname is None:
            continue

        if not _should_ignore(qualname):
            used.add(qualname)

    return used


def names_tested(tests_dir: Path | None = None) -> set[str]:
    """Collect all public scipy names accessed in test files.

    Args:
        tests_dir: Path to the tests directory. Defaults to `tests/` relative to
            this script's parent directory.

    Returns:
        A set of fully qualified scipy names (e.g., "scipy.stats.pearsonr").
    """
    if tests_dir is None:
        tests_dir = Path(__file__).parent.parent / "tests"

    assert tests_dir.exists(), f"Tests directory not found: {tests_dir}"

    all_names: set[str] = set()
    for pyi_file in tests_dir.rglob("*.pyi"):
        tree = ast.parse(pyi_file.read_text(), type_comments=True)
        imports = _parse_scipy_imports(tree)
        all_names.update(_find_scipy_names_in_tree(tree, imports))

    return all_names


def names_public() -> set[str]:
    """Collect all public scipy API names by inspecting `__all__` at runtime.

    Returns:
        A set of fully qualified scipy names (e.g., "scipy.stats.pearsonr").
    """
    all_names: set[str] = set()

    for subpkg in _PACKAGES_PUBLIC:
        qualname = f"scipy.{subpkg}"
        try:
            module = importlib.import_module(qualname)
        except ImportError:
            continue

        exported = getattr(module, "__all__", None)
        if exported is not None:
            all_names.update(
                f"{qualname}.{name}"
                for name in exported
                if not _should_ignore(f"{qualname}.{name}")
            )

    return all_names


def main() -> int:
    public = names_public()
    tested = names_tested() & public

    package = "scipy.none"
    package_public: int = 0
    package_tested: int = 0

    def _print_coverage(n_tested: int, n_public: int) -> None:
        if package_public:
            print()
            print(f"Coverage: {n_tested} / {n_public} ({n_tested / n_public:.1%})")

    for name in sorted(public):
        if not name.startswith(package):
            _print_coverage(package_tested, package_public)
            print("</details>")

            package = ".".join(name.split(".", 2)[:2])
            package_public = package_tested = 0

            print(f"<details>\n<summary><code>{package}</code></summary>\n")

        package_public += 1
        package_tested += name in tested
        x = "x" if name in tested else " "
        print(f"- [{x}] `{name}`")

    _print_coverage(package_tested, package_public)
    print("</details>")

    _print_coverage(len(tested), len(public))

    return 0


if __name__ == "__main__":
    sys.exit(main())
