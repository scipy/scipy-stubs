"""Walk through tests/ and collect the qualnames of accessed scipy.* names."""

# ruff: noqa: S101, T201

import ast
import importlib
import sys
from pathlib import Path
from typing import Final

_SCIPY_PREFIX: Final = "scipy."
_IGNORED_SUFFIXES: Final = (".__class__",)

_SCIPY_SUBPACKAGES: Final = (
    "cluster",
    "cluster.hierarchy",
    "cluster.vq",
    "constants",
    "datasets",
    "differentiate",
    "fft",
    "fftpack",
    "integrate",
    "integrate._rules",
    "interpolate",
    "io",
    "io.arff",
    "io.matlab",
    "linalg",
    "ndimage",
    "odr",
    "optimize",
    "signal",
    "signal.windows",
    "sparse",
    "sparse.csgraph",
    "sparse.linalg",
    "spatial",
    "spatial.transform",
    "special",
    "stats",
    "stats.contingency",
    "stats.mstats",
    "stats.qmc",
    "stats.sampling",
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

    return parts


def _resolve_to_qualname(parts: list[str], imports: dict[str, str]) -> str | None:
    """Resolve an attribute chain to a fully qualified scipy name."""
    if not parts:
        return None

    full_name = ".".join(parts)

    # direct scipy.* access
    if full_name.startswith(_SCIPY_PREFIX):
        return full_name

    # resolve through imports
    if parts[0] in imports:
        return f"{imports[parts[0]]}.{'.'.join(parts[1:])}"

    return None


def _parse_scipy_imports(tree: ast.AST) -> dict[str, str]:
    """Extract scipy imports, returning a mapping of local name -> qualified name."""
    imports: dict[str, str] = {}

    for node in ast.walk(tree):
        match node:
            case ast.ImportFrom(module=str(m), names=aliases) if m.startswith(
                _SCIPY_PREFIX
            ):
                for alias in aliases:
                    imports[alias.asname or alias.name] = f"{m}.{alias.name}"

            case ast.Import(names=aliases):
                for alias in aliases:
                    if alias.name.startswith(_SCIPY_PREFIX):
                        imports[alias.asname or alias.name] = alias.name

            case _:
                pass

    return imports


def _should_ignore(qualname: str) -> bool:
    """Check if a qualified name should be ignored (private or bare package)."""
    parts = qualname.split(".")
    is_private = any(part.startswith("_") for part in parts[1:])
    is_bare_package = len(parts) < 3  # noqa: PLR2004
    return qualname.endswith(_IGNORED_SUFFIXES) or is_private or is_bare_package


def _find_scipy_names_in_tree(tree: ast.AST, imports: dict[str, str]) -> set[str]:
    """Find all scipy names that are actually used in the AST."""
    used: set[str] = set()

    for node in ast.walk(tree):
        qualname: str | None = None

        if isinstance(node, ast.Name) and node.id in imports:
            qualname = imports[node.id]
        elif isinstance(node, ast.Attribute):
            qualname = _resolve_to_qualname(_extract_attribute_chain(node), imports)

        if qualname and not _should_ignore(qualname):
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

    for subpkg in _SCIPY_SUBPACKAGES:
        qualname = f"scipy.{subpkg}"
        try:
            module = importlib.import_module(qualname)
        except ImportError:
            continue

        exported = getattr(module, "__all__", None)
        if exported is not None:
            all_names.update(f"{qualname}.{name}" for name in exported)

    return all_names


def main() -> int:
    public = names_public()
    tested = names_tested() & public

    package = "scipy.none"
    package_public: int = 0
    package_tested: int = 0

    def _print_package_coverage() -> None:
        if package_public:
            print()
            print(
                f"Coverage `{package}`: "
                f"{package_tested} / {package_public} "
                f"({package_tested / package_public:.2%})"
            )

    for name in sorted(public):
        if not name.startswith(package):
            _print_package_coverage()
            package = ".".join(name.split(".", 2)[:2])
            package_public = package_tested = 0
            print(f"\n## `{package}`\n")

        package_public += 1
        package_tested += name in tested
        x = "x" if name in tested else " "
        print(f"- [{x}] `{name}`")

    _print_package_coverage()

    print()
    print(
        f"Total coverage: "
        f"{len(tested)} / {len(public)} "
        f"({len(tested) / len(public):.2%})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
