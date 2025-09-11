"""Test that the `scipy.version` stub literals match runtime."""

# ruff: noqa: S101

import importlib.util
import sys
import types
from pathlib import Path

from scipy import version as version_scipy

PATH_STUBS = Path(__file__).parent.parent / "scipy-stubs"


def _import_pyi(name: str, path: str | Path) -> types.ModuleType:
    """Hack to import a `.pyi` file as a module."""
    spec = importlib.util.spec_from_loader(name, loader=None)
    if spec is None:
        raise ImportError(name=name, path=str(path))
    module = importlib.util.module_from_spec(spec)

    source = Path(path).read_text(encoding="utf-8")
    exec(source, module.__dict__)  # noqa: S102

    sys.modules[spec.name] = module
    return module


def main() -> None:
    version_scipyi = _import_pyi("scipyi.version", PATH_STUBS / "version.pyi")
    literals_scipyi = {
        n: v
        for n, v in vars(version_scipyi).items()
        if not n.startswith("_") and v is not Ellipsis
    }
    assert literals_scipyi, "No literals found in scipy-stubs/version.pyi"

    for name, val in literals_scipyi.items():
        val_expect = getattr(version_scipy, name, "<MISSING>")
        qname = f"scipy.version.{name}"
        assert val == val_expect, f"Expected `{qname} = {val_expect!r}`, got {val!r}."


if __name__ == "__main__":
    main()
