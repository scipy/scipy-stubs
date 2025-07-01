# scipy-stubs

Precise type hints for **all** of <a href="https://github.com/scipy/scipy">SciPy</a>.

[![PyPI](https://img.shields.io/pypi/v/scipy-stubs?color=blue&style=flat-square)](https://pypi.org/project/scipy-stubs/)
[![scipy-stubs - conda-forge](https://anaconda.org/conda-forge/scipy-stubs/badges/version.svg)](https://anaconda.org/conda-forge/scipy-stubs)
![Python Versions](https://img.shields.io/pypi/pyversions/scipy-stubs?color=blue&style=flat-square)
![license](https://img.shields.io/github/license/scipy/scipy-stubs?color=violet&style=flat-square)
![PyPI Downloads](https://img.shields.io/pypi/dm/scipy-stubs?color=violet&style=flat-square)

[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright](https://img.shields.io/badge/basedpyright-checked-42b983)](https://detachhead.github.io/basedpyright)
[![pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://github.com/microsoft/pyright)
[![mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
![typed](https://img.shields.io/pypi/types/scipy-stubs?color=white)

## Quick Start

Install `scipy-stubs` and start getting better type hints immediately:

```bash
pip install scipy-stubs
```

That's it! Your IDE and type checker will now provide precise type information for SciPy functions:

## Examples

Prevent mistakes with precise type hints:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/e50eb3db-7cb5-41e7-a56b-a563e9bd28d6">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/7905d0ea-693c-4b5f-aaf9-4af0d11a520f">
  <img alt="bug prevention demo" src="https://github.com/user-attachments/assets/7905d0ea-693c-4b5f-aaf9-4af0d11a520f">
</picture>

Accurate annotations for dtypes and shapes:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c8a5d204-13ca-4fe8-8d83-95d55ca9b9df">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/2170df07-3491-480e-985f-24494e7d5d3f">
  <img alt="precise type inference demo" src="https://github.com/user-attachments/assets/2170df07-3491-480e-985f-24494e7d5d3f">
</picture>

## Why use scipy-stubs?

### Enhanced Development Experience

- **Better IDE support**: Get accurate autocompletion, parameter hints, and return type information
- **Catch errors early**: Type checkers can detect mistakes before runtime
- **Improved code documentation**: Type hints serve as inline documentation for function signatures

### Zero Configuration Required

- **Drop-in replacement**: Works immediately after installation, no configuration needed
- **No runtime impact**: Type stubs are only used during development and type checking
- **IDE agnostic**: Works with VSCode, PyCharm, Vim, Emacs, and any editor with Python language server support

### Precise and Complete

- **Array shape awareness**: Many functions include shape-type information for better array handling
- **Generic types**: Comprehensive generic classes for sparse arrays, distributions, linear operators, and more
- **Complete coverage**: Type hints are provided for the entire SciPy API

<!-- NOTE: SciPy permalinks to the following `#installation` anchor; don't modify it! -->

## Installation

The source code is hosted on GitHub at [github.com/scipy/scipy-stubs](https://github.com/scipy/scipy-stubs/).

Binary distributions are available on [PyPI](https://pypi.org/project/scipy-stubs/) and
[conda-forge](https://anaconda.org/conda-forge/scipy-stubs).

### Using pip (PyPI)

To install from the [PyPI](https://pypi.org/project/scipy-stubs/), run:

```bash
pip install scipy-stubs
```

In case you haven't installed `scipy` yet, both can be installed with:

```bash
pip install scipy-stubs[scipy]
```

### Using conda (conda-forge)

To install using Conda from the [conda-forge channel](https://anaconda.org/conda-forge/scipy-stubs), run:

```bash
conda install conda-forge::scipy-stubs
```

It's also possible to install both `scipy` and `scipy-stubs` together through the bundled
[`scipy-typed`](https://anaconda.org/conda-forge/scipy-typed) package:

```bash
conda install conda-forge::scipy-typed
```

## Frequently Asked Questions

### Q: Do I need to change my existing code?

**A:** No! `scipy-stubs` works with your existing code without any modifications.
Just install it and your type checker and IDE will automatically use the type information.

### Q: Will this slow down my code?

**A:** Not at all. Type stubs are only used during development and type checking.
They have zero runtime overhead since they're not imported when your code runs.

### Q: What if I don't use type hints in my code?

**A:** You'll still benefit! Your IDE will provide better autocompletion and error detection
even without explicit type annotations in your code.

### Q: Can I use this with Jupyter notebooks?

**A:** Yes! Most modern Jupyter environments (JupyterLab, VS Code notebooks) support
type checking and will benefit from `scipy-stubs`.

### Q: What's the difference between this and the built-in scipy typing?

**A:** SciPy itself has limited type annotations. `scipy-stubs` provides comprehensive,
precise type information for the entire SciPy API, including shape-typing and advanced type features.

### Q: How do I know if it's working?

**A:** You should see improved autocompletion in your IDE and more precise type information.
You can also run `pyright` or another type checker on your code to see type checking in action.

### Q: How much of SciPy is covered?

**A:** All of it! If you find any missing or incorrect type annotations, please open an issue on [GitHub](https://github.com/scipy/scipy-stubs/issues).

### Q: What static type-checkers are supported?

**A:** `scipy-stubs` is compatible with [`pyright`](https://pyright.readthedocs.io/en/latest/index.html) (a.k.a. pylance),
[`basedpyright`](https://github.com/DetachHead/basedpyright), and [`mypy`](https://github.com/python/mypy).
We only support the latest versions of these type-checkers, so make sure to keep them up to date.

## Versioning and requirements

The versioning scheme of `scipy-stubs` includes the compatible `scipy` version as `{scipy_version}.{stubs_version}`.
Even though `scipy-stubs` doesn't enforce an upper bound on the `scipy` version, later `scipy` versions aren't guaranteed to be
fully compatible.

There are no additional restrictions enforced by `scipy-stubs` on the `numpy` requirements.
For `scipy-stubs==1.16.*` that is `numpy >= 1.25.2`.

Currently, `scipy-stubs` has one required dependency: [`optype`](https://github.com/jorenham/optype).
This is essential for `scipy-stubs` to work properly, as it relies heavily on it for annotating (shaped) array-likes,
scalar-likes, shape-typing in general, and much more. At the moment, `scipy-stubs` requires the latest version `optype`.

The exact version requirements are specified in the [`pyproject.toml`](pyproject.toml).

## Generics

`scipy-stubs` provides many generic classes that enable precise type checking for SciPy's complex APIs.

All generic type parameters are optional and can be omitted if not needed.

Note that not all classes are subscriptable at runtime, as that requires the `__class_getitem__` method to be implemented in `scipy`.
This can be worked around with `from __future__ import annotations` or by manually stringifying the generic annotations.
We are working on improving this in future versions of SciPy.
See the `scipy` columns below for which classes are subscriptable at runtime.

### `scipy.integrate`

| generic type             | `scipy-stubs` | `scipy`  |                                                                                               |
| ------------------------ | ------------- | -------- | --------------------------------------------------------------------------------------------- |
| `BDF[T: f64 \| c128]`    | `>=1.14.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html)         |
| `DOP853[T: f64 \| c128]` | `>=1.14.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html)      |
| `RK23[T: f64 \| c128]`   | `>=1.14.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK23.html)        |
| `RK45[T: f64 \| c128]`   | `>=1.14.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html)        |
| `ode[*ArgTs]`            | `>=1.14.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html)         |
| `complex_ode[*ArgTs]`    | `>=1.14.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.html) |

### `scipy.interpolate`

| generic type                                       | `scipy-stubs` | `scipy`  |                                                                                                                |
| -------------------------------------------------- | ------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| `AAA[T: inexact]`                                  | `>=1.15.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.AAA.html)                        |
| `BarycentricInterpolator[T: f64 \| c128]`          | `>=1.16.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.html)    |
| `BPoly[T: f64 \| c128]`                            | `>=1.14.1.4`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.html)                      |
| `BSpline[T: f64 \| c128]`                          | `>=1.14.1.6`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)                    |
| `CubicHermiteSpline[T: f64 \| c128]`               | `>=1.14.1.4`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html)         |
| `CubicSpline[T: f64 \| c128]`                      | `>=1.14.1.4`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)                |
| `FloaterHormannInterpolator[T: f64 \| c128]`       | `>=1.15.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.FloaterHormannInterpolator.html) |
| `KroghInterpolator[T: f64 \| c128, S: (int, ...)]` | `>=1.16.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.KroghInterpolator.html)          |
| `LinearNDInterpolator[T: f64 \| c128]`             | `>=1.15.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html)       |
| `NdBSpline[T: f64 \| c128]`                        | `>=1.15.2.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdBSpline.html)                  |
| `NdPPoly[T: f64 \| c128]`                          | `>=1.14.1.4`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.html)                    |
| `NearestNDInterpolator[T: f64 \| c128]`            | `>=1.14.1.6`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html)      |
| `PPoly[T: f64 \| c128]`                            | `>=1.14.1.4`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.html)                      |
| `RBFInterpolator[T: f64 \| c128, S: (int, ...)]`   | `>=1.16.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)            |
| `RegularGridInterpolator[T: f64 \| c128]`          | `>=1.14.1.6`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html)    |

### `scipy.optimize`

| generic type                            | `scipy-stubs` | `scipy`  |                                                                                                  |
| --------------------------------------- | ------------- | -------- | ------------------------------------------------------------------------------------------------ |
| `BroydenFirst[T: inexact]`              | `>=1.15.2.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.html)    |
| `InverseJacobian[T: inexact]`           | `>=1.15.2.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.InverseJacobian.html) |
| `KrylovJacobian[T: inexact]`            | `>=1.15.2.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.KrylovJacobian.html)  |
| `Bounds[S: (int, int, ...), T: scalar]` | `>=1.16.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html)          |

### `scipy.signal`

| generic type                                         | `scipy-stubs` | `scipy`  |                                                                                                 |
| ---------------------------------------------------- | ------------- | -------- | ----------------------------------------------------------------------------------------------- |
| `ShortTimeFFT[T: inexact]`                           | `>=1.16.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html)     |
| `StateSpace[Z: inexact, P: floating, D: scalar]`     | `>=1.15.2.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.html)       |
| `TransferFunction[P: floating, D: scalar]`           | `>=1.15.2.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html) |
| `ZerosPolesGain[Z: inexact, P: floating, D: scalar]` | `>=1.15.2.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZerosPolesGain.html)   |
| `lti[Z: inexact, P: floating]`                       | `>=1.15.2.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.html)              |
| `dlti[Z: inexact, P: floating, D: scalar]`           | `>=1.15.2.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.html)             |

### `scipy.sparse`

| generic type                                    | `scipy-stubs` | `scipy`  |                                                                                           |
| ----------------------------------------------- | ------------- | -------- | ----------------------------------------------------------------------------------------- |
| `bsr_array[T: scalar]`                          | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.html)  |
| `bsr_matrix[T: scalar]`                         | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html) |
| `coo_array[T: scalar, S: (int, ...)]`           | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html)  |
| `coo_matrix[T: scalar]`                         | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html) |
| `csc_array[T: scalar]`                          | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html)  |
| `csc_matrix[T: scalar]`                         | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html) |
| `csr_array[T: scalar, S: (int,) \| (int, int)]` | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html)  |
| `csr_matrix[T: scalar]`                         | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) |
| `dia_array[T: scalar]`                          | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.html)  |
| `dia_matrix[T: scalar]`                         | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html) |
| `dok_array[T: scalar, S: (int,) \| (int, int)]` | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.html)  |
| `dok_matrix[T: scalar]`                         | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html) |
| `lil_array[T: scalar]`                          | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.html)  |
| `lil_matrix[T: scalar]`                         | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html) |
| `sparray[T: scalar, S: (int, ...)]`             | `>=1.15.2.0`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.sparray.html)    |
| `spmatrix[T: scalar]`                           | `>=1.14.1.6`  | `>=1.16` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.html)   |

#### `scipy.sparse.linalg`

| generic type                | `scipy-stubs` | `scipy`  |                                                                                                      |
| --------------------------- | ------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `LaplacianNd[T: real]`      | `>=1.14.1.6`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.html)    |
| `LinearOperator[T: scalar]` | `>=1.14.1.6`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) |
| `SuperLU[T: inexact]`       | `>=1.16.0.1`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html)        |

### `scipy.stats`

| generic type                                   | `scipy-stubs` | `scipy`  |                                                                                          |
| ---------------------------------------------- | ------------- | -------- | ---------------------------------------------------------------------------------------- |
| `Covariance[T: real]`                          | `>=1.14.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Covariance.html) |
| `Uniform[S: (int, ...), T: floating]`          | `>=1.15.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Uniform.html)    |
| `Normal[S: (int, ...), T: floating]`           | `>=1.15.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Normal.html)     |
| `Binomial[S: (int, ...), T: floating]`         | `>=1.16.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Binomial.html)   |
| `Mixture[T: floating]`                         | `>=1.15.0.0`  | `>=1.17` | [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Mixture.html)    |
| `rv_frozen[D: rv_generic, T: scalar or array]` | `>=1.14.0.0`  | `>=1.17` |                                                                                          |
| `multi_rv_frozen[D: rv_generic]`               | `>=1.14.0.0`  | `>=1.17` |                                                                                          |

## Contributing

We welcome contributions from the community! There are many ways to help improve `scipy-stubs`:

### Ways to Contribute

- **Report issues**: Found a bug or incorrect type annotation? [Open an issue](https://github.com/scipy/scipy-stubs/issues)
- **Improve stubs**: Fix or enhance `.pyi` files (see [CONTRIBUTING.md](https://github.com/scipy/scipy-stubs/blob/master/CONTRIBUTING.md))
- **Add tests**: Help with type-testing (see the `README.md` in [`tests/`](https://github.com/scipy/scipy-stubs/tree/master/tests))
- **Documentation**: Write guides, examples, or improve existing documentation
- **Spread the word**: Help others discover `scipy-stubs`

### Development Setup

See the [CONTRIBUTING.md](https://github.com/scipy/scipy-stubs/blob/master/CONTRIBUTING.md) for detailed instructions.

## License

`scipy-stubs` is licensed under the [BSD 3-Clause License](https://github.com/scipy/scipy-stubs/blob/master/LICENSE),
the same as SciPy itself.
