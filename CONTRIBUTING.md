# scipy-stubs pull request guidelines

Pull requests are always welcome, and the SciPy community appreciates any help you
provide. Note that a [Code of Conduct][coc] applies to all spaces managed by the
SciPy project, including issues and pull requests.

When submitting a pull request, we ask you to check the following:

1. [Tests](#testing), [documentation](#documentation), and [code style](#code-style)
   are in order, and no errors are reported by type-checkers and stubgen.
   For details, see the [*Local development*](#local-development) section.

   It's also OK to submit work in progress if you're unsure of what this exactly means,
   in which case you'll likely be asked to make some further changes.

1. The contributed code will be **licensed under scipy-stubs' [license]**.
   If you did not write the code yourself, you must ensure the existing license is
   compatible and include the license information in the contributed files, or obtain
   permission from the original author to relicense the contributed code.

## Local development

Ensure you have [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
installed. Now you can install the project with the dev dependencies:

```shell
uv sync --python 3.11
```

By installing the lowest support Python version (3.11 in this example), it prevents
your IDE from e.g. auto-importing unsupported `typing` features.

### Tox

The linters, type-checkers, and `stubtest` can easily be run with
[tox](https://github.com/tox-dev/tox). It can be installed as a `uv` tool:

```shell
uv tool install tox --with tox-uv --upgrade
```

To run all environments (in parallel), run:

```shell
uvx tox p
```

<details>
<summary>Output:</summary>

```plaintext
lint: OK ✔ in 0.52 seconds
3.11: OK ✔ in 21.59 seconds
mypy: OK ✔ in 21.62 seconds
pyright: OK ✔ in 25.23 seconds
3.10: OK ✔ in 25.4 seconds
3.12: OK ✔ in 38.71 seconds
  lint: OK (0.52=setup[0.04]+cmd[0.41,0.03,0.05] seconds)
  pyright: OK (25.23=setup[0.03]+cmd[25.20] seconds)
  mypy: OK (21.62=setup[0.03]+cmd[21.59] seconds)
  3.13: OK (53.28=setup[0.03]+cmd[53.25] seconds)
  3.12: OK (38.71=setup[0.03]+cmd[38.68] seconds)
  3.11: OK (21.59=setup[0.03]+cmd[21.55] seconds)
  3.10: OK (25.40=setup[0.03]+cmd[25.36] seconds)
  congratulations :) (53.35 seconds)
```

</details>

## Documentation

All [documentation] lives in the `README.md`. Please read it carefully before proposing
any changes. Ensure that the markdown is formatted correctly with
[markdownlint](https://github.com/DavidAnson/markdownlint/tree/main).

## Testing

See the `README.md` in [`scipy-stubs/tests`][tests].

## Code style

See <https://typing.readthedocs.io/en/latest/guides/writing_stubs.html#style-guide>.

## Commit message style

Scipy-stubs recommends using [Gitmoji](https://gitmoji.dev/) for commit messages and PR
titles. For VSCode and VSCodium users, it is recommended to use the
[`gitmoji-vscode` extension](https://github.com/seatonjiang/gitmoji-vscode) for this.

[coc]: https://docs.scipy.org/doc/scipy/dev/conduct/code_of_conduct.html
[license]: https://github.com/scipy/scipy-stubs/blob/master/LICENSE
[tests]: https://github.com/scipy/scipy-stubs/tree/master/tests
