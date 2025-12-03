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
uv sync --exact --python 3.11
```

Installing the lowest supported Python version (3.11 in this example) prevents
your IDE from e.g. auto-importing unsupported `typing` features.

## Lefthook

[Lefthook](https://github.com/evilmartians/lefthook) is a modern Git hooks manager,
which automatically lints and formats your code before you commit it. It will also sync
your `uv` environment with the lockfile when you `git pull` or `git checkout`.

To install it as a `uv` tool, run

```shell
uv tool install lefthook --upgrade
```

To set it up, navigate to the root of the `scipy-stubs` repo, and run

```shell
uvx lefthook install
```

Now let's see if it all works:

```bash
$ uvx lefthook validate
All good
```

See <https://lefthook.dev/> for more information.

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
ty: OK ✔ in 0.52 seconds
lint: OK ✔ in 0.99 seconds
pyrefly: OK ✔ in 2.08 seconds
mypy: OK ✔ in 15.63 seconds
pyright: OK ✔ in 17.06 seconds
3.14: OK ✔ in 41.55 seconds
3.13: OK ✔ in 42.99 seconds
3.12: OK ✔ in 44.69 seconds
  lint: OK (0.99=setup[0.24]+cmd[0.60,0.04,0.04,0.06] seconds)
  ty: OK (0.52=setup[0.20]+cmd[0.32] seconds)
  pyrefly: OK (2.08=setup[0.31]+cmd[1.77] seconds)
  pyright: OK (17.06=setup[0.23]+cmd[16.83] seconds)
  mypy: OK (15.63=setup[0.19]+cmd[15.44] seconds)
  3.11: OK (45.98=setup[0.46]+cmd[18.92,26.60] seconds)
  3.12: OK (44.69=setup[0.35]+cmd[18.91,25.42] seconds)
  3.13: OK (42.99=setup[0.20]+cmd[18.45,24.35] seconds)
  3.14: OK (41.55=setup[0.45]+cmd[18.65,22.45] seconds)
  congratulations :) (46.00 seconds)
```

</details>

## Documentation

All [documentation] lives in the `README.md`. Please read it carefully before proposing
any changes. Ensure that the markdown is formatted correctly with
[dprint][dprint] by running:

```shell
uv run dprint fmt
```

## Testing

See the `README.md` in [`scipy-stubs/tests`][tests].

## Code style

See <https://typing.python.org/en/latest/guides/writing_stubs.html#style-guide>.

## Commit message style

scipy-stubs recommends using [Gitmoji](https://gitmoji.dev/) for commit messages and PR
titles. For VSCode and VSCodium users, it can be convenient to use the
[`gitmoji-vscode`](https://github.com/seatonjiang/gitmoji-vscode) extension for this.

[coc]: https://docs.scipy.org/doc/scipy/dev/conduct/code_of_conduct.html
[license]: https://github.com/scipy/scipy-stubs/blob/master/LICENSE
[tests]: https://github.com/scipy/scipy-stubs/tree/master/tests
[dprint]: https://github.com/dprint/dprint
