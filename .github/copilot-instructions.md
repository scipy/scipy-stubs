# Copilot Instructions for scipy-stubs

Whenever you run a command in the terminal, pipe the output to a file, `output.txt`,
that you can read from. Do not attempt to read the output of commands from the terminal
directly, as Copilot will not be able to read it correctly. Make sure to overwrite each
time so that it doesn't grow too big. There is a bug in the current version of Copilot
that causes it to not read the output of commands correctly. This workaround allows you
to read the output from the temporary file instead. Be sure to delete `output.txt` after
you are done, as it is not included in `.gitignore`, and might contain sensitive
information.

## Type annotations

Use `A | B` instead of `Union[A, B]`.
Use `*Ts` instead of `Unpack[Ts]`.
Use `type[T]` and `list[T]` instead of `Type[T]` and `List[T]`.
Import `Sequence`, `Mapping`, `Iterable`, `Iterator`, `Generator`, and `Callable` from
`collections.abc` instead of `typing`.
The `typing_extensions` module is used for type annotations that are not yet available
in the standard library, which is needed for `T = TypeAlias("T", default=...)`,
`Alias = TypeAliasType("Alias", ...)`, `@deprecated`, etc.
We currently require Python 3.11 or higher.

## Environment Setup

We use UV to manage the project dependencies. To install the dependencies, run:

```bash
uv install
```

## Linting and formatting

For linting, we use `ruff`. To run the linter, execute:

```bash
uv run ruff check --fix
```

To format the code, we use `black`. To format the code, execute:

```bash
uv run ruff format
```

Be sure to format the code before committing any changes.

Lines can be 130 characters long in `*.pyi` stubs.

## Static Type Checking

Static type checking of `scipy-stubs`, and its type-tests, is performed using
`basedpyright` and `mypy`. To run these checks, execute the following commands:

```bash
uv run basedpyright
uv run --no-editable mypy .
```

If only the tests should be checked, you can run:

```bash
uv run basedpyright tests
uv run --no-editable mypy tests
```

Note that the errors reported by the vscode mypy plugin can take a long time to update.
If you think that could be the case, then feel free to ignore those errors.

## Stubtest

To run stubtest, execute the following command:

```bash
poe stubtest
```

If `poe` is not available, then you can find the full stubtest command under the
`[tool.poe.tasks.stubtest]` section in the `pyproject.toml` at the root of the project.

## Tox

If you want to run all checks in parallel and with multiple Python versions, run:

```bash
uvx tox p
```
