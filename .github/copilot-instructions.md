# Copilot Instructions for scipy-stubs

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for environment setup, the full check suite
(`tox`), the `lefthook` git hooks (which lint and format on commit), markdown formatting
(`dprint`), code style, and commit-message conventions. The notes below only cover
what's specific to writing the stubs and running individual checks.

## Type annotations

Use `*Ts` instead of `Unpack[Ts]`.
The `typing_extensions` module is used for type annotations that are not yet available
in the standard library, which is needed for `T = TypeVar("T", default=...)`,
`@deprecated`, etc.

## Linting and formatting

```bash
uv run ruff check   # lint
uv run ruff format  # format
```

## Static Type Checking

Static type checking is performed with four type-checkers: `basedpyright`, `mypy`,
`pyrefly`, and `ty`. The stubs are checked by all four; the type-tests by all except
`ty` (which runs on the stubs only).

To check the stubs, run:

```bash
uv run basedpyright scipy-stubs
uv run --no-editable mypy scipy-stubs
uv run pyrefly check scipy-stubs
uv run ty check scipy-stubs
```

To check the type-tests instead, run:

```bash
uv run basedpyright tests
uv run --no-editable mypy tests
uv run pyrefly check tests
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
