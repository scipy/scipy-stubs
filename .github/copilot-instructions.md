We use UV to manage the project dependencies. To install the dependencies, run:

```bash
uv install
```

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

To run stubtest, execute the following command:

```bash
poe stubtest
```

For linting, we use `ruff`. To run the linter, execute:

```bash
uv run ruff check --fix
```

To format the code, we use `black`. To format the code, execute:

```bash
uv run ruff format
```

If you want to run all checks in parallel, you can use `tox`:

```bash
uvx tox p
```

This will run all the checks in parallel, including linting, type checking, and stub testing.
