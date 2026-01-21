"""
Collect the number of overloads for each function in scipy-stubs/** as a JSON file of
fqname -> overload count.
"""

import argparse
import ast
import json
from pathlib import Path
from typing_extensions import override

STUBS_PATH = Path(__file__).parent.parent / "scipy-stubs"


class OverloadCounter(ast.NodeVisitor):
    def __init__(self, module: str, counts: dict[str, int]) -> None:
        super().__init__()
        self.module = module
        self.counts = counts
        self.class_stack: list[str] = []

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        _ = self.class_stack.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        fqname = ".".join([self.module, *self.class_stack, node.name])
        if any(self._is_overload(decorator) for decorator in node.decorator_list):
            self.counts[fqname] = self.counts.get(fqname, 0) + 1
        elif fqname not in self.counts:
            self.counts[fqname] = 1

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        fqname = ".".join([self.module, *self.class_stack, node.name])
        if any(self._is_overload(decorator) for decorator in node.decorator_list):
            self.counts[fqname] = self.counts.get(fqname, 0) + 1
        elif fqname not in self.counts:
            self.counts[fqname] = 1

    @staticmethod
    def _is_overload(node: ast.expr) -> bool:
        match node:
            case ast.Name(id="overload"):
                return True
            case ast.Attribute(attr="overload"):
                return True
            case _:
                return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect overload counts in scipy-stubs"
    )
    _ = parser.add_argument(
        "output", nargs="?", default="overload_stats.json", help="Output JSON file path"
    )
    args = parser.parse_args()

    counts: dict[str, int] = {}
    for pyi_path in STUBS_PATH.rglob("*.pyi"):
        rel = pyi_path.relative_to(STUBS_PATH)
        parts = list(rel.parts)
        if parts[-1] == "__init__.pyi":
            module_parts = parts[:-1]
        else:
            module_parts = [*parts[:-1], Path(parts[-1]).stem]

        module = ".".join(["scipy", *module_parts]) if module_parts else "scipy"
        tree = ast.parse(pyi_path.read_text(encoding="utf-8"), type_comments=True)
        OverloadCounter(module, counts).visit(tree)

    output_path = Path(args.output)
    _ = output_path.write_text(
        json.dumps(dict(sorted(counts.items())), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
