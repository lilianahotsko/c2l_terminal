"""Built-in task: function/method body generation (``code_gen``).

Reference implementation of a *non-test* task on top of the same C2L adapter
machinery. Given a function signature plus its docstring (and the preceding
file context), predict the function body.

Extraction mirrors the assertion extractor's "full-file prefix" convention so
the repository context the model sees matches the paper pipeline:

* ``prefix`` -- the file content up to the first real body statement, including
  the leading indentation of that statement (i.e. signature + docstring + the
  in-line indent the body starts at).
* ``target`` -- the function body (first real statement through the end of the
  function).
"""

from __future__ import annotations

import ast
import warnings
from typing import List, Optional

from .base import QnA, Task

# Practical bounds so targets stay learnable / scorable.
MIN_BODY_CHARS = 8
MAX_BODY_LINES = 40
MIN_SIGNATURE_DOCSTRING = False  # if True, require a docstring


def _docstring_index(node: ast.AST) -> int:
    """Return index of the first non-docstring statement in ``node.body``."""
    body = getattr(node, "body", [])
    if not body:
        return 0
    first = body[0]
    if (isinstance(first, ast.Expr)
            and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)):
        return 1
    return 0


def _is_test_function(name: str, path: str) -> bool:
    if name.startswith("test_") or name == "test":
        return True
    low = path.lower()
    return "test" in low and (low.endswith("_test.py") or "/test" in low
                              or low.startswith("test"))


class CodeGenTask(Task):
    task_id = "code_gen"
    task_index = 1
    description = "Generate a function/method body from its signature + docstring."
    mines_test_files = False

    def extract_from_source(self, source: str, path: str = "") -> List[QnA]:
        if not source:
            return []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return []

        lines = source.splitlines(keepends=True)
        out: List[QnA] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_test_function(node.name, path):
                continue
            qna = self._function_qna(source, lines, node, path)
            if qna is not None:
                out.append(qna)
        return out

    def _function_qna(self, source: str, lines: List[str],
                      node: ast.AST, path: str) -> Optional[QnA]:
        body = getattr(node, "body", [])
        start = _docstring_index(node)
        if MIN_SIGNATURE_DOCSTRING and start == 0:
            return None
        real_body = body[start:]
        if not real_body:
            return None
        first = real_body[0]
        first_line = getattr(first, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        col = getattr(first, "col_offset", 0)
        if not first_line or not end_line or first_line < 1:
            return None
        if (end_line - first_line + 1) > MAX_BODY_LINES:
            return None

        # Skip trivial bodies (e.g. a single ``pass`` / ``...``).
        if len(real_body) == 1 and isinstance(first, (ast.Pass,)):
            return None
        if (len(real_body) == 1 and isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and first.value.value is Ellipsis):
            return None

        pre = "".join(lines[: first_line - 1])
        body_first = lines[first_line - 1] if first_line - 1 < len(lines) else ""
        indent = body_first[:col] if col <= len(body_first) else body_first
        prefix = pre + indent

        target_first = body_first[col:] if col <= len(body_first) else ""
        rest = "".join(lines[first_line:end_line])
        target = (target_first + rest).rstrip("\n")
        if len(target.strip()) < MIN_BODY_CHARS:
            return None

        return QnA(
            task=self.task_id,
            prefix=prefix,
            target=target,
            file=path,
            function=node.name,
            lineno=first_line,
            col_offset=col,
            kind="function_body",
        )

    def metric(self, pred: str, target: str) -> dict:
        # Multi-line aware EM/EditSim/CodeBLEU (compute_metrics keeps multiline
        # targets intact and only collapses single-line ones).
        from ..metrics import compute_metrics
        return compute_metrics(pred, target)


__all__ = ["CodeGenTask"]
