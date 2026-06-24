"""Built-in task: test-assertion right-hand-side completion (``assert_rhs``).

This is the original Code2LoRA v2 task: given a test file up to an assertion's
right-hand side, predict the RHS. Wraps the AST extractor in
:mod:`c2l.assertions` so it stays identical to the paper pipeline.
"""

from __future__ import annotations

from typing import List

from ..assertions import extract_assertions_from_source
from .base import QnA, Task


class AssertRhsTask(Task):
    task_id = "assert_rhs"
    task_index = 0          # default task -> task-embedding row 0 (back-compat)
    description = "Complete the right-hand side of a test assertion."
    mines_test_files = True

    def extract_from_source(self, source: str, path: str = "") -> List[QnA]:
        out: List[QnA] = []
        for ex in extract_assertions_from_source(source):
            out.append(QnA(
                task=self.task_id,
                prefix=ex.prefix,
                target=ex.target,
                file=path,
                function=ex.test_function,
                lineno=ex.lineno,
                col_offset=ex.col_offset,
                kind=ex.assertion_type,
            ))
        return out


__all__ = ["AssertRhsTask"]
