"""Task abstraction for the C2L multi-task framework.

A *task* defines what the generated adapter is asked to do. Crucially, the
repository -> adapter generation (the GRU walk over commits) is **task
agnostic** -- the task only affects:

1. which (``prefix``, ``target``) pairs are mined for training / evaluation
   (:meth:`Task.extract_from_source`), and
2. which learned task embedding conditions the LoRA head (``task_index``),
3. how a prediction is scored (:meth:`Task.metric`).

New capabilities are therefore added as small registry plugins, not forks.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class QnA:
    """One supervised example: predict ``target`` given ``prefix``."""

    task: str
    prefix: str
    target: str
    # provenance (best-effort; used for display + dedup + execution checks)
    file: str = ""
    function: str = ""
    lineno: int = 0
    col_offset: int = 0
    kind: str = ""            # e.g. assertion type, or "function_body"
    meta: Dict[str, str] = field(default_factory=dict)


class Task(abc.ABC):
    """Base class for a C2L task plugin."""

    #: stable registry id, e.g. "assert_rhs"
    task_id: str = ""
    #: stable integer index used by the multi-task head's task embedding.
    #: MUST be stable across releases once a checkpoint is trained on it.
    task_index: int = 0
    #: one-line human description
    description: str = ""
    #: True if QnAs are mined from *test* files, False for production code.
    mines_test_files: bool = True

    @abc.abstractmethod
    def extract_from_source(self, source: str, path: str = "") -> List[QnA]:
        """Mine (prefix, target) QnAs from one source file's text."""

    def format(self, qna: QnA) -> "tuple[str, str]":
        """Return the (prompt_prefix, target) actually fed to the model.

        Default is identity; tasks may add instruction wrappers here.
        """
        return qna.prefix, qna.target

    def metric(self, pred: str, target: str) -> Dict:
        """Score a prediction against the target. Default: EM/EditSim/CodeBLEU."""
        from ..metrics import compute_metrics
        return compute_metrics(pred, target)


__all__ = ["QnA", "Task"]
