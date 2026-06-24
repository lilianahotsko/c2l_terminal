"""Pluggable task registry for C2L.

Register a task once; every surface (CLI, API, trainer, evaluator) can then
look it up by id. ``task_index`` is the stable integer used by the multi-task
head's task embedding and MUST stay constant once a checkpoint is trained.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .assert_rhs import AssertRhsTask
from .base import QnA, Task
from .code_gen import CodeGenTask

_REGISTRY: Dict[str, Task] = {}


def register(task: Task) -> Task:
    """Register a task instance (idempotent on id)."""
    if not task.task_id:
        raise ValueError("Task is missing a task_id")
    existing = _REGISTRY.get(task.task_id)
    if existing is not None and type(existing) is type(task):
        return existing
    # Guard against two tasks claiming the same head index.
    for other in _REGISTRY.values():
        if other.task_id != task.task_id and other.task_index == task.task_index:
            raise ValueError(
                f"task_index clash: {task.task_id!r} and {other.task_id!r} "
                f"both use index {task.task_index}")
    _REGISTRY[task.task_id] = task
    return task


def get_task(task_id: str) -> Task:
    try:
        return _REGISTRY[task_id]
    except KeyError:
        raise KeyError(
            f"Unknown task {task_id!r}. Registered: {sorted(_REGISTRY)}") from None


def list_tasks() -> List[str]:
    return sorted(_REGISTRY)


def num_tasks() -> int:
    """Number of task-embedding rows required (max index + 1)."""
    if not _REGISTRY:
        return 0
    return max(t.task_index for t in _REGISTRY.values()) + 1


def task_index(task_id: str) -> int:
    return get_task(task_id).task_index


# Register the built-ins.
register(AssertRhsTask())
register(CodeGenTask())


__all__ = [
    "QnA",
    "Task",
    "AssertRhsTask",
    "CodeGenTask",
    "register",
    "get_task",
    "list_tasks",
    "num_tasks",
    "task_index",
]
