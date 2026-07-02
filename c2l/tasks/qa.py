"""Built-in task: repository question-answering (``qa``).

Unlike ``assert_rhs`` / ``code_gen``, QnAs for this task are **not mined from a
single source file** -- they are questions about a *change* (a commit diff),
answered in free-form natural language. The repository -> adapter generation is
task-agnostic (the GRU walks the same commit diff stream), so the only thing
this task contributes is:

* ``task_index = 1`` -- selects the QA row of the multi-task head's task
  embedding. This MUST match the order the checkpoint was trained with
  (``--tasks assert_rhs qa`` -> assert_rhs=0, qa=1), which is how the
  ``c2l_evo_multitask`` checkpoint was finetuned.
* a QA-appropriate ``metric`` (token-F1 / ROUGE-L when available) -- the
  code-oriented EM/EditSim/CodeBLEU under-rate free-form answers.

At inference the user simply types a question; the QA-conditioned LoRA steers
the frozen base model to answer it for the adapted repository. There is no
source mining, so :meth:`extract_from_source` returns an empty list.
"""

from __future__ import annotations

import re
import string
from typing import Dict, List

from .base import QnA, Task

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    s = (s or "").lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _token_f1(pred: str, ref: str) -> float:
    p, r = _normalize(pred).split(), _normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = 0
    ref_counts: Dict[str, int] = {}
    for t in r:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    seen: Dict[str, int] = {}
    for t in p:
        seen[t] = seen.get(t, 0) + 1
        if seen[t] <= ref_counts.get(t, 0):
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


class QaTask(Task):
    task_id = "qa"
    task_index = 1          # MUST match training order: assert_rhs=0, qa=1
    description = "Answer a natural-language question about the repository / a change."
    mines_test_files = False

    def extract_from_source(self, source: str, path: str = "") -> List[QnA]:
        # QA examples come from commit diffs (generated offline), not from a
        # single source file -- nothing to mine here.
        return []

    def format(self, qna: QnA) -> "tuple[str, str]":
        # Prompt is the question verbatim (matches the training prefix format).
        return qna.prefix, qna.target

    def metric(self, pred: str, target: str) -> Dict:
        f1 = _token_f1(pred, target)
        out: Dict[str, float] = {"token_f1": f1}
        try:
            from rouge_score import rouge_scorer
            rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(
                target or "", pred or "")
            rougeL = rs["rougeL"].fmeasure
            out["rougeL"] = rougeL
            out["qa_quality"] = 0.5 * (rougeL + f1)
        except Exception:
            out["qa_quality"] = f1
        return out


__all__ = ["QaTask"]
