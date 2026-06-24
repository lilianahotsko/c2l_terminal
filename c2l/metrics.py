"""Evaluation metrics for the demo.

Faithful port of ``evaluation/metrics.py`` (exact match / edit similarity /
postprocessing) so the demo scores predictions exactly the way the paper does.
The optional ``codebleu`` dependency is not required; a tokenized BLEU
fallback is used when it is absent.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")

_CODE_TOKEN_RE = re.compile(
    r"[a-zA-Z_]\w*|0[xXoObB][\da-fA-F_]+|\d[\d_]*\.?\d*[eE]?[\d_]*"
    r"|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[^\s]"
)


def strip_fim_tokens(s: str) -> str:
    for tok in FIM_TOKENS:
        s = s.replace(tok, "")
    return s.strip()


def strip_comments(s: str) -> str:
    return s.split("#")[0].strip()


def normalize_for_match(s: str) -> str:
    s = s.strip().rstrip(":, \t")
    s = s.replace(", ", ",").replace(" ,", ",")
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> bool:
    return normalize_for_match(pred) == normalize_for_match(ref)


def edit_similarity(pred: str, ref: str) -> float:
    return SequenceMatcher(None, pred, ref).ratio()


def _tokenize_code(s: str):
    return _CODE_TOKEN_RE.findall(s)


def _ngram_bleu(pred_tokens, ref_tokens, max_n: int = 4) -> float:
    from collections import Counter
    import math

    if not pred_tokens or not ref_tokens:
        return 0.0
    brevity = min(1.0, len(pred_tokens) / len(ref_tokens))
    log_bleu = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = max(len(pred_tokens) - n + 1, 1)
        precision = (matches + 1) / (total + 1)
        log_bleu += math.log(precision) / max_n
    return brevity * math.exp(log_bleu)


def code_bleu_score(pred: str, ref: str, lang: str = "python") -> float:
    if not pred.strip() or not ref.strip():
        return 0.0
    if pred.strip() == ref.strip():
        return 1.0
    try:
        from codebleu import calc_codebleu
        return calc_codebleu([ref], [pred], lang=lang)["codebleu"]
    except Exception:
        return _ngram_bleu(_tokenize_code(pred), _tokenize_code(ref))


def postprocess_prediction(pred: str, target: str) -> str:
    pred = strip_fim_tokens(pred)
    if "\n" not in target and "\n" in pred:
        pred = pred.split("\n")[0]
    pred = strip_comments(pred)
    ref = strip_comments(target).strip()
    if ref:
        if "," not in ref and "," in pred:
            pred = pred.split(",")[0].strip()
        if len(ref.split()) == 1 and len(pred.strip().split()) > 1:
            pred = pred.strip().split()[0]
    return pred.strip()


def compute_metrics(pred: str, ref: str) -> dict:
    pred_clean = postprocess_prediction(pred, ref)
    ref_clean = strip_comments(ref)
    return {
        "exact_match": exact_match(pred_clean, ref_clean),
        "edit_similarity": edit_similarity(pred_clean, ref_clean),
        "code_bleu": code_bleu_score(pred_clean, ref_clean),
        "pred_clean": pred_clean,
        "ref_clean": ref_clean,
    }


__all__ = [
    "compute_metrics",
    "postprocess_prediction",
    "exact_match",
    "edit_similarity",
    "code_bleu_score",
    "normalize_for_match",
    "strip_comments",
]
