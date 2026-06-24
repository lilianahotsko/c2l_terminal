"""Assertion extraction from Python test files.

Faithful port of the AST-based assertion parsers in
``create_dataset/create_qnas.py`` plus the full-file-prefix extractor
``extract_assertions_from_source`` from
``create_dataset/build_commit_parquet_db.py``.

A QnA pair is one assertion turned into (prefix, target):

* ``prefix`` -- the full test-file content up to (but not including) the
  assertion's right-hand side, plus the in-line prefix (e.g. ``assert x == ``).
* ``target`` -- the right-hand side the model must predict (e.g. ``42``).
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass
from typing import List, Optional

SELF_ASSERT_METHODS = {
    "assertEqual", "assertNotEqual", "assertTrue", "assertFalse",
    "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone",
    "assertIn", "assertNotIn", "assertIsInstance", "assertNotIsInstance",
    "assertRaises", "assertRaisesRegex", "assertAlmostEqual", "assertNotAlmostEqual",
    "assertDictEqual", "assertListEqual", "assertTupleEqual", "assertSetEqual",
    "assertSequenceEqual", "assertRegex", "assertNotRegex", "assertCountEqual",
    "assertWarns", "assertLogs", "assertMultiLineEqual",
    "assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual",
}


@dataclass
class AssertionPair:
    assertion_type: str
    prefix: str
    target: str
    lineno: int
    col_offset: int


def get_source_segment(source: str, node: ast.AST) -> Optional[str]:
    if hasattr(ast, "get_source_segment"):
        return ast.get_source_segment(source, node)
    lines = source.splitlines()
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1) - 1
    if start < 0 or end >= len(lines):
        return None
    return "\n".join(lines[start:end + 1])


def flatten_to_oneliner(s: str) -> str:
    if not s or "\n" not in s:
        return s
    return " ".join(s.split())


# --- 1. plain assert -------------------------------------------------------

def _parse_plain_assert(source, node, lines):
    src = get_source_segment(source, node)
    if not src:
        return None
    lineno, col = node.lineno, node.col_offset
    test = node.test
    if isinstance(test, ast.Compare):
        left_src = get_source_segment(source, test.left)
        if not left_src:
            return None
        op = test.ops[0]
        comparators = test.comparators
        if not comparators:
            return None
        rhs_src = get_source_segment(source, comparators[0])
        if not rhs_src:
            return None
        op_map = {
            ast.Eq: " == ", ast.NotEq: " != ", ast.Lt: " < ", ast.LtE: " <= ",
            ast.Gt: " > ", ast.GtE: " >= ", ast.In: " in ", ast.NotIn: " not in ",
            ast.Is: " is ", ast.IsNot: " is not ",
        }
        op_str = op_map.get(type(op), " ??? ")
        prefix = f"assert {left_src}{op_str}"
        target = rhs_src
    else:
        prefix = "assert "
        target = get_source_segment(source, test) or ""
    return AssertionPair("assert", prefix, target, lineno, col)


# --- 2. self.assert* -------------------------------------------------------

def _parse_self_assert(source, node, lines):
    if not isinstance(node.func, ast.Attribute):
        return None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "self":
        return None
    method = node.func.attr
    if method not in SELF_ASSERT_METHODS:
        return None
    args = node.args
    if not args:
        return None
    if not get_source_segment(source, node):
        return None

    if method == "assertRaises":
        exc_src = get_source_segment(source, args[0])
        if exc_src:
            return AssertionPair(f"self.{method}", "self.assertRaises(",
                                 exc_src + ")", node.lineno, node.col_offset)
        return None
    if method == "assertRaisesRegex":
        if len(args) >= 2:
            exc_src = get_source_segment(source, args[0])
            pat_src = get_source_segment(source, args[1])
            if exc_src and pat_src:
                return AssertionPair(f"self.{method}", "self.assertRaisesRegex(",
                                     f"{exc_src}, {pat_src})",
                                     node.lineno, node.col_offset)
        return None
    if method in ("assertTrue", "assertFalse", "assertIsNone", "assertIsNotNone"):
        arg_src = get_source_segment(source, args[0])
        if arg_src:
            return AssertionPair(f"self.{method}", f"self.{method}(",
                                 arg_src + ")", node.lineno, node.col_offset)
        return None
    if len(args) >= 2:
        last_src = get_source_segment(source, args[-1])
        if last_src:
            prefix_parts = [f"self.{method}("]
            for i, arg in enumerate(args[:-1]):
                arg_src = get_source_segment(source, arg)
                if arg_src:
                    prefix_parts.append(arg_src)
                    if i < len(args) - 2:
                        prefix_parts.append(", ")
            prefix = "".join(prefix_parts) + ", "
            return AssertionPair(f"self.{method}", prefix, last_src + ")",
                                 node.lineno, node.col_offset)
    return None


# --- 3. pytest.raises ------------------------------------------------------

def _parse_pytest_raises(source, node, lines):
    if not isinstance(node.func, ast.Attribute):
        return None
    val = node.func.value
    if not isinstance(val, ast.Name) or val.id != "pytest" or node.func.attr != "raises":
        return None
    args = node.args
    if not args:
        return None
    exc_src = get_source_segment(source, args[0])
    if not exc_src:
        return None
    kw_parts = []
    for kw in node.keywords:
        if kw.arg and kw.value:
            kw_src = get_source_segment(source, kw.value)
            if kw_src:
                kw_parts.append(f', {kw.arg}={kw_src}')
    target = exc_src + "".join(kw_parts) + ")"
    return AssertionPair("pytest.raises", "pytest.raises(", target,
                         node.lineno, node.col_offset)


# --- 4. pytest.approx ------------------------------------------------------

def _parse_pytest_approx(source, node, lines):
    if not isinstance(node.func, ast.Attribute):
        return None
    val = node.func.value
    if not isinstance(val, ast.Name) or val.id != "pytest" or node.func.attr != "approx":
        return None
    args = node.args
    if not args:
        return None
    arg_src = get_source_segment(source, args[0])
    if not arg_src:
        return None
    kw_parts = []
    for kw in node.keywords:
        if kw.arg and kw.value:
            kw_src = get_source_segment(source, kw.value)
            if kw_src:
                kw_parts.append(f', {kw.arg}={kw_src}')
    target = arg_src + "".join(kw_parts) + ")"
    return AssertionPair("pytest.approx", "pytest.approx(", target,
                         node.lineno, node.col_offset)


# --- 5. assert_* (numpy/testing) ------------------------------------------

def _parse_assert_underscore(source, node, lines):
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
        if not name.startswith("assert_") or name == "assert_":
            return None
        if isinstance(func.value, ast.Attribute):
            if func.value.attr == "mock" or (
                isinstance(func.value.value, ast.Name)
                and func.value.value.id == "unittest"
            ):
                return None
    elif isinstance(func, ast.Name):
        name = func.id
        if not name.startswith("assert_") or name == "assert_":
            return None
    else:
        return None
    args = node.args
    if not args:
        return None
    last_src = get_source_segment(source, args[-1])
    if not last_src:
        return None
    prefix_parts = [f"{get_source_segment(source, func) or name}("]
    for i, arg in enumerate(args[:-1]):
        arg_src = get_source_segment(source, arg)
        if arg_src:
            prefix_parts.append(arg_src)
            if i < len(args) - 2:
                prefix_parts.append(", ")
    prefix = "".join(prefix_parts) + ", "
    return AssertionPair("assert_*", prefix, last_src + ")",
                         node.lineno, node.col_offset)


def _find_enclosing_test(tree, node):
    node_start = getattr(node, "lineno", 0)
    node_end = getattr(node, "end_lineno", node_start)
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "test" in n.name.lower():
                s, e = getattr(n, "lineno", 0), getattr(n, "end_lineno", 0)
                if s <= node_start and node_end <= e:
                    if best is None or (getattr(best, "lineno", 0) < s):
                        best = n
        if isinstance(n, ast.ClassDef) and "test" in n.name.lower():
            s, e = getattr(n, "lineno", 0), getattr(n, "end_lineno", 0)
            if s <= node_start and node_end <= e:
                for item in n.body:
                    if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name.startswith("test_")):
                        is_, ie = getattr(item, "lineno", 0), getattr(item, "end_lineno", 0)
                        if is_ <= node_start and node_end <= ie:
                            if best is None or getattr(best, "lineno", 0) < is_:
                                best = item
    return best


# ---------------------------------------------------------------------------
# Full-file-prefix extractor (build_commit_parquet_db.py)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedAssertion:
    assertion_type: str
    prefix: str
    target: str
    lineno: int
    col_offset: int
    test_function: str


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_id(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", (s or "").strip())


def _full_prefix_for(source_lines: List[str], pair: AssertionPair) -> str:
    lineno = pair.lineno
    col = pair.col_offset
    if lineno < 1:
        return pair.prefix
    pre = source_lines[: lineno - 1]
    last_line = source_lines[lineno - 1] if lineno - 1 < len(source_lines) else ""
    if col < 0:
        col = 0
    last_line_prefix = last_line[:col] if col <= len(last_line) else last_line
    return "".join(pre) + last_line_prefix + pair.prefix


def extract_assertions_from_source(source: str) -> List[ExtractedAssertion]:
    """Walk the AST and return full-file-prefix assertions for one file."""
    if not source:
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    lines = source.splitlines(keepends=True)
    out: List[ExtractedAssertion] = []

    for node in ast.walk(tree):
        pair: Optional[AssertionPair] = None
        if isinstance(node, ast.Assert):
            pair = _parse_plain_assert(source, node, lines)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                val = func.value
                val_id = getattr(val, "id", None) if isinstance(val, ast.Name) else None
                if val_id == "self" and func.attr in SELF_ASSERT_METHODS:
                    pair = _parse_self_assert(source, node, lines)
                elif val_id == "pytest":
                    if func.attr == "raises":
                        pair = _parse_pytest_raises(source, node, lines)
                    elif func.attr == "approx":
                        pair = _parse_pytest_approx(source, node, lines)
                    else:
                        pair = _parse_assert_underscore(source, node, lines)
                else:
                    pair = _parse_assert_underscore(source, node, lines)
            elif isinstance(func, ast.Name) and func.id.startswith("assert_"):
                pair = _parse_assert_underscore(source, node, lines)

        if pair is None or not pair.target.strip():
            continue
        target = pair.target
        if target.lstrip().startswith(","):
            continue
        if "\n" in target:
            target = flatten_to_oneliner(target)
        if not target.strip():
            continue

        full_prefix = _full_prefix_for(lines, pair)
        test_node = _find_enclosing_test(tree, node)
        test_name = test_node.name if test_node else ""
        out.append(ExtractedAssertion(
            assertion_type=pair.assertion_type,
            prefix=full_prefix,
            target=target,
            lineno=pair.lineno,
            col_offset=pair.col_offset,
            test_function=test_name,
        ))
    return out


__all__ = [
    "AssertionPair",
    "ExtractedAssertion",
    "extract_assertions_from_source",
    "normalize_for_id",
    "SELF_ASSERT_METHODS",
]
