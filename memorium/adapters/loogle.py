"""LooGLE v2 prompt and scoring utilities (benchmark revision 4365a5c)."""
from __future__ import annotations

import ast
import re
from typing import Any

GENERIC_ANSWER_FIRST_SUFFIX = (
    "\nDo not explain. Your entire response must be one line beginning exactly "
    "`The correct answer is ` followed only by the final answer requested above."
)


def option_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def official_prompt(item: dict[str, Any], context: str) -> str:
    return str(item["instruction"]).format(context=context, options=option_text(item.get("options")),
                                           question=str(item["question"]))


def extract_first_match(response: str, pattern: str) -> str | None:
    match = re.search(r"[Tt]he correct answer is[:\s]+(.*)", response)
    if not match:
        return None
    matches = re.findall(pattern, match.group(1))
    return matches[0] if matches else None


def extract_answer(response: str, task: str, source: str) -> Any:
    response = str(response).replace("*", "")
    if task == "Legal Case Retrieval":
        pattern = r"CASE_\d+"
    elif task == "Legal Article Extraction":
        pattern = r"LAW_\d+"
    elif task == "Version Control":
        return list(set(re.findall(r"[\w/]+\.py", response)))
    elif source == "Finance" and task == "Trend Analysis":
        return extract_first_match(response, r"\b\d{4}-\d{4}\b")
    elif source == "Finance":
        return extract_first_match(response, r"\d{1,6}(?:,\d{3})*(?:\.\d+)?(?:[M%])?")
    else:
        pattern = r"[A-D]"
    match = re.search(rf"[Tt]he correct answer is[:\s]*[\(<]*({pattern})", response)
    return match.group(1) if match else None


def normalize_number(value: Any) -> float | None:
    if isinstance(value, list):
        value = value[0] if value else ""
    if not value:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").strip().removesuffix("M").removesuffix("%"))
    except ValueError:
        return None


def judge_answer(prediction: Any, answer: Any, task: str) -> bool | float:
    if task in {"Metric Calculation", "Cross-Company Comparison"}:
        pred_value, gold_value = normalize_number(prediction), normalize_number(answer)
        if pred_value is not None and gold_value is not None:
            if gold_value == 0 and task == "Cross-Company Comparison":
                return prediction == answer
            return abs(pred_value - gold_value) / abs(gold_value) < 0.05
        if task == "Metric Calculation":
            return False
    if task == "Version Control":
        gold = ast.literal_eval(answer) if isinstance(answer, str) else answer
        left, right = set(gold or []), set(prediction or [])
        return round(100 * len(left & right) / len(left | right), 2) if left and right else 0.0
    return prediction == answer


def judge_score(value: bool | float) -> float:
    return float(value) if isinstance(value, bool) else float(value) / 100.0
