"""Token overlap and exact-match metrics."""
import re
from typing import Any

def normalize_answer(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

def token_f1(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(gold).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = {}
    for token in pred:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in ref:
        if common.get(token, 0) > 0:
            common[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0
