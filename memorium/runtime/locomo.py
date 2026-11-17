import hashlib
import importlib.util
from pathlib import Path
import re
from memorium.runtime.rendering import sha
NO_INFO = 'No information available'
LOCOMO_PROTOCOLS = ('legacy_mcq_v1', 'native_freeform_v1')


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def choose_options(q, seed):
    choices = [str(q['official_qa']['adversarial_answer']), NO_INFO]
    if int(text_sha(f"{seed}:{q['id']}:options"), 16) % 2:
        choices.reverse()
    assert choices[0] != choices[1]
    return choices

def locomo_question(q, seed, protocol='legacy_mcq_v1'):
    if protocol not in LOCOMO_PROTOCOLS:
        raise ValueError('unknown LoCoMo protocol')
    question = q['official_qa']['question']
    if protocol == 'native_freeform_v1':
        return question
    category = q['official_qa']['category']
    if category == 2:
        question += ' Use DATE of CONVERSATION to answer with an approximate date.'
    if category == 5:
        a, b = choose_options(q, seed)
        question += f'\n(a) {a}\n(b) {b}'
    return question

def parse_choice(raw, choices):
    value = raw.strip()
    exact = [choice for choice in choices if value.casefold() == choice.casefold()]
    if len(exact) == 1:
        return exact[0]
    matches = re.findall(r'\(([ab])\)|\boption\s+([ab])\b|^\s*(?:answer\s*:\s*)?([ab])(?:[\s.)\]:-]|$)', value, re.I)
    letters = {part.lower() for match in matches for part in match if part}
    return choices[ord(next(iter(letters))) - ord('a')] if len(letters) == 1 else None

def official_scorer(path):
    expected = '8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd'
    if sha(path) != expected:
        raise ValueError(f"unexpected LoCoMo evaluator revision: {path}")
    spec = importlib.util.spec_from_file_location('locomo_pinned_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
