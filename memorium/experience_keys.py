"""Question families and entity cues for experience reuse."""
import re

_ENTITY = re.compile(r"\b[A-Z][a-zA-Z'\-]+\b")
_DATE = re.compile(r"\b(?:\d{1,2} )?(?:January|February|March|April|May|June|July|August|September|October|"
                   r"November|December)(?: \d{1,2})?(?:,? \d{4})?\b|\b(?:19|20)\d{2}\b")
_INTENT = re.compile(r"^\W*(who|whom|whose|when|where|what|which|why|how many|how much|how|did|does|do|is|are|"
                     r"was|were|can|could|would|will|has|have)\b", re.I)
_STOPWORD_CAPS = {"What", "When", "Where", "Who", "Which", "Why", "How", "Did", "Does", "Do", "Is", "Are", "Was",
                  "Were", "Can", "Could", "Would", "Will", "Has", "Have", "The", "A", "An", "In", "On", "At", "I",
                  "Answer", "Evidence", "Question"}


def entities(text: str) -> frozenset[str]:
    caps = {w for w in _ENTITY.findall(text) if w not in _STOPWORD_CAPS}
    return frozenset(caps | {d.strip() for d in _DATE.findall(text)})


def family(question: str) -> str:
    """Frozen runtime rule: intent word + the question's entities. Never uses generator-side labels."""
    match = _INTENT.match(question)
    intent = match.group(1).lower() if match else "other"
    return intent + "|" + ",".join(sorted(entities(question)))
