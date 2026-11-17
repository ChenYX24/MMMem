"""Deterministic structure for conversational memory: dates, segments, timelines."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Callable, Iterable, Sequence

MONTHS = {name: index for index, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), 1)}
MONTHS.update({name[:3]: index for name, index in list(MONTHS.items())})
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "a couple of": 2, "couple of": 2, "three": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

_LOCOMO_TIME = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ampm>am|pm)\s+on\s+(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+),?\s+(?P<y>\d{4})",
    re.I)
_ISO = re.compile(r"(?P<y>\d{4})-(?P<mo>\d{1,2})-(?P<d>\d{1,2})")


def parse_anchor(value: str | None) -> date | None:
    """Parse a session timestamp (LoCoMo '1:56 pm on 8 May, 2023' or ISO '2024-06-17')."""
    if not value:
        return None
    match = _LOCOMO_TIME.search(value)
    if match and match["mon"].lower() in MONTHS:
        return date(int(match["y"]), MONTHS[match["mon"].lower()], int(match["d"]))
    match = _ISO.search(value)
    if match:
        return date(int(match["y"]), int(match["mo"]), int(match["d"]))
    return None


def fmt(day: date) -> str:
    return f"{day.day} {day.strftime('%B')} {day.year}"


def _month_shift(day: date, months: int) -> tuple[int, int]:
    total = day.year * 12 + day.month - 1 + months
    return total // 12, total % 12 + 1


def _month_name(year: int, month: int) -> str:
    return f"{date(year, month, 1).strftime('%B')} {year}"


_NUM = r"(a couple of|couple of|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)"
_RULES: list[tuple[re.Pattern[str], Callable[[re.Match[str], date], str | None]]] = []


def _rule(pattern: str):
    def register(function):
        _RULES.append((re.compile(pattern, re.I), function))
        return function
    return register


def _count(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else NUMBERS[token]


@_rule(r"\bthe day before yesterday\b")
def _day_before_yesterday(_, anchor):
    return fmt(anchor - timedelta(days=2))


@_rule(r"\b(yesterday|last night)\b")
def _yesterday(_, anchor):
    return fmt(anchor - timedelta(days=1))


@_rule(r"\b(today|tonight|this morning|this afternoon|this evening)\b")
def _today(_, anchor):
    return fmt(anchor)


@_rule(r"\btomorrow\b")
def _tomorrow(_, anchor):
    return fmt(anchor + timedelta(days=1))


@_rule(r"\b(last|this past|past)\s+(" + "|".join(WEEKDAYS) + r")\b")
def _last_weekday(match, anchor):
    target = WEEKDAYS.index(match[2].lower())
    delta = (anchor.weekday() - target) % 7 or 7
    day = anchor - timedelta(days=delta)
    return f"{match[2].capitalize()}, {fmt(day)}"


@_rule(r"\b(next|this coming|coming)\s+(" + "|".join(WEEKDAYS) + r")\b")
def _next_weekday(match, anchor):
    target = WEEKDAYS.index(match[2].lower())
    delta = (target - anchor.weekday()) % 7 or 7
    day = anchor + timedelta(days=delta)
    return f"{match[2].capitalize()}, {fmt(day)}"


@_rule(r"\blast weekend\b")
def _last_weekend(_, anchor):
    return f"the weekend before {fmt(anchor)}"


@_rule(r"\bthis weekend\b")
def _this_weekend(_, anchor):
    return f"the weekend of {fmt(anchor)}"


@_rule(r"\blast week\b")
def _last_week(_, anchor):
    return f"the week before {fmt(anchor)}"


@_rule(r"\bthis week\b")
def _this_week(_, anchor):
    return f"the week of {fmt(anchor)}"


@_rule(r"\bnext week\b")
def _next_week(_, anchor):
    return f"the week after {fmt(anchor)}"


@_rule(r"\blast month\b")
def _last_month(_, anchor):
    return _month_name(*_month_shift(anchor, -1))


@_rule(r"\bthis month\b")
def _this_month(_, anchor):
    return _month_name(anchor.year, anchor.month)


@_rule(r"\bnext month\b")
def _next_month(_, anchor):
    return _month_name(*_month_shift(anchor, 1))


@_rule(r"\blast year\b")
def _last_year(_, anchor):
    return str(anchor.year - 1)


@_rule(r"\bthis year\b")
def _this_year(_, anchor):
    return str(anchor.year)


@_rule(r"\bnext year\b")
def _next_year(_, anchor):
    return str(anchor.year + 1)


@_rule(r"\b" + _NUM + r"\s+(day|week|month|year)s?\s+ago\b")
def _ago(match, anchor):
    count, unit = _count(match[1]), match[2].lower()
    if unit == "day":
        return fmt(anchor - timedelta(days=count))
    if unit == "week":
        return f"the week of {fmt(anchor - timedelta(weeks=count))}"
    if unit == "month":
        return _month_name(*_month_shift(anchor, -count))
    return str(anchor.year - count)


@_rule(r"\b(a few|few|several)\s+(day|week|month|year)s?\s+ago\b")
def _vague_ago(match, anchor):
    return f"a few {match[2].lower()}s before {fmt(anchor)}"


def annotate_relative_time(text: str, anchor: date | None) -> str:
    """Append the absolute date after each relative time phrase; the original words are kept."""
    if anchor is None or not text:
        return text
    spans: list[tuple[int, int, str]] = []
    for pattern, function in _RULES:
        for match in pattern.finditer(text):
            if any(start < match.end() and match.start() < end for start, end, _ in spans):
                continue
            value = function(match, anchor)
            if value:
                spans.append((match.start(), match.end(), value))
    for start, end, value in sorted(spans, reverse=True):
        text = f"{text[:end]} [= {value}]{text[end:]}"
    return text


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic stand-in for a tokenizer; runs pass a real counter when available."""
    return max(1, round(len(re.findall(r"\w+|[^\w\s]", text)) * 1.15))


@dataclass(frozen=True)
class Turn:
    id: str
    ordinal: int
    session: str
    text: str


def segment_session(turns: Sequence[Turn], *, max_tokens: int = 400, min_turns: int = 3, max_turns: int = 6,
                    count_tokens: Callable[[str], int] = estimate_tokens) -> list[list[Turn]]:
    """Split one session into contiguous segments of up to max_turns within a token cap.

    A turn that alone exceeds the cap becomes its own segment rather than being truncated.
    """
    segments: list[list[Turn]] = []
    current: list[Turn] = []
    used = 0
    for turn in turns:
        cost = count_tokens(turn.text)
        if current and (len(current) >= max_turns or used + cost > max_tokens):
            segments.append(current)
            current, used = [], 0
        current.append(turn)
        used += cost
    if current:
        # Fold a short tail into its predecessor when that stays within the cap.
        if (segments and len(current) < min_turns
                and len(segments[-1]) + len(current) <= max_turns
                and sum(count_tokens(t.text) for t in segments[-1] + current) <= max_tokens):
            segments[-1].extend(current)
        else:
            segments.append(current)
    return segments


def chunk_lines(lines: Iterable[str], *, max_tokens: int,
                count_tokens: Callable[[str], int] = estimate_tokens) -> list[list[str]]:
    """Pack lines into chunks under a token cap without splitting any line."""
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0
    for line in lines:
        cost = count_tokens(line)
        if current and used + cost > max_tokens:
            chunks.append(current)
            current, used = [], 0
        current.append(line)
        used += cost
    if current:
        chunks.append(current)
    return chunks


_CAPITALISED = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b")
_STOP = {"The", "This", "That", "What", "When", "Where", "Which", "Who", "How", "Why", "And", "But", "Yes", "Yeah",
         "Wow", "Hey", "Thanks", "Thank", "Sure", "Great", "Nice", "Awesome", "Haha", "Well", "Oh", "Also", "Just",
         "Really", "Good", "Glad", "Sounds", "Let", "It's", "I'm", "You", "Your", "They", "She", "He", "We", "Our",
         "Have", "Had", "Did", "Do", "Can", "Could", "Would", "Will", "Not", "Maybe", "Hope", "Love", "Take", "Keep",
         "Congrats", "Congratulations", "Definitely", "Absolutely", "Totally", "Right", "Okay", "So", "Speaking",
         *(day.capitalize() for day in WEEKDAYS), *(name.capitalize() for name in MONTHS)}


def candidate_entities(turns: Sequence[Turn], speakers: Iterable[str], *, min_mentions: int = 2) -> dict[str, list[str]]:
    """Deterministic fallback entity finder: repeated capitalised names that are not the speakers.

    Speakers are excluded because they appear in every turn and would link the whole unit.
    """
    excluded = set(speakers) | _STOP
    mentions: dict[str, list[str]] = {}
    for turn in turns:
        for match in _CAPITALISED.finditer(turn.text):
            name = match[1]
            before = turn.text[:match.start()].rstrip()
            # Sentence-initial capitals carry no entity signal ("Agreed", "Anything", "Bye").
            if not before or before[-1] in ".!?\"'(:;-—":
                continue
            if name.split()[0] in excluded or name in excluded:
                continue
            ids = mentions.setdefault(name, [])
            if turn.id not in ids:
                ids.append(turn.id)
    return {name: ids for name, ids in mentions.items() if len(ids) >= min_mentions}


def session_date(value: str | None) -> str:
    anchor = parse_anchor(value)
    return fmt(anchor) if anchor else (value or "unknown date")


def to_datetime(value: str | None) -> datetime | None:
    anchor = parse_anchor(value)
    return datetime(anchor.year, anchor.month, anchor.day) if anchor else None
