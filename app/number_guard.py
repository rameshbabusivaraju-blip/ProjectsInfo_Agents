"""Number guard for the agent loop (ADR-022).

The rule: a number in the final answer must come from the data. This module checks it. A
number is allowed if it appears in the user's question or anywhere in the tool results (rows
and document passages). Anything else is "unsupported": the model invented it, rounded it or
calculated it. The check is strict on purpose. Numbers are compared by value, so 5 matches
5.0 and 1,234 matches 1234.

What it does not check: numbers written as words ("five"), and whether a supported number is
used in the right place. It only checks that the number exists in the data.

The loop uses it like this: check the answer, ask the model once to rewrite it if numbers are
unsupported, and if they are still unsupported, add a notice with flag_answer().
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

# A number such as 7, 12.5, 1,234 or 1,234.5. It must not touch a letter, digit or dot before
# it, so "Q3", "p50" and the "3" in "1.2.3" are not read as numbers. A "-" before it is fine,
# so the 14 in AGENTS-14 and the parts of 2026-10-07 are numbers.
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

# "1. " or "2) " at the start of a line is a list marker, not a number in the answer.
_LIST_MARKER = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]+")


@dataclass(frozen=True)
class GuardResult:
    """Outcome of one check. unsupported holds the numbers as written, once each, in order."""

    ok: bool
    unsupported: list[str]


def numbers_in(text: str) -> list[str]:
    """Return every number in the text as written, in order, without list markers."""
    return _NUMBER.findall(_LIST_MARKER.sub("", text))


def _value(number: str) -> Decimal | None:
    """The numeric value of a number as written, or None if it cannot be read."""
    try:
        return Decimal(number.replace(",", ""))
    except InvalidOperation:
        return None


def _allowed_values(question: str, tool_results: Sequence[Any]) -> set[Decimal]:
    """Values of all numbers in the question and in the tool results."""
    text = question + "\n" + json.dumps(list(tool_results), default=str, ensure_ascii=False)
    values: set[Decimal] = set()
    for number in numbers_in(text):
        value = _value(number)
        if value is not None:
            values.add(value)
    return values


def check_answer(answer: str, question: str, tool_results: Sequence[Any]) -> GuardResult:
    """Find the numbers in the answer that appear in neither the question nor the tool results."""
    allowed = _allowed_values(question, tool_results)
    unsupported: list[str] = []
    for number in numbers_in(answer):
        value = _value(number)
        if value is not None and value not in allowed and number not in unsupported:
            unsupported.append(number)
    return GuardResult(ok=not unsupported, unsupported=unsupported)


def retry_message(unsupported: Sequence[str]) -> str:
    """The message that asks the model to rewrite its answer once."""
    return (
        "Your answer contains numbers that are not in the tool results or the question: "
        + ", ".join(unsupported)
        + ". Rewrite the answer using only numbers that appear in the tool results. "
        "Do not round, add or estimate. If the data does not give the number, say so."
    )


def flag_answer(answer: str, unsupported: Sequence[str]) -> str:
    """Add a notice to an answer that still has unsupported numbers after the retry."""
    return (
        answer
        + "\n\nCheck these numbers before using them. They were not found in the project data: "
        + ", ".join(unsupported)
        + "."
    )
