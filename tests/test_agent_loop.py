"""Tests for app/agent_loop.py. The prompt can only be judged by a model, so these tests guard
its structure. The behaviour (an off-topic question gets a refusal and no tool call) is
checked in the golden set and in the loop tests."""

from app import tools
from app.agent_loop import SYSTEM_PROMPT


def test_prompt_names_every_tool_the_model_can_call() -> None:
    """A tool added to TOOLS must also be explained in the prompt."""
    for tool in tools.TOOLS:
        assert tool.name in SYSTEM_PROMPT


def test_prompt_does_not_repeat_the_lists_in_the_tool_descriptions() -> None:
    """Metric keys and document types live in app/tools.py only.

    Only names with an underscore are checked. Single words such as velocity or general are
    also ordinary English and may appear in the rules.
    """
    for name in [*tools.METRIC_DESCRIPTIONS, *tools.DOC_TYPE_DESCRIPTIONS]:
        if "_" in name:
            assert name not in SYSTEM_PROMPT


def test_prompt_numbers_its_rules_without_gaps() -> None:
    """Rules 1 to 10 are all present, so a deleted rule is noticed."""
    for number in range(1, 11):
        assert f"\n{number}. " in SYSTEM_PROMPT


def test_prompt_has_no_stray_line_continuations() -> None:
    """The prompt is one block of text; a backslash left in it would show to the model."""
    assert "\\" not in SYSTEM_PROMPT
