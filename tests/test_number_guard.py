"""Tests for app/number_guard.py. Pure functions, no database or model."""

from datetime import datetime

from app.number_guard import check_answer, flag_answer, numbers_in, retry_message

ROWS = [{"sprint": "Sprint 3", "delivered": 21.0}, {"sprint": "Sprint 4", "delivered": 34.5}]


def test_numbers_in_reads_plain_decimal_and_comma_numbers() -> None:
    assert numbers_in("7 items, 12.5 hours, 1,234 lines and 1,234.5 points") == [
        "7",
        "12.5",
        "1,234",
        "1,234.5",
    ]


def test_numbers_in_reads_ticket_keys_percentages_and_dates() -> None:
    assert numbers_in("AGENTS-14 failed 12.5% of runs on 2026-10-07") == [
        "14",
        "12.5",
        "2026",
        "10",
        "07",
    ]


def test_numbers_in_skips_numbers_attached_to_letters_and_versions() -> None:
    assert numbers_in("Q3 p50 v2 version 1.2.3") == ["1.2"]


def test_numbers_in_skips_list_markers() -> None:
    text = "1. Sprint 3 delivered 21\n2) Sprint 4 delivered 34.5\n  3. done"
    assert numbers_in(text) == ["3", "21", "4", "34.5"]


def test_answer_with_only_data_numbers_is_ok() -> None:
    result = check_answer("Sprint 4 delivered 34.5 points.", "velocity?", [{"rows": ROWS}])

    assert result.ok
    assert result.unsupported == []


def test_numbers_are_compared_by_value() -> None:
    """21 matches 21.0 and 1234 matches 1,234."""
    rows = [{"lines": "1,234", "points": 21.0}]

    assert check_answer("21 points and 1234 lines", "q", [{"rows": rows}]).ok
    assert check_answer("21.00 points and 1,234 lines", "q", [{"rows": rows}]).ok


def test_a_number_from_the_question_is_allowed() -> None:
    assert check_answer("Looking at the last 3 sprints.", "Show the last 3 sprints", []).ok


def test_a_number_from_a_document_passage_is_allowed() -> None:
    sources = [{"title": "RAID Log", "text": "R2 has no owner. R7 is due in 14 days."}]

    assert check_answer("R7 is due in 14 days.", "which risks?", [{"sources": sources}]).ok


def test_a_rounded_number_is_unsupported() -> None:
    rows = [{"sprint": "Sprint 4", "delivered": 34.5}]

    result = check_answer("Sprint 4 delivered about 35 points.", "q", [{"rows": rows}])

    assert not result.ok
    assert result.unsupported == ["35"]


def test_a_calculated_number_is_unsupported() -> None:
    """21 + 34.5 = 55.5 is not in the data, so it is flagged."""
    result = check_answer("Together they delivered 55.5 points.", "q", [{"rows": ROWS}])

    assert result.unsupported == ["55.5"]


def test_unsupported_numbers_are_listed_once_in_order() -> None:
    result = check_answer("It was 99, then 77, then 99 again.", "q", [{"rows": ROWS}])

    assert result.unsupported == ["99", "77"]


def test_any_number_is_unsupported_when_there_are_no_tool_results() -> None:
    assert check_answer("The velocity was 21.", "What was the velocity?", []).unsupported == ["21"]


def test_an_answer_without_numbers_is_ok() -> None:
    assert check_answer("The data does not answer this.", "q", []).ok


def test_list_markers_in_the_answer_do_not_count() -> None:
    answer = "1. Sprint 3 delivered 21\n2. Sprint 4 delivered 34.5"

    assert check_answer(answer, "q", [{"rows": ROWS}]).ok


def test_results_that_json_cannot_write_are_still_read() -> None:
    """A date in a row is turned into text, so its numbers are allowed."""
    result = check_answer("Merged on 2026-10-07.", "q", [{"rows": [{"at": datetime(2026, 10, 7)}]}])

    assert result.ok


def test_retry_message_lists_the_numbers() -> None:
    message = retry_message(["35", "55.5"])

    assert "35, 55.5" in message
    assert "only numbers that appear in the tool results" in message


def test_flag_answer_keeps_the_answer_and_adds_the_notice() -> None:
    flagged = flag_answer("About 35 points.", ["35"])

    assert flagged.startswith("About 35 points.")
    assert flagged.endswith("They were not found in the project data: 35.")
