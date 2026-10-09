"""Tests for app/agent_loop.py.

The prompt can only be judged by a model, so its tests guard its structure. The loop is tested
with a scripted fake model that returns prepared replies, so nothing here calls a provider.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app import agent_loop, tools
from app.agent_loop import MAX_STEPS, SYSTEM_PROMPT


class ScriptedModel:
    """Stands in for the bound model. Returns the prepared replies in order and keeps what it saw.

    With repeat=True the last reply is returned again whenever the script runs out.
    """

    def __init__(self, replies: list[AIMessage], repeat: bool = False) -> None:
        self.replies = replies
        self.repeat = repeat
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        index = len(self.calls) - 1
        reply = self.replies[min(index, len(self.replies) - 1) if self.repeat else index]
        # A fresh copy each time, as a real model gives: the loop gives every message an id.
        return reply.model_copy(update={"id": None})


def _asks(*calls: tuple[str, dict[str, Any], str], tokens: int = 10) -> AIMessage:
    """A model reply that asks for tools. Each call is (tool name, arguments, call id)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            for name, args, call_id in calls
        ],
        usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )


def _says(text: str, tokens: int = 10) -> AIMessage:
    """A model reply with a final answer and no tool call."""
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )


VELOCITY = {"metric_key": "velocity", "rows": [{"sprint": "Sprint 4", "delivered": 34.5}]}


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ScriptedModel]:
    """Install a scripted model, and a fake run_tool that returns VELOCITY, in the loop."""

    def install(*replies: AIMessage, repeat: bool = False) -> ScriptedModel:
        model = ScriptedModel(list(replies), repeat)
        monkeypatch.setattr(agent_loop, "_model", lambda: model)
        monkeypatch.setattr(agent_loop, "run_tool", lambda name, args: VELOCITY)
        return model

    return install


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
    """Rules 1 to 11 are all present, so a deleted rule is noticed."""
    for number in range(1, 12):
        assert f"\n{number}. " in SYSTEM_PROMPT


def test_prompt_has_no_stray_line_continuations() -> None:
    """The prompt is one block of text; a backslash left in it would show to the model."""
    assert "\\" not in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_the_graph_has_the_shape_in_adr_022() -> None:
    """Two nodes, tools back to the agent, and the agent either calls tools or ends."""
    graph = agent_loop.loop.get_graph()

    assert {"agent_node", "tools_node"} <= set(graph.nodes)
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert edges == {
        ("__start__", "agent_node"),
        ("agent_node", "tools_node"),
        ("agent_node", "agent_node"),
        ("agent_node", "__end__"),
        ("tools_node", "agent_node"),
    }


def test_model_is_the_strong_tier_with_the_tools_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """_model asks for the strong model and binds every tool to it."""
    seen: dict[str, Any] = {}

    class FakeLlm:
        def bind_tools(self, bound: list[Any]) -> str:
            seen["tools"] = bound
            return "bound model"

    def fake_get_llm(tier: str) -> FakeLlm:
        seen["tier"] = tier
        return FakeLlm()

    agent_loop._model.cache_clear()
    monkeypatch.setattr(agent_loop, "get_llm", fake_get_llm)
    try:
        assert agent_loop._model() == "bound model"
    finally:
        agent_loop._model.cache_clear()

    assert seen == {"tier": "strong", "tools": tools.TOOLS}


def test_a_question_that_needs_no_tool_is_answered_in_one_call(
    script: Callable[..., ScriptedModel],
) -> None:
    model = script(_says("Velocity is the story points a team finishes in one sprint."))

    state = agent_loop.run_loop("What is velocity?")

    assert state["answer"] == "Velocity is the story points a team finishes in one sprint."
    assert state["steps"] == 1
    assert state["tool_results"] == []
    first_call = model.calls[0]
    assert first_call[0] == SystemMessage(content=SYSTEM_PROMPT)
    assert isinstance(first_call[1], HumanMessage)
    assert first_call[1].content == "What is velocity?"


def test_a_tool_result_goes_back_to_the_model_before_the_answer(
    script: Callable[..., ScriptedModel],
) -> None:
    model = script(
        _asks(("get_metric", {"metric_key": "velocity"}, "call_1")),
        _says("Sprint 4 delivered 34.5 points."),
    )

    state = agent_loop.run_loop("What was the velocity of the last sprint?")

    assert state["answer"] == "Sprint 4 delivered 34.5 points."
    assert state["steps"] == 2
    assert state["tool_results"] == [VELOCITY]
    returned = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert len(returned) == 1
    assert returned[0].tool_call_id == "call_1"
    assert returned[0].name == "get_metric"
    assert json.loads(str(returned[0].content)) == VELOCITY


def test_two_tools_asked_in_one_reply_run_in_the_same_step(
    script: Callable[..., ScriptedModel],
) -> None:
    model = script(
        _asks(
            ("get_metric", {"metric_key": "velocity"}, "call_a"),
            ("search_documents", {"query": "velocity change", "doc_type": "retro"}, "call_b"),
        ),
        _says("Sprint 4 delivered 34.5 points."),
    )

    state = agent_loop.run_loop("What was the velocity, and why did it change?")

    assert state["steps"] == 2
    assert len(state["tool_results"]) == 2
    returned = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in returned] == ["call_a", "call_b"]


def test_a_tool_error_is_given_to_the_model_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real run_tool turns an unknown tool into an error result that the model can read."""
    model = ScriptedModel(
        [_asks(("run_sql", {"sql": "select 1"}, "call_1")), _says("I could not get that data.")]
    )
    monkeypatch.setattr(agent_loop, "_model", lambda: model)

    state = agent_loop.run_loop("Run some SQL")

    assert state["answer"] == "I could not get that data."
    returned = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert "Unknown tool 'run_sql'" in str(returned[0].content)


def test_the_loop_stops_at_the_step_limit(script: Callable[..., ScriptedModel]) -> None:
    """A model that keeps asking for tools is stopped, and the answer says why."""
    model = script(
        _asks(("get_metric", {"metric_key": "velocity"}, "call_1")),
        repeat=True,
    )

    state = agent_loop.run_loop("A question the model never finishes")

    assert len(model.calls) == MAX_STEPS
    assert state["steps"] == MAX_STEPS
    assert state["answer"] == agent_loop.LIMIT_ANSWER
    assert f"{MAX_STEPS} model calls" in state["answer"]
    assert len(state["tool_results"]) == MAX_STEPS - 1  # the last request is not run


def test_an_unsupported_number_gets_one_rewrite(script: Callable[..., ScriptedModel]) -> None:
    model = script(
        _asks(("get_metric", {"metric_key": "velocity"}, "call_1")),
        _says("Sprint 4 delivered about 35 points."),
        _says("Sprint 4 delivered 34.5 points."),
    )

    state = agent_loop.run_loop("What was the velocity of the last sprint?")

    assert state["answer"] == "Sprint 4 delivered 34.5 points."
    assert state["guard_retried"] is True
    assert state["steps"] == 3
    last_message = model.calls[2][-1]
    assert isinstance(last_message, HumanMessage)
    assert "35" in str(last_message.content)


def test_an_answer_that_is_still_unsupported_after_the_rewrite_is_flagged(
    script: Callable[..., ScriptedModel],
) -> None:
    model = script(
        _asks(("get_metric", {"metric_key": "velocity"}, "call_1")),
        _says("About 35 points."),
        _says("Roughly 36 points."),
    )

    state = agent_loop.run_loop("What was the velocity of the last sprint?")

    assert len(model.calls) == 3  # one rewrite, not more
    assert state["answer"].startswith("Roughly 36 points.")
    assert "They were not found in the project data: 36." in state["answer"]


def test_no_rewrite_is_asked_when_it_would_pass_the_step_limit(
    monkeypatch: pytest.MonkeyPatch,
    script: Callable[..., ScriptedModel],
) -> None:
    """With a limit of 2 the second call is the last, so a bad answer is flagged at once."""
    model = script(
        _asks(("get_metric", {"metric_key": "velocity"}, "call_1")),
        _says("About 35 points."),
    )
    monkeypatch.setattr(agent_loop, "MAX_STEPS", 2)

    state = agent_loop.run_loop("What was the velocity of the last sprint?")

    assert len(model.calls) == 2
    assert "They were not found in the project data: 35." in state["answer"]


def test_an_empty_reply_gives_a_plain_message(script: Callable[..., ScriptedModel]) -> None:
    script(_says(""))

    assert agent_loop.run_loop("Anything")["answer"] == agent_loop.NO_ANSWER


def test_a_reply_made_of_text_blocks_is_read(script: Callable[..., ScriptedModel]) -> None:
    """Some providers return a list of blocks instead of a string."""
    script(AIMessage(content=[{"type": "text", "text": "Hello from blocks."}]))

    assert agent_loop.run_loop("Hi")["answer"] == "Hello from blocks."


def test_tokens_are_added_up_and_logged(
    script: Callable[..., ScriptedModel], caplog: pytest.LogCaptureFixture
) -> None:
    script(
        _asks(("get_metric", {"metric_key": "velocity"}, "call_1"), tokens=100),
        _says("Sprint 4 delivered 34.5 points.", tokens=40),
    )

    with caplog.at_level(logging.INFO, logger="app.agent_loop"):
        state = agent_loop.run_loop("What was the velocity of the last sprint?")

    assert state["tokens"] == 140
    assert "agent loop: 2 model calls, 140 tokens, 1 tool calls" in caplog.text


def test_missing_token_counts_are_treated_as_zero(script: Callable[..., ScriptedModel]) -> None:
    script(AIMessage(content="No numbers here."))

    assert agent_loop.run_loop("Hi")["tokens"] == 0


# ---------------------------------------------------------------------------
# loop_result: the final state in the fields /ask returns
# ---------------------------------------------------------------------------

PASSAGE = {
    "title": "RAID Log",
    "url": "https://wiki.example/raid",
    "text": "R2 unowned",
    "score": 0.9,
}


def _state(
    *outcomes: tuple[str, dict[str, Any]], answer: str = "An answer."
) -> agent_loop.LoopState:
    """A finished loop state whose tool messages hold the given (tool name, result) pairs."""
    messages: list[BaseMessage] = [HumanMessage(content="A question")]
    for index, (name, result) in enumerate(outcomes):
        messages.append(
            ToolMessage(content=json.dumps(result), tool_call_id=f"call_{index}", name=name)
        )
    return {"question": "A question", "messages": messages, "answer": answer}


def test_result_for_a_metric_answer() -> None:
    result = agent_loop.loop_result(_state(("get_metric", VELOCITY)))

    assert result == {
        "question": "A question",
        "answer": "An answer.",
        "metric_key": "velocity",
        "rows": VELOCITY["rows"],
        "sources": [],
    }


def test_result_for_a_document_answer() -> None:
    result = agent_loop.loop_result(
        _state(("search_documents", {"doc_type": "raid_log", "sources": [PASSAGE]}))
    )

    assert result["metric_key"] == "narrative"
    assert result["rows"] == []
    assert result["sources"] == [PASSAGE]


def test_result_for_a_number_with_a_reason_is_hybrid() -> None:
    result = agent_loop.loop_result(
        _state(
            ("get_metric", VELOCITY),
            ("search_documents", {"doc_type": "retro", "sources": [PASSAGE]}),
        )
    )

    assert result["metric_key"] == "hybrid"
    assert result["rows"] == VELOCITY["rows"]
    assert result["sources"] == [PASSAGE]


def test_result_for_an_export_carries_the_file_and_the_rows() -> None:
    export = {
        "file_path": "exports/velocity_last.xlsx",
        "rows": VELOCITY["rows"],
        "rows_written": 1,
    }

    result = agent_loop.loop_result(_state(("export_excel", export)))

    assert result["metric_key"] == "export_excel"
    assert result["file_path"] == "exports/velocity_last.xlsx"
    assert result["rows"] == VELOCITY["rows"]


def test_result_without_any_tool_is_other_and_has_no_file() -> None:
    """A refusal, a definition or a clarifying question used no tool."""
    result = agent_loop.loop_result(_state())

    assert result["metric_key"] == "other"
    assert result["rows"] == []
    assert result["sources"] == []
    assert "file_path" not in result


def test_result_takes_the_rows_of_the_latest_call_that_returned_rows() -> None:
    dora = {"metric_key": "dora_per_sprint", "rows": [{"sprint": "Sprint 4", "deployments": 1}]}
    empty = {"metric_key": "unreviewed_prs", "rows": [], "message": "no rows"}

    result = agent_loop.loop_result(
        _state(("get_metric", VELOCITY), ("get_metric", dora), ("get_metric", empty))
    )

    assert result["rows"] == dora["rows"]
    assert result["metric_key"] == "unreviewed_prs"  # the key the model asked about last


def test_result_lists_each_passage_once() -> None:
    """Repeated searches often return the same passage."""
    search = {"doc_type": "retro", "sources": [PASSAGE]}

    result = agent_loop.loop_result(
        _state(("search_documents", search), ("search_documents", search))
    )

    assert result["sources"] == [PASSAGE]


def test_result_ignores_tool_errors() -> None:
    result = agent_loop.loop_result(
        _state(("get_metric", {"error": "Unknown metric_key 'x'."}), ("get_metric", VELOCITY))
    )

    assert result["metric_key"] == "velocity"
    assert result["rows"] == VELOCITY["rows"]


def test_result_without_an_answer_uses_the_plain_message() -> None:
    state = _state()
    del state["answer"]

    assert agent_loop.loop_result(state)["answer"] == agent_loop.NO_ANSWER
