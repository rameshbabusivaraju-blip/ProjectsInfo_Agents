# ADR-022 — Tool-calling loop for the agent

**Status:** Accepted

**Context.** Today `app/agent.py` classifies each question once with a fast model, then
runs one fixed path (metric, export, narrative, hybrid or refuse), and a strong model
writes the answer. There is no loop. A question that needs two steps, such as "what was
the velocity, and why did it change?", cannot be answered, and every new kind of
question needs a new key, a new prompt line and sometimes a new path.

**Decision.**

1. **Loop shape.** Two nodes. `agent_node` calls the strong model with the tools bound.
   `tools_node` runs the tool calls the model asked for. A conditional edge after
   `agent_node` sends the graph to `tools_node` when the model asked for a tool, and back
   from `tools_node` to `agent_node`. When the model asks for no tool, the graph ends.
2. **Step limit.** At most 6 model calls per question. At the limit the loop stops and
   the answer says that the limit was reached.
3. **State.** The list of messages (question, model replies, tool results) is kept in
   state. The existing keys `rows`, `sources` and `file_path` are filled from the tool
   results, so the `/ask` response keeps its current shape.
4. **Tools.** `get_metric` (wraps `_QUERY_MAP`, with the ticket key bound as a
   parameter), `search_documents` (wraps `search()`) and `export_excel`. The model only
   chooses a tool and its arguments. The reviewed SQL stays in code, and the model never
   writes SQL.
5. **System prompt.** One prompt, stored in one place. Rules: numbers only from tool
   results; say plainly when the data does not answer; name the documents used; refuse
   questions that are not about the project.
6. **Number guard.** After the final answer, every number in it must appear in the tool
   results. If one does not, retry once, then flag the answer.
7. **Cost control.** The step limit, a log of tokens per question, and the fast model for
   any cheap pre-check.
8. **Fixed paths stay for now.** The current fixed paths remain in the code until the
   golden set shows that the loop passes at least as many questions. They are removed in
   a separate ticket after that.
9. **Provider.** See the check below.

**Why.** The loop answers multi-step questions and new question types without new code
paths. The tools wrap the queries that were already reviewed (ADR-015), so numbers still
come from reviewed SQL. The number guard protects the rule that an answer never contains a
number the data does not contain.

**Provider check.** Run on 8 Oct 2026 with the models set in `app/llm/provider.py`. One
tool, `get_metric`, was offered with the question "What was the velocity of the last
sprint?". A provider passes if the model asks for the tool with a sensible argument, and
then answers from the tool result.

| Provider | Tier and model | Asked for the tool | Answered from the result | Total tokens, whole test |
|---|---|---|---|---|
| Anthropic | strong, claude-sonnet-5 | Yes | Yes | 592 |
| Anthropic | fast, claude-haiku-4-5 | Yes | Yes | 694 |
| OpenAI | strong, gpt-4o | Yes | Yes | 125 |
| OpenAI | fast, gpt-4o-mini | Yes | Yes | 125 |
| Gemini | strong, gemini-3.1-pro-preview | Not tested | Not tested | The call failed with a quota error (429) |
| Gemini | fast, gemini-3.8-flash | Yes | Yes | 255 |

Findings:

- **Anthropic and OpenAI.** Tool calling works on all four models. Every model asked for
  `get_metric` with `metric_key` set to `velocity`, and answered 41 from the result.
- **Token difference.** For the same task Anthropic used 592 to 694 tokens in total and OpenAI
  used 125. The two providers count the tool and system text differently, and a loop
  repeats that text at every step. Price per token differs by provider, so this
  is not yet a cost comparison.
- **Gemini.** The two Gemini models set in `app/llm/provider.py` (`gemini-2.5-pro` and
  `gemini-2.5-flash`) return a 404 error: they are no longer available to new users. The
  check was re-run with the newer names that the error suggested and that the account's
  model list confirmed. `gemini-3.8-flash` passes: it asked for `get_metric` with
  `metric_key` set to `velocity` and answered 41. `gemini-3.1-pro-preview` is a preview
  model, and its call failed with a quota error (429, quota exceeded), so its tool calling
  is not tested. Updating the names in `provider.py` is a separate small code change,
  outside this decision.

**Default provider for the loop: Anthropic, strong tier (claude-sonnet-5).** The check
shows that tool calling works on both Anthropic and OpenAI, so it does not separate them.
Anthropic is chosen because the loop must follow rules over several steps, such as using
only numbers from tool results, and Claude is expected to follow such rules well. This is a
judgement, not a measurement. The provider stays a `.env` setting, so switching
is configuration, not code, and the golden-set run on multi-step questions (plan item A7)
checks the choice.

**Consequences.** Cost per question rises, by an expected 5 to 20 times, because each step
sends the conversation again. Results differ between runs, so the golden set is run three
times and steps and cost per question are recorded. The loop is harder to debug than a
fixed path, so tests use a scripted fake model that returns a set sequence of tool calls.

**Revisit when:** the loop's pass rate on the golden set is lower than the fixed paths, or
the cost per question makes everyday use too expensive.
