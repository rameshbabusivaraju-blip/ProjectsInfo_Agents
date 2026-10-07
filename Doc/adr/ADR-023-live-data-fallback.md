# ADR-023 — Live-data fallback

**Status:** Accepted

**Context.** The agent answers from a stored copy: SQLite for Jira, GitHub and pipeline
data, and FAISS for documents. The copy changes only when a connector is run by hand. A
ticket created after the last run is missing, and a status that changed since then is
wrong. The agent needs a way to read the live source when the copy cannot answer, and it
must do so safely.

**Decision.**

1. **Triggers.** The agent may go to the live source when any one of these is true:
   - **No rows for a named item.** The question names one ticket, pull request or page,
     and the stored data has nothing for it. An empty result for a general question, such
     as "which pull requests were merged without a review?", is a valid answer and does
     not trigger the fallback.
   - **No matching passage.** The document search returns no passage that answers the
     question.
   - **Stale data.** The stored data for the source is older than 24 hours. The limit is
     one setting, `MAX_STORED_AGE_HOURS`, kept in one place. Each connector records
     `last_synced_at` for its source.
2. **Read-only.** Live tools send reads only: fetch one ticket, a restricted JQL search,
   recent pull requests and commits, and find a Confluence page. There is no code path
   that creates, updates, deletes or transitions anything.
3. **Limits.** At most 3 live calls per question. A timeout of 10 seconds per call. A
   dropped connection is retried up to 3 times, as in the Actions connector. A 401 or 429
   response is not retried.
4. **When live data is unreachable.** The agent says so plainly and answers from stored
   data, with its age.
5. **Label.** An answer that used live data says "live data" and the time it was fetched.
6. **Nothing is stored.** Live results are not written to the database. The connectors
   stay the only code that writes to it.
7. **Secrets.** The live tools reuse the existing tokens from `.env`. No key is put in a
   tracked file (ADR-020).

**Why.** The stored copy stays the main source, because it is fast, free of rate limits and
covered by tests. Live reads cover only the gaps, so cost and rate-limit risk stay small.
Read-only access and no write-back mean a wrong tool call cannot change Jira, GitHub or
Confluence, or corrupt the copy.

**Consequences.** Some answers are slower, because they wait for a live call. Live calls
can fail, so each tool is tested with mocked HTTP for a timeout, a 401, a 429 and an empty
result. A short age limit means more live calls. If the limit is 12 hours and the stored
copy is refreshed once a day, about half of all questions would trigger a live call. The
limit is one setting, so it can change without touching the logic.

**Revisit when:** the fallback fires on a large share of questions. That means the stored
copy is refreshed too rarely, and the better fix is to run the connectors on a schedule.
