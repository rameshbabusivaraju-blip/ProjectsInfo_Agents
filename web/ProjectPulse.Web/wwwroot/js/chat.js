// Chat screen behaviour: show each question, send it to the page's Ask handler
// (which calls the API from the server), and show the answer or a plain error.
(function () {
    "use strict";

    const form = document.getElementById("chat-form");
    const input = document.getElementById("question");
    const sendButton = form.querySelector("button[type=submit]");
    const list = document.getElementById("messages");
    const emptyState = document.getElementById("empty-state");
    const token = form.querySelector("input[name=__RequestVerificationToken]").value;
    let busy = false;

    /** Adds one message bubble to the list and scrolls it into view. Returns the bubble. */
    function addMessage(role, text) {
        if (emptyState) {
            emptyState.remove();
        }
        const item = document.createElement("li");
        item.className = "chat-message chat-" + role;
        item.textContent = text;
        list.appendChild(item);
        item.scrollIntoView({ block: "end" });
        return item;
    }

    const MAX_ROWS = 50;
    const EXCERPT_LENGTH = 280;

    /** Creates an element with optional class and text. Text is set with textContent, so it is never read as HTML. */
    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) {
            node.className = className;
        }
        if (text !== undefined && text !== null) {
            node.textContent = text;
        }
        return node;
    }

    /** Builds a table from the answer's rows; the columns are the names in the first row. */
    function buildTable(rows) {
        const columns = Object.keys(rows[0]);
        const table = el("table", "table table-sm table-striped chat-table");
        const head = table.createTHead().insertRow();
        columns.forEach(function (name) {
            head.appendChild(el("th", null, name.replace(/_/g, " ")));
        });
        const body = table.createTBody();
        rows.slice(0, MAX_ROWS).forEach(function (row) {
            const tr = body.insertRow();
            columns.forEach(function (name) {
                const value = row[name];
                tr.appendChild(el("td", null, value === null || value === undefined ? "" : String(value)));
            });
        });
        const wrapper = el("div", "chat-table-wrap");
        wrapper.appendChild(table);
        if (rows.length > MAX_ROWS) {
            wrapper.appendChild(el("div", "text-muted small", "Showing the first " + MAX_ROWS + " of " + rows.length + " rows."));
        }
        return wrapper;
    }

    // Words that say little about what a question is about; they are not used to pick an excerpt.
    const STOPWORDS = new Set(["what", "does", "were", "where", "which", "there", "their", "would", "should",
        "could", "have", "been", "with", "without", "from", "that", "this", "your", "into", "about", "say",
        "says", "decision", "log", "documents", "document", "also", "than", "then", "when", "whose", "were"]);

    /** Removes markdown marks from a passage and collapses the spacing. */
    function plainText(text) {
        return text
            .replace(/^#+\s*/gm, "")
            .replace(/^-{3,}$/gm, " ")
            .replace(/\*\*/g, "")
            .replace(/\s+/g, " ")
            .trim();
    }

    /** The meaningful words of the question, cut to five letters so "reviews" also finds "reviewed". */
    function keyTerms(question) {
        const words = (String(question || "").toLowerCase().match(/[a-z0-9]{4,}/g) || [])
            .filter(function (word) { return !STOPWORDS.has(word); });
        return Array.from(new Set(words.map(function (word) { return word.length > 5 ? word.slice(0, 5) : word; })));
    }

    /**
     * A short excerpt of a passage. It is the stretch of EXCERPT_LENGTH characters that contains the most
     * of the question's key words (the start of the passage if none match), so the reader sees the lines
     * that answer the question rather than the top of a long chunk.
     */
    function excerpt(text, terms) {
        const plain = plainText(text);
        if (plain.length <= EXCERPT_LENGTH) {
            return plain;
        }
        const lower = plain.toLowerCase();
        let bestStart = 0;
        let bestScore = 0;
        for (let start = 0; start + EXCERPT_LENGTH <= plain.length; start += 20) {
            const window = lower.slice(start, start + EXCERPT_LENGTH);
            const score = terms.filter(function (term) { return window.indexOf(term) !== -1; }).length;
            if (score > bestScore) {
                bestScore = score;
                bestStart = start;
            }
        }
        // Begin at a word boundary so the excerpt does not open in the middle of a word.
        if (bestStart > 0) {
            const space = plain.indexOf(" ", bestStart);
            bestStart = space === -1 ? bestStart : space + 1;
        }
        const end = bestStart + EXCERPT_LENGTH;
        const piece = plain.slice(bestStart, end).trimEnd();
        return (bestStart > 0 ? "…" : "") + piece + (end < plain.length ? "…" : "");
    }

    /** Groups passages by document, keeping the order they came in (best match first). */
    function groupByDocument(sources) {
        const groups = [];
        sources.forEach(function (source) {
            const key = source.title + "|" + (source.url || "");
            let group = groups.find(function (candidate) { return candidate.key === key; });
            if (!group) {
                group = { key: key, title: source.title, url: source.url, passages: [] };
                groups.push(group);
            }
            group.passages.push(source.text);
        });
        return groups;
    }

    /**
     * Builds the "Why, from these documents" panel: one entry per document, showing the best passage as a
     * short excerpt. Further passages from the same document sit in a collapsed list under it.
     */
    function buildSources(sources, question) {
        const terms = keyTerms(question);
        const panel = el("details", "chat-sources");
        panel.open = true;
        panel.appendChild(el("summary", null, "Why, from these documents"));
        groupByDocument(sources).forEach(function (group) {
            const entry = el("div", "chat-source");
            if (group.url) {
                const link = el("a", null, group.title);
                link.href = group.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                entry.appendChild(el("strong")).appendChild(link);
            } else {
                entry.appendChild(el("strong", null, group.title));
            }
            entry.appendChild(el("div", "text-muted small", excerpt(group.passages[0], terms)));
            if (group.passages.length > 1) {
                const more = el("details", "chat-more");
                const count = group.passages.length - 1;
                more.appendChild(el("summary", "small", count + " more passage" + (count === 1 ? "" : "s") + " from this document"));
                group.passages.slice(1).forEach(function (passage) {
                    more.appendChild(el("div", "text-muted small chat-more-item", excerpt(passage, terms)));
                });
                entry.appendChild(more);
            }
            panel.appendChild(entry);
        });
        return panel;
    }

    /** Fills the assistant bubble: the answer text, then a table, the sources and a download link when present. */
    function renderAnswer(bubble, data) {
        const parts = [el("div", "chat-answer", data.answer)];
        if (data.rows && data.rows.length > 0) {
            parts.push(buildTable(data.rows));
        }
        if (data.sources && data.sources.length > 0) {
            parts.push(buildSources(data.sources, data.question));
        }
        if (data.file_url) {
            const name = decodeURIComponent(data.file_url.split("/").pop());
            const link = el("a", "btn btn-outline-primary btn-sm chat-download", "Download " + name);
            link.href = "?handler=File&name=" + encodeURIComponent(name);
            parts.push(link);
        }
        bubble.replaceChildren.apply(bubble, parts);
        bubble.classList.add("chat-rich");
    }

    /** Turns the input and Send button on or off while an answer is being fetched. */
    function setBusy(value) {
        busy = value;
        input.disabled = value;
        sendButton.disabled = value;
        if (!value) {
            input.focus();
        }
    }

    /** Sends one question and fills the assistant bubble with the answer or an error. */
    async function send(question) {
        if (busy) {
            return;
        }
        addMessage("user", question);
        const reply = addMessage("assistant", "Thinking…");
        setBusy(true);
        try {
            const response = await fetch("?handler=Ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "RequestVerificationToken": token
                },
                body: JSON.stringify({ question: question })
            });
            const data = await response.json().catch(function () { return {}; });
            if (response.ok) {
                renderAnswer(reply, data);
            } else {
                reply.textContent = data.error || "Something went wrong. Please try again.";
                reply.classList.add("chat-error");
            }
        } catch (error) {
            reply.textContent = "Could not reach the server. Please try again.";
            reply.classList.add("chat-error");
        } finally {
            setBusy(false);
            reply.scrollIntoView({ block: "end" });
        }
    }

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        const question = input.value.trim();
        if (!question) {
            return;
        }
        input.value = "";
        send(question);
    });

    // Enter sends; Shift+Enter adds a new line.
    input.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    // Example questions send straight away.
    document.querySelectorAll(".example").forEach(function (button) {
        button.addEventListener("click", function () {
            send(button.textContent.trim());
        });
    });
})();
