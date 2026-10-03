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

    /** Builds the "Why, from these documents" panel: one entry per document passage. */
    function buildSources(sources) {
        const panel = el("details", "chat-sources");
        panel.open = true;
        panel.appendChild(el("summary", null, "Why, from these documents"));
        sources.forEach(function (source) {
            const entry = el("div", "chat-source");
            if (source.url) {
                const link = el("a", null, source.title);
                link.href = source.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                entry.appendChild(el("strong")).appendChild(link);
            } else {
                entry.appendChild(el("strong", null, source.title));
            }
            entry.appendChild(el("div", "text-muted small", source.text));
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
            parts.push(buildSources(data.sources));
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
