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
                reply.textContent = data.answer;
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
