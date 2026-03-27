function $(id) {
  return document.getElementById(id);
}

console.log("AGENT JS NEW VERSION LOADED");

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderSimpleMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

function appendMessage(role, text) {
  const wrap = $("chatMessages");
  if (!wrap) return;

  const msg = document.createElement("div");
  msg.className = `chat-msg ${role === "user" ? "chat-msg-user" : "chat-msg-agent"}`;

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";

  if (role === "agent") {
    console.log("RAW AGENT TEXT:", text);
    console.log("RENDERED HTML:", renderSimpleMarkdown(text));
    bubble.innerHTML = renderSimpleMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  msg.appendChild(bubble);
  wrap.appendChild(msg);
  wrap.scrollTop = wrap.scrollHeight;
}

function setSendingState(isSending) {
  const input = $("agentInput");
  const button = $("agentSendBtn");

  if (input) input.disabled = isSending;
  if (button) {
    button.disabled = isSending;
    button.textContent = isSending ? "Sending..." : "Send";
  }
}

async function sendAgentMessage(message) {
  const res = await fetch("/agent/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      context: {},
      history: [],
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Agent request failed");
  }

  return await res.json();
}

function initAgentChat() {
  const form = $("agentForm");
  const input = $("agentInput");

  if (!form || !input) return;

  appendMessage("agent", "Test **bold** message");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = input.value.trim();
    if (!text) return;

    appendMessage("user", text);
    input.value = "";
    setSendingState(true);

    try {
      const payload = await sendAgentMessage(text);
      appendMessage("agent", payload.answer || "No response from agent.");
    } catch (err) {
      appendMessage("agent", `Error: ${err.message}`);
    } finally {
      setSendingState(false);
      input.focus();
    }
  });
}

initAgentChat();