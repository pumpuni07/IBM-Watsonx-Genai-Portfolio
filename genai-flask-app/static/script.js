/*
 * Chat frontend logic for the GenAI Flask app.
 *
 * Own implementation written against the element IDs of the lab-provided
 * index.html (the lab's original gist-hosted assets are not redistributed
 * here). Handles: sending messages to POST /generate, rendering the
 * structured JSON response (summary, sentiment, response, next_step,
 * duration), loading states, Enter-to-send, and clearing the chat.
 */

(function () {
  "use strict";

  const form = document.getElementById("chatForm");
  const input = document.getElementById("messageInput");
  const modelSelect = document.getElementById("modelSelect");
  const sendButton = document.getElementById("sendButton");
  const sendIcon = document.getElementById("sendIcon");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const messagesEnd = document.getElementById("messagesEnd");
  const clearBtn = document.getElementById("clearBtn");

  let busy = false;

  function scrollToBottom() {
    messagesEnd.scrollIntoView({ behavior: "smooth" });
  }

  function setBusy(state) {
    busy = state;
    sendButton.disabled = state;
    input.disabled = state;
    sendIcon.style.display = state ? "none" : "inline";
    loadingSpinner.style.display = state ? "inline-block" : "none";
    loadingIndicator.style.display = state ? "block" : "none";
  }

  function hideWelcome() {
    if (welcomeScreen) welcomeScreen.style.display = "none";
    clearBtn.style.display = "inline-flex";
  }

  function addUserMessage(text) {
    const el = document.createElement("div");
    el.className = "message message-user";
    el.innerHTML = '<div class="bubble bubble-user"></div>';
    el.querySelector(".bubble").textContent = text;
    messagesContainer.appendChild(el);
    scrollToBottom();
  }

  function sentimentClass(score) {
    if (score >= 67) return "sentiment-positive";
    if (score >= 34) return "sentiment-neutral";
    return "sentiment-negative";
  }

  function addAIMessage(data, modelName) {
    const el = document.createElement("div");
    el.className = "message message-ai";

    if (data.error) {
      el.innerHTML = '<div class="bubble bubble-ai bubble-error"></div>';
      el.querySelector(".bubble").textContent = "Error: " + data.error;
    } else {
      const bubble = document.createElement("div");
      bubble.className = "bubble bubble-ai";

      const meta = document.createElement("div");
      meta.className = "ai-meta";
      const duration = typeof data.duration === "number" ? data.duration.toFixed(2) + "s" : "";
      meta.textContent = modelName + (duration ? " · " + duration : "");
      bubble.appendChild(meta);

      const fields = [
        ["Summary", data.summary],
        ["Response", data.response],
        ["Next step", data.next_step],
      ];
      fields.forEach(function (pair) {
        if (pair[1] === undefined || pair[1] === null) return;
        const row = document.createElement("div");
        row.className = "ai-field";
        const label = document.createElement("span");
        label.className = "ai-label";
        label.textContent = pair[0] + ": ";
        row.appendChild(label);
        row.appendChild(document.createTextNode(String(pair[1])));
        bubble.appendChild(row);
      });

      if (typeof data.sentiment === "number") {
        const badge = document.createElement("span");
        badge.className = "sentiment-badge " + sentimentClass(data.sentiment);
        badge.textContent = "Sentiment: " + data.sentiment + "/100";
        bubble.appendChild(badge);
      }

      el.appendChild(bubble);
    }

    messagesContainer.appendChild(el);
    scrollToBottom();
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || busy) return;

    hideWelcome();
    addUserMessage(text);
    input.value = "";
    autoResize();
    setBusy(true);

    const modelName = modelSelect.options[modelSelect.selectedIndex].text;

    try {
      const res = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, model: modelSelect.value }),
      });
      const data = await res.json();
      addAIMessage(data, modelName);
    } catch (err) {
      addAIMessage({ error: String(err) }, modelName);
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    sendMessage();
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener("input", autoResize);

  clearBtn.addEventListener("click", function () {
    messagesContainer.innerHTML = "";
    clearBtn.style.display = "none";
    if (welcomeScreen) welcomeScreen.style.display = "block";
    input.focus();
  });

  input.focus();
})();
