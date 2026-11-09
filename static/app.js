// AI Justice Department Assistant — app.js
// Handles conversational thread, streaming SSE, follow-up history, citations, and feedback.

(function () {
  "use strict";

  // ── DOM References ──────────────────────────────────────────────────────────
  const thread = document.getElementById("thread");
  const starterSection = document.getElementById("starter-section");
  const form = document.getElementById("question-form");
  const questionInput = document.getElementById("question-input");
  const submitBtn = document.getElementById("submit-btn");
  const statusBar = document.getElementById("status-bar");
  const statusText = document.getElementById("status-text");
  const stopBtn = document.getElementById("stop-btn");
  const newConversationBtn = document.getElementById("new-conversation-btn");
  const languageSelect = document.getElementById("language-select");

  // ── State ───────────────────────────────────────────────────────────────────
  let conversationHistory = []; // list of { question, answer }
  let currentAbortController = null;
  let turnCounter = 0;
  let lastFailedQuestion = null;

  // ── Helpers ─────────────────────────────────────────────────────────────────

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Convert plain text into paragraphs and turn [1], [2] markers into clickable citation links
  function formatAnswerWithCitations(rawText, turnId) {
    if (!rawText) return "";

    // Replace [1], [2], [1, 2] with clickable links
    const citationPattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
    const withLinks = rawText.replace(citationPattern, function (match, digits) {
      const parts = digits.split(/\s*,\s*/).map(function (num) {
        const id = "source-" + turnId + "-" + num.trim();
        return (
          '<a href="#' +
          id +
          '" class="citation-link" data-source-id="' +
          id +
          '">[' +
          num.trim() +
          ']</a>'
        );
      });
      return parts.join(" ");
    });

    // Split on double newlines for paragraphs
    const paragraphs = withLinks.split(/\n{2,}/);
    return paragraphs
      .map(function (p) {
        const trimmed = p.trim();
        return trimmed ? "<p>" + trimmed.replace(/\n/g, "<br>") + "</p>" : "";
      })
      .join("");
  }

  function scrollToBottom() {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: "smooth",
    });
  }

  function autoResizeTextarea() {
    questionInput.style.height = "auto";
    const newHeight = Math.min(questionInput.scrollHeight, 130);
    questionInput.style.height = (newHeight > 42 ? newHeight : 42) + "px";
  }

  function setStatus(text, isVisible) {
    if (isVisible) {
      statusText.textContent = text;
      statusBar.hidden = false;
      submitBtn.disabled = true;
    } else {
      statusText.textContent = "";
      statusBar.hidden = true;
      submitBtn.disabled = false;
    }
  }

  function updateStarterVisibility() {
    if (starterSection) {
      starterSection.hidden = thread.children.length > 0;
    }
  }

  // ── Sources Rendering ───────────────────────────────────────────────────────

  function renderSourcesPanel(sources, turnId) {
    if (!sources || sources.length === 0) {
      return null;
    }

    const panel = document.createElement("div");
    panel.className = "sources-panel";

    const heading = document.createElement("div");
    heading.className = "sources-heading";
    heading.textContent = "Sources";
    panel.appendChild(heading);

    const ol = document.createElement("ol");
    ol.className = "sources-list";

    sources.forEach(function (s) {
      const idx = s.index || 1;
      const li = document.createElement("li");
      li.className = "source-item";
      li.id = "source-" + turnId + "-" + idx;

      const mainDiv = document.createElement("div");
      mainDiv.className = "source-main";

      const sectionText = s.section && s.section !== "N/A" ? ", " + s.section : "";
      const fileText = s.file && s.file !== "N/A" ? s.file : "";
      const pageText =
        s.page !== undefined && s.page !== null && s.page !== "N/A"
          ? "page " + s.page
          : "";
      const fileRef = [fileText, pageText].filter(Boolean).join(", ");
      const citeStr =
        s.act_name + sectionText + (fileRef ? " (" + fileRef + ")" : "");

      const titleSpan = document.createElement("span");
      titleSpan.className = "source-title";
      titleSpan.textContent = citeStr;
      mainDiv.appendChild(titleSpan);

      if (s.passage) {
        const passageId = "passage-" + turnId + "-" + idx;
        const toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "toggle-passage-btn";
        toggleBtn.textContent = "Show passage";
        toggleBtn.setAttribute("aria-expanded", "false");
        toggleBtn.setAttribute("aria-controls", passageId);

        const passageDiv = document.createElement("div");
        passageDiv.className = "source-passage";
        passageDiv.id = passageId;
        passageDiv.hidden = true;
        passageDiv.textContent = s.passage;

        if (s.score !== undefined && s.score !== null) {
          const scoreSpan = document.createElement("span");
          scoreSpan.className = "source-score";
          scoreSpan.textContent = "Distance: " + s.score;
          passageDiv.appendChild(scoreSpan);
        }

        toggleBtn.addEventListener("click", function () {
          const isHidden = passageDiv.hidden;
          passageDiv.hidden = !isHidden;
          toggleBtn.textContent = isHidden ? "Hide passage" : "Show passage";
          toggleBtn.setAttribute("aria-expanded", String(isHidden));
        });

        mainDiv.appendChild(toggleBtn);
        li.appendChild(mainDiv);
        li.appendChild(passageDiv);
      } else {
        li.appendChild(mainDiv);
      }

      ol.appendChild(li);
    });

    panel.appendChild(ol);
    return panel;
  }

  // ── Answer Actions (Copy & Feedback) ────────────────────────────────────────

  function renderActions(turnId, questionText, getAnswerText, sources) {
    const actions = document.createElement("div");
    actions.className = "answer-actions";

    // Copy answer button
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "action-btn";
    copyBtn.textContent = "Copy answer";

    copyBtn.addEventListener("click", function () {
      const answerText = getAnswerText();
      let fullCopy = answerText;

      if (sources && sources.length > 0) {
        fullCopy += "\n\nSources:\n";
        sources.forEach(function (s, i) {
          const sec = s.section && s.section !== "N/A" ? ", " + s.section : "";
          const file = s.file && s.file !== "N/A" ? s.file : "";
          const page =
            s.page !== undefined && s.page !== null && s.page !== "N/A"
              ? "page " + s.page
              : "";
          const fileRef = [file, page].filter(Boolean).join(", ");
          fullCopy += (i + 1) + ". " + s.act_name + sec + (fileRef ? " (" + fileRef + ")" : "") + "\n";
        });
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(fullCopy).then(
          function () {
            copyBtn.textContent = "Copied";
            setTimeout(function () {
              copyBtn.textContent = "Copy answer";
            }, 2000);
          },
          function () {
            copyBtn.textContent = "Unable to copy";
          }
        );
      }
    });
    actions.appendChild(copyBtn);

    // Separator
    const sep = document.createElement("span");
    sep.className = "action-sep";
    sep.textContent = "|";
    actions.appendChild(sep);

    // Feedback wrapper
    const fbWrapper = document.createElement("span");
    fbWrapper.className = "feedback-wrapper";

    const fbLabel = document.createElement("span");
    fbLabel.textContent = "Was this helpful?";
    fbWrapper.appendChild(fbLabel);

    const yesBtn = document.createElement("button");
    yesBtn.type = "button";
    yesBtn.className = "action-btn";
    yesBtn.textContent = "Yes";

    const noBtn = document.createElement("button");
    noBtn.type = "button";
    noBtn.className = "action-btn";
    noBtn.textContent = "No";

    function sendFeedback(isHelpful) {
      yesBtn.disabled = true;
      noBtn.disabled = true;

      fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: questionText,
          answer: getAnswerText(),
          helpful: isHelpful,
        }),
      })
        .then(function () {
          fbWrapper.innerHTML = '<span class="feedback-confirm">Thank you for your feedback.</span>';
        })
        .catch(function () {
          fbWrapper.innerHTML = '<span class="feedback-confirm">Feedback recorded.</span>';
        });
    }

    yesBtn.addEventListener("click", function () {
      sendFeedback(true);
    });
    noBtn.addEventListener("click", function () {
      sendFeedback(false);
    });

    fbWrapper.appendChild(yesBtn);
    fbWrapper.appendChild(noBtn);
    actions.appendChild(fbWrapper);

    return actions;
  }

  // ── Click delegation for citations ──────────────────────────────────────────

  document.addEventListener("click", function (event) {
    const link = event.target.closest(".citation-link");
    if (!link) return;

    event.preventDefault();
    const targetId = link.getAttribute("data-source-id");
    if (!targetId) return;

    const targetEl = document.getElementById(targetId);
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: "smooth", block: "center" });
      targetEl.classList.remove("highlighted");
      // Force DOM reflow so re-clicking re-triggers the flash animation
      void targetEl.offsetWidth;
      targetEl.classList.add("highlighted");
    }
  });

  // ── Core Streaming Submit ───────────────────────────────────────────────────

  async function submitQuestion(question) {
    question = question.trim();
    if (!question) return;

    lastFailedQuestion = null;
    turnCounter += 1;
    const turnId = turnCounter;

    // Build the visual turn group
    const turnGroup = document.createElement("div");
    turnGroup.className = "turn-group";
    turnGroup.id = "turn-" + turnId;

    // User block
    const userBlock = document.createElement("div");
    userBlock.className = "user-block";

    const userRole = document.createElement("span");
    userRole.className = "role-title";
    userRole.textContent = "You";
    userBlock.appendChild(userRole);

    const userQuery = document.createElement("div");
    userQuery.className = "user-query";
    userQuery.textContent = question;
    userBlock.appendChild(userQuery);

    turnGroup.appendChild(userBlock);

    // Assistant block
    const assistantBlock = document.createElement("div");
    assistantBlock.className = "assistant-block";

    const assistantRole = document.createElement("span");
    assistantRole.className = "role-title";
    assistantRole.textContent = "Assistant";
    assistantBlock.appendChild(assistantRole);

    const answerBody = document.createElement("div");
    answerBody.className = "answer-body";
    assistantBlock.appendChild(answerBody);

    turnGroup.appendChild(assistantBlock);
    thread.appendChild(turnGroup);
    updateStarterVisibility();
    scrollToBottom();

    // Prepare state for this turn
    let accumulatedText = "";
    let retrievedSources = [];
    const language = (languageSelect && languageSelect.value) || "en";
    const historyPayload = conversationHistory.slice(-4);

    setStatus("Searching the documents...", true);
    currentAbortController = new AbortController();

    try {
      const response = await fetch("/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          history: historyPayload,
          language: language,
        }),
        signal: currentAbortController.signal,
      });

      if (!response.ok) {
        let errorMsg = "The server returned an error (" + response.status + ").";
        try {
          const errData = await response.json();
          if (errData.detail) errorMsg = errData.detail;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // keep partial chunk

        for (const block of lines) {
          const trimmed = block.trim();
          if (!trimmed.startsWith("data:")) continue;

          const jsonStr = trimmed.slice(5).trim();
          if (!jsonStr) continue;

          try {
            const eventObj = JSON.parse(jsonStr);

            if (eventObj.type === "status") {
              if (eventObj.data === "searching") {
                setStatus("Searching the documents...", true);
              } else if (eventObj.data === "writing") {
                setStatus("Writing the answer...", true);
              }
            } else if (eventObj.type === "sources") {
              retrievedSources = eventObj.data || [];
            } else if (eventObj.type === "token") {
              accumulatedText += eventObj.data;
              answerBody.innerHTML = formatAnswerWithCitations(accumulatedText, turnId);
              scrollToBottom();
            } else if (eventObj.type === "error") {
              throw new Error(eventObj.data);
            } else if (eventObj.type === "done") {
              // Finalizing turn
            }
          } catch (parseErr) {
            if (parseErr.message && parseErr.message !== "Unexpected end of JSON input") {
              throw parseErr;
            }
          }
        }
      }

      // Check if no answer was returned
      if (!accumulatedText.trim()) {
        accumulatedText = "I couldn't find this in the available legal documents.";
        answerBody.innerHTML = formatAnswerWithCitations(accumulatedText, turnId);
      }

      // If nothing relevant was retrieved, hide sources
      const isNotFound = accumulatedText.indexOf("couldn't find this in the available legal documents") !== -1;

      if (!isNotFound && retrievedSources.length > 0) {
        const sourcesPanel = renderSourcesPanel(retrievedSources, turnId);
        if (sourcesPanel) {
          assistantBlock.appendChild(sourcesPanel);
        }
      }

      // Append copy & feedback actions
      const actions = renderActions(
        turnId,
        question,
        function () {
          return accumulatedText;
        },
        isNotFound ? [] : retrievedSources
      );
      assistantBlock.appendChild(actions);

      // Save turn to history
      conversationHistory.push({
        question: question,
        answer: accumulatedText,
      });

    } catch (err) {
      if (err.name === "AbortError") {
        if (!accumulatedText.trim()) {
          accumulatedText = "The request was stopped.";
          answerBody.innerHTML = "<p><em>" + escapeHtml(accumulatedText) + "</em></p>";
        }
      } else {
        lastFailedQuestion = question;
        renderErrorState(assistantBlock, err.message || "Could not reach the server.");
      }
    } finally {
      setStatus("", false);
      currentAbortController = null;
      scrollToBottom();
    }
  }

  function renderErrorState(container, message) {
    const errorCard = document.createElement("div");
    errorCard.className = "error-card";

    const msgP = document.createElement("p");
    msgP.textContent = message;
    errorCard.appendChild(msgP);

    const actionDiv = document.createElement("div");
    actionDiv.className = "error-actions";

    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "retry-btn";
    retryBtn.textContent = "Try again";
    retryBtn.addEventListener("click", function () {
      errorCard.remove();
      if (lastFailedQuestion) {
        submitQuestion(lastFailedQuestion);
      }
    });

    actionDiv.appendChild(retryBtn);
    errorCard.appendChild(actionDiv);
    container.appendChild(errorCard);
  }

  // ── Event Listeners ─────────────────────────────────────────────────────────

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const q = questionInput.value.trim();
    if (!q) {
      questionInput.focus();
      return;
    }
    questionInput.value = "";
    autoResizeTextarea();
    submitQuestion(q);
  });

  questionInput.addEventListener("input", autoResizeTextarea);

  // Enter sends, Shift+Enter adds newline
  questionInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // Stop button
  stopBtn.addEventListener("click", function () {
    if (currentAbortController) {
      currentAbortController.abort();
    }
  });

  // New conversation button
  newConversationBtn.addEventListener("click", function () {
    if (currentAbortController) {
      currentAbortController.abort();
    }
    conversationHistory = [];
    thread.innerHTML = "";
    lastFailedQuestion = null;
    questionInput.value = "";
    autoResizeTextarea();
    updateStarterVisibility();
    questionInput.focus();
  });

  // Starter questions buttons
  document.querySelectorAll(".starter-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const q = btn.getAttribute("data-question");
      if (q) {
        submitQuestion(q);
      }
    });
  });

  // Initial setup
  updateStarterVisibility();
  questionInput.focus();
})();
