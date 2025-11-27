// web/static/js/chat_widget.js

(function () {
  const CHAT_STORAGE_KEY = "jf_chat_history_v1";

  let chatOpen = false;
  let chatHistory = [];

  const widget = document.getElementById("chat-widget");
  if (!widget) return;

  const panel = document.getElementById("chat-panel");
  const toggleBtn = document.getElementById("chat-toggle");
  const closeBtn = document.getElementById("chat-close");
  const messagesEl = document.getElementById("chat-messages");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const statusEl = document.getElementById("chat-status");

  if (!panel || !toggleBtn || !closeBtn || !messagesEl || !form || !input || !sendBtn) {
    return;
  }

  // ====== Helpers ======

  function loadHistoryFromStorage() {
    try {
      const raw = sessionStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed;
    } catch (e) {
      console.warn("Cannot load chat history", e);
      return [];
    }
  }

  function saveHistoryToStorage() {
    try {
      sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatHistory));
    } catch (e) {
      console.warn("Cannot save chat history", e);
    }
  }

  function scrollToBottom() {
    setTimeout(() => {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }, 10);
  }

  function renderHistory() {
    messagesEl.innerHTML = "";

    chatHistory.forEach((m) => {
      const item = document.createElement("div");
      item.className =
        "chat-msg mb-2 flex " +
        (m.role === "user" ? "justify-end" : "justify-start");

      const bubble = document.createElement("div");
      bubble.className =
        "inline-block px-3 py-2 rounded-2xl text-sm max-w-[80%] " +
        (m.role === "user"
          ? "bg-indigo-600 text-white"
          : "bg-white border border-gray-200 text-gray-800");

      if (m.role === "assistant") {
        // cho phép HTML (đáp án từ server có thể chứa link)
        bubble.innerHTML = m.content;
      } else {
        bubble.textContent = m.content;
      }

      item.appendChild(bubble);
      messagesEl.appendChild(item);
    });

    scrollToBottom();
  }

  function setSendingState(isSending) {
    if (isSending) {
      sendBtn.disabled = true;
      input.disabled = true;
      statusEl.textContent = "Đang gửi...";
    } else {
      sendBtn.disabled = false;
      input.disabled = false;
      statusEl.textContent = "";
    }
  }

  function openChat() {
    chatOpen = true;
    panel.classList.remove("hidden");
  }

  function closeChat() {
    chatOpen = false;
    panel.classList.add("hidden");
  }

  // ====== Init ======

  // bật input (vì trước đây bạn disable trong HTML)
  input.disabled = false;
  sendBtn.disabled = false;
  input.placeholder = "Nhập câu hỏi về công việc, lương, kỹ năng...";
  statusEl.textContent = "";

  if (toggleBtn) {
    toggleBtn.textContent = "Chat";
    toggleBtn.title = "Mở chat";
    toggleBtn.classList.remove("bg-gray-400");
    toggleBtn.classList.add("bg-indigo-600");
  }

  chatHistory = loadHistoryFromStorage();
  renderHistory();

  // ====== Events ======

  toggleBtn.addEventListener("click", function () {
    if (chatOpen) {
      closeChat();
    } else {
      openChat();
      scrollToBottom();
    }
  });

  closeBtn.addEventListener("click", function () {
    closeChat();
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const text = (input.value || "").trim();
    if (!text) return;

    // lấy history hiện tại để gửi lên server
    const historyToSend = chatHistory.slice();

    // thêm message user vào UI ngay (local)
    chatHistory.push({ role: "user", content: text });
    renderHistory();
    saveHistoryToStorage();

    input.value = "";
    setSendingState(true);

    // Nếu đang ở trang chi tiết job, bạn có thể set biến global:
    // window.JF_CURRENT_JOB_ID = {{ job.job_id }}
    const currentJobId =
      typeof window.JF_CURRENT_JOB_ID !== "undefined"
        ? window.JF_CURRENT_JOB_ID
        : null;

    fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: text,
        history: historyToSend,
        current_job_id: currentJobId,
      }),
    })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error("HTTP " + res.status);
        }
        return res.json();
      })
      .then((data) => {
        const answer = data.answer || "(Không có câu trả lời)";
        const newHistory = Array.isArray(data.history) ? data.history : null;

        if (newHistory) {
          chatHistory = newHistory;
        } else {
          // fallback: append assistant message
          chatHistory.push({ role: "assistant", content: answer });
        }

        renderHistory();
        saveHistoryToStorage();

        // TODO: nếu muốn hiển thị danh sách job gợi ý riêng trong chat,
        // có thể dùng data.jobs ở đây.
      })
      .catch((err) => {
        console.error("Chat error", err);
        chatHistory.push({
          role: "assistant",
          content: "Xin lỗi, chatbot đang gặp sự cố. Bạn thử lại sau nhé.",
        });
        renderHistory();
        saveHistoryToStorage();
      })
      .finally(() => {
        setSendingState(false);
        input.focus();
      });
  });
})();
