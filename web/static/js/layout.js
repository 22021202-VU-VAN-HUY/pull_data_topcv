// web/static/js/layout.js
(function (window, document) {
  // ----- FORMAT TEXT CHUNG (dùng cho chat/detail nếu cần) -----
  function escapeHtml(s) {
    return (s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatChatMarkup(text) {
    const escaped = escapeHtml(text || "");
    const lines = escaped.split(/\r?\n/);
    const html = [];
    let inList = false;

    const openList = () => {
      if (!inList) {
        html.push('<ul class="list-disc pl-5 my-2">');
        inList = true;
      }
    };
    const closeList = () => {
      if (inList) {
        html.push("</ul>");
        inList = false;
      }
    };

    const boldify = (t) => t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    const mdLink = (t) =>
      t.replace(
        /\[([^\]]+)\]\(([^)]+)\)/g,
        '<a href="$2" rel="noopener noreferrer" class="text-indigo-600 underline">$1</a>'
      );
    const linkify = (t) =>
      t.replace(
        /(https?:\/\/[^\s)]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-indigo-600 underline">$1</a>'
      );

    for (const raw of lines) {
      const trimmed = raw.trim();
      if (trimmed.startsWith("* ")) {
        openList();
        const item = trimmed.replace(/^\*+\s+/, "");
        let content = mdLink(boldify(item));
        if (!content.includes("<a ")) content = linkify(content);
        html.push("<li>" + content + "</li>");
      } else if (trimmed === "") {
        closeList();
        html.push("<br/>");
      } else {
        closeList();
        let content = mdLink(boldify(raw));
        if (!content.includes("<a ")) content = linkify(content);
        html.push("<p>" + content + "</p>");
      }
    }
    closeList();
    return html.join("");
  }

  window.formatChatMarkup = formatChatMarkup;

  // ----- NAV USER (check /api/me) -----
  async function initAuthNav() {
    const navLogin = document.getElementById("nav-login");
    const navRegister = document.getElementById("nav-register");
    const navUserMenu = document.getElementById("nav-user-menu");
    const navUserName = document.getElementById("nav-user-name");
    const navLogout = document.getElementById("nav-logout");
    const navUserBtn = document.getElementById("nav-user-button");
    const navDropdown = document.getElementById("nav-user-dropdown");

    let dropdownOpen = false;

    function setDropdown(open) {
      dropdownOpen = open;
      if (!navDropdown) return;
      navDropdown.classList.toggle("hidden", !open);
    }

    document.addEventListener("click", (e) => {
      if (!navUserMenu || !navDropdown || !dropdownOpen) return;
      if (!navUserMenu.contains(e.target)) {
        setDropdown(false);
      }
    });

    if (navUserBtn) {
      navUserBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        setDropdown(!dropdownOpen);
      });
    }

    // Gọi /api/me để biết đã login chưa
    try {
      const data = await window.apiClient.get("/api/me");
      if (data && data.user) {
        if (navLogin) navLogin.classList.add("hidden");
        if (navRegister) navRegister.classList.add("hidden");
        if (navUserMenu) navUserMenu.classList.remove("hidden");
        if (navUserName && data.user.full_name) {
          navUserName.textContent = data.user.full_name;
        }
      } else {
        if (navLogin) navLogin.classList.remove("hidden");
        if (navRegister) navRegister.classList.remove("hidden");
        if (navUserMenu) navUserMenu.classList.add("hidden");
      }
    } catch (e) {
      // không sao, giữ giao diện mặc định (chưa login)
    }

    if (navLogout) {
      navLogout.addEventListener("click", async () => {
        try {
          await window.apiClient.post("/api/logout", {});
        } catch (err) {
          // ignore
        }
        window.location.href = "/";
      });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    initAuthNav();
    initBookmarkButtons();
  });
  // ----- BOOKMARK JOBS (★) -----
  function setBookmarkState(btn, starred) {
    if (!btn) return;
    btn.classList.toggle("text-yellow-400", !!starred);
    btn.classList.toggle("text-gray-300", !starred);
    btn.dataset.starred = starred ? "1" : "0";
  }

  function initBookmarkButtons() {
    const buttons = document.querySelectorAll(".home-bookmark-button");
    if (!buttons || buttons.length === 0) return;

    buttons.forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();

        const jobId = btn.getAttribute("data-job-id");
        if (!jobId) return;

        // tránh spam
        if (btn.dataset.loading === "1") return;
        btn.dataset.loading = "1";

        try {
          const res = await window.apiClient.post(
            `/api/star?job_id=${encodeURIComponent(jobId)}`,
            {}
          );
          setBookmarkState(btn, !!(res && res.starred));
        } catch (err) {
          if (err && err.status === 401) {
            window.location.href = "/login";
          } else {
            console.error("Không lưu được bookmark", err);
          }
        } finally {
          btn.dataset.loading = "0";
        }
      });

      // đồng bộ trạng thái ban đầu (dựa vào class do server render)
      const starred = btn.classList.contains("text-yellow-400");
      setBookmarkState(btn, starred);
    });
  }
})(window, document);
