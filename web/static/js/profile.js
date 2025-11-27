// web/static/js/profile.js
(function (window, document) {
  function initProfileForm() {
    const form = document.getElementById("profile-form");
    if (!form) return;

    const fullNameEl = document.getElementById("profile-fullname");
    const phoneEl = document.getElementById("profile-phone");
    const statusEl = document.getElementById("profile-status");
    const btnSave = document.getElementById("btn-profile-save");

    btnSave.addEventListener("click", async () => {
      const full_name = (fullNameEl.value || "").trim();
      const phone = (phoneEl.value || "").trim();
      statusEl.textContent = "Đang lưu...";
      try {
        await window.apiClient.post("/api/me/update", {
          full_name,
          phone
        });
        statusEl.textContent = "Đã lưu thông tin.";
      } catch (err) {
        statusEl.textContent =
          (err.data && err.data.detail) || "Không lưu được thông tin.";
      }
    });
  }

  function initPasswordForm() {
    const form = document.getElementById("password-form");
    if (!form) return;

    const oldEl = document.getElementById("pw-old");
    const newEl = document.getElementById("pw-new");
    const new2El = document.getElementById("pw-new2");
    const statusEl = document.getElementById("pw-status");
    const btnChange = document.getElementById("btn-pw-change");

    btnChange.addEventListener("click", async () => {
      const old_pw = (oldEl.value || "").trim();
      const new_pw = (newEl.value || "").trim();
      const new_pw2 = (new2El.value || "").trim();

      if (!old_pw || !new_pw || !new_pw2) {
        statusEl.textContent = "Vui lòng nhập đầy đủ.";
        return;
      }
      if (new_pw !== new_pw2) {
        statusEl.textContent = "Mật khẩu mới nhập lại không khớp.";
        return;
      }

      statusEl.textContent = "Đang đổi mật khẩu...";
      try {
        await window.apiClient.post("/api/me/change_password", {
          old_password: old_pw,
          new_password: new_pw
        });
        statusEl.textContent = "Đổi mật khẩu thành công.";
      } catch (err) {
        statusEl.textContent =
          (err.data && err.data.detail) || "Không đổi được mật khẩu.";
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initProfileForm();
    initPasswordForm();
  });
})(window, document);
