// web/static/js/auth.js
(function (window, document) {
  function initLoginForm() {
    const emailEl = document.getElementById("login-email");
    const passEl = document.getElementById("login-password");
    const statusEl = document.getElementById("login-status");
    const btnLogin = document.getElementById("btn-login");

    if (!btnLogin) return;

    async function doLogin() {
      const email = (emailEl.value || "").trim();
      const password = (passEl.value || "").trim();
      if (!email || !password) {
        statusEl.textContent = "Vui lòng nhập email và mật khẩu.";
        return;
      }
      statusEl.textContent = "Đang đăng nhập...";
      try {
        await window.apiClient.post("/api/login", { email, password });
        window.location.href = "/";
      } catch (err) {
        statusEl.textContent =
          (err.data && err.data.detail) || "Đăng nhập thất bại.";
      }
    }

    btnLogin.addEventListener("click", doLogin);
  }

  function initRegisterForm() {
    const fullNameEl = document.getElementById("reg-fullname");
    const emailEl = document.getElementById("reg-email");
    const phoneEl = document.getElementById("reg-phone");
    const p1 = document.getElementById("reg-password");
    const p2 = document.getElementById("reg-password2");
    const statusEl = document.getElementById("reg-status");
    const btnRegister = document.getElementById("btn-register");

    if (!btnRegister) return;

    async function doRegister() {
      const full_name = (fullNameEl.value || "").trim();
      const email = (emailEl.value || "").trim();
      const phone = (phoneEl.value || "").trim();
      const pw1 = (p1.value || "").trim();
      const pw2 = (p2.value || "").trim();

      if (!full_name || !email || !pw1 || !pw2) {
        statusEl.textContent = "Vui lòng nhập đầy đủ thông tin.";
        return;
      }
      if (pw1 !== pw2) {
        statusEl.textContent = "Mật khẩu nhập lại không khớp.";
        return;
      }

      statusEl.textContent = "Đang đăng ký...";
      try {
        await window.apiClient.post("/api/register", {
          full_name,
          email,
          phone,
          password: pw1
        });
        window.location.href = "/";
      } catch (err) {
        statusEl.textContent =
          (err.data && err.data.detail) || "Đăng ký thất bại.";
      }
    }

    btnRegister.addEventListener("click", doRegister);
  }

  document.addEventListener("DOMContentLoaded", () => {
    initLoginForm();
    initRegisterForm();
  });
})(window, document);
