// web/static/js/api_client.js
(function (window) {
  async function request(method, url, body) {
    const options = {
      method: method.toUpperCase(),
      headers: {}
    };
    if (body !== undefined && body !== null) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const res = await fetch(url, options);
    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      data = null;
    }
    if (!res.ok) {
      const err = new Error((data && data.detail) || "Request failed");
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function get(url) {
    return request("GET", url);
  }

  function post(url, body) {
    return request("POST", url, body);
  }

  window.apiClient = { get, post };
})(window);
