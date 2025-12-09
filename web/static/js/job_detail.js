document.addEventListener('DOMContentLoaded', function () {
  const bodies = document.querySelectorAll('.detail-section-body');
  bodies.forEach(function (el) {
    const raw = el.innerHTML || '';
    let cleaned = raw
      // Normalize invalid closing <\/br> tags to <br>
      .replace(/<\/(br)\s*>/gi, '<br>')
      // Collapse multiple consecutive <br> tags into one
      .replace(/(<br\s*\/?>(\s|&nbsp;)*){2,}/gi, '<br>')
      // Trim leading and trailing <br>
      .replace(/^(<br\s*\/?>(\s|&nbsp;)*)+/i, '')
      .replace(/(<br\s*\/?>(\s|&nbsp;)*)+$/i, '')
      // Drop empty paragraphs that only contain whitespace or <br>
      .replace(/<p>(\s|&nbsp;|<br\s*\/??>)*<\/p>/gi, '');

    if (cleaned !== raw) {
      el.innerHTML = cleaned;
    }
  });
});