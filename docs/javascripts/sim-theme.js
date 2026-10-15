// Keep the embedded reactor simulator (iframe.alc-sim) in the docs' colour scheme.
// The simulator themes itself through <html data-theme="light|dark"> and has its own toggle (saved in its
// localStorage); this sets that attribute from the docs scheme (slate -> dark, default -> light) whenever
// the iframe loads, the docs palette is switched, or an instant navigation swaps the page in.
// Same-origin only: if the iframe document is not reachable, it does nothing.
(function () {
  function docsTheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "light";
  }

  function apply(frame) {
    try {
      var doc = frame.contentDocument;
      if (doc && doc.documentElement) doc.documentElement.dataset.theme = docsTheme();
    } catch (e) { /* cross-origin or not loaded: leave the simulator as it is */ }
  }

  function bind() {
    document.querySelectorAll("iframe.alc-sim").forEach(function (frame) {
      if (!frame.dataset.simThemeBound) {
        frame.dataset.simThemeBound = "1";
        frame.addEventListener("load", function () { apply(frame); });
      }
      apply(frame); // already loaded (the load event fired before this ran)
    });
  }

  function start() {
    bind();
    new MutationObserver(bind).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
    // Instant navigation replaces the page content without a reload: bind the new iframe.
    if (typeof document$ !== "undefined") document$.subscribe(bind);
  }

  if (document.body) start();
  else document.addEventListener("DOMContentLoaded", start);
})();
