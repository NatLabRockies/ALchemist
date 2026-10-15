// Keep the embedded reactor simulator (iframe.alc-sim) in the docs' colour scheme.
// The simulator themes itself through <html data-theme="light|dark"> from an in-memory value it loads from its
// own localStorage (key alchemist-reactor-sim/v2, field theme), and has its own toggle (#theme-toggle). In the
// docs embed the docs palette owns the theme, so this
//   - sets data-theme from the docs scheme (slate -> dark, default -> light) whenever the iframe loads, the docs
//     palette is switched, or an instant navigation swaps the page in;
//   - hides the simulator's own toggle (it only flips the in-memory value, so it would disagree with the display);
//   - writes the docs theme into the simulator's localStorage field so a reload starts in the right theme.
// Same-origin only: if the iframe document is not reachable, it does nothing.
(function () {
  if (window.alcSimThemeStarted) return; // the script ran twice: one observer is enough
  window.alcSimThemeStarted = true;

  var SIM_STORE = "alchemist-reactor-sim/v2";

  function docsTheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "light";
  }

  function apply(frame) {
    var theme = docsTheme();
    try {
      var doc = frame.contentDocument;
      if (!doc || !doc.documentElement) return;
      doc.documentElement.dataset.theme = theme;
      if (doc.head && !doc.getElementById("alc-sim-theme-style")) {
        var style = doc.createElement("style");
        style.id = "alc-sim-theme-style";
        style.textContent = "#theme-toggle { display: none !important; }";
        doc.head.appendChild(style);
      }
    } catch (e) { /* cross-origin or not loaded: leave the simulator as it is */ }
    try {
      var store = frame.contentWindow.localStorage;
      var state = JSON.parse(store.getItem(SIM_STORE) || "{}");
      if (state && typeof state === "object" && state.theme !== theme) {
        state.theme = theme;
        store.setItem(SIM_STORE, JSON.stringify(state));
      }
    } catch (e) { /* storage blocked or unreadable: the attribute is already set */ }
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
