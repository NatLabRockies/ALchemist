window.MathJax = {
  // Preload \boldsymbol at startup rather than leaving it to autoload mid-typeset.
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]], processEscapes: true, processEnvironments: true,
  },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
};
// Re-typeset after each instant navigation. MathJax's own startup typesets the first page, so skip
// document$'s first emission: re-typesetting there either throws (startup not finished, output null)
// or typesets every formula twice.
var alcFirstDocument = true;
document$.subscribe(() => {
  if (alcFirstDocument) { alcFirstDocument = false; return; }
  if (!(window.MathJax.startup && window.MathJax.startup.promise)) return;
  MathJax.startup.promise = MathJax.startup.promise.then(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
