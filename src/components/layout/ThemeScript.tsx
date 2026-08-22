import Script from "next/script";

/**
 * Applies the stored theme before hydration so there is no flash of the wrong
 * palette. It has to be an inline, blocking script — anything asynchronous
 * paints the default theme first.
 */
const script = `(function () {
  try {
    var raw = localStorage.getItem("pytorch-prep:v1");
    var theme = raw ? (JSON.parse(raw).prefs || {}).theme : null;
    if (theme !== "light" && theme !== "dark") {
      theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    document.documentElement.classList.toggle("dark", theme === "dark");
  } catch (e) {
    /* first visit or blocked storage: fall back to the light palette */
  }
})();`;

export function ThemeScript() {
  return (
    // The lint rule below predates the App Router, where `beforeInteractive`
    // in the root layout is the documented way to run a blocking script.
    // eslint-disable-next-line @next/next/no-before-interactive-script-outside-document
    <Script id="theme-init" strategy="beforeInteractive">
      {script}
    </Script>
  );
}
