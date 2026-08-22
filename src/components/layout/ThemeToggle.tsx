"use client";

import { useEffect } from "react";
import { useAppState } from "@/hooks/useAppState";
import { setPreference } from "@/lib/storage";
import type { ThemePreference } from "@/lib/types";
import { cn } from "@/lib/utils";

const OPTIONS: { value: ThemePreference; label: string; icon: string }[] = [
  { value: "light", label: "Light", icon: "☀" },
  { value: "dark", label: "Dark", icon: "☾" },
  { value: "system", label: "System", icon: "⌘" },
];

function apply(theme: ThemePreference) {
  const dark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
}

export function ThemeToggle() {
  const { prefs } = useAppState();
  const theme = prefs.theme;

  useEffect(() => {
    apply(theme);
    if (theme !== "system") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => apply("system");
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [theme]);

  return (
    <div
      className="flex items-center gap-0.5 rounded-lg border border-border-base bg-surface-2 p-0.5"
      role="radiogroup"
      aria-label="Colour theme"
    >
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={theme === option.value}
          aria-label={`${option.label} theme`}
          title={`${option.label} theme`}
          onClick={() => setPreference("theme", option.value)}
          className={cn(
            "rounded-md px-2 py-1 text-xs transition-colors",
            theme === option.value
              ? "bg-surface text-text shadow-[var(--shadow-card)]"
              : "text-muted hover:text-text",
          )}
        >
          <span aria-hidden>{option.icon}</span>
        </button>
      ))}
    </div>
  );
}
