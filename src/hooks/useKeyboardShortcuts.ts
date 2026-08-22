"use client";

import { useEffect, useRef } from "react";

export interface ShortcutHandler {
  /** Lower-case `event.key`, e.g. "k", "enter", "h", "n". */
  key: string;
  meta?: boolean;
  shift?: boolean;
  /** Fire even when focus is inside an input/textarea. Default false. */
  allowInInput?: boolean;
  handler: (event: KeyboardEvent) => void;
  description?: string;
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
}

/**
 * Global keyboard shortcuts. Single-letter shortcuts are suppressed while the
 * learner is typing an answer; Cmd/Ctrl combos still fire.
 */
export function useKeyboardShortcuts(shortcuts: ShortcutHandler[], enabled = true) {
  const ref = useRef(shortcuts);

  // Refs must not be written during render; this keeps the handler list fresh
  // without re-subscribing the listener on every render.
  useEffect(() => {
    ref.current = shortcuts;
  });

  useEffect(() => {
    if (!enabled) return;
    function onKeyDown(event: KeyboardEvent) {
      const typing = isTypingTarget(event.target);
      for (const shortcut of ref.current) {
        if (event.key.toLowerCase() !== shortcut.key.toLowerCase()) continue;
        const metaPressed = event.metaKey || event.ctrlKey;
        if (Boolean(shortcut.meta) !== metaPressed) continue;
        if (shortcut.shift !== undefined && shortcut.shift !== event.shiftKey) continue;
        if (typing && !shortcut.meta && !shortcut.allowInInput) continue;
        if (event.altKey) continue;
        event.preventDefault();
        shortcut.handler(event);
        return;
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled]);
}
