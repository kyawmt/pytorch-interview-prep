"use client";

import { useEffect, useRef } from "react";
import { HighlightedCode } from "@/components/common/CodeBlock";
import { cn } from "@/lib/utils";

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  /** Cmd/Ctrl + Enter. */
  onSubmit?: () => void;
  placeholder?: string;
  minRows?: number;
  disabled?: boolean;
  autoFocus?: boolean;
  ariaLabel?: string;
  tone?: "default" | "correct" | "incorrect";
}

/**
 * A lightweight syntax-highlighted editor: a transparent <textarea> layered on
 * a highlighted <pre>. Monaco would add ~1MB for inputs that are rarely more
 * than a handful of lines, and would need its own SSR handling.
 *
 * The two layers must share identical typography — see `.editor-shared`.
 */
export function CodeEditor({
  value,
  onChange,
  onSubmit,
  placeholder = "Type your PyTorch code…",
  minRows = 3,
  disabled = false,
  autoFocus = false,
  ariaLabel = "Your answer",
  tone = "default",
}: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const preRef = useRef<HTMLPreElement>(null);

  const lineCount = value.split("\n").length;
  const rows = Math.max(minRows, Math.min(lineCount + 1, 24));

  useEffect(() => {
    if (autoFocus) textareaRef.current?.focus();
  }, [autoFocus]);

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      onSubmit?.();
      return;
    }
    // Tab inserts four spaces instead of moving focus — but Shift+Tab still
    // escapes the editor so keyboard navigation is not trapped.
    if (event.key === "Tab" && !event.shiftKey) {
      event.preventDefault();
      const target = event.currentTarget;
      const { selectionStart, selectionEnd } = target;
      const next = `${value.slice(0, selectionStart)}    ${value.slice(selectionEnd)}`;
      onChange(next);
      requestAnimationFrame(() => {
        target.selectionStart = target.selectionEnd = selectionStart + 4;
      });
    }
  }

  function syncScroll() {
    if (preRef.current && textareaRef.current) {
      preRef.current.scrollTop = textareaRef.current.scrollTop;
      preRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  }

  const toneClass =
    tone === "correct"
      ? "border-success/50"
      : tone === "incorrect"
        ? "border-danger/50"
        : "border-[var(--code-border)] focus-within:border-accent";

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border bg-[var(--code-bg)] transition-colors",
        toneClass,
        disabled && "opacity-70",
      )}
    >
      <pre
        ref={preRef}
        aria-hidden
        className="editor-shared pointer-events-none absolute inset-0 overflow-hidden"
      >
        <code className="font-mono">
          <HighlightedCode code={value} />
          {"\n"}
        </code>
      </pre>
      <textarea
        ref={textareaRef}
        value={value}
        rows={rows}
        disabled={disabled}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        autoComplete="off"
        aria-label={ariaLabel}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        onScroll={syncScroll}
        className="editor-shared editor-input relative block w-full overflow-auto outline-none"
      />
    </div>
  );
}
