"use client";

import { useState } from "react";
import { useCopy } from "@/hooks/useCopy";
import { TOKEN_CLASS, tokenizePython } from "@/lib/highlight";
import { cn } from "@/lib/utils";

interface CodeBlockProps {
  code: string;
  /** Show a gutter with line numbers. */
  lineNumbers?: boolean;
  copyable?: boolean;
  /** Render behind a "Show code" curtain — used by the cheat sheet's recall mode. */
  hidden?: boolean;
  /** Offer a Hide button even when the block starts visible. */
  allowHide?: boolean;
  label?: string;
  className?: string;
  /** Highlight decoration only; e.g. "danger" for the wrong-code panel. */
  tone?: "default" | "danger" | "success";
}

/** Renders Python with the in-house tokenizer. No client-side syntax library. */
export function HighlightedCode({ code }: { code: string }) {
  const tokens = tokenizePython(code);
  return (
    <>
      {tokens.map((token, index) => (
        <span key={index} className={TOKEN_CLASS[token.kind]}>
          {token.value}
        </span>
      ))}
    </>
  );
}

export function CodeBlock({
  code,
  lineNumbers = false,
  copyable = true,
  hidden = false,
  allowHide = false,
  label,
  className,
  tone = "default",
}: CodeBlockProps) {
  const { copied, copy } = useCopy();
  const [revealed, setRevealed] = useState(!hidden);
  const lines = code.split("\n");

  const toneClass =
    tone === "danger"
      ? "border-danger/35"
      : tone === "success"
        ? "border-success/35"
        : "border-[var(--code-border)]";

  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-lg border bg-[var(--code-bg)]",
        toneClass,
        className,
      )}
    >
      {label ? (
        <div className="flex items-center justify-between border-b border-[var(--code-border)] px-3 py-1.5">
          <span className="font-mono text-[11px] uppercase tracking-wide text-faint">{label}</span>
        </div>
      ) : null}

      {revealed ? (
        <div className="relative overflow-x-auto">
          <pre className="editor-shared">
            <code className="font-mono">
              {lineNumbers
                ? lines.map((line, index) => (
                    <span key={index} className="block">
                      <span className="mr-4 inline-block w-6 select-none text-right text-[var(--tok-comment)]">
                        {index + 1}
                      </span>
                      <HighlightedCode code={line} />
                    </span>
                  ))
                : <HighlightedCode code={code} />}
            </code>
          </pre>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setRevealed(true)}
          className="flex w-full items-center justify-center gap-2 px-4 py-6 text-xs text-muted transition-colors hover:text-accent"
        >
          <span aria-hidden>👁</span> Show code
          <span className="text-faint">({lines.length} {lines.length === 1 ? "line" : "lines"})</span>
        </button>
      )}

      <div className="absolute right-1.5 top-1.5 flex gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
        {(hidden || allowHide) && revealed ? (
          <button
            type="button"
            onClick={() => setRevealed(false)}
            className="rounded-md border border-border-base bg-surface px-1.5 py-0.5 text-[10px] text-muted hover:text-text"
            aria-label="Hide code"
          >
            Hide
          </button>
        ) : null}
        {copyable && revealed ? (
          <button
            type="button"
            onClick={() => copy(code)}
            className="rounded-md border border-border-base bg-surface px-1.5 py-0.5 text-[10px] text-muted hover:text-text"
            aria-label="Copy code"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        ) : null}
      </div>
    </div>
  );
}
