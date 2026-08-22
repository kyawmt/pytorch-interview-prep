"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { kindLabel, search, type SearchResult } from "@/lib/search";
import { cn } from "@/lib/utils";

const SUGGESTIONS = [
  "zero_grad",
  "CrossEntropyLoss",
  "reshape",
  "no_grad",
  "DataLoader",
  "device",
  "detach",
  "permute",
];

interface SearchDialogProps {
  onClose: () => void;
}

/** Rendered only while open (see AppShell), so mount === opening. */
export function SearchDialog({ onClose }: SearchDialogProps) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const router = useRouter();

  const results = useMemo<SearchResult[]>(() => search(query, 24), [query]);

  useEffect(() => {
    const id = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, []);

  // Keep the highlighted row visible while arrowing through results.
  useEffect(() => {
    const node = listRef.current?.children[cursor] as HTMLElement | undefined;
    node?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  function onKeyDown(event: React.KeyboardEvent) {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      setCursor((c) => Math.min(c + 1, Math.max(results.length - 1, 0)));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setCursor((c) => Math.max(c - 1, 0));
    } else if (event.key === "Enter" && results[cursor]) {
      event.preventDefault();
      router.push(results[cursor].href);
      onClose();
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 px-4 pt-[10vh] backdrop-blur-sm"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Search PyTorch Prep"
        className="w-full max-w-2xl overflow-hidden rounded-xl border border-border-base bg-surface shadow-2xl animate-fade-in-up"
        onKeyDown={onKeyDown}
      >
        <div className="flex items-center gap-3 border-b border-border-base px-4">
          <span aria-hidden className="text-faint">
            ⌕
          </span>
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setCursor(0);
            }}
            placeholder="Search syntax, exercises, interview questions…"
            aria-label="Search query"
            className="w-full bg-transparent py-3.5 text-sm text-text outline-none placeholder:text-faint"
          />
          <kbd className="hidden rounded border border-border-base px-1.5 py-0.5 font-mono text-[10px] text-faint sm:block">
            esc
          </kbd>
        </div>

        {query.trim().length < 2 ? (
          <div className="px-4 py-5">
            <p className="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-faint">
              Try searching for
            </p>
            <div className="flex flex-wrap gap-1.5">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => {
                    setQuery(suggestion);
                    setCursor(0);
                    inputRef.current?.focus();
                  }}
                  className="rounded-md border border-border-base bg-surface-2 px-2 py-1 font-mono text-xs text-muted transition-colors hover:border-accent hover:text-accent"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : results.length === 0 ? (
          <p className="px-4 py-8 text-center text-sm text-muted">
            No matches for <span className="font-mono text-text">{query}</span>
          </p>
        ) : (
          <ul ref={listRef} className="max-h-[55vh] overflow-y-auto py-1.5" role="listbox">
            {results.map((result, index) => (
              <li key={`${result.kind}-${result.id}`} role="option" aria-selected={index === cursor}>
                <Link
                  href={result.href}
                  onClick={onClose}
                  onMouseEnter={() => setCursor(index)}
                  className={cn(
                    "flex items-center gap-3 px-4 py-2",
                    index === cursor ? "bg-accent-soft" : "hover:bg-surface-hover",
                  )}
                >
                  <span className="w-[72px] shrink-0 text-[10px] font-semibold uppercase tracking-wide text-faint">
                    {kindLabel(result.kind)}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm text-text">{result.title}</span>
                    <span className="block truncate text-xs text-faint">{result.subtitle}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}

        <div className="flex items-center gap-4 border-t border-border-base bg-surface-2 px-4 py-2 text-[11px] text-faint">
          <span>
            <kbd className="font-mono">↑↓</kbd> navigate
          </span>
          <span>
            <kbd className="font-mono">↵</kbd> open
          </span>
          <span className="ml-auto">{results.length ? `${results.length} results` : ""}</span>
        </div>
      </div>
    </div>
  );
}
