"use client";

import Link from "next/link";
import { useState } from "react";
import { Badge } from "@/components/common/Badge";
import { CodeBlock } from "@/components/common/CodeBlock";
import { InlineText } from "@/components/common/InlineCode";
import { useAppState } from "@/hooks/useAppState";
import { useCopy } from "@/hooks/useCopy";
import { toggleBookmark } from "@/lib/storage";
import type { CheatSheetEntry } from "@/lib/types";
import { cn } from "@/lib/utils";

export function CheatSheetCard({
  entry,
  recallMode,
}: {
  entry: CheatSheetEntry;
  /** Start with the example hidden so the learner recalls it first. */
  recallMode: boolean;
}) {
  const state = useAppState();
  const { copied, copy } = useCopy();
  const [open, setOpen] = useState(true);
  const bookmarkKey = `cheat:${entry.id}`;
  const bookmarked = state.bookmarks.includes(bookmarkKey);
  const practiceHref = entry.relatedExercises?.length
    ? `/practice/${entry.topic}?exercise=${entry.relatedExercises[0]}`
    : `/practice/${entry.topic}`;

  return (
    <article id={entry.id} className="card scroll-mt-20 overflow-hidden">
      <header className="flex flex-wrap items-center gap-2 border-b border-border-base px-4 py-2.5">
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          <span aria-hidden className={cn("text-[10px] text-faint transition-transform", open && "rotate-90")}>
            ▶
          </span>
          <h3 className="truncate font-mono text-sm font-semibold text-text">{entry.title}</h3>
        </button>
        <div className="flex shrink-0 items-center gap-1.5">
          {entry.importance === "high" ? <Badge tone="accent">🔥</Badge> : null}
          <button
            type="button"
            onClick={() => toggleBookmark(bookmarkKey)}
            aria-pressed={bookmarked}
            aria-label={bookmarked ? "Remove bookmark" : "Bookmark"}
            className={cn(
              "rounded px-1 text-sm transition-colors",
              bookmarked ? "text-accent" : "text-faint hover:text-text",
            )}
          >
            {bookmarked ? "★" : "☆"}
          </button>
          <button
            type="button"
            onClick={() => copy(entry.example)}
            className="rounded-md border border-border-base px-1.5 py-0.5 text-[10px] text-muted transition-colors hover:text-text"
          >
            {copied ? "Copied" : "Copy"}
          </button>
          <Link
            href={practiceHref}
            className="rounded-md border border-border-base px-1.5 py-0.5 text-[10px] text-muted transition-colors hover:border-accent hover:text-accent"
          >
            Practise
          </Link>
        </div>
      </header>

      {open ? (
        <div className="px-4 py-3.5">
          <p className="text-sm leading-relaxed text-muted">
            <InlineText>{entry.description}</InlineText>
          </p>

          <p className="mt-3 overflow-x-auto rounded-md border border-border-base bg-surface-2 px-2.5 py-1.5 font-mono text-xs text-text">
            {entry.syntax}
          </p>

          <CodeBlock
            code={entry.example}
            className="mt-3"
            hidden={recallMode}
            allowHide
            lineNumbers={entry.example.split("\n").length > 3}
          />

          {entry.result ? (
            <div className="mt-2">
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-faint">Output</p>
              <pre className="overflow-x-auto rounded-md border border-border-base bg-surface-2 px-2.5 py-1.5 font-mono text-xs text-muted">
                {entry.result}
              </pre>
            </div>
          ) : null}

          {entry.interviewNote ? (
            <p className="mt-3 rounded-lg border border-accent/25 bg-accent-soft px-3 py-2 text-xs leading-relaxed text-text">
              <span className="font-semibold text-accent">Interview note · </span>
              <InlineText>{entry.interviewNote}</InlineText>
            </p>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}
