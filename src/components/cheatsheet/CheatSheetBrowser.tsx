"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/common/Button";
import { groupBySection } from "@/data/cheatsheet";
import { useAppState } from "@/hooks/useAppState";
import { setPreference } from "@/lib/storage";
import type { CheatSheetEntry } from "@/lib/types";
import { cn } from "@/lib/utils";
import { CheatSheetCard } from "./CheatSheetCard";

/**
 * Filter + render a set of cheat-sheet entries. Used by the topic pages and by
 * the "all entries" view on the index.
 */
export function CheatSheetBrowser({ entries }: { entries: CheatSheetEntry[] }) {
  const state = useAppState();
  const [query, setQuery] = useState("");
  const [highOnly, setHighOnly] = useState(false);
  const [bookmarkedOnly, setBookmarkedOnly] = useState(false);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const recallMode = state.prefs.recallMode;

  const toggleSection = (section: string) =>
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(section)) next.delete(section);
      else next.add(section);
      return next;
    });

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return entries.filter((entry) => {
      if (highOnly && entry.importance !== "high") return false;
      if (bookmarkedOnly && !state.bookmarks.includes(`cheat:${entry.id}`)) return false;
      if (!needle) return true;
      return [entry.title, entry.description, entry.syntax, entry.example, entry.section, ...(entry.tags ?? [])]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });
  }, [entries, query, highOnly, bookmarkedOnly, state.bookmarks]);

  const sections = groupBySection(filtered);
  const allCollapsed = sections.length > 0 && sections.every((s) => collapsed.has(s.section));

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <div className="relative min-w-[200px] flex-1">
          <label htmlFor="cheatsheet-filter" className="sr-only">
            Filter entries
          </label>
          <input
            id="cheatsheet-filter"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter this page… (e.g. reshape, no_grad)"
            className="w-full rounded-lg border border-border-base bg-surface px-3 py-2 text-sm text-text outline-none transition-colors focus:border-accent placeholder:text-faint"
          />
        </div>

        <button
          type="button"
          onClick={() => setHighOnly((value) => !value)}
          aria-pressed={highOnly}
          className={cn(
            "rounded-lg border px-2.5 py-2 text-xs font-medium transition-colors",
            highOnly
              ? "border-accent bg-accent-soft text-accent"
              : "border-border-base bg-surface text-muted hover:text-text",
          )}
        >
          🔥 High priority
        </button>

        <button
          type="button"
          onClick={() => setBookmarkedOnly((value) => !value)}
          aria-pressed={bookmarkedOnly}
          className={cn(
            "rounded-lg border px-2.5 py-2 text-xs font-medium transition-colors",
            bookmarkedOnly
              ? "border-accent bg-accent-soft text-accent"
              : "border-border-base bg-surface text-muted hover:text-text",
          )}
        >
          ★ Bookmarked
        </button>

        <Button
          size="sm"
          onClick={() =>
            setCollapsed(allCollapsed ? new Set() : new Set(sections.map((s) => s.section)))
          }
          className="py-2"
        >
          {allCollapsed ? "Expand all" : "Collapse all"}
        </Button>

        <Button
          size="sm"
          onClick={() => setPreference("recallMode", !recallMode)}
          aria-pressed={recallMode}
          className={cn("py-2", recallMode && "border-accent text-accent")}
          title="Hide every example so you have to recall it first"
        >
          {recallMode ? "Recall mode: on" : "Recall mode: off"}
        </Button>
      </div>

      {filtered.length === 0 ? (
        <p className="card px-6 py-10 text-center text-sm text-muted">
          Nothing matches those filters.
        </p>
      ) : (
        sections.map((section) => {
          const isCollapsed = collapsed.has(section.section);
          return (
            <section key={section.section} className="mb-7">
              <h2>
                <button
                  type="button"
                  onClick={() => toggleSection(section.section)}
                  aria-expanded={!isCollapsed}
                  className="mb-2.5 flex w-full items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-faint transition-colors hover:text-text"
                >
                  <span
                    aria-hidden
                    className={cn("text-[9px] transition-transform", !isCollapsed && "rotate-90")}
                  >
                    ▶
                  </span>
                  {section.section}
                  <span className="font-normal normal-case tracking-normal">
                    {section.items.length}
                  </span>
                </button>
              </h2>
              {isCollapsed ? null : (
                <div className="space-y-2.5">
                  {section.items.map((entry) => (
                    <CheatSheetCard key={entry.id} entry={entry} recallMode={recallMode} />
                  ))}
                </div>
              )}
            </section>
          );
        })
      )}
    </div>
  );
}
