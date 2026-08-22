"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { CodeBlock } from "@/components/common/CodeBlock";
import { PageHeader } from "@/components/common/PageHeader";
import { MISTAKES } from "@/data/mistakes";
import { topicTitle } from "@/data/topics";
import { cn } from "@/lib/utils";

export default function MistakesPage() {
  const [query, setQuery] = useState("");
  const [highOnly, setHighOnly] = useState(false);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return MISTAKES.filter((mistake) => {
      if (highOnly && mistake.importance !== "high") return false;
      if (!needle) return true;
      return [mistake.title, mistake.symptom, mistake.explanation, mistake.wrong, mistake.right, ...(mistake.tags ?? [])]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });
  }, [query, highOnly]);

  return (
    <div>
      <PageHeader
        eyebrow="Common mistakes"
        title="The bugs that cost interviews"
        description={`${MISTAKES.length} mistakes that show up constantly in real PyTorch code. Each one starts with the SYMPTOM you would actually observe — because in an interview you are given the symptom, not the diff.`}
      />

      <div className="mb-5 flex flex-wrap items-center gap-2">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter by symptom or error message…"
          aria-label="Filter mistakes"
          className="min-w-[200px] flex-1 rounded-lg border border-border-base bg-surface px-3 py-2 text-sm outline-none transition-colors focus:border-accent placeholder:text-faint"
        />
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
        <span className="ml-auto text-xs text-faint">{filtered.length} mistakes</span>
      </div>

      <div className="space-y-3">
        {filtered.map((mistake) => (
          <article key={mistake.id} id={mistake.id} className="card scroll-mt-20 overflow-hidden">
            <header className="flex flex-wrap items-center gap-2 border-b border-border-base bg-surface-2 px-4 py-2.5">
              <h2 className="text-sm font-semibold text-text">{mistake.title}</h2>
              <span className="ml-auto flex items-center gap-1.5">
                <Badge tone="neutral">{topicTitle(mistake.topic)}</Badge>
                {mistake.importance === "high" ? <Badge tone="accent">🔥</Badge> : null}
              </span>
            </header>

            <div className="px-4 py-3.5">
              <p className="rounded-lg border border-warning/25 bg-warning-soft px-3 py-2 text-xs leading-relaxed text-text">
                <span className="font-semibold text-warning">Symptom · </span>
                {mistake.symptom}
              </p>

              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <div>
                  <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-danger">
                    ✗ What people write
                  </p>
                  <CodeBlock code={mistake.wrong} tone="danger" copyable={false} />
                </div>
                <div>
                  <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-success">
                    ✓ What it should be
                  </p>
                  <CodeBlock code={mistake.right} tone="success" />
                </div>
              </div>

              <p className="mt-3 text-sm leading-relaxed text-muted">{mistake.explanation}</p>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
