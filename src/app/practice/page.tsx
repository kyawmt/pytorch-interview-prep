"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { LinkButton } from "@/components/common/Button";
import { PageHeader } from "@/components/common/PageHeader";
import {
  applyFilters,
  DEFAULT_FILTERS,
  ExerciseFilters,
  type FilterState,
} from "@/components/exercises/ExerciseFilters";
import { ExerciseRunner } from "@/components/exercises/ExerciseRunner";
import { ProgressBar } from "@/components/progress/ProgressBar";
import { EXERCISES } from "@/data/exercises";
import { TOPIC_MAP } from "@/data/topics";
import { useAppState } from "@/hooks/useAppState";
import { levelStats } from "@/lib/progress";

export default function PracticePage() {
  const state = useAppState();
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [mode, setMode] = useState<"browse" | "run">("browse");

  const filtered = useMemo(() => applyFilters(EXERCISES, filters, state), [filters, state]);
  const levels = levelStats(state);

  return (
    <div>
      <PageHeader
        eyebrow="Practice"
        title="Exercise browser"
        description={`${EXERCISES.length} exercises across eight levels. Filter down to what you need, then run through them one at a time — check with ⌘↵, hint with H, next with N.`}
        actions={
          <>
            <LinkButton href="/daily" variant="primary">
              Daily session
            </LinkButton>
            <LinkButton href="/memory">Write from memory</LinkButton>
          </>
        }
      />

      <ExerciseFilters value={filters} onChange={setFilters} resultCount={filtered.length} />

      <div className="mb-6 flex items-center gap-2">
        <button
          type="button"
          onClick={() => setMode("browse")}
          aria-pressed={mode === "browse"}
          className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
            mode === "browse"
              ? "border-accent bg-accent-soft text-accent"
              : "border-border-base bg-surface text-muted hover:text-text"
          }`}
        >
          Browse by level
        </button>
        <button
          type="button"
          onClick={() => setMode("run")}
          aria-pressed={mode === "run"}
          className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
            mode === "run"
              ? "border-accent bg-accent-soft text-accent"
              : "border-border-base bg-surface text-muted hover:text-text"
          }`}
        >
          Run these {filtered.length} exercises
        </button>
      </div>

      {mode === "run" ? (
        <ExerciseRunner
          exercises={filtered}
          emptyMessage="No exercises match these filters. Try widening them."
        />
      ) : (
        <div className="space-y-4">
          {levels.map((level) => {
            const inLevel = filtered.filter((exercise) => exercise.level === level.level);
            return (
              <section key={level.level} className="card p-4">
                <div className="flex flex-wrap items-baseline gap-2">
                  <span className="font-mono text-xs text-accent">Level {level.level}</span>
                  <h2 className="text-sm font-semibold text-text">{level.title}</h2>
                  <span className="ml-auto font-mono text-xs text-faint">
                    {level.solved}/{level.total}
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted">{level.blurb}</p>
                <ProgressBar value={level.completion} showValue={false} size="sm" className="mt-2.5" />

                <div className="mt-3 flex flex-wrap gap-1.5">
                  {level.topics.map((topicId) => {
                    const topic = TOPIC_MAP[topicId];
                    return (
                      <Link
                        key={topicId}
                        href={`/practice/${topicId}`}
                        className="rounded-md border border-border-base bg-surface-2 px-2 py-1 text-xs text-muted transition-colors hover:border-accent hover:text-accent"
                      >
                        {topic?.icon} {topic?.title}
                      </Link>
                    );
                  })}
                  {inLevel.length > 0 && inLevel.length !== level.total ? (
                    <Badge tone="info">{inLevel.length} match the filters</Badge>
                  ) : null}
                </div>
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}
