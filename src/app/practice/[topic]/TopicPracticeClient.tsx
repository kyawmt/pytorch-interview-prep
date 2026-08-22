"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
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
import { cheatSheetByTopic } from "@/data/cheatsheet";
import { exercisesByTopic } from "@/data/exercises";
import { TOPIC_MAP } from "@/data/topics";
import { useAppState } from "@/hooks/useAppState";
import { topicStats } from "@/lib/progress";
import type { TopicId } from "@/lib/types";

function TopicPracticeInner({ topic }: { topic: TopicId }) {
  const state = useAppState();
  const searchParams = useSearchParams();
  const initialId = searchParams.get("exercise") ?? undefined;
  const [filters, setFilters] = useState<FilterState>({ ...DEFAULT_FILTERS, topic });

  const all = useMemo(() => exercisesByTopic(topic), [topic]);
  const filtered = useMemo(
    () => applyFilters(all, { ...filters, topic }, state),
    [all, filters, topic, state],
  );

  const meta = TOPIC_MAP[topic];
  const stats = topicStats(state, topic);
  const hasCheatSheet = cheatSheetByTopic(topic).length > 0;

  return (
    <div>
      <nav aria-label="Breadcrumb" className="mb-3 text-xs text-muted">
        <Link href="/practice" className="hover:text-accent">
          Practice
        </Link>
        <span className="mx-1.5 text-faint">/</span>
        <span className="text-text">{meta.title}</span>
      </nav>

      <PageHeader
        title={meta.title}
        description={meta.summary}
        actions={
          hasCheatSheet ? <LinkButton href={`/cheatsheet/${topic}`}>Cheat sheet</LinkButton> : null
        }
      />

      <div className="card mb-5 flex flex-wrap items-center gap-4 p-3.5">
        <ProgressBar
          value={stats.completion}
          label={`${stats.correct} of ${stats.total} solved`}
          className="min-w-[180px] flex-1"
        />
        <div className="flex gap-5 text-xs">
          <span>
            <span className="block text-faint">Accuracy</span>
            <span className="font-mono text-sm text-text">{stats.accuracy}%</span>
          </span>
          <span>
            <span className="block text-faint">Mastery</span>
            <span className="font-mono text-sm text-accent">{stats.mastery}%</span>
          </span>
          <span>
            <span className="block text-faint">To review</span>
            <span className="font-mono text-sm text-text">{stats.needsReview}</span>
          </span>
        </div>
      </div>

      <ExerciseFilters
        value={{ ...filters, topic }}
        onChange={(next) => setFilters({ ...next, topic })}
        baseline={{ ...DEFAULT_FILTERS, topic }}
        hideTopic
        resultCount={filtered.length}
      />

      <ExerciseRunner
        exercises={filtered}
        initialId={initialId}
        emptyMessage="No exercises match these filters in this topic."
        emptyAction={{ href: "/practice", label: "Browse all exercises" }}
      />
    </div>
  );
}

export function TopicPracticeClient({ topic }: { topic: TopicId }) {
  return (
    <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
      <TopicPracticeInner topic={topic} />
    </Suspense>
  );
}
