"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { Button } from "@/components/common/Button";
import { PageHeader } from "@/components/common/PageHeader";
import { ProgressBar } from "@/components/progress/ProgressBar";
import { StatsCard } from "@/components/progress/StatsCard";
import { TopicProgress } from "@/components/progress/TopicProgress";
import { PROJECTS } from "@/data/projects";
import { useAppState, useHydrated } from "@/hooks/useAppState";
import { allTopicStats, learningPath, levelStats, overallStats } from "@/lib/progress";
import { exportState, importState, resetAll } from "@/lib/storage";
import { cn, formatRelativeDate } from "@/lib/utils";

export default function ProgressPage() {
  const state = useAppState();
  const hydrated = useHydrated();
  const stats = overallStats(state);
  const topics = allTopicStats(state);
  const levels = levelStats(state);
  const path = learningPath(state);
  const fileRef = useRef<HTMLInputElement>(null);
  const [message, setMessage] = useState<string | null>(null);

  function download() {
    const blob = new Blob([exportState()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `pytorch-prep-progress-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function upload(file: File) {
    const text = await file.text();
    setMessage(importState(text) ? "Progress imported." : "That file could not be read.");
  }

  return (
    <div>
      <PageHeader
        eyebrow="Progress"
        title="Where you stand"
        description="Mastery blends coverage with accuracy and discounts questions where you needed a hint or revealed the answer — so it reflects what you could reproduce under interview pressure."
        actions={
          <>
            <Button size="sm" onClick={download}>
              Export
            </Button>
            <Button size="sm" onClick={() => fileRef.current?.click()}>
              Import
            </Button>
            <input
              ref={fileRef}
              type="file"
              accept="application/json"
              className="hidden"
              aria-label="Import progress file"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void upload(file);
                event.target.value = "";
              }}
            />
          </>
        }
      />

      {message ? (
        <p className="mb-4 rounded-lg border border-info/25 bg-info-soft px-3 py-2 text-sm text-text">
          {message}
        </p>
      ) : null}

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatsCard label="Solved" value={hydrated ? stats.solved : "—"} hint={`of ${stats.totalExercises}`} />
        <StatsCard label="Attempted" value={hydrated ? stats.attempted : "—"} hint="unique exercises" />
        <StatsCard label="Accuracy" value={hydrated ? `${stats.accuracy}%` : "—"} hint="all attempts" />
        <StatsCard label="Streak" value={hydrated ? `${stats.streak}d` : "—"} hint={`best ${stats.longestStreak}d`} tone="accent" />
        <StatsCard label="Interview Qs" value={hydrated ? stats.interviewKnown : "—"} hint={`of ${stats.interviewTotal} known`} />
        <StatsCard label="Due now" value={hydrated ? stats.dueCount : "—"} hint="spaced repetition" href="/review" />
      </section>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <section className="card p-4" aria-labelledby="mastery-heading">
          <h2 id="mastery-heading" className="mb-3 text-sm font-semibold text-text">
            Topic mastery
          </h2>
          <div className="-mx-2">
            {topics.map((topic) => (
              <TopicProgress key={topic.topic} stats={topic} />
            ))}
          </div>
        </section>

        <div className="space-y-5">
          <section className="card p-4" aria-labelledby="levels-heading">
            <h2 id="levels-heading" className="mb-3 text-sm font-semibold text-text">
              Levels
            </h2>
            <div className="space-y-3">
              {levels.map((level) => (
                <div key={level.level}>
                  <div className="flex items-baseline justify-between text-xs">
                    <span className="text-text">
                      <span className="font-mono text-accent">L{level.level}</span> {level.title}
                    </span>
                    <span className="font-mono text-faint">
                      {level.solved}/{level.total}
                    </span>
                  </div>
                  <ProgressBar value={level.completion} showValue={false} size="sm" className="mt-1" />
                </div>
              ))}
            </div>
          </section>

          <section className="card p-4" aria-labelledby="path-heading">
            <h2 id="path-heading" className="mb-3 text-sm font-semibold text-text">
              Recommended learning path
            </h2>
            <ol className="space-y-1">
              {path.map((step) => (
                <li key={step.topic}>
                  <Link
                    href={`/practice/${step.topic}`}
                    className="flex items-center gap-2.5 rounded-md px-1.5 py-1.5 transition-colors hover:bg-surface-hover"
                  >
                    <span
                      aria-hidden
                      className={cn(
                        "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold",
                        step.status === "done"
                          ? "bg-success text-white"
                          : step.status === "active"
                            ? "bg-accent text-accent-text"
                            : "bg-surface-2 text-faint ring-1 ring-border-base",
                      )}
                    >
                      {step.status === "done" ? "✓" : step.step}
                    </span>
                    <span className="flex-1 truncate text-sm text-text">{step.title}</span>
                    <span className="font-mono text-[11px] text-faint">{step.completion}%</span>
                  </Link>
                </li>
              ))}
            </ol>
          </section>
        </div>
      </div>

      <section className="mt-6 card p-4" aria-labelledby="projects-heading">
        <h2 id="projects-heading" className="mb-3 text-sm font-semibold text-text">
          Mini projects
        </h2>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {PROJECTS.map((project) => {
            const progress = state.projects[project.slug];
            const completed = project.steps.filter((step) => progress?.steps[step.id]).length;
            return (
              <Link
                key={project.slug}
                href={`/projects/${project.slug}`}
                className="rounded-lg border border-border-base bg-surface-2 p-2.5 transition-colors hover:border-border-strong"
              >
                <p className="truncate text-xs font-medium text-text">{project.title}</p>
                <ProgressBar
                  value={(completed / project.steps.length) * 100}
                  showValue={false}
                  size="sm"
                  className="mt-1.5"
                />
                <p className="mt-1 text-[11px] text-faint">
                  {completed}/{project.steps.length} steps
                  {progress?.updatedAt ? ` · ${formatRelativeDate(progress.updatedAt)}` : ""}
                </p>
              </Link>
            );
          })}
        </div>
      </section>

      {state.challenges.length > 0 ? (
        <section className="mt-6 card p-4" aria-labelledby="challenges-heading">
          <h2 id="challenges-heading" className="mb-3 text-sm font-semibold text-text">
            Challenge history
          </h2>
          <ul className="space-y-1.5">
            {state.challenges.slice(0, 6).map((challenge) => (
              <li
                key={challenge.id}
                className="flex flex-wrap items-center gap-2 rounded-lg border border-border-base bg-surface-2 px-3 py-2 text-xs"
              >
                <span className="font-mono text-sm text-text">
                  {challenge.score}/{challenge.total}
                </span>
                <Badge tone={challenge.score / challenge.total >= 0.8 ? "success" : "warning"}>
                  {Math.round((challenge.score / challenge.total) * 100)}%
                </Badge>
                <span className="text-faint">{formatRelativeDate(challenge.date)}</span>
                <span className="ml-auto text-faint">
                  {Math.floor(challenge.durationSeconds / 60)}m {challenge.durationSeconds % 60}s
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="mt-6 card border-danger/25 p-4">
        <h2 className="text-sm font-semibold text-text">Reset</h2>
        <p className="mt-1 text-xs text-muted">
          Progress lives in this browser&rsquo;s localStorage. Export first if you want to keep it.
        </p>
        <Button
          variant="danger"
          size="sm"
          className="mt-3"
          onClick={() => {
            if (window.confirm("Delete all local progress? This cannot be undone.")) {
              resetAll();
              setMessage("All progress cleared.");
            }
          }}
        >
          Clear all progress
        </Button>
      </section>
    </div>
  );
}
