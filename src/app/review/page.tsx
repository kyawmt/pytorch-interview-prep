"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { LinkButton } from "@/components/common/Button";
import { PageHeader } from "@/components/common/PageHeader";
import { ExerciseRunner } from "@/components/exercises/ExerciseRunner";
import { EXERCISES } from "@/data/exercises";
import { INTERVIEW_QUESTIONS } from "@/data/interview";
import { InterviewQuestionCard } from "@/components/interview/InterviewQuestionCard";
import { topicTitle } from "@/data/topics";
import { useAppState } from "@/hooks/useAppState";
import { reviewQueue, type ReviewFilter } from "@/lib/progress";
import { cn, formatRelativeDate } from "@/lib/utils";

const FILTERS: { value: ReviewFilter; label: string; hint: string }[] = [
  { value: "all", label: "All weak", hint: "Wrong, hinted or revealed" },
  { value: "wrong", label: "Wrong answers", hint: "Last attempt was incorrect" },
  { value: "hint", label: "Used hint", hint: "You needed a nudge" },
  { value: "revealed", label: "Revealed answer", hint: "You looked at the solution" },
  { value: "flagged", label: "Flagged", hint: "Marked with Again / Need review" },
  { value: "due", label: "Due now", hint: "Spaced repetition says it is time" },
];

export default function ReviewPage() {
  const state = useAppState();
  const [filter, setFilter] = useState<ReviewFilter>("all");
  const [mode, setMode] = useState<"list" | "practise">("list");

  const queue = useMemo(() => reviewQueue(state, filter), [state, filter]);
  const bookmarkedExercises = useMemo(
    () => EXERCISES.filter((exercise) => state.bookmarks.includes(`ex:${exercise.id}`)),
    [state.bookmarks],
  );
  const interviewReview = useMemo(
    () =>
      INTERVIEW_QUESTIONS.filter(
        (question) =>
          state.interview[question.id]?.status === "review" ||
          state.bookmarks.includes(`iq:${question.id}`),
      ),
    [state],
  );

  return (
    <div>
      <PageHeader
        eyebrow="Review"
        title="Everything you got wrong, in one place"
        description="Questions land here automatically when you answer incorrectly, use a hint, reveal the solution, or rate a card Again. This is the highest-value list on the site."
        actions={
          queue.length > 0 ? (
            <button
              type="button"
              onClick={() => setMode(mode === "practise" ? "list" : "practise")}
              className="rounded-lg border border-transparent bg-accent px-3.5 py-2 text-sm font-medium text-accent-text transition-colors hover:bg-accent-hover"
            >
              {mode === "practise" ? "Back to list" : `Practise ${queue.length} weak questions`}
            </button>
          ) : null
        }
      />

      <div className="mb-5 flex flex-wrap gap-1.5">
        {FILTERS.map((item) => {
          const count = reviewQueue(state, item.value).length;
          return (
            <button
              key={item.value}
              type="button"
              title={item.hint}
              onClick={() => {
                setFilter(item.value);
                setMode("list");
              }}
              aria-pressed={filter === item.value}
              className={cn(
                "rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors",
                filter === item.value
                  ? "border-accent bg-accent-soft text-accent"
                  : "border-border-base bg-surface text-muted hover:border-border-strong hover:text-text",
              )}
            >
              {item.label}
              <span className="ml-1.5 font-mono text-[10px] opacity-70">{count}</span>
            </button>
          );
        })}
      </div>

      {mode === "practise" ? (
        <ExerciseRunner exercises={queue} emptyMessage="Nothing to review right now." />
      ) : queue.length === 0 ? (
        <div className="card px-6 py-12 text-center">
          <p className="text-sm text-muted">
            Nothing in this bucket. Practise some exercises and anything you stumble on will appear
            here automatically.
          </p>
          <LinkButton href="/practice" variant="primary" className="mt-4">
            Go to practice
          </LinkButton>
        </div>
      ) : (
        <ul className="space-y-1.5">
          {queue.map((exercise) => {
            const entry = state.exercises[exercise.id];
            return (
              <li key={exercise.id}>
                <a
                  href={`/practice/${exercise.topic}?exercise=${exercise.id}`}
                  className="card flex flex-wrap items-center gap-2 px-3.5 py-2.5 transition-colors hover:border-border-strong hover:bg-surface-hover"
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm text-text">{exercise.title}</span>
                    <span className="block truncate text-xs text-faint">
                      {topicTitle(exercise.topic)} · {exercise.difficulty}
                      {entry?.lastAttempt ? ` · ${formatRelativeDate(entry.lastAttempt)}` : ""}
                    </span>
                  </span>
                  <span className="flex shrink-0 flex-wrap items-center gap-1.5">
                    {entry && !entry.correct ? <Badge tone="danger">Wrong</Badge> : null}
                    {entry?.revealedAnswer ? <Badge tone="warning">Revealed</Badge> : null}
                    {entry?.usedHint ? <Badge tone="info">Hint</Badge> : null}
                    {entry?.flagged ? <Badge tone="accent">Flagged</Badge> : null}
                    <span className="font-mono text-[11px] text-faint">
                      {entry ? `${entry.correctAttempts}/${entry.attempts}` : ""}
                    </span>
                  </span>
                </a>
              </li>
            );
          })}
        </ul>
      )}

      {bookmarkedExercises.length > 0 && mode === "list" ? (
        <section className="mt-8">
          <h2 className="mb-2.5 text-sm font-semibold text-text">
            Bookmarked exercises
            <span className="ml-2 font-mono text-xs text-faint">{bookmarkedExercises.length}</span>
          </h2>
          <ul className="space-y-1.5">
            {bookmarkedExercises.map((exercise) => (
              <li key={exercise.id}>
                <a
                  href={`/practice/${exercise.topic}?exercise=${exercise.id}`}
                  className="card flex items-center gap-2 px-3.5 py-2.5 text-sm transition-colors hover:border-border-strong hover:bg-surface-hover"
                >
                  <span className="text-accent">★</span>
                  <span className="min-w-0 flex-1 truncate text-text">{exercise.title}</span>
                  <span className="shrink-0 text-xs text-faint">{topicTitle(exercise.topic)}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {interviewReview.length > 0 && mode === "list" ? (
        <section className="mt-8">
          <h2 className="mb-2.5 text-sm font-semibold text-text">
            Interview questions to revisit
            <span className="ml-2 font-mono text-xs text-faint">{interviewReview.length}</span>
          </h2>
          <div className="space-y-2.5">
            {interviewReview.map((question) => (
              <InterviewQuestionCard key={question.id} question={question} />
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
