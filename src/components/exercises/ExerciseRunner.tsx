"use client";

import { useCallback, useState } from "react";
import { Button, LinkButton } from "@/components/common/Button";
import { ProgressBar } from "@/components/progress/ProgressBar";
import type { Exercise } from "@/lib/types";
import { ExerciseCard } from "./ExerciseCard";

interface ExerciseRunnerProps {
  exercises: Exercise[];
  /** Jump straight to this exercise id on mount (used by search deep links). */
  initialId?: string;
  emptyMessage?: string;
  emptyAction?: { href: string; label: string };
  onComplete?: () => void;
}

export function ExerciseRunner({
  exercises,
  initialId,
  emptyMessage = "No exercises match these filters.",
  emptyAction,
  onComplete,
}: ExerciseRunnerProps) {
  const [rawIndex, setIndex] = useState(() => {
    if (!initialId) return 0;
    const position = exercises.findIndex((exercise) => exercise.id === initialId);
    return position >= 0 ? position : 0;
  });
  const [seen, setSeen] = useState<Set<string>>(new Set());
  const [finished, setFinished] = useState(false);

  // Filters can shrink the list underneath us — clamp during render rather than
  // syncing with an effect.
  const index = Math.min(rawIndex, Math.max(exercises.length - 1, 0));
  const current = exercises[index];

  const next = useCallback(() => {
    setSeen((previous) => {
      if (!current) return previous;
      const updated = new Set(previous);
      updated.add(current.id);
      return updated;
    });
    if (index + 1 >= exercises.length) {
      setFinished(true);
      onComplete?.();
    } else {
      setIndex(index + 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, [current, index, exercises.length, onComplete]);

  if (exercises.length === 0) {
    return (
      <div className="card px-6 py-12 text-center">
        <p className="text-sm text-muted">{emptyMessage}</p>
        {emptyAction ? (
          <LinkButton href={emptyAction.href} variant="primary" className="mt-4">
            {emptyAction.label}
          </LinkButton>
        ) : null}
      </div>
    );
  }

  if (finished) {
    return (
      <div className="card px-6 py-12 text-center">
        <p className="text-3xl" aria-hidden>
          ✓
        </p>
        <h2 className="mt-2 text-lg font-semibold text-text">Session complete</h2>
        <p className="mt-1 text-sm text-muted">
          You worked through {exercises.length} {exercises.length === 1 ? "exercise" : "exercises"}.
        </p>
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          <Button
            variant="primary"
            onClick={() => {
              setFinished(false);
              setIndex(0);
              setSeen(new Set());
            }}
          >
            Practise again
          </Button>
          <LinkButton href="/review">Review weak questions</LinkButton>
          <LinkButton href="/practice">All topics</LinkButton>
        </div>
      </div>
    );
  }

  return (
    <div>
      <ProgressBar
        value={((index + (seen.has(current.id) ? 1 : 0)) / exercises.length) * 100}
        label={`Exercise ${index + 1} of ${exercises.length}`}
        showValue={false}
        size="sm"
        className="mb-4"
      />
      <ExerciseCard
        key={current.id}
        exercise={current}
        index={index}
        total={exercises.length}
        onNext={next}
        onPrevious={index > 0 ? () => setIndex(index - 1) : undefined}
      />
    </div>
  );
}
