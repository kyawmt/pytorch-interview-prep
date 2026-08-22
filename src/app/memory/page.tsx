"use client";

import { useMemo, useState } from "react";
import { PageHeader } from "@/components/common/PageHeader";
import { ExerciseCard } from "@/components/exercises/ExerciseCard";
import { EXERCISES } from "@/data/exercises";
import { MEMORY_DRILLS } from "@/data/memory-drills";
import { useAppState } from "@/hooks/useAppState";
import { cn } from "@/lib/utils";

export default function MemoryPage() {
  const state = useAppState();
  const [index, setIndex] = useState(0);

  // The dedicated drills plus any memory-type exercises from the main bank.
  const drills = useMemo(() => {
    const extra = EXERCISES.filter(
      (exercise) => exercise.type === "memory" && !MEMORY_DRILLS.some((d) => d.id === exercise.id),
    );
    return [...MEMORY_DRILLS, ...extra];
  }, []);

  const drill = drills[index];

  return (
    <div>
      <PageHeader
        eyebrow="Write from memory"
        title="Reproduce the patterns without looking"
        description="These are graded on STRUCTURE, not on exact text — `outputs = network(inputs)` counts the same as `pred = model(X)`. Blank editor, no starter code: this is what a live-coding round actually feels like."
      />

      <div className="mb-5 flex flex-wrap gap-1.5">
        {drills.map((item, itemIndex) => {
          const solved = (state.exercises[item.id]?.correctAttempts ?? 0) > 0;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setIndex(itemIndex)}
              aria-pressed={index === itemIndex}
              className={cn(
                "rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors",
                index === itemIndex
                  ? "border-accent bg-accent-soft text-accent"
                  : solved
                    ? "border-success/40 bg-success-soft text-success"
                    : "border-border-base bg-surface text-muted hover:border-border-strong hover:text-text",
              )}
            >
              {solved ? "✓ " : ""}
              {item.title}
            </button>
          );
        })}
      </div>

      <ExerciseCard
        key={drill.id}
        exercise={drill}
        index={index}
        total={drills.length}
        onNext={index < drills.length - 1 ? () => setIndex(index + 1) : undefined}
        onPrevious={index > 0 ? () => setIndex(index - 1) : undefined}
      />
    </div>
  );
}
