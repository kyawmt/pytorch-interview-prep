"use client";

import { useState } from "react";
import { PageHeader } from "@/components/common/PageHeader";
import { ExerciseCard } from "@/components/exercises/ExerciseCard";
import { BUILDER_DRILLS } from "@/data/builder-drills";
import { cn } from "@/lib/utils";

export default function TrainingLoopBuilderPage() {
  const [index, setIndex] = useState(0);
  const drill = BUILDER_DRILLS[index];

  return (
    <div>
      <PageHeader
        eyebrow="Training loop builder"
        title="Assemble the loop from its parts"
        description="Click blocks to build the sequence. Some of the offered blocks are deliberately wrong — recognising what does NOT belong is exactly what an interviewer is testing when they hand you a broken loop."
      />

      <div className="mb-5 flex flex-wrap gap-1.5">
        {BUILDER_DRILLS.map((item, itemIndex) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setIndex(itemIndex)}
            aria-pressed={index === itemIndex}
            className={cn(
              "rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors",
              index === itemIndex
                ? "border-accent bg-accent-soft text-accent"
                : "border-border-base bg-surface text-muted hover:border-border-strong hover:text-text",
            )}
          >
            {item.title}
          </button>
        ))}
      </div>

      <ExerciseCard
        key={drill.id}
        exercise={drill}
        index={index}
        total={BUILDER_DRILLS.length}
        onNext={index < BUILDER_DRILLS.length - 1 ? () => setIndex(index + 1) : undefined}
        onPrevious={index > 0 ? () => setIndex(index - 1) : undefined}
        autoFocus={false}
      />
    </div>
  );
}
