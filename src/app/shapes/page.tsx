"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/common/Button";
import { PageHeader } from "@/components/common/PageHeader";
import { ExerciseRunner } from "@/components/exercises/ExerciseRunner";
import { SHAPE_DRILL_EXERCISES } from "@/data/shape-drills";
import type { Difficulty } from "@/lib/types";
import { cn, hashString, seededRandom, shuffle } from "@/lib/utils";

const LEVELS: Array<Difficulty | "all"> = ["all", "easy", "medium", "hard"];

export default function ShapePlaygroundPage() {
  const [difficulty, setDifficulty] = useState<Difficulty | "all">("all");
  const [round, setRound] = useState(0);

  // Seeded rather than Math.random so the server-rendered order matches the
  // first client render; "Reshuffle" just advances the seed.
  const drills = useMemo(() => {
    const pool =
      difficulty === "all"
        ? SHAPE_DRILL_EXERCISES
        : SHAPE_DRILL_EXERCISES.filter((drill) => drill.difficulty === difficulty);
    return shuffle(pool, seededRandom(hashString(difficulty) + round * 7919 + 1));
  }, [difficulty, round]);

  return (
    <div>
      <PageHeader
        eyebrow="Shape playground"
        title="Predict the shape before you run the code"
        description="Shape reasoning is the skill interviewers probe most and the one that breaks real models silently. Work through these until the answer is instant. Answers are flexible: (32, 128), 32x128 and torch.Size([32, 128]) all count."
        actions={
          <Button size="sm" onClick={() => setRound((value) => value + 1)}>
            ↻ Reshuffle
          </Button>
        }
      />

      <div className="mb-5 flex flex-wrap items-center gap-1.5">
        {LEVELS.map((level) => (
          <button
            key={level}
            type="button"
            onClick={() => {
              setDifficulty(level);
              setRound((value) => value + 1);
            }}
            aria-pressed={difficulty === level}
            className={cn(
              "rounded-lg border px-2.5 py-1.5 text-xs font-medium capitalize transition-colors",
              difficulty === level
                ? "border-accent bg-accent-soft text-accent"
                : "border-border-base bg-surface text-muted hover:border-border-strong hover:text-text",
            )}
          >
            {level === "all" ? "All" : level}
          </button>
        ))}
        <span className="ml-auto text-xs text-faint">{drills.length} drills</span>
      </div>

      <ExerciseRunner key={`${difficulty}-${round}`} exercises={drills} />

      <section className="card mt-6 p-4">
        <h2 className="text-sm font-semibold text-text">Rules worth memorising</h2>
        <ul className="mt-2.5 space-y-1.5 text-xs leading-relaxed text-muted">
          <li>
            <span className="font-mono text-text">Conv2d/pool:</span> out = (in + 2·padding − kernel) //
            stride + 1. kernel 3 with padding 1 preserves the size; MaxPool2d(2) halves it.
          </li>
          <li>
            <span className="font-mono text-text">Linear:</span> only the LAST dimension changes; every
            leading dimension is treated as batch.
          </li>
          <li>
            <span className="font-mono text-text">Reductions:</span> the dim you name disappears (unless
            keepdim=True). For softmax, the dim you name is the one that sums to 1.
          </li>
          <li>
            <span className="font-mono text-text">Broadcasting:</span> right-align the shapes; each pair
            must be equal or contain a 1.
          </li>
          <li>
            <span className="font-mono text-text">cat vs stack:</span> cat keeps the rank, stack adds a
            dimension.
          </li>
        </ul>
      </section>
    </div>
  );
}
