"use client";

import { CodeBlock } from "@/components/common/CodeBlock";
import { InlineParagraphs, InlineText } from "@/components/common/InlineCode";
import type { ValidationResult } from "@/lib/validation";
import type { Exercise, Rating } from "@/lib/types";
import { cn } from "@/lib/utils";

const RATINGS: { value: Rating; label: string; hint: string }[] = [
  { value: "again", label: "Again", hint: "Show this again today" },
  { value: "hard", label: "Hard", hint: "~1 day" },
  { value: "good", label: "Good", hint: "~2-4 days" },
  { value: "easy", label: "Easy", hint: "~1 week+" },
];

export function ExerciseFeedback({
  result,
  exercise,
  revealed,
  onRate,
  rated,
}: {
  result: ValidationResult | null;
  exercise: Exercise;
  revealed: boolean;
  onRate?: (rating: Rating) => void;
  rated?: Rating | null;
}) {
  if (!result && !revealed) return null;
  const correct = Boolean(result?.correct);

  return (
    <div
      className={cn(
        "mt-4 overflow-hidden rounded-lg border",
        correct
          ? "border-success/35 bg-success-soft animate-pop"
          : revealed && !result
            ? "border-border-base bg-surface-2"
            : "border-danger/35 bg-danger-soft animate-shake",
      )}
      role="status"
      aria-live="polite"
    >
      <div className="px-4 py-3">
        {result ? (
          <p
            className={cn(
              "flex items-center gap-2 text-sm font-semibold",
              correct ? "text-success" : "text-danger",
            )}
          >
            <span aria-hidden>{correct ? "✓" : "✗"}</span>
            {correct ? "Correct" : result.feedback}
          </p>
        ) : (
          <p className="text-sm font-semibold text-text">Solution</p>
        )}

        {result?.nearMiss ? <p className="mt-1.5 text-sm text-muted">{result.nearMiss}</p> : null}

        {result?.missing?.length ? (
          <div className="mt-2">
            <p className="text-xs text-muted">Your answer is missing:</p>
            <ul className="mt-1 space-y-0.5">
              {result.missing.map((item) => (
                <li key={item} className="font-mono text-xs text-danger">
                  · {item}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {result?.warnings?.length
          ? result.warnings.map((warning) => (
              <p key={warning} className="mt-2 rounded-md bg-warning-soft px-2.5 py-1.5 text-xs text-warning">
                ⚠ {warning}
              </p>
            ))
          : null}

        {(correct || revealed) && (
          <div className="mt-3 space-y-3">
            {revealed || correct ? (
              <div>
                <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-faint">
                  Reference solution
                </p>
                <CodeBlock code={exercise.solution} />
              </div>
            ) : null}

            <InlineParagraphs className="text-sm leading-relaxed text-text">
              {exercise.explanation}
            </InlineParagraphs>

            {exercise.interviewNote ? (
              <p className="rounded-md border border-accent/25 bg-accent-soft px-3 py-2 text-xs leading-relaxed text-text">
                <span className="font-semibold text-accent">Interview note · </span>
                <InlineText>{exercise.interviewNote}</InlineText>
              </p>
            ) : null}
          </div>
        )}

        {!correct && !revealed && result && exercise.hint ? (
          <p className="mt-2 text-sm text-muted">
            <span className="font-medium text-text">Hint: </span>
            {exercise.hint}
          </p>
        ) : null}
      </div>

      {onRate && (correct || revealed) ? (
        <div className="flex flex-wrap items-center gap-1.5 border-t border-border-base bg-surface/60 px-4 py-2">
          <span className="mr-1 text-[11px] text-muted">When should this come back?</span>
          {RATINGS.map((rating) => (
            <button
              key={rating.value}
              type="button"
              title={rating.hint}
              onClick={() => onRate(rating.value)}
              className={cn(
                "rounded-md border px-2 py-1 text-[11px] font-medium transition-colors",
                rated === rating.value
                  ? "border-accent bg-accent-soft text-accent"
                  : "border-border-base bg-surface text-muted hover:border-border-strong hover:text-text",
              )}
            >
              {rating.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
