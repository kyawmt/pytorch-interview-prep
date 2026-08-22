"use client";

import { useMemo, useState } from "react";
import { Badge, DifficultyBadge, ImportanceBadge, TypeBadge } from "@/components/common/Badge";
import { Button } from "@/components/common/Button";
import { CodeBlock, HighlightedCode } from "@/components/common/CodeBlock";
import { InlineText } from "@/components/common/InlineCode";
import { topicTitle } from "@/data/topics";
import { useAppState } from "@/hooks/useAppState";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import {
  markAnswerRevealed,
  markHintUsed,
  recordAttempt,
  scheduleReview,
  toggleBookmark,
} from "@/lib/storage";
import type { Exercise, Rating } from "@/lib/types";
import { validateAnswer, type ValidationResult } from "@/lib/validation";
import { cn } from "@/lib/utils";
import { CodeEditor } from "./CodeEditor";
import { ExerciseFeedback } from "./ExerciseFeedback";
import { OrderingInput } from "./OrderingInput";

interface ExerciseCardProps {
  exercise: Exercise;
  onNext?: () => void;
  onPrevious?: () => void;
  index?: number;
  total?: number;
  /** Quiz mode: hide the solution/hint affordances and report the result up. */
  examMode?: boolean;
  onAnswered?: (correct: boolean) => void;
  autoFocus?: boolean;
}

export function ExerciseCard({
  exercise,
  onNext,
  onPrevious,
  index,
  total,
  examMode = false,
  onAnswered,
  autoFocus = true,
}: ExerciseCardProps) {
  const state = useAppState();
  const [answer, setAnswer] = useState(exercise.starterCode ?? "");
  const [choice, setChoice] = useState<number | null>(null);
  const [order, setOrder] = useState<number[]>([]);
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [rated, setRated] = useState<Rating | null>(null);
  const [attempts, setAttempts] = useState(0);

  const progress = state.exercises[exercise.id];
  const bookmarkKey = `ex:${exercise.id}`;
  const bookmarked = state.bookmarks.includes(bookmarkKey);

  // NOTE: every caller renders this component with key={exercise.id}, so React
  // remounts it (and resets all of the state above) when the exercise changes.

  const currentAnswer = useMemo(() => {
    if (exercise.type === "multiple-choice") return choice ?? -1;
    if (exercise.type === "ordering") return order.map((i) => exercise.blocks?.[i] ?? "");
    return answer;
  }, [exercise, answer, choice, order]);

  const answered = result?.correct === true || revealed;
  const canCheck =
    exercise.type === "multiple-choice"
      ? choice !== null
      : exercise.type === "ordering"
        ? order.length > 0
        : answer.trim().length > 0;

  function check() {
    if (!canCheck || answered) return;
    const validation = validateAnswer(currentAnswer, exercise);
    setResult(validation);
    setAttempts((n) => n + 1);
    recordAttempt(exercise.id, validation.correct);
    onAnswered?.(validation.correct);
  }

  function reveal() {
    setRevealed(true);
    markAnswerRevealed(exercise.id);
    if (!result) {
      recordAttempt(exercise.id, false);
      onAnswered?.(false);
    }
  }

  function revealHint() {
    setShowHint(true);
    markHintUsed(exercise.id);
  }

  function rate(rating: Rating) {
    setRated(rating);
    scheduleReview(exercise.id, rating);
  }

  useKeyboardShortcuts([
    { key: "Enter", meta: true, allowInInput: true, handler: check },
    { key: "h", handler: () => !examMode && exercise.hint && revealHint() },
    { key: "n", handler: () => onNext?.() },
    { key: "b", handler: () => toggleBookmark(bookmarkKey) },
  ]);

  return (
    <article className="card overflow-hidden">
      <header className="flex flex-wrap items-center gap-2 border-b border-border-base bg-surface-2 px-4 py-2.5">
        {index !== undefined && total !== undefined ? (
          <span className="font-mono text-xs text-faint">
            {index + 1}/{total}
          </span>
        ) : null}
        <span className="text-sm font-medium text-text">{exercise.title}</span>
        <span className="ml-auto flex flex-wrap items-center gap-1.5">
          <Badge tone="neutral">{topicTitle(exercise.topic)}</Badge>
          <TypeBadge type={exercise.type} />
          <DifficultyBadge difficulty={exercise.difficulty} />
          {exercise.importance === "high" ? <ImportanceBadge importance="high" /> : null}
          {progress?.correctAttempts ? (
            <Badge tone="success" title="Solved before">
              ✓
            </Badge>
          ) : null}
          <button
            type="button"
            onClick={() => toggleBookmark(bookmarkKey)}
            aria-pressed={bookmarked}
            aria-label={bookmarked ? "Remove bookmark" : "Bookmark this exercise"}
            className={cn(
              "rounded px-1 text-sm transition-colors",
              bookmarked ? "text-accent" : "text-faint hover:text-text",
            )}
          >
            {bookmarked ? "★" : "☆"}
          </button>
        </span>
      </header>

      <div className="px-4 py-4">
        <p className="text-[15px] leading-relaxed text-text">
          <InlineText>{exercise.question}</InlineText>
        </p>

        {exercise.context ? (
          <CodeBlock
            code={exercise.context}
            className="mt-3"
            label={exercise.type === "debug" ? "given" : "context"}
            tone={exercise.type === "debug" ? "danger" : "default"}
          />
        ) : null}

        <div className="mt-4">
          {exercise.type === "multiple-choice" ? (
            <fieldset disabled={answered}>
              <legend className="sr-only">Choose an answer</legend>
              <div className="space-y-1.5">
                {exercise.options?.map((option, optionIndex) => {
                  const isChosen = choice === optionIndex;
                  const isRight = answered && optionIndex === exercise.correctOption;
                  const isWrongChoice = answered && isChosen && !isRight;
                  return (
                    <label
                      key={option}
                      className={cn(
                        "flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2.5 text-sm transition-colors",
                        isRight
                          ? "border-success/50 bg-success-soft"
                          : isWrongChoice
                            ? "border-danger/50 bg-danger-soft"
                            : isChosen
                              ? "border-accent bg-accent-soft"
                              : "border-border-base hover:border-border-strong hover:bg-surface-hover",
                        answered && "cursor-default",
                      )}
                    >
                      <input
                        type="radio"
                        name={`mc-${exercise.id}`}
                        checked={isChosen}
                        onChange={() => setChoice(optionIndex)}
                        className="mt-0.5 accent-[var(--accent)]"
                      />
                      <span className="font-mono text-xs leading-5 text-faint">
                        {String.fromCharCode(65 + optionIndex)}
                      </span>
                      <span className="flex-1 text-text">
                        <InlineText>{option}</InlineText>
                      </span>
                      {isRight ? <span className="text-success">✓</span> : null}
                    </label>
                  );
                })}
              </div>
            </fieldset>
          ) : exercise.type === "ordering" ? (
            <OrderingInput
              blocks={exercise.blocks ?? []}
              order={order}
              onChange={setOrder}
              disabled={answered}
            />
          ) : exercise.type === "shape" || exercise.type === "fill-blank" ? (
            <div>
              <label htmlFor={`answer-${exercise.id}`} className="sr-only">
                Your answer
              </label>
              <input
                id={`answer-${exercise.id}`}
                value={answer}
                onChange={(event) => setAnswer(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    check();
                  }
                }}
                disabled={answered}
                autoFocus={autoFocus}
                spellCheck={false}
                autoComplete="off"
                placeholder={exercise.type === "shape" ? "(32, 128)" : "your answer"}
                className={cn(
                  "w-full max-w-sm rounded-lg border bg-[var(--code-bg)] px-3 py-2 font-mono text-sm text-text outline-none transition-colors placeholder:text-faint",
                  result?.correct
                    ? "border-success/50"
                    : result
                      ? "border-danger/50"
                      : "border-[var(--code-border)] focus:border-accent",
                )}
              />
            </div>
          ) : (
            <CodeEditor
              value={answer}
              onChange={setAnswer}
              onSubmit={check}
              disabled={answered}
              autoFocus={autoFocus}
              minRows={exercise.type === "memory" ? 8 : 3}
              tone={result?.correct ? "correct" : result ? "incorrect" : "default"}
              placeholder={exercise.type === "memory" ? "Write it from memory…" : "Type your PyTorch code…"}
            />
          )}
        </div>

        {showHint && exercise.hint && !answered ? (
          <p className="mt-3 rounded-lg border border-info/25 bg-info-soft px-3 py-2 text-sm text-text animate-fade-in-up">
            <span className="font-semibold text-info">Hint · </span>
            {exercise.hint}
          </p>
        ) : null}

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Button variant="primary" onClick={check} disabled={!canCheck || answered}>
            Check answer
            <kbd className="ml-1 hidden font-mono text-[10px] opacity-70 sm:inline">⌘↵</kbd>
          </Button>

          {!examMode && exercise.hint && !answered ? (
            <Button onClick={revealHint} disabled={showHint}>
              Hint
              <kbd className="ml-1 hidden font-mono text-[10px] opacity-60 sm:inline">H</kbd>
            </Button>
          ) : null}

          {!examMode && !answered ? (
            <Button onClick={reveal} disabled={attempts === 0} title={attempts === 0 ? "Try once first" : undefined}>
              Show answer
            </Button>
          ) : null}

          {answered && !examMode ? (
            <Button
              onClick={() => {
                setResult(null);
                setRevealed(false);
                setAnswer(exercise.starterCode ?? "");
                setChoice(null);
                setOrder([]);
                setShowHint(false);
              }}
            >
              Try again
            </Button>
          ) : null}

          <div className="ml-auto flex items-center gap-2">
            {onPrevious ? (
              <Button variant="ghost" onClick={onPrevious} aria-label="Previous exercise">
                ← Prev
              </Button>
            ) : null}
            {onNext ? (
              <Button variant={answered ? "primary" : "secondary"} onClick={onNext}>
                Next
                <kbd className="ml-1 hidden font-mono text-[10px] opacity-60 sm:inline">N</kbd>
              </Button>
            ) : null}
          </div>
        </div>

        {!examMode ? (
          <ExerciseFeedback
            result={result}
            exercise={exercise}
            revealed={revealed}
            onRate={rate}
            rated={rated}
          />
        ) : result ? (
          <p
            className={cn(
              "mt-3 text-sm font-semibold",
              result.correct ? "text-success" : "text-danger",
            )}
            role="status"
          >
            {result.correct ? "✓ Correct" : "✗ Recorded — review after the challenge"}
          </p>
        ) : null}

        {exercise.type === "ordering" && answered && !examMode ? (
          <div className="mt-3 rounded-lg border border-border-base bg-surface-2 px-3 py-2">
            <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-faint">
              Correct order
            </p>
            <pre className="overflow-x-auto font-mono text-xs">
              <HighlightedCode
                code={(exercise.correctOrder ?? [])
                  .map((i) => exercise.blocks?.[i] ?? "")
                  .join("\n")}
              />
            </pre>
          </div>
        ) : null}
      </div>
    </article>
  );
}
