"use client";

import { useState } from "react";
import { Button } from "@/components/common/Button";
import { CodeBlock } from "@/components/common/CodeBlock";
import { InlineText } from "@/components/common/InlineCode";
import { CodeEditor } from "@/components/exercises/CodeEditor";
import { setProjectStep } from "@/lib/storage";
import type { ProjectStep } from "@/lib/types";
import { validateProjectStep, type ValidationResult } from "@/lib/validation";
import { cn } from "@/lib/utils";

export function ProjectStepCard({
  step,
  slug,
  index,
  total,
  done,
  onDone,
}: {
  step: ProjectStep;
  slug: string;
  index: number;
  total: number;
  done: boolean;
  onDone: () => void;
}) {
  const [answer, setAnswer] = useState(step.starterCode ?? "");
  const [choice, setChoice] = useState<number | null>(null);
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [attempts, setAttempts] = useState(0);

  // ProjectRunner renders this with key={step.id}, so changing step remounts
  // the component and resets the state above.

  const solved = result?.correct === true;

  function check() {
    const value = step.kind === "multiple-choice" ? (choice ?? -1) : answer;
    const validation = validateProjectStep(value, step);
    setResult(validation);
    setAttempts((n) => n + 1);
    if (validation.correct) {
      setProjectStep(slug, step.id, true);
      onDone();
    }
  }

  function reveal() {
    setRevealed(true);
    setProjectStep(slug, step.id, true);
    onDone();
  }

  return (
    <section className="card overflow-hidden">
      <header className="flex flex-wrap items-center gap-2 border-b border-border-base bg-surface-2 px-4 py-2.5">
        <span
          className={cn(
            "flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-semibold",
            done || solved ? "bg-success text-white" : "bg-surface text-muted ring-1 ring-border-base",
          )}
          aria-hidden
        >
          {done || solved ? "✓" : index + 1}
        </span>
        <h2 className="text-sm font-medium text-text">{step.title}</h2>
        <span className="ml-auto font-mono text-xs text-faint">
          Step {index + 1}/{total}
        </span>
      </header>

      <div className="px-4 py-4">
        <p className="text-sm leading-relaxed text-text">
          <InlineText>{step.brief}</InlineText>
        </p>

        {step.context ? <CodeBlock code={step.context} className="mt-3" label="context" /> : null}

        <div className="mt-3.5">
          {step.kind === "multiple-choice" ? (
            <fieldset disabled={solved || revealed}>
              <legend className="sr-only">Choose an answer</legend>
              <div className="space-y-1.5">
                {step.options?.map((option, optionIndex) => {
                  const chosen = choice === optionIndex;
                  const isRight = (solved || revealed) && optionIndex === step.correctOption;
                  return (
                    <label
                      key={option}
                      className={cn(
                        "flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2.5 text-sm transition-colors",
                        isRight
                          ? "border-success/50 bg-success-soft"
                          : chosen
                            ? "border-accent bg-accent-soft"
                            : "border-border-base hover:border-border-strong",
                      )}
                    >
                      <input
                        type="radio"
                        name={`step-${step.id}`}
                        checked={chosen}
                        onChange={() => setChoice(optionIndex)}
                        className="mt-0.5 accent-[var(--accent)]"
                      />
                      <span className="flex-1 text-text">
                        <InlineText>{option}</InlineText>
                      </span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
          ) : step.kind === "shape" ? (
            <input
              value={answer}
              onChange={(event) => setAnswer(event.target.value)}
              disabled={solved || revealed}
              placeholder="(32, 128)"
              aria-label="Your answer"
              spellCheck={false}
              className="w-full max-w-sm rounded-lg border border-[var(--code-border)] bg-[var(--code-bg)] px-3 py-2 font-mono text-sm outline-none focus:border-accent"
            />
          ) : (
            <CodeEditor
              value={answer}
              onChange={setAnswer}
              onSubmit={check}
              disabled={solved || revealed}
              minRows={4}
              tone={result?.correct ? "correct" : result ? "incorrect" : "default"}
            />
          )}
        </div>

        {showHint && step.hint && !solved && !revealed ? (
          <p className="mt-3 rounded-lg border border-info/25 bg-info-soft px-3 py-2 text-sm text-text">
            <span className="font-semibold text-info">Hint · </span>
            {step.hint}
          </p>
        ) : null}

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Button variant="primary" onClick={check} disabled={solved || revealed}>
            Check
          </Button>
          {step.hint && !solved && !revealed ? (
            <Button onClick={() => setShowHint(true)} disabled={showHint}>
              Hint
            </Button>
          ) : null}
          {!solved && !revealed ? (
            <Button onClick={reveal} disabled={attempts === 0} title={attempts === 0 ? "Try once first" : undefined}>
              Show solution
            </Button>
          ) : null}
        </div>

        {result && !result.correct && !revealed ? (
          <div className="mt-3 rounded-lg border border-danger/35 bg-danger-soft px-3 py-2.5 animate-shake">
            <p className="text-sm font-semibold text-danger">✗ {result.feedback}</p>
            {result.nearMiss ? <p className="mt-1 text-xs text-muted">{result.nearMiss}</p> : null}
            {result.missing?.length ? (
              <ul className="mt-1.5 space-y-0.5">
                {result.missing.map((item) => (
                  <li key={item} className="font-mono text-xs text-danger">
                    · missing {item}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}

        {solved || revealed ? (
          <div
            className={cn(
              "mt-3 rounded-lg border px-3 py-3",
              solved ? "border-success/35 bg-success-soft animate-pop" : "border-border-base bg-surface-2",
            )}
          >
            {solved ? <p className="mb-2 text-sm font-semibold text-success">✓ Correct</p> : null}
            <CodeBlock code={step.solution} label="solution" />
            <p className="mt-2.5 text-sm leading-relaxed text-text">{step.explanation}</p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
