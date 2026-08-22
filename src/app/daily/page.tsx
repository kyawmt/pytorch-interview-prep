"use client";

import { useMemo, useState } from "react";
import { LinkButton } from "@/components/common/Button";
import { PageHeader } from "@/components/common/PageHeader";
import { ExerciseCard } from "@/components/exercises/ExerciseCard";
import { InterviewQuestionCard } from "@/components/interview/InterviewQuestionCard";
import { ProgressBar } from "@/components/progress/ProgressBar";
import { useAppState, useHydrated } from "@/hooks/useAppState";
import { buildDailySession } from "@/lib/session";
import { todayKey } from "@/lib/storage";
import { cn } from "@/lib/utils";

type Stage = "syntax" | "concepts" | "interview" | "done";

const STAGE_LABEL: Record<Exclude<Stage, "done">, string> = {
  syntax: "Syntax recall",
  concepts: "Concepts",
  interview: "Interview questions",
};

export default function DailyPracticePage() {
  const state = useAppState();
  const hydrated = useHydrated();
  // The page is prerendered, so "today" must come from the client only.
  const dateKey = hydrated ? todayKey() : "";

  // Built once per day per browser: stable across reloads, fresh each morning.
  const session = useMemo(() => buildDailySession(state, dateKey), [dateKey, hydrated]); // eslint-disable-line react-hooks/exhaustive-deps

  const [stage, setStage] = useState<Stage>("syntax");
  const [index, setIndex] = useState(0);

  const exercises = stage === "syntax" ? session.syntax : stage === "concepts" ? session.concepts : [];
  const totalItems = session.syntax.length + session.concepts.length + session.interview.length;
  const doneItems =
    stage === "syntax"
      ? index
      : stage === "concepts"
        ? session.syntax.length + index
        : stage === "interview"
          ? session.syntax.length + session.concepts.length
          : totalItems;

  function advance() {
    if (stage === "interview") {
      setStage("done");
      return;
    }
    if (index + 1 < exercises.length) {
      setIndex(index + 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    setIndex(0);
    setStage(stage === "syntax" ? "concepts" : "interview");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div>
      <PageHeader
        eyebrow="Daily practice"
        title="Today's session"
        description="Five syntax questions, three concept questions and two interview questions — chosen from what you got wrong, what is due for review and what you have not seen. The set is fixed for today, so a reload will not reshuffle it."
      />

      <div className="mb-5 flex flex-wrap items-center gap-2">
        {(["syntax", "concepts", "interview"] as const).map((item) => (
          <span
            key={item}
            className={cn(
              "rounded-lg border px-2.5 py-1.5 text-xs font-medium",
              stage === item
                ? "border-accent bg-accent-soft text-accent"
                : "border-border-base bg-surface text-muted",
            )}
          >
            {STAGE_LABEL[item]}
            <span className="ml-1.5 font-mono text-[10px] opacity-70">
              {item === "syntax"
                ? session.syntax.length
                : item === "concepts"
                  ? session.concepts.length
                  : session.interview.length}
            </span>
          </span>
        ))}
        <span className="ml-auto font-mono text-xs text-faint">{dateKey || "…"}</span>
      </div>

      <ProgressBar
        value={(doneItems / Math.max(totalItems, 1)) * 100}
        showValue={false}
        size="sm"
        className="mb-4"
      />

      {stage === "done" ? (
        <div className="card px-6 py-12 text-center">
          <p className="text-3xl" aria-hidden>
            ✓
          </p>
          <h2 className="mt-2 text-lg font-semibold text-text">Session complete</h2>
          <p className="mt-1 text-sm text-muted">
            That is today&rsquo;s ten. Come back tomorrow for a fresh set, or keep going below.
          </p>
          <div className="mt-5 flex flex-wrap justify-center gap-2">
            <LinkButton href="/review" variant="primary">
              Review weak questions
            </LinkButton>
            <LinkButton href="/challenge">Timed challenge</LinkButton>
            <LinkButton href="/practice">Free practice</LinkButton>
          </div>
        </div>
      ) : stage === "interview" ? (
        <div className="space-y-3">
          {session.interview.map((question) => (
            <InterviewQuestionCard key={question.id} question={question} />
          ))}
          <div className="flex justify-end">
            <button
              type="button"
              onClick={advance}
              className="rounded-lg bg-accent px-3.5 py-2 text-sm font-medium text-accent-text transition-colors hover:bg-accent-hover"
            >
              Finish session
            </button>
          </div>
        </div>
      ) : exercises.length === 0 ? (
        <div className="card px-6 py-12 text-center">
          <p className="text-sm text-muted">Nothing scheduled for this stage.</p>
          <button
            type="button"
            onClick={advance}
            className="mt-4 rounded-lg bg-accent px-3.5 py-2 text-sm font-medium text-accent-text"
          >
            Continue
          </button>
        </div>
      ) : (
        <ExerciseCard
          key={exercises[index].id}
          exercise={exercises[index]}
          index={index}
          total={exercises.length}
          onNext={advance}
        />
      )}
    </div>
  );
}
