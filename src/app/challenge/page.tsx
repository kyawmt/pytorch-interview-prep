"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { Button, LinkButton } from "@/components/common/Button";
import { PageHeader } from "@/components/common/PageHeader";
import { ProgressBar } from "@/components/progress/ProgressBar";
import { ExerciseCard } from "@/components/exercises/ExerciseCard";
import { topicTitle } from "@/data/topics";
import { useAppState } from "@/hooks/useAppState";
import { buildChallenge, DEFAULT_CHALLENGE } from "@/lib/session";
import { saveChallengeResult } from "@/lib/storage";
import type { ChallengeResult, Exercise } from "@/lib/types";
import { cn, formatDuration, percent } from "@/lib/utils";

type Phase = "intro" | "running" | "done";

export default function ChallengePage() {
  const state = useAppState();
  const [phase, setPhase] = useState<Phase>("intro");
  const [seed, setSeed] = useState(0);
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, boolean>>({});
  const [secondsLeft, setSecondsLeft] = useState(DEFAULT_CHALLENGE.minutes * 60);
  const startedAt = useRef<number>(0);
  const [result, setResult] = useState<ChallengeResult | null>(null);

  const questions = useMemo<Exercise[]>(
    () => (seed === 0 ? [] : buildChallenge(seed)),
    [seed],
  );

  const finish = useCallback(
    (finalAnswers: Record<string, boolean>) => {
      const byTopic: Record<string, { correct: number; total: number }> = {};
      const missed: string[] = [];
      let score = 0;

      for (const question of questions) {
        const bucket = (byTopic[question.topic] ??= { correct: 0, total: 0 });
        bucket.total += 1;
        if (finalAnswers[question.id]) {
          bucket.correct += 1;
          score += 1;
        } else {
          missed.push(question.id);
        }
      }

      const payload: ChallengeResult = {
        id: `challenge-${Date.now()}`,
        date: new Date().toISOString(),
        score,
        total: questions.length,
        durationSeconds: Math.round((Date.now() - startedAt.current) / 1000),
        byTopic,
        missedExerciseIds: missed,
      };
      setResult(payload);
      saveChallengeResult(payload);
      setPhase("done");
    },
    [questions],
  );

  // Countdown; auto-submits when the clock runs out.
  useEffect(() => {
    if (phase !== "running") return;
    const timer = setInterval(() => {
      setSecondsLeft((remaining) => {
        if (remaining <= 1) {
          clearInterval(timer);
          finish(answers);
          return 0;
        }
        return remaining - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [phase, answers, finish]);

  function start() {
    setSeed(Date.now());
    setIndex(0);
    setAnswers({});
    setResult(null);
    setSecondsLeft(DEFAULT_CHALLENGE.minutes * 60);
    startedAt.current = Date.now();
    setPhase("running");
  }

  if (phase === "intro") {
    const previous = state.challenges[0];
    return (
      <div>
        <PageHeader
          eyebrow="Interview challenge"
          title="20 questions. 30 minutes. No hints."
          description="A balanced mix of syntax, shape reasoning, debugging scenarios, concepts and code completion — weighted toward the topics interviewers actually ask about. You get a per-topic breakdown and a recommended review list at the end."
        />

        <div className="card mx-auto max-w-lg px-6 py-8 text-center">
          <div className="flex justify-center gap-8">
            <div>
              <p className="font-mono text-3xl font-semibold text-accent">
                {DEFAULT_CHALLENGE.questionCount}
              </p>
              <p className="text-xs text-muted">questions</p>
            </div>
            <div>
              <p className="font-mono text-3xl font-semibold text-accent">
                {DEFAULT_CHALLENGE.minutes}
              </p>
              <p className="text-xs text-muted">minutes</p>
            </div>
          </div>

          <ul className="mx-auto mt-6 max-w-xs space-y-1 text-left text-xs text-muted">
            <li>· No hints, no revealing the solution</li>
            <li>· Answers are recorded either way</li>
            <li>· Everything you miss goes to your review queue</li>
          </ul>

          <Button variant="primary" className="mt-6 w-full" onClick={start}>
            Start the challenge
          </Button>

          {previous ? (
            <p className="mt-4 text-xs text-muted">
              Last attempt:{" "}
              <span className="font-mono text-text">
                {previous.score}/{previous.total}
              </span>{" "}
              in {formatDuration(previous.durationSeconds)}
            </p>
          ) : null}
        </div>
      </div>
    );
  }

  if (phase === "running") {
    const question = questions[index];
    const answered = Object.keys(answers).length;

    return (
      <div>
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <span
            className={cn(
              "rounded-lg border px-3 py-1.5 font-mono text-sm tabular-nums",
              secondsLeft < 120
                ? "border-danger/40 bg-danger-soft text-danger"
                : "border-border-base bg-surface text-text",
            )}
            role="timer"
            aria-live="off"
          >
            ⏱ {formatDuration(secondsLeft)}
          </span>
          <span className="text-xs text-muted">
            {answered} of {questions.length} answered
          </span>
          <Button
            size="sm"
            variant="ghost"
            className="ml-auto"
            onClick={() => {
              if (window.confirm("End the challenge and see your score?")) finish(answers);
            }}
          >
            Finish early
          </Button>
        </div>

        <ProgressBar
          value={((index + 1) / questions.length) * 100}
          showValue={false}
          size="sm"
          className="mb-4"
        />

        <ExerciseCard
          key={question.id}
          exercise={question}
          index={index}
          total={questions.length}
          examMode
          onAnswered={(correct) => setAnswers((current) => ({ ...current, [question.id]: correct }))}
          onNext={() => {
            if (index + 1 >= questions.length) finish(answers);
            else {
              setIndex(index + 1);
              window.scrollTo({ top: 0, behavior: "smooth" });
            }
          }}
        />
      </div>
    );
  }

  if (!result) return null;

  const pct = percent(result.score, result.total);
  const missed = result.missedExerciseIds
    .map((id) => questions.find((question) => question.id === id))
    .filter(Boolean) as Exercise[];

  return (
    <div>
      <PageHeader eyebrow="Challenge complete" title={`Score: ${result.score} / ${result.total}`} />

      <div className="grid gap-5 lg:grid-cols-2">
        <section className="card p-5">
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-4xl font-semibold text-accent">{pct}%</span>
            <span className="text-sm text-muted">in {formatDuration(result.durationSeconds)}</span>
          </div>
          <p className="mt-2 text-sm text-muted">
            {pct >= 90
              ? "Interview-ready on this material. Move to the mini projects and the from-memory drills."
              : pct >= 70
                ? "Solid. Close the gaps in the weakest topics below and retake it."
                : "Plenty of upside here — work through the recommended review list, then take it again."}
          </p>

          <h2 className="mt-5 mb-2 text-sm font-semibold text-text">By topic</h2>
          <div className="space-y-2">
            {Object.entries(result.byTopic)
              .sort((a, b) => a[1].correct / a[1].total - b[1].correct / b[1].total)
              .map(([topic, bucket]) => (
                <div key={topic}>
                  <div className="flex items-baseline justify-between text-xs">
                    <span className="text-text">{topicTitle(topic)}</span>
                    <span className="font-mono text-faint">
                      {bucket.correct}/{bucket.total}
                    </span>
                  </div>
                  <ProgressBar
                    value={percent(bucket.correct, bucket.total)}
                    showValue={false}
                    size="sm"
                    tone={bucket.correct === bucket.total ? "success" : "accent"}
                    className="mt-1"
                  />
                </div>
              ))}
          </div>
        </section>

        <section className="card p-5">
          <h2 className="mb-2.5 text-sm font-semibold text-text">Recommended review</h2>
          {missed.length === 0 ? (
            <p className="text-sm text-muted">Perfect score — nothing to review.</p>
          ) : (
            <ul className="space-y-1.5">
              {missed.map((exercise) => (
                <li key={exercise.id}>
                  <Link
                    href={`/practice/${exercise.topic}?exercise=${exercise.id}`}
                    className="flex items-center gap-2 rounded-lg border border-border-base bg-surface-2 px-3 py-2 text-sm transition-colors hover:border-accent"
                  >
                    <span className="min-w-0 flex-1 truncate text-text">{exercise.title}</span>
                    <Badge tone="neutral">{topicTitle(exercise.topic)}</Badge>
                  </Link>
                </li>
              ))}
            </ul>
          )}

          <div className="mt-5 flex flex-wrap gap-2">
            <Button variant="primary" onClick={start}>
              Take it again
            </Button>
            <LinkButton href="/review">Review queue</LinkButton>
            <LinkButton href="/progress">Progress</LinkButton>
          </div>
        </section>
      </div>
    </div>
  );
}
