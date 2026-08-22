"use client";

import Link from "next/link";
import { Badge } from "@/components/common/Badge";
import { LinkButton } from "@/components/common/Button";
import { PageHeader } from "@/components/common/PageHeader";
import { ProgressBar } from "@/components/progress/ProgressBar";
import { StatsCard } from "@/components/progress/StatsCard";
import { TopicProgress } from "@/components/progress/TopicProgress";
import { EXERCISES } from "@/data/exercises";
import { INTERVIEW_QUESTIONS } from "@/data/interview";
import { PROJECTS } from "@/data/projects";
import { CHEATSHEET } from "@/data/cheatsheet";
import { useAppState, useHydrated } from "@/hooks/useAppState";
import { allTopicStats, continueLearning, overallStats, weakAreas } from "@/lib/progress";

const QUICK_LINKS = [
  { href: "/cheatsheet", label: "Cheat sheet", detail: `${CHEATSHEET.length} syntax cards`, icon: "▤" },
  { href: "/practice", label: "Practice", detail: `${EXERCISES.length} exercises`, icon: "▶" },
  { href: "/interview", label: "Interview questions", detail: `${INTERVIEW_QUESTIONS.length} answers to rehearse`, icon: "◎" },
  { href: "/projects", label: "Mini projects", detail: `${PROJECTS.length} guided builds`, icon: "▦" },
  { href: "/shapes", label: "Shape playground", detail: "Predict tensor shapes", icon: "⤢" },
  { href: "/rapid-review", label: "15-minute review", detail: "Read before the interview", icon: "⚡" },
];

export default function DashboardPage() {
  const state = useAppState();
  const hydrated = useHydrated();
  const stats = overallStats(state);
  const topics = allTopicStats(state);
  const weak = weakAreas(state, 4);
  const next = continueLearning(state);
  const started = stats.attempted > 0;

  return (
    <div>
      <PageHeader
        eyebrow="Dashboard"
        title={started ? "Keep the streak going" : "Start with active recall"}
        description={
          started
            ? "Your weakest topics are surfaced below. Ten focused minutes on those beats an hour of re-reading."
            : "This site is built around one loop: see a concept, hide the syntax, recall it, type it, check it. Pick a starting point below."
        }
        actions={
          <>
            <LinkButton href="/daily" variant="primary">
              Daily practice
            </LinkButton>
            <LinkButton href="/challenge">Interview challenge</LinkButton>
          </>
        }
      />

      <section aria-label="Statistics" className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatsCard label="Solved" value={hydrated ? stats.solved : "—"} hint={`of ${stats.totalExercises}`} href="/practice" />
        <StatsCard label="Streak" value={hydrated ? `${stats.streak}d` : "—"} hint={`best ${stats.longestStreak}d`} tone="accent" />
        <StatsCard label="Accuracy" value={hydrated ? `${stats.accuracy}%` : "—"} hint="all attempts" />
        <StatsCard label="Topics mastered" value={hydrated ? stats.topicsMastered : "—"} hint={`of ${topics.length}`} href="/progress" />
        <StatsCard
          label="Needs review"
          value={hydrated ? stats.needingReview : "—"}
          hint="wrong or hinted"
          href="/review"
          tone={stats.needingReview > 0 ? "danger" : "default"}
        />
        <StatsCard
          label="Projects"
          value={hydrated ? `${stats.projectsCompleted}/${PROJECTS.length}` : "—"}
          hint={`${stats.projectSteps}/${stats.projectStepsTotal} steps`}
          href="/projects"
        />
      </section>

      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        <section className="card p-4 lg:col-span-2" aria-labelledby="topic-progress-heading">
          <div className="mb-3 flex items-baseline justify-between">
            <h2 id="topic-progress-heading" className="text-sm font-semibold text-text">
              Topic mastery
            </h2>
            <Link href="/progress" className="text-xs text-muted hover:text-accent">
              Full breakdown →
            </Link>
          </div>
          <ProgressBar value={stats.overallProgress} label="Overall progress" className="mb-3" />
          <div className="-mx-2">
            {topics.slice(0, 9).map((topic) => (
              <TopicProgress key={topic.topic} stats={topic} />
            ))}
          </div>
        </section>

        <div className="space-y-5">
          <section className="card p-4" aria-labelledby="continue-heading">
            <h2 id="continue-heading" className="text-sm font-semibold text-text">
              Continue learning
            </h2>
            <p className="mt-2.5 text-lg font-semibold text-accent">{next.title}</p>
            <p className="text-xs text-muted">
              {next.correct} / {next.total} exercises completed
            </p>
            <ProgressBar value={next.completion} showValue={false} size="sm" className="mt-2.5" />
            <div className="mt-3.5 flex gap-2">
              <LinkButton href={`/practice/${next.topic}`} variant="primary" size="sm">
                Practise
              </LinkButton>
              <LinkButton href={`/cheatsheet/${next.topic}`} size="sm">
                Cheat sheet
              </LinkButton>
            </div>
          </section>

          <section className="card p-4" aria-labelledby="weak-heading">
            <h2 id="weak-heading" className="text-sm font-semibold text-text">
              Weak areas
            </h2>
            {weak.length === 0 ? (
              <p className="mt-2.5 text-xs text-muted">
                {started
                  ? "Nothing is flagged yet — keep practising and weak spots will surface here automatically."
                  : "Answer a few exercises and your weakest topics will be listed here."}
              </p>
            ) : (
              <ol className="mt-2.5 space-y-1.5">
                {weak.map((topic, index) => (
                  <li key={topic.topic}>
                    <Link
                      href={`/practice/${topic.topic}`}
                      className="flex items-center gap-2 rounded-md px-1.5 py-1 text-sm transition-colors hover:bg-surface-hover"
                    >
                      <span className="font-mono text-xs text-faint">{index + 1}.</span>
                      <span className="flex-1 truncate text-text">{topic.title}</span>
                      <Badge tone={topic.mastery < 40 ? "danger" : "warning"}>{topic.mastery}%</Badge>
                    </Link>
                  </li>
                ))}
              </ol>
            )}
            {stats.needingReview > 0 ? (
              <LinkButton href="/review" size="sm" className="mt-3">
                Practise {stats.needingReview} weak {stats.needingReview === 1 ? "question" : "questions"}
              </LinkButton>
            ) : null}
          </section>
        </div>
      </div>

      <section className="mt-6" aria-labelledby="quick-links-heading">
        <h2 id="quick-links-heading" className="mb-3 text-sm font-semibold text-text">
          Jump in
        </h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {QUICK_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="card flex items-center gap-3 p-3.5 transition-colors hover:border-border-strong hover:bg-surface-hover"
            >
              <span aria-hidden className="text-lg text-accent">
                {link.icon}
              </span>
              <span className="min-w-0">
                <span className="block text-sm font-medium text-text">{link.label}</span>
                <span className="block truncate text-xs text-muted">{link.detail}</span>
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="mt-6 card p-4" aria-labelledby="loop-heading">
        <h2 id="loop-heading" className="text-sm font-semibold text-text">
          How to use this site
        </h2>
        <ol className="mt-3 grid gap-2 text-xs text-muted sm:grid-cols-3 lg:grid-cols-6">
          {[
            "See the concept",
            "Hide the syntax",
            "Recall it",
            "Type the code",
            "Check the answer",
            "Review what you missed",
          ].map((step, index) => (
            <li key={step} className="rounded-lg border border-border-base bg-surface-2 px-2.5 py-2">
              <span className="font-mono text-accent">{index + 1}</span>
              <span className="mt-0.5 block text-text">{step}</span>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
