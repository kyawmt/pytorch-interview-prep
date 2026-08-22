"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
import { PageHeader } from "@/components/common/PageHeader";
import { InterviewQuestionCard } from "@/components/interview/InterviewQuestionCard";
import { INTERVIEW_CATEGORIES, INTERVIEW_QUESTIONS } from "@/data/interview";
import { useAppState } from "@/hooks/useAppState";
import { cn } from "@/lib/utils";

type StatusFilter = "all" | "unseen" | "knew" | "review" | "bookmarked";

function InterviewInner() {
  const state = useAppState();
  const searchParams = useSearchParams();
  const focusId = searchParams.get("q");
  const [category, setCategory] = useState<string>("all");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [highOnly, setHighOnly] = useState(false);
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return INTERVIEW_QUESTIONS.filter((question) => {
      if (category !== "all" && question.category !== category) return false;
      if (highOnly && question.importance !== "high") return false;
      const progress = state.interview[question.id];
      if (status === "unseen" && progress) return false;
      if (status === "knew" && progress?.status !== "knew") return false;
      if (status === "review" && progress?.status !== "review") return false;
      if (status === "bookmarked" && !state.bookmarks.includes(`iq:${question.id}`)) return false;
      if (!needle) return true;
      return [question.question, question.shortAnswer, question.detailedAnswer, ...(question.tags ?? [])]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });
  }, [category, status, highOnly, query, state]);

  const known = Object.values(state.interview).filter((entry) => entry.status === "knew").length;

  return (
    <div>
      <PageHeader
        eyebrow="Interview questions"
        title="Practise the spoken answer, not just the code"
        description={`${INTERVIEW_QUESTIONS.length} questions taken from real AI/ML engineer interviews. Each one gives a concise answer you could actually say out loud, a deeper explanation for study, and the related PyTorch syntax.`}
        actions={
          <span className="rounded-lg border border-border-base bg-surface px-3 py-2 text-xs text-muted">
            <span className="font-mono text-text">{known}</span> / {INTERVIEW_QUESTIONS.length} marked known
          </span>
        }
      />

      <div className="mb-5 flex flex-wrap items-center gap-2">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter questions…"
          aria-label="Filter questions"
          className="min-w-[180px] flex-1 rounded-lg border border-border-base bg-surface px-3 py-2 text-sm outline-none transition-colors focus:border-accent placeholder:text-faint"
        />
        <select
          value={category}
          onChange={(event) => setCategory(event.target.value)}
          aria-label="Category"
          className="rounded-lg border border-border-base bg-surface px-2.5 py-2 text-xs text-text outline-none hover:border-border-strong focus:border-accent"
        >
          <option value="all">All categories</option>
          {INTERVIEW_CATEGORIES.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <select
          value={status}
          onChange={(event) => setStatus(event.target.value as StatusFilter)}
          aria-label="Status"
          className="rounded-lg border border-border-base bg-surface px-2.5 py-2 text-xs text-text outline-none hover:border-border-strong focus:border-accent"
        >
          <option value="all">All statuses</option>
          <option value="unseen">Not seen yet</option>
          <option value="knew">Marked known</option>
          <option value="review">Needs review</option>
          <option value="bookmarked">Bookmarked</option>
        </select>
        <button
          type="button"
          onClick={() => setHighOnly((value) => !value)}
          aria-pressed={highOnly}
          className={cn(
            "rounded-lg border px-2.5 py-2 text-xs font-medium transition-colors",
            highOnly
              ? "border-accent bg-accent-soft text-accent"
              : "border-border-base bg-surface text-muted hover:text-text",
          )}
        >
          🔥 High priority
        </button>
        <span className="ml-auto text-xs text-faint">{filtered.length} questions</span>
      </div>

      {filtered.length === 0 ? (
        <p className="card px-6 py-10 text-center text-sm text-muted">
          No questions match those filters.
        </p>
      ) : (
        <div className="space-y-2.5">
          {filtered.map((question) => (
            <InterviewQuestionCard
              key={question.id}
              question={question}
              defaultOpen={question.id === focusId}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function InterviewPage() {
  return (
    <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
      <InterviewInner />
    </Suspense>
  );
}
