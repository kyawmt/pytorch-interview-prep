"use client";

import { useState } from "react";
import { Badge } from "@/components/common/Badge";
import { Button } from "@/components/common/Button";
import { CodeBlock } from "@/components/common/CodeBlock";
import { InlineParagraphs, InlineText } from "@/components/common/InlineCode";
import { topicTitle } from "@/data/topics";
import { useAppState } from "@/hooks/useAppState";
import { setInterviewStatus, toggleBookmark } from "@/lib/storage";
import type { InterviewQuestion } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Think → reveal → self-assess. The two-stage answer (spoken version first,
 * detail second) mirrors how the question is actually answered in an interview.
 */
export function InterviewQuestionCard({
  question,
  defaultOpen = false,
}: {
  question: InterviewQuestion;
  defaultOpen?: boolean;
}) {
  const state = useAppState();
  const [revealed, setRevealed] = useState(defaultOpen);
  const progress = state.interview[question.id];
  const bookmarkKey = `iq:${question.id}`;
  const bookmarked = state.bookmarks.includes(bookmarkKey);

  return (
    <article id={question.id} className="card scroll-mt-20 overflow-hidden">
      <div className="px-4 py-3.5">
        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          <Badge tone="neutral">{question.category}</Badge>
          <Badge tone="neutral">{topicTitle(question.topic)}</Badge>
          {question.importance === "high" ? <Badge tone="accent">🔥 Interview essential</Badge> : null}
          {progress?.status === "knew" ? <Badge tone="success">Knew it</Badge> : null}
          {progress?.status === "review" ? <Badge tone="warning">Needs review</Badge> : null}
          <button
            type="button"
            onClick={() => toggleBookmark(bookmarkKey)}
            aria-pressed={bookmarked}
            aria-label={bookmarked ? "Remove bookmark" : "Bookmark this question"}
            className={cn(
              "ml-auto rounded px-1 text-sm transition-colors",
              bookmarked ? "text-accent" : "text-faint hover:text-text",
            )}
          >
            {bookmarked ? "★" : "☆"}
          </button>
        </div>

        <h3 className="text-[15px] font-medium leading-relaxed text-text">
          <InlineText>{question.question}</InlineText>
        </h3>

        {!revealed ? (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <p className="flex-1 text-xs text-muted">
              Answer it out loud first — then check yourself.
            </p>
            <Button variant="primary" size="sm" onClick={() => setRevealed(true)}>
              Reveal answer
            </Button>
          </div>
        ) : (
          <div className="mt-3 space-y-3 animate-fade-in-up">
            <div className="rounded-lg border border-success/25 bg-success-soft px-3 py-2.5">
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-success">
                Say this
              </p>
              <InlineParagraphs className="text-sm leading-relaxed text-text">
                {question.shortAnswer}
              </InlineParagraphs>
            </div>

            <div>
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-faint">
                Deeper explanation
              </p>
              <InlineParagraphs className="text-sm leading-relaxed text-muted">
                {question.detailedAnswer}
              </InlineParagraphs>
            </div>

            {question.code ? (
              <div>
                <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-faint">
                  Related syntax
                </p>
                <CodeBlock code={question.code} />
              </div>
            ) : null}

            {question.followUps?.length ? (
              <div>
                <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-faint">
                  Likely follow-ups
                </p>
                <ul className="space-y-0.5">
                  {question.followUps.map((followUp) => (
                    <li key={followUp} className="text-xs text-muted">
                      → {followUp}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-2 border-t border-border-base pt-3">
              <Button
                size="sm"
                onClick={() => setInterviewStatus(question.id, "knew")}
                className={cn(progress?.status === "knew" && "border-success text-success")}
              >
                ✓ I knew this
              </Button>
              <Button
                size="sm"
                onClick={() => setInterviewStatus(question.id, "review")}
                className={cn(progress?.status === "review" && "border-warning text-warning")}
              >
                ↺ Need review
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setRevealed(false)}>
                Hide
              </Button>
            </div>
          </div>
        )}
      </div>
    </article>
  );
}
