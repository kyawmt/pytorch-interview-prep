"use client";

import { useState } from "react";
import { Button, LinkButton } from "@/components/common/Button";
import { CodeBlock } from "@/components/common/CodeBlock";
import { PageHeader } from "@/components/common/PageHeader";
import { ProgressBar } from "@/components/progress/ProgressBar";
import { REVISION_CARDS } from "@/data/revision";
import { topicTitle } from "@/data/topics";
import { cn } from "@/lib/utils";

export default function RapidReviewPage() {
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [mode, setMode] = useState<"cards" | "list">("cards");

  const card = REVISION_CARDS[index];

  function advance(delta: number) {
    setFlipped(false);
    setIndex((current) => (current + delta + REVISION_CARDS.length) % REVISION_CARDS.length);
  }

  return (
    <div>
      <PageHeader
        eyebrow="15-minute review"
        title="Read this in the hour before the interview"
        description={`${REVISION_CARDS.length} cards covering everything that actually comes up. Try to answer each one out loud before flipping it — even a failed recall attempt is worth more than re-reading.`}
        actions={
          <div className="flex gap-1.5">
            <Button
              size="sm"
              onClick={() => setMode("cards")}
              className={cn(mode === "cards" && "border-accent text-accent")}
            >
              Flashcards
            </Button>
            <Button
              size="sm"
              onClick={() => setMode("list")}
              className={cn(mode === "list" && "border-accent text-accent")}
            >
              Read all
            </Button>
          </div>
        }
      />

      {mode === "cards" ? (
        <div className="mx-auto max-w-2xl">
          <ProgressBar
            value={((index + 1) / REVISION_CARDS.length) * 100}
            label={`Card ${index + 1} of ${REVISION_CARDS.length}`}
            showValue={false}
            size="sm"
            className="mb-4"
          />

          <article className="card min-h-[280px] p-6">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-accent">
              {topicTitle(card.topic)}
            </p>
            <h2 className="mt-2 text-lg font-medium leading-relaxed text-text">{card.prompt}</h2>

            {flipped ? (
              <div className="mt-4 space-y-3 animate-fade-in-up">
                <p className="text-sm leading-relaxed text-muted">{card.answer}</p>
                {card.code ? <CodeBlock code={card.code} /> : null}
              </div>
            ) : (
              <p className="mt-4 text-sm text-faint">Answer it out loud, then reveal.</p>
            )}
          </article>

          <div className="mt-4 flex items-center justify-between">
            <Button onClick={() => advance(-1)}>← Previous</Button>
            <Button variant="primary" onClick={() => setFlipped((value) => !value)}>
              {flipped ? "Hide answer" : "Reveal answer"}
            </Button>
            <Button onClick={() => advance(1)}>Next →</Button>
          </div>

          {index === REVISION_CARDS.length - 1 && flipped ? (
            <div className="card mt-5 px-6 py-6 text-center">
              <p className="text-sm text-text">That is the whole deck.</p>
              <div className="mt-3 flex flex-wrap justify-center gap-2">
                <LinkButton href="/challenge" variant="primary" size="sm">
                  Take the timed challenge
                </LinkButton>
                <LinkButton href="/mistakes" size="sm">
                  Common mistakes
                </LinkButton>
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="space-y-2.5">
          {REVISION_CARDS.map((item, itemIndex) => (
            <article key={item.id} className="card p-4">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-xs text-faint">
                  {String(itemIndex + 1).padStart(2, "0")}
                </span>
                <h2 className="text-sm font-medium text-text">{item.prompt}</h2>
                <span className="ml-auto text-[11px] text-faint">{topicTitle(item.topic)}</span>
              </div>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{item.answer}</p>
              {item.code ? <CodeBlock code={item.code} className="mt-2.5" /> : null}
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
