import Link from "next/link";
import { notFound } from "next/navigation";
import { LinkButton } from "@/components/common/Button";
import { PageHeader } from "@/components/common/PageHeader";
import { CheatSheetBrowser } from "@/components/cheatsheet/CheatSheetBrowser";
import { cheatSheetByTopic } from "@/data/cheatsheet";
import { exercisesByTopic } from "@/data/exercises";
import { TOPICS, TOPIC_MAP } from "@/data/topics";
import type { TopicId } from "@/lib/types";

/** Only topics that actually have cheat-sheet content get a page. */
const CHEATSHEET_TOPICS = TOPICS.filter((topic) => cheatSheetByTopic(topic.id).length > 0);

export function generateStaticParams() {
  return CHEATSHEET_TOPICS.map((topic) => ({ topic: topic.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ topic: string }> }) {
  const { topic } = await params;
  const meta = TOPIC_MAP[topic as TopicId];
  return meta ? { title: `${meta.title} — Cheat Sheet`, description: meta.summary } : {};
}

export default async function CheatSheetTopicPage({
  params,
}: {
  params: Promise<{ topic: string }>;
}) {
  const { topic } = await params;
  const meta = TOPIC_MAP[topic as TopicId];
  const entries = cheatSheetByTopic(topic as TopicId);
  if (!meta || entries.length === 0) notFound();

  const exerciseCount = exercisesByTopic(topic as TopicId).length;
  const order = CHEATSHEET_TOPICS.findIndex((item) => item.id === topic);
  const previous = CHEATSHEET_TOPICS[order - 1];
  const next = CHEATSHEET_TOPICS[order + 1];

  return (
    <div>
      <nav aria-label="Breadcrumb" className="mb-3 text-xs text-muted">
        <Link href="/cheatsheet" className="hover:text-accent">
          Cheat sheet
        </Link>
        <span className="mx-1.5 text-faint">/</span>
        <span className="text-text">{meta.title}</span>
      </nav>

      <PageHeader
        title={meta.title}
        description={meta.summary}
        actions={
          exerciseCount > 0 ? (
            <LinkButton href={`/practice/${topic}`} variant="primary">
              Practise {exerciseCount} exercises
            </LinkButton>
          ) : null
        }
      />

      <CheatSheetBrowser entries={entries} />

      <nav className="mt-8 flex items-center justify-between border-t border-border-base pt-4 text-sm">
        {previous ? (
          <Link href={`/cheatsheet/${previous.id}`} className="text-muted hover:text-accent">
            ← {previous.title}
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/cheatsheet/${next.id}`} className="text-muted hover:text-accent">
            {next.title} →
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </div>
  );
}
